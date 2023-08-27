// See LICENSE for license details.

#include "config.h"
#include "mmu.h"
#include "arith.h"
#include "simif.h"
#include "processor.h"

mmu_t::mmu_t(simif_t* sim, endianness_t endianness, processor_t* proc)
 : sim(sim), proc(proc),
#ifdef RISCV_ENABLE_DUAL_ENDIAN
  target_big_endian(endianness == endianness_big),
#endif
  check_triggers_fetch(false),
  check_triggers_load(false),
  check_triggers_store(false),
  matched_trigger(NULL)
{
#ifndef RISCV_ENABLE_DUAL_ENDIAN
  assert(endianness == endianness_little);
#endif
  flush_tlb();
  yield_load_reservation();
}

mmu_t::~mmu_t()
{
}

void mmu_t::flush_icache()
{
  for (size_t i = 0; i < ICACHE_ENTRIES; i++)
    icache[i].tag = -1;
}

void mmu_t::flush_tlb()
{
  memset(tlb_insn_tag, -1, sizeof(tlb_insn_tag));
  memset(tlb_load_tag, -1, sizeof(tlb_load_tag));
  memset(tlb_store_tag, -1, sizeof(tlb_store_tag));

  flush_icache();
}

void throw_access_exception(bool virt, reg_t addr, access_type type)
{
  switch (type) {
    case FETCH: throw trap_instruction_access_fault(virt, addr, 0, 0);
    case LOAD: throw trap_load_access_fault(virt, addr, 0, 0);
    case STORE: throw trap_store_access_fault(virt, addr, 0, 0);
    default: abort();
  }
}

reg_t mmu_t::translate(mem_access_info_t access_info, reg_t len)
{
  reg_t addr = access_info.vaddr;
  access_type type = access_info.type;
  if (!proc)
    return addr;

  bool virt = access_info.effective_virt;
  reg_t mode = (reg_t) access_info.effective_priv;

  reg_t paddr = walk(access_info) | (addr & (PGSIZE-1));
  if (!pmp_ok(paddr, len, type, mode))
    throw_access_exception(virt, addr, type);
  return paddr;
}

tlb_entry_t mmu_t::fetch_slow_path(reg_t vaddr)
{
  auto access_info = generate_access_info(vaddr, FETCH, {false, false, false});
  check_triggers(triggers::OPERATION_EXECUTE, vaddr, access_info.effective_virt);

  tlb_entry_t result;
  reg_t vpn = vaddr >> PGSHIFT;
  if (unlikely(tlb_insn_tag[vpn % TLB_ENTRIES] != (vpn | TLB_CHECK_TRIGGERS))) {
    reg_t paddr = translate(access_info, sizeof(fetch_temp));
    if (auto host_addr = sim->addr_to_mem(paddr)) {
      result = refill_tlb(vaddr, paddr, host_addr, FETCH);
    } else {
      if (!mmio_fetch(paddr, sizeof fetch_temp, (uint8_t*)&fetch_temp))
        throw trap_instruction_access_fault(proc->state.v, vaddr, 0, 0);
      result = {(char*)&fetch_temp - vaddr, paddr - vaddr};
    }
  } else {
    result = tlb_data[vpn % TLB_ENTRIES];
  }

  check_triggers(triggers::OPERATION_EXECUTE, vaddr, access_info.effective_virt, from_le(*(const uint16_t*)(result.host_offset + vaddr)));

  return result;
}

reg_t reg_from_bytes(size_t len, const uint8_t* bytes)
{
  switch (len) {
    case 1:
      return bytes[0];
    case 2:
      return bytes[0] |
        (((reg_t) bytes[1]) << 8);
    case 4:
      return bytes[0] |
        (((reg_t) bytes[1]) << 8) |
        (((reg_t) bytes[2]) << 16) |
        (((reg_t) bytes[3]) << 24);
    case 8:
      return bytes[0] |
        (((reg_t) bytes[1]) << 8) |
        (((reg_t) bytes[2]) << 16) |
        (((reg_t) bytes[3]) << 24) |
        (((reg_t) bytes[4]) << 32) |
        (((reg_t) bytes[5]) << 40) |
        (((reg_t) bytes[6]) << 48) |
        (((reg_t) bytes[7]) << 56);
  }
  abort();
}

bool mmu_t::mmio_ok(reg_t paddr, access_type UNUSED type)
{
  // Disallow access to debug region when not in debug mode
  if (paddr >= DEBUG_START && paddr <= DEBUG_END && proc && !proc->state.debug_mode)
    return false;

  return true;
}

bool mmu_t::mmio_fetch(reg_t paddr, size_t len, uint8_t* bytes)
{
  if (!mmio_ok(paddr, FETCH))
    return false;

  return sim->mmio_fetch(paddr, len, bytes);
}

bool mmu_t::mmio_load(reg_t paddr, size_t len, uint8_t* bytes)
{
  return mmio(paddr, len, bytes, LOAD);
}

bool mmu_t::mmio_store(reg_t paddr, size_t len, const uint8_t* bytes)
{
  return mmio(paddr, len, const_cast<uint8_t*>(bytes), STORE);
}

bool mmu_t::mmio(reg_t paddr, size_t len, uint8_t* bytes, access_type type)
{
  bool power_of_2 = (len & (len - 1)) == 0;
  bool naturally_aligned = (paddr & (len - 1)) == 0;

  if (power_of_2 && naturally_aligned) {
    if (!mmio_ok(paddr, type))
      return false;

    if (type == STORE)
      return sim->mmio_store(paddr, len, bytes);
    else
      return sim->mmio_load(paddr, len, bytes);
  }

  for (size_t i = 0; i < len; i++) {
    if (!mmio(paddr + i, 1, bytes + i, type))
      return false;
  }

  return true;
}

void mmu_t::check_triggers(triggers::operation_t operation, reg_t address, bool virt, reg_t tval, std::optional<reg_t> data)
{
  if (matched_trigger || !proc)
    return;

  auto match = proc->TM.detect_memory_access_match(operation, address, data);

  if (match.has_value())
    switch (match->timing) {
      case triggers::TIMING_BEFORE:
        throw triggers::matched_t(operation, tval, match->action, virt);

      case triggers::TIMING_AFTER:
        // We want to take this exception on the next instruction.  We check
        // whether to do so in the I$ refill path, so flush the I$.
        flush_icache();
        matched_trigger = new triggers::matched_t(operation, tval, match->action, virt);
    }
}

void mmu_t::load_slow_path_intrapage(reg_t len, uint8_t* bytes, mem_access_info_t access_info)
{
  reg_t addr = access_info.vaddr;
  reg_t vpn = addr >> PGSHIFT;
  if (!access_info.flags.is_special_access() && vpn == (tlb_load_tag[vpn % TLB_ENTRIES] & ~TLB_CHECK_TRIGGERS)) {
    auto host_addr = tlb_data[vpn % TLB_ENTRIES].host_offset + addr;
    memcpy(bytes, host_addr, len);
    return;
  }

  reg_t paddr = translate(access_info, len);

  if (access_info.flags.lr && !sim->reservable(paddr)) {
    throw trap_load_access_fault(access_info.effective_virt, addr, 0, 0);
  }

  if (auto host_addr = sim->addr_to_mem(paddr)) {
    memcpy(bytes, host_addr, len);
    if (tracer.interested_in_range(paddr, paddr + PGSIZE, LOAD))
      tracer.trace(paddr, len, LOAD);
    else if (!access_info.flags.is_special_access())
      refill_tlb(addr, paddr, host_addr, LOAD);

  } else if (!mmio_load(paddr, len, bytes)) {
    throw trap_load_access_fault(access_info.effective_virt, addr, 0, 0);
  }

  if (access_info.flags.lr) {
    load_reservation_address = paddr;
  }
}

void mmu_t::load_slow_path(reg_t addr, reg_t len, uint8_t* bytes, xlate_flags_t xlate_flags)
{
  auto access_info = generate_access_info(addr, LOAD, xlate_flags);
  check_triggers(triggers::OPERATION_LOAD, addr, access_info.effective_virt);

  if ((addr & (len - 1)) == 0) {
    load_slow_path_intrapage(len, bytes, access_info);
  } else {
    bool gva = access_info.effective_virt;
    if (!is_misaligned_enabled())
      throw trap_load_address_misaligned(gva, addr, 0, 0);

    if (access_info.flags.lr)
      throw trap_load_access_fault(gva, addr, 0, 0);

    reg_t len_page0 = std::min(len, PGSIZE - addr % PGSIZE);
    load_slow_path_intrapage(len_page0, bytes, access_info);
    if (len_page0 != len)
      load_slow_path_intrapage(len - len_page0, bytes + len_page0, access_info.split_misaligned_access(len_page0));
  }

  while (len > sizeof(reg_t)) {
    check_triggers(triggers::OPERATION_LOAD, addr, access_info.effective_virt, reg_from_bytes(sizeof(reg_t), bytes));
    len -= sizeof(reg_t);
    bytes += sizeof(reg_t);
  }
  check_triggers(triggers::OPERATION_LOAD, addr, access_info.effective_virt, reg_from_bytes(len, bytes));
}

void mmu_t::store_slow_path_intrapage(reg_t len, const uint8_t* bytes, mem_access_info_t access_info, bool actually_store)
{
  reg_t addr = access_info.vaddr;
  reg_t vpn = addr >> PGSHIFT;
  if (!access_info.flags.is_special_access() && vpn == (tlb_store_tag[vpn % TLB_ENTRIES] & ~TLB_CHECK_TRIGGERS)) {
    if (actually_store) {
      auto host_addr = tlb_data[vpn % TLB_ENTRIES].host_offset + addr;
      memcpy(host_addr, bytes, len);
    }
    return;
  }

  reg_t paddr = translate(access_info, len);

  if (actually_store) {
    if (auto host_addr = sim->addr_to_mem(paddr)) {
      memcpy(host_addr, bytes, len);
      if (tracer.interested_in_range(paddr, paddr + PGSIZE, STORE))
        tracer.trace(paddr, len, STORE);
      else if (!access_info.flags.is_special_access())
        refill_tlb(addr, paddr, host_addr, STORE);
    } else if (!mmio_store(paddr, len, bytes)) {
      throw trap_store_access_fault(access_info.effective_virt, addr, 0, 0);
    }
  }
}

void mmu_t::store_slow_path(reg_t addr, reg_t len, const uint8_t* bytes, xlate_flags_t xlate_flags, bool actually_store, bool UNUSED require_alignment)
{
  auto access_info = generate_access_info(addr, STORE, xlate_flags);
  if (actually_store) {
    reg_t trig_len = len;
    const uint8_t* trig_bytes = bytes;
    while (trig_len > sizeof(reg_t)) {
      check_triggers(triggers::OPERATION_STORE, addr, access_info.effective_virt, reg_from_bytes(sizeof(reg_t), trig_bytes));
      trig_len -= sizeof(reg_t);
      trig_bytes += sizeof(reg_t);
    }
    check_triggers(triggers::OPERATION_STORE, addr, access_info.effective_virt, reg_from_bytes(trig_len, trig_bytes));
  }

  if (addr & (len - 1)) {
    bool gva = access_info.effective_virt;
    if (!is_misaligned_enabled())
      throw trap_store_address_misaligned(gva, addr, 0, 0);

    if (require_alignment)
      throw trap_store_access_fault(gva, addr, 0, 0);

    reg_t len_page0 = std::min(len, PGSIZE - addr % PGSIZE);
    store_slow_path_intrapage(len_page0, bytes, access_info, actually_store);
    if (len_page0 != len)
      store_slow_path_intrapage(len - len_page0, bytes + len_page0, access_info.split_misaligned_access(len_page0), actually_store);
  } else {
    store_slow_path_intrapage(len, bytes, access_info, actually_store);
  }
}

tlb_entry_t mmu_t::refill_tlb(reg_t vaddr, reg_t paddr, char* host_addr, access_type type)
{
  reg_t idx = (vaddr >> PGSHIFT) % TLB_ENTRIES;
  reg_t expected_tag = vaddr >> PGSHIFT;

  tlb_entry_t entry = {host_addr - vaddr, paddr - vaddr};

  if (in_mprv())
    return entry;

  if ((tlb_load_tag[idx] & ~TLB_CHECK_TRIGGERS) != expected_tag)
    tlb_load_tag[idx] = -1;
  if ((tlb_store_tag[idx] & ~TLB_CHECK_TRIGGERS) != expected_tag)
    tlb_store_tag[idx] = -1;
  if ((tlb_insn_tag[idx] & ~TLB_CHECK_TRIGGERS) != expected_tag)
    tlb_insn_tag[idx] = -1;

  if ((check_triggers_fetch && type == FETCH) ||
      (check_triggers_load && type == LOAD) ||
      (check_triggers_store && type == STORE))
    expected_tag |= TLB_CHECK_TRIGGERS;

  if (pmp_homogeneous(paddr & ~reg_t(PGSIZE - 1), PGSIZE)) {
    if (type == FETCH) tlb_insn_tag[idx] = expected_tag;
    else if (type == STORE) tlb_store_tag[idx] = expected_tag;
    else tlb_load_tag[idx] = expected_tag;
  }

  tlb_data[idx] = entry;
  return entry;
}

bool mmu_t::pmp_ok(reg_t addr, reg_t len, access_type type, reg_t mode)
{
  if (!proc || proc->n_pmp == 0)
    return true;

  for (size_t i = 0; i < proc->n_pmp; i++) {
    // Check each 4-byte sector of the access
    bool any_match = false;
    bool all_match = true;
    for (reg_t offset = 0; offset < len; offset += 1 << PMP_SHIFT) {
      reg_t cur_addr = addr + offset;
      bool match = proc->state.pmpaddr[i]->match4(cur_addr);
      any_match |= match;
      all_match &= match;
    }

    if (any_match) {
      // If the PMP matches only a strict subset of the access, fail it
      if (!all_match)
        return false;

      return proc->state.pmpaddr[i]->access_ok(type, mode);
    }
  }

  // in case matching region is not found
  const bool mseccfg_mml = proc->state.mseccfg->get_mml();
  const bool mseccfg_mmwp = proc->state.mseccfg->get_mmwp();
  return ((mode == PRV_M) && !mseccfg_mmwp
          && (!mseccfg_mml || ((type == LOAD) || (type == STORE))));
}

reg_t mmu_t::pmp_homogeneous(reg_t addr, reg_t len)
{
  if ((addr | len) & (len - 1))
    abort();

  if (!proc)
    return true;

  for (size_t i = 0; i < proc->n_pmp; i++)
    if (proc->state.pmpaddr[i]->subset_match(addr, len))
      return false;

  return true;
}

reg_t mmu_t::s2xlate(reg_t gva, reg_t gpa, access_type type, access_type trap_type, bool virt, bool hlvx, bool is_for_vs_pt_addr)
{
  if (!virt)
    return gpa;

  vm_info vm = decode_vm_info(proc->get_const_xlen(), true, 0, proc->get_state()->hgatp->read());
  if (vm.levels == 0)
    return gpa;

  int maxgpabits = vm.levels * vm.idxbits + vm.widenbits + PGSHIFT;
  reg_t maxgpa = (1ULL << maxgpabits) - 1;

  bool mxr = proc->state.sstatus->readvirt(false) & MSTATUS_MXR;
  // tinst is set to 0x3000/0x3020 - for RV64 read/write respectively for
  // VS-stage address translation (for spike HSXLEN == VSXLEN always) else
  // tinst is set to 0x2000/0x2020 - for RV32 read/write respectively for
  // VS-stage address translation else set to 0
  int tinst = 0;
  tinst |= (is_for_vs_pt_addr == true) ? 0x2000 : 0;
  tinst |= ((proc->get_const_xlen() == 64) && (is_for_vs_pt_addr == true)) ? 0x1000 : 0;
  tinst |= ((type == STORE) && (is_for_vs_pt_addr == true)) ? 0x0020 : 0;

  reg_t base = vm.ptbase;
  if ((gpa & ~maxgpa) == 0) {
    for (int i = vm.levels - 1; i >= 0; i--) {
      int ptshift = i * vm.idxbits;
      int idxbits = (i == (vm.levels - 1)) ? vm.idxbits + vm.widenbits : vm.idxbits;
      reg_t idx = (gpa >> (PGSHIFT + ptshift)) & ((reg_t(1) << idxbits) - 1);

      // check that physical address of PTE is legal
      auto pte_paddr = base + idx * vm.ptesize;
      reg_t pte = pte_load(pte_paddr, gva, virt, trap_type, vm.ptesize);
      reg_t ppn = (pte & ~reg_t(PTE_ATTR)) >> PTE_PPN_SHIFT;
      bool pbmte = proc->get_state()->menvcfg->read() & MENVCFG_PBMTE;
      bool hade = proc->get_state()->menvcfg->read() & MENVCFG_ADUE;

      if (pte & PTE_RSVD) {
        break;
      } else if (!proc->extension_enabled(EXT_SVNAPOT) && (pte & PTE_N)) {
        break;
      } else if (!pbmte && (pte & PTE_PBMT)) {
        break;
      } else if ((pte & PTE_PBMT) == PTE_PBMT) {
        break;
      } else if (PTE_TABLE(pte)) { // next level of page table
        if (pte & (PTE_D | PTE_A | PTE_U | PTE_N | PTE_PBMT))
          break;
        base = ppn << PGSHIFT;
      } else if (!(pte & PTE_V) || (!(pte & PTE_R) && (pte & PTE_W))) {
        break;
      } else if (!(pte & PTE_U)) {
        break;
      } else if (type == FETCH || hlvx ? !(pte & PTE_X) :
                 type == LOAD          ? !(pte & PTE_R) && !(mxr && (pte & PTE_X)) :
                                         !((pte & PTE_R) && (pte & PTE_W))) {
        break;
      } else if ((ppn & ((reg_t(1) << ptshift) - 1)) != 0) {
        break;
      } else {
        reg_t ad = PTE_A | ((type == STORE) * PTE_D);

        if ((pte & ad) != ad) {
          if (hade) {
            // set accessed and possibly dirty bits
            pte_store(pte_paddr, pte | ad, gva, virt, type, vm.ptesize);
          } else {
            // take exception if access or possibly dirty bit is not set.
            break;
          }
        }

        reg_t vpn = gpa >> PGSHIFT;
        reg_t page_mask = (reg_t(1) << PGSHIFT) - 1;

        int napot_bits = ((pte & PTE_N) ? (ctz(ppn) + 1) : 0);
        if (((pte & PTE_N) && (ppn == 0 || i != 0)) || (napot_bits != 0 && napot_bits != 4))
          break;

        reg_t page_base = ((ppn & ~((reg_t(1) << napot_bits) - 1))
                          | (vpn & ((reg_t(1) << napot_bits) - 1))
                          | (vpn & ((reg_t(1) << ptshift) - 1))) << PGSHIFT;
        return page_base | (gpa & page_mask);
      }
    }
  }

  switch (trap_type) {
    case FETCH: throw trap_instruction_guest_page_fault(gva, gpa >> 2, tinst);
    case LOAD: throw trap_load_guest_page_fault(gva, gpa >> 2, tinst);
    case STORE: throw trap_store_guest_page_fault(gva, gpa >> 2, tinst);
    default: abort();
  }
}

reg_t mmu_t::walk(mem_access_info_t access_info)
{
  access_type type = access_info.type;
  reg_t addr = access_info.vaddr;
  bool virt = access_info.effective_virt;
  bool hlvx = access_info.flags.hlvx;
  reg_t mode = access_info.effective_priv;
  reg_t page_mask = (reg_t(1) << PGSHIFT) - 1;
  reg_t satp = proc->get_state()->satp->readvirt(virt);
  vm_info vm = decode_vm_info(proc->get_const_xlen(), false, mode, satp);
  if (vm.levels == 0)
    return s2xlate(addr, addr & ((reg_t(2) << (proc->xlen-1))-1), type, type, virt, hlvx, false) & ~page_mask; // zero-extend from xlen

  bool s_mode = mode == PRV_S;
  bool sum = proc->state.sstatus->readvirt(virt) & MSTATUS_SUM;
  bool mxr = (proc->state.sstatus->readvirt(false) | proc->state.sstatus->readvirt(virt)) & MSTATUS_MXR;

  // verify bits xlen-1:va_bits-1 are all equal
  int va_bits = PGSHIFT + vm.levels * vm.idxbits;
  reg_t mask = (reg_t(1) << (proc->xlen - (va_bits-1))) - 1;
  reg_t masked_msbs = (addr >> (va_bits-1)) & mask;
  if (masked_msbs != 0 && masked_msbs != mask)
    vm.levels = 0;

  reg_t base = vm.ptbase;
  for (int i = vm.levels - 1; i >= 0; i--) {
    int ptshift = i * vm.idxbits;
    reg_t idx = (addr >> (PGSHIFT + ptshift)) & ((1 << vm.idxbits) - 1);

    // check that physical address of PTE is legal
    auto pte_paddr = s2xlate(addr, base + idx * vm.ptesize, LOAD, type, virt, false, true);
    reg_t pte = pte_load(pte_paddr, addr, virt, type, vm.ptesize);
    reg_t ppn = (pte & ~reg_t(PTE_ATTR)) >> PTE_PPN_SHIFT;
    bool pbmte = virt ? (proc->get_state()->henvcfg->read() & HENVCFG_PBMTE) : (proc->get_state()->menvcfg->read() & MENVCFG_PBMTE);
    bool hade = virt ? (proc->get_state()->henvcfg->read() & HENVCFG_ADUE) : (proc->get_state()->menvcfg->read() & MENVCFG_ADUE);

    if (pte & PTE_RSVD) {
      break;
    } else if (!proc->extension_enabled(EXT_SVNAPOT) && (pte & PTE_N)) {
      break;
    } else if (!pbmte && (pte & PTE_PBMT)) {
      break;
    } else if ((pte & PTE_PBMT) == PTE_PBMT) {
      break;
    } else if (PTE_TABLE(pte)) { // next level of page table
      if (pte & (PTE_D | PTE_A | PTE_U | PTE_N | PTE_PBMT))
        break;
      base = ppn << PGSHIFT;
    } else if ((pte & PTE_U) ? s_mode && (type == FETCH || !sum) : !s_mode) {
      break;
    } else if (!(pte & PTE_V) || (!(pte & PTE_R) && (pte & PTE_W))) {
      break;
    } else if (type == FETCH || hlvx ? !(pte & PTE_X) :
               type == LOAD          ? !(pte & PTE_R) && !(mxr && (pte & PTE_X)) :
                                       !((pte & PTE_R) && (pte & PTE_W))) {
      break;
    } else if ((ppn & ((reg_t(1) << ptshift) - 1)) != 0) {
      break;
    } else {
      reg_t ad = PTE_A | ((type == STORE) * PTE_D);

      if ((pte & ad) != ad) {
        if (hade) {
          // Check for write permission to the first-stage PT in second-stage
          // PTE and set the D bit in the second-stage PTE if needed
          s2xlate(addr, base + idx * vm.ptesize, STORE, type, virt, false, true);
          // set accessed and possibly dirty bits.
          pte_store(pte_paddr, pte | ad, addr, virt, type, vm.ptesize);
        } else {
          // take exception if access or possibly dirty bit is not set.
          break;
        }
      }

      // for superpage or Svnapot NAPOT mappings, make a fake leaf PTE for the TLB's benefit.
      reg_t vpn = addr >> PGSHIFT;

      int napot_bits = ((pte & PTE_N) ? (ctz(ppn) + 1) : 0);
      if (((pte & PTE_N) && (ppn == 0 || i != 0)) || (napot_bits != 0 && napot_bits != 4))
        break;

      reg_t page_base = ((ppn & ~((reg_t(1) << napot_bits) - 1))
                        | (vpn & ((reg_t(1) << napot_bits) - 1))
                        | (vpn & ((reg_t(1) << ptshift) - 1))) << PGSHIFT;
      reg_t phys = page_base | (addr & page_mask);
      return s2xlate(addr, phys, type, type, virt, hlvx, false) & ~page_mask;
    }
  }

  switch (type) {
    case FETCH: throw trap_instruction_page_fault(virt, addr, 0, 0);
    case LOAD: throw trap_load_page_fault(virt, addr, 0, 0);
    case STORE: throw trap_store_page_fault(virt, addr, 0, 0);
    default: abort();
  }
}

void mmu_t::register_memtracer(memtracer_t* t)
{
  flush_tlb();
  tracer.hook(t);
}

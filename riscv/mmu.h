// See LICENSE for license details.

#ifndef _RISCV_MMU_H
#define _RISCV_MMU_H

#include "decode.h"
#include "trap.h"
#include "common.h"
#include "simif.h"
#include "processor.h"
#include "memtracer.h"
#include "../fesvr/byteorder.h"
#include "triggers.h"
#include "cfg.h"
#include <stdlib.h>
#include <vector>

// virtual memory configuration
#define PGSHIFT 12
const reg_t PGSIZE = 1 << PGSHIFT;
const reg_t PGMASK = ~(PGSIZE-1);
#define MAX_PADDR_BITS 56 // imposed by Sv39 / Sv48

struct insn_fetch_t
{
  insn_func_t func;
  insn_t insn;
};

struct icache_entry_t {
  reg_t tag;
  struct icache_entry_t* next;
  insn_fetch_t data;
};

struct tlb_entry_t {
  char* host_offset;
  reg_t target_offset;
};

struct xlate_flags_t {
  const bool forced_virt : 1;
  const bool hlvx : 1;
  const bool lr : 1;

  bool is_special_access() const {
    return forced_virt || hlvx || lr;
  }
};

struct mem_access_info_t {
  const reg_t vaddr;
  const reg_t effective_priv;
  const bool effective_virt;
  const xlate_flags_t flags;
  const access_type type;

  mem_access_info_t split_misaligned_access(reg_t offset) const {
    return {vaddr + offset, effective_priv, effective_virt, flags, type};
  }
};

void throw_access_exception(bool virt, reg_t addr, access_type type);

// this class implements a processor's port into the virtual memory system.
// an MMU and instruction cache are maintained for simulator performance.
class mmu_t
{
private:
  std::map<reg_t, reg_t> alloc_cache;
  std::vector<std::pair<reg_t, reg_t >> addr_tbl;

  mem_access_info_t generate_access_info(reg_t addr, access_type type, xlate_flags_t xlate_flags) {
    if (!proc)
      return {addr, 0, false, {false, false, false}, type};
    bool virt = proc->state.v;
    reg_t mode = proc->state.prv;
    if (type != FETCH) {
      if (in_mprv()) {
        mode = get_field(proc->state.mstatus->read(), MSTATUS_MPP);
        if (get_field(proc->state.mstatus->read(), MSTATUS_MPV) && mode != PRV_M)
          virt = true;
      }
      if (xlate_flags.forced_virt) {
        virt = true;
        mode = get_field(proc->state.hstatus->read(), HSTATUS_SPVP);
      }
    }
    return {addr, mode, virt, xlate_flags, type};
  }

public:
  mmu_t(simif_t* sim, endianness_t endianness, processor_t* proc);
  ~mmu_t();

  template<typename T>
  T ALWAYS_INLINE load(reg_t addr, xlate_flags_t xlate_flags = {false, false, false}) {
    target_endian<T> res;
    reg_t vpn = addr >> PGSHIFT;
    bool aligned = (addr & (sizeof(T) - 1)) == 0;
    bool tlb_hit = tlb_load_tag[vpn % TLB_ENTRIES] == vpn;

    if (likely(!xlate_flags.is_special_access() && aligned && tlb_hit)) {
      res = *(target_endian<T>*)(tlb_data[vpn % TLB_ENTRIES].host_offset + addr);
    } else {
      load_slow_path(addr, sizeof(T), (uint8_t*)&res, xlate_flags);
    }

    if (unlikely(proc && proc->get_log_commits_enabled()))
      proc->state.log_mem_read.push_back(std::make_tuple(addr, 0, sizeof(T)));

    return from_target(res);
  }

  template<typename T>
  T load_reserved(reg_t addr) {
    bool forced_virt = false;
    bool hlvx = false;
    bool lr = true;
    return load<T>(addr, {forced_virt, hlvx, lr});
  }

  template<typename T>
  T guest_load(reg_t addr) {
    bool forced_virt = true;
    bool hlvx = false;
    bool lr = false;
    return load<T>(addr, {forced_virt, hlvx, lr});
  }

  template<typename T>
  T guest_load_x(reg_t addr) {
    bool forced_virt = true;
    bool hlvx = true;
    bool lr = false;
    return load<T>(addr, {forced_virt, hlvx, lr});
  }

  template<typename T>
  void ALWAYS_INLINE store(reg_t addr, T val, xlate_flags_t xlate_flags = {false, false, false}) {
    reg_t vpn = addr >> PGSHIFT;
    bool aligned = (addr & (sizeof(T) - 1)) == 0;
    bool tlb_hit = tlb_store_tag[vpn % TLB_ENTRIES] == vpn;

    if (!xlate_flags.is_special_access() && likely(aligned && tlb_hit)) {
      *(target_endian<T>*)(tlb_data[vpn % TLB_ENTRIES].host_offset + addr) = to_target(val);
    } else {
      target_endian<T> target_val = to_target(val);
      store_slow_path(addr, sizeof(T), (const uint8_t*)&target_val, xlate_flags, true, false);
    }

    if (unlikely(proc && proc->get_log_commits_enabled()))
      proc->state.log_mem_write.push_back(std::make_tuple(addr, val, sizeof(T)));
  }

  template<typename T>
  void guest_store(reg_t addr, T val) {
    bool forced_virt = true;
    bool hlvx = false;
    bool lr = false;
    store(addr, val, {forced_virt, hlvx, lr});
  }

  // AMO/Zicbom faults should be reported as store faults
  #define convert_load_traps_to_store_traps(BODY) \
    try { \
      BODY \
    } catch (trap_load_address_misaligned& t) { \
      /* Misaligned fault will not be triggered by Zicbom */ \
      throw trap_store_address_misaligned(t.has_gva(), t.get_tval(), t.get_tval2(), t.get_tinst()); \
    } catch (trap_load_page_fault& t) { \
      throw trap_store_page_fault(t.has_gva(), t.get_tval(), t.get_tval2(), t.get_tinst()); \
    } catch (trap_load_access_fault& t) { \
      throw trap_store_access_fault(t.has_gva(), t.get_tval(), t.get_tval2(), t.get_tinst()); \
    } catch (trap_load_guest_page_fault& t) { \
      throw trap_store_guest_page_fault(t.get_tval(), t.get_tval2(), t.get_tinst()); \
    }

  // template for functions that perform an atomic memory operation
  template<typename T, typename op>
  T amo(reg_t addr, op f) {
    convert_load_traps_to_store_traps({
      store_slow_path(addr, sizeof(T), nullptr, {false, false, false}, false, true);
      auto lhs = load<T>(addr);
      store<T>(addr, f(lhs));
      return lhs;
    })
  }

  template<typename T>
  T amo_compare_and_swap(reg_t addr, T comp, T swap) {
    convert_load_traps_to_store_traps({
      store_slow_path(addr, sizeof(T), nullptr, {false, false, false}, false, true);
      auto lhs = load<T>(addr);
      if (lhs == comp)
        store<T>(addr, swap);
      return lhs;
    })
  }

  void store_float128(reg_t addr, float128_t val)
  {
    if (unlikely(addr & (sizeof(float128_t)-1)) && !is_misaligned_enabled()) {
      throw trap_store_address_misaligned((proc) ? proc->state.v : false, addr, 0, 0);
    }

    store<uint64_t>(addr, val.v[0]);
    store<uint64_t>(addr + 8, val.v[1]);
  }

  float128_t load_float128(reg_t addr)
  {
    if (unlikely(addr & (sizeof(float128_t)-1)) && !is_misaligned_enabled()) {
      throw trap_load_address_misaligned((proc) ? proc->state.v : false, addr, 0, 0);
    }

    return (float128_t){load<uint64_t>(addr), load<uint64_t>(addr + 8)};
  }

  void cbo_zero(reg_t addr) {
    auto base = addr & ~(blocksz - 1);
    for (size_t offset = 0; offset < blocksz; offset += 1) {
      check_triggers(triggers::OPERATION_STORE, base + offset, false, addr, std::nullopt);
      store<uint8_t>(base + offset, 0);
    }
  }

  void clean_inval(reg_t addr, bool clean, bool inval) {
    auto base = addr & ~(blocksz - 1);
    for (size_t offset = 0; offset < blocksz; offset += 1)
      check_triggers(triggers::OPERATION_STORE, base + offset, false, addr, std::nullopt);
    convert_load_traps_to_store_traps({
      const reg_t paddr = translate(generate_access_info(addr, LOAD, {false, false, false}), 1);
      if (sim->reservable(paddr)) {
        if (tracer.interested_in_range(paddr, paddr + PGSIZE, LOAD))
          tracer.clean_invalidate(paddr, blocksz, clean, inval);
      } else {
        throw trap_store_access_fault((proc) ? proc->state.v : false, addr, 0, 0);
      }
    })
  }

  inline void yield_load_reservation()
  {
    load_reservation_address = (reg_t)-1;
  }

  inline bool check_load_reservation(reg_t vaddr, size_t size)
  {
    if (vaddr & (size-1)) {
      // Raise either access fault or misaligned exception
      store_slow_path(vaddr, size, nullptr, {false, false, false}, false, true);
    }

    reg_t paddr = translate(generate_access_info(vaddr, STORE, {false, false, false}), 1);
    if (sim->reservable(paddr))
      return load_reservation_address == paddr;
    else
      throw trap_store_access_fault((proc) ? proc->state.v : false, vaddr, 0, 0);
  }

  template<typename T>
  bool store_conditional(reg_t addr, T val)
  {
    bool have_reservation = check_load_reservation(addr, sizeof(T));

    if (have_reservation)
      store(addr, val);

    yield_load_reservation();

    return have_reservation;
  }

  static const reg_t ICACHE_ENTRIES = 1024;

  inline size_t icache_index(reg_t addr)
  {
    return (addr / PC_ALIGN) % ICACHE_ENTRIES;
  }

  template<typename T>
  T ALWAYS_INLINE fetch_jump_table(reg_t addr) {
    auto tlb_entry = translate_insn_addr(addr);
    return from_target(*(target_endian<T>*)(tlb_entry.host_offset + addr));
  }

  inline icache_entry_t* refill_icache(reg_t addr, icache_entry_t* entry)
  {
    if (matched_trigger)
      throw *matched_trigger;

    auto tlb_entry = translate_insn_addr(addr);
    insn_bits_t insn = from_le(*(uint16_t*)(tlb_entry.host_offset + addr));
    int length = insn_length(insn);

    if (likely(length == 4)) {
      insn |= (insn_bits_t)from_le(*(const uint16_t*)translate_insn_addr_to_host(addr + 2)) << 16;
    } else if (length == 2) {
      // entire instruction already fetched
    } else if (length == 6) {
      insn |= (insn_bits_t)from_le(*(const uint16_t*)translate_insn_addr_to_host(addr + 2)) << 16;
      insn |= (insn_bits_t)from_le(*(const uint16_t*)translate_insn_addr_to_host(addr + 4)) << 32;
    } else {
      static_assert(sizeof(insn_bits_t) == 8, "insn_bits_t must be uint64_t");
      insn |= (insn_bits_t)from_le(*(const uint16_t*)translate_insn_addr_to_host(addr + 2)) << 16;
      insn |= (insn_bits_t)from_le(*(const uint16_t*)translate_insn_addr_to_host(addr + 4)) << 32;
      insn |= (insn_bits_t)from_le(*(const uint16_t*)translate_insn_addr_to_host(addr + 6)) << 48;
    }

    insn_fetch_t fetch = {proc->decode_insn(insn), insn};
    entry->tag = addr;
    entry->next = &icache[icache_index(addr + length)];
    entry->data = fetch;

    reg_t paddr = tlb_entry.target_offset + addr;;
    if (tracer.interested_in_range(paddr, paddr + 1, FETCH)) {
      entry->tag = -1;
      tracer.trace(paddr, length, FETCH);
    }
    return entry;
  }

  inline icache_entry_t* access_icache(reg_t addr)
  {
    icache_entry_t* entry = &icache[icache_index(addr)];
    if (likely(entry->tag == addr))
      return entry;
    return refill_icache(addr, entry);
  }

  inline insn_fetch_t load_insn(reg_t addr)
  {
    icache_entry_t entry;
    return refill_icache(addr, &entry)->data;
  }

  void flush_tlb();
  void flush_icache();

  void register_memtracer(memtracer_t*);

  int is_misaligned_enabled()
  {
    return proc && proc->get_cfg().misaligned;
  }

  bool is_target_big_endian()
  {
    return target_big_endian;
  }

  template<typename T> inline T from_target(target_endian<T> n) const
  {
    return target_big_endian? n.from_be() : n.from_le();
  }

  template<typename T> inline target_endian<T> to_target(T n) const
  {
    return target_big_endian? target_endian<T>::to_be(n) : target_endian<T>::to_le(n);
  }

  void set_cache_blocksz(reg_t size)
  {
    blocksz = size;
  }

private:
  simif_t* sim;
  processor_t* proc;
  memtracer_list_t tracer;
  reg_t load_reservation_address;
  uint16_t fetch_temp;
  reg_t blocksz;

  // implement an instruction cache for simulator performance
  icache_entry_t icache[ICACHE_ENTRIES];

  // implement a TLB for simulator performance
  static const reg_t TLB_ENTRIES = 256;
  // If a TLB tag has TLB_CHECK_TRIGGERS set, then the MMU must check for a
  // trigger match before completing an access.
  static const reg_t TLB_CHECK_TRIGGERS = reg_t(1) << 63;
  tlb_entry_t tlb_data[TLB_ENTRIES];
  reg_t tlb_insn_tag[TLB_ENTRIES];
  reg_t tlb_load_tag[TLB_ENTRIES];
  reg_t tlb_store_tag[TLB_ENTRIES];

  // finish translation on a TLB miss and update the TLB
  tlb_entry_t refill_tlb(reg_t vaddr, reg_t paddr, char* host_addr, access_type type);
  const char* fill_from_mmio(reg_t vaddr, reg_t paddr);

  // perform a stage2 translation for a given guest address
  reg_t s2xlate(reg_t gva, reg_t gpa, access_type type, access_type trap_type, bool virt, bool hlvx, bool is_for_vs_pt_addr);

  // perform a page table walk for a given VA; set referenced/dirty bits
  reg_t walk(mem_access_info_t access_info);

  // handle uncommon cases: TLB misses, page faults, MMIO
  tlb_entry_t fetch_slow_path(reg_t addr);
  void load_slow_path(reg_t addr, reg_t len, uint8_t* bytes, xlate_flags_t xlate_flags);
  void load_slow_path_intrapage(reg_t len, uint8_t* bytes, mem_access_info_t access_info);
  void store_slow_path(reg_t addr, reg_t len, const uint8_t* bytes, xlate_flags_t xlate_flags, bool actually_store, bool require_alignment);
  void store_slow_path_intrapage(reg_t len, const uint8_t* bytes, mem_access_info_t access_info, bool actually_store);
  bool mmio_fetch(reg_t paddr, size_t len, uint8_t* bytes);
  bool mmio_load(reg_t paddr, size_t len, uint8_t* bytes);
  bool mmio_store(reg_t paddr, size_t len, const uint8_t* bytes);
  bool mmio(reg_t paddr, size_t len, uint8_t* bytes, access_type type);
  bool mmio_ok(reg_t paddr, access_type type);
  void check_triggers(triggers::operation_t operation, reg_t address, bool virt, std::optional<reg_t> data = std::nullopt) {
    check_triggers(operation, address, virt, address, data);
  }
  void check_triggers(triggers::operation_t operation, reg_t address, bool virt, reg_t tval, std::optional<reg_t> data);
  reg_t translate(mem_access_info_t access_info, reg_t len);

  reg_t pte_load(reg_t pte_paddr, reg_t addr, bool virt, access_type trap_type, size_t ptesize) {
    if (ptesize == 4)
      return pte_load<uint32_t>(pte_paddr, addr, virt, trap_type);
    else
      return pte_load<uint64_t>(pte_paddr, addr, virt, trap_type);
  }

  void pte_store(reg_t pte_paddr, reg_t new_pte, reg_t addr, bool virt, access_type trap_type, size_t ptesize) {
    if (ptesize == 4)
      return pte_store<uint32_t>(pte_paddr, new_pte, addr, virt, trap_type);
    else
      return pte_store<uint64_t>(pte_paddr, new_pte, addr, virt, trap_type);
  }

  template<typename T> inline reg_t pte_load(reg_t pte_paddr, reg_t addr, bool virt, access_type trap_type)
  {
    const size_t ptesize = sizeof(T);

    if (!pmp_ok(pte_paddr, ptesize, LOAD, PRV_S))
      throw_access_exception(virt, addr, trap_type);

    void* host_pte_addr = sim->addr_to_mem(pte_paddr);
    target_endian<T> target_pte;
    if (host_pte_addr) {
      memcpy(&target_pte, host_pte_addr, ptesize);
    } else if (!mmio_load(pte_paddr, ptesize, (uint8_t*)&target_pte)) {
      throw_access_exception(virt, addr, trap_type);
    }
    return from_target(target_pte);
  }

  template<typename T> inline void pte_store(reg_t pte_paddr, reg_t new_pte, reg_t addr, bool virt, access_type trap_type)
  {
    const size_t ptesize = sizeof(T);

    if (!pmp_ok(pte_paddr, ptesize, STORE, PRV_S))
      throw_access_exception(virt, addr, trap_type);

    void* host_pte_addr = sim->addr_to_mem(pte_paddr);
    target_endian<T> target_pte = to_target((T)new_pte);
    if (host_pte_addr) {
      memcpy(host_pte_addr, &target_pte, ptesize);
    } else if (!mmio_store(pte_paddr, ptesize, (uint8_t*)&target_pte)) {
      throw_access_exception(virt, addr, trap_type);
    }
  }

  // ITLB lookup
  inline tlb_entry_t translate_insn_addr(reg_t addr) {
    reg_t vpn = addr >> PGSHIFT;
    if (likely(tlb_insn_tag[vpn % TLB_ENTRIES] == vpn))
      return tlb_data[vpn % TLB_ENTRIES];
    return fetch_slow_path(addr);
  }

  inline const uint16_t* translate_insn_addr_to_host(reg_t addr) {
    return (uint16_t*)(translate_insn_addr(addr).host_offset + addr);
  }

  inline bool in_mprv()
  {
    return proc != nullptr
           && !(proc->state.mnstatus && !get_field(proc->state.mnstatus->read(), MNSTATUS_NMIE))
           && !proc->state.debug_mode
           && get_field(proc->state.mstatus->read(), MSTATUS_MPRV);
  }

  reg_t pmp_homogeneous(reg_t addr, reg_t len);
  bool pmp_ok(reg_t addr, reg_t len, access_type type, reg_t mode);

#ifdef RISCV_ENABLE_DUAL_ENDIAN
  bool target_big_endian;
#else
  static const bool target_big_endian = false;
#endif
  bool check_triggers_fetch;
  bool check_triggers_load;
  bool check_triggers_store;
  // The exception describing a matched trigger, or NULL.
  triggers::matched_t *matched_trigger;

  friend class processor_t;
};

struct vm_info {
  int levels;
  int idxbits;
  int widenbits;
  int ptesize;
  reg_t ptbase;
};

inline vm_info decode_vm_info(int xlen, bool stage2, reg_t prv, reg_t satp)
{
  if (prv == PRV_M) {
    return {0, 0, 0, 0, 0};
  } else if (!stage2 && prv <= PRV_S && xlen == 32) {
    switch (get_field(satp, SATP32_MODE)) {
      case SATP_MODE_OFF: return {0, 0, 0, 0, 0};
      case SATP_MODE_SV32: return {2, 10, 0, 4, (satp & SATP32_PPN) << PGSHIFT};
      default: abort();
    }
  } else if (!stage2 && prv <= PRV_S && xlen == 64) {
    switch (get_field(satp, SATP64_MODE)) {
      case SATP_MODE_OFF: return {0, 0, 0, 0, 0};
      case SATP_MODE_SV39: return {3, 9, 0, 8, (satp & SATP64_PPN) << PGSHIFT};
      case SATP_MODE_SV48: return {4, 9, 0, 8, (satp & SATP64_PPN) << PGSHIFT};
      case SATP_MODE_SV57: return {5, 9, 0, 8, (satp & SATP64_PPN) << PGSHIFT};
      case SATP_MODE_SV64: return {6, 9, 0, 8, (satp & SATP64_PPN) << PGSHIFT};
      default: abort();
    }
  } else if (stage2 && xlen == 32) {
    switch (get_field(satp, HGATP32_MODE)) {
      case HGATP_MODE_OFF: return {0, 0, 0, 0, 0};
      case HGATP_MODE_SV32X4: return {2, 10, 2, 4, (satp & HGATP32_PPN) << PGSHIFT};
      default: abort();
    }
  } else if (stage2 && xlen == 64) {
    switch (get_field(satp, HGATP64_MODE)) {
      case HGATP_MODE_OFF: return {0, 0, 0, 0, 0};
      case HGATP_MODE_SV39X4: return {3, 9, 2, 8, (satp & HGATP64_PPN) << PGSHIFT};
      case HGATP_MODE_SV48X4: return {4, 9, 2, 8, (satp & HGATP64_PPN) << PGSHIFT};
      case HGATP_MODE_SV57X4: return {5, 9, 2, 8, (satp & HGATP64_PPN) << PGSHIFT};
      default: abort();
    }
  } else {
    abort();
  }
}

#endif

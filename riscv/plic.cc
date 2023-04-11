#include <sys/time.h>
#include <sstream>
#include "devices.h"
#include "processor.h"
#include "simif.h"
#include "sim.h"
#include "dts.h"

#define PLIC_MAX_CONTEXTS 15872

/*
 * The PLIC consists of memory-mapped control registers, with a memory map
 * as follows:
 *
 * base + 0x000000: Reserved (interrupt source 0 does not exist)
 * base + 0x000004: Interrupt source 1 priority
 * base + 0x000008: Interrupt source 2 priority
 * ...
 * base + 0x000FFC: Interrupt source 1023 priority
 * base + 0x001000: Pending 0
 * base + 0x001FFF: Pending
 * base + 0x002000: Enable bits for sources 0-31 on context 0
 * base + 0x002004: Enable bits for sources 32-63 on context 0
 * ...
 * base + 0x0020FC: Enable bits for sources 992-1023 on context 0
 * base + 0x002080: Enable bits for sources 0-31 on context 1
 * ...
 * base + 0x002100: Enable bits for sources 0-31 on context 2
 * ...
 * base + 0x1F1F80: Enable bits for sources 992-1023 on context 15871
 * base + 0x1F1F84: Reserved
 * ...		    (higher context IDs would fit here, but wouldn't fit
 *		     inside the per-context priority vector)
 * base + 0x1FFFFC: Reserved
 * base + 0x200000: Priority threshold for context 0
 * base + 0x200004: Claim/complete for context 0
 * base + 0x200008: Reserved
 * ...
 * base + 0x200FFC: Reserved
 * base + 0x201000: Priority threshold for context 1
 * base + 0x201004: Claim/complete for context 1
 * ...
 * base + 0xFFE000: Priority threshold for context 15871
 * base + 0xFFE004: Claim/complete for context 15871
 * base + 0xFFE008: Reserved
 * ...
 * base + 0xFFFFFC: Reserved
 */

/* Each interrupt source has a priority register associated with it. */
#define PRIORITY_BASE           0
#define PRIORITY_PER_ID         4

/* Each interrupt source has a pending bit associated with it. */
#define PENDING_BASE            0x1000

/*
 * Each hart context has a vector of interupt enable bits associated with it.
 * There's one bit for each interrupt source.
 */
#define ENABLE_BASE             0x2000
#define ENABLE_PER_HART         0x80

/*
 * Each hart context has a set of control registers associated with it.  Right
 * now there's only two: a source priority threshold over which the hart will
 * take an interrupt, and a register to claim interrupts.
 */
#define CONTEXT_BASE            0x200000
#define CONTEXT_PER_HART        0x1000
#define CONTEXT_THRESHOLD       0
#define CONTEXT_CLAIM           4

#define REG_SIZE                0x1000000

plic_t::plic_t(const simif_t* sim, uint32_t ndev)
  : num_ids(ndev + 1), num_ids_word(((ndev + 1) + (32 - 1)) / 32),
  max_prio((1UL << PLIC_PRIO_BITS) - 1), priority{}, level{}
{
  // PLIC contexts are contiguous in memory even if harts are discontiguous.
  for (const auto& [hart_id, hart] : sim->get_harts()) {
    contexts.push_back(plic_context_t(hart, true));

    if (hart->extension_enabled_const('S')) {
      contexts.push_back(plic_context_t(hart, false));
    }
  }
}

uint32_t plic_t::context_best_pending(const plic_context_t *c)
{
  uint8_t best_id_prio = 0;
  uint32_t best_id = 0;

  for (uint32_t i = 0; i < num_ids_word; i++) {
    if (!c->pending[i]) {
      continue;
    }

    for (uint32_t j = 0; j < 32; j++) {
      uint32_t id = i * 32 + j;
      if ((num_ids <= id) ||
          !(c->pending[i] & (1 << j)) ||
          (c->claimed[i] & (1 << j))) {
        continue;
      }

      if (!best_id ||
          (best_id_prio < c->pending_priority[id])) {
        best_id = id;
        best_id_prio = c->pending_priority[id];
      }
    }
  }

  return best_id;
}

void plic_t::context_update(const plic_context_t *c)
{
  uint32_t best_id = context_best_pending(c);
  reg_t mask = c->mmode ? MIP_MEIP : MIP_SEIP;

  c->proc->state.mip->backdoor_write_with_mask(mask, best_id ? mask : 0);
}

uint32_t plic_t::context_claim(plic_context_t *c)
{
  uint32_t best_id = context_best_pending(c);
  uint32_t best_id_word = best_id / 32;
  uint32_t best_id_mask = (1 << (best_id % 32));

  if (best_id) {
    c->claimed[best_id_word] |= best_id_mask;
  }

  context_update(c);
  return best_id;
}

bool plic_t::priority_read(reg_t offset, uint32_t *val)
{
  uint32_t id = (offset >> 2);

  if (id > 0 && id < num_ids)
    *val = priority[id];
  else
    *val = 0;

  return true;
}

bool plic_t::priority_write(reg_t offset, uint32_t val)
{
  uint32_t id = (offset >> 2);

  if (id > 0 && id < num_ids) {
    val &= ((1 << PLIC_PRIO_BITS) - 1);
    priority[id] = val;
  }

  return true;
}

bool plic_t::pending_read(reg_t offset, uint32_t *val)
{
  uint32_t id_word = (offset >> 2);

  if (id_word < num_ids_word) {
    *val = 0;
    for (auto context: contexts) {
        *val |= context.pending[id_word];
    }
  } else
    *val = 0;

  return true;
}

bool plic_t::context_enable_read(const plic_context_t *c,
                                 reg_t offset, uint32_t *val)
{
  uint32_t id_word = offset >> 2;

  if (id_word < num_ids_word)
    *val = c->enable[id_word];
  else
    *val = 0;

  return true;
}

bool plic_t::context_enable_write(plic_context_t *c,
                                  reg_t offset, uint32_t val)
{
  uint32_t id_word = offset >> 2;

  if (id_word >= num_ids_word)
    return true;

  uint32_t old_val = c->enable[id_word];
  uint32_t new_val = id_word == 0 ? val & ~(uint32_t)1 : val;
  uint32_t xor_val = old_val ^ new_val;

  c->enable[id_word] = new_val;

  for (uint32_t i = 0; i < 32; i++) {
    uint32_t id = id_word * 32 + i;
    uint32_t id_mask = 1 << i;
    uint8_t id_prio = priority[id];
    if (!(xor_val & id_mask)) {
      continue;
    }
    if ((new_val & id_mask) &&
        (level[id_word] & id_mask)) {
      c->pending[id_word] |= id_mask;
      c->pending_priority[id] = id_prio;
    } else if (!(new_val & id_mask)) {
      c->pending[id_word] &= ~id_mask;
      c->pending_priority[id] = 0;
      c->claimed[id_word] &= ~id_mask;
    }
  }

  context_update(c);
  return true;
}

bool plic_t::context_read(plic_context_t *c,
                          reg_t offset, uint32_t *val)
{
  switch (offset) {
    case CONTEXT_THRESHOLD:
      *val = c->priority_threshold;
      return true;
    case CONTEXT_CLAIM:
      *val = context_claim(c);
      return true;
    default:
      return true;
  };
}

bool plic_t::context_write(plic_context_t *c,
                           reg_t offset, uint32_t val)
{
  bool ret = true, update = false;

  switch (offset) {
    case CONTEXT_THRESHOLD:
      val &= ((1 << PLIC_PRIO_BITS) - 1);
      if (val <= max_prio) {
        c->priority_threshold = val;
        update = true;
      } else {
        ret = false;
      }
      break;
    case CONTEXT_CLAIM: {
      uint32_t id_word = val / 32;
      uint32_t id_mask = 1 << (val % 32);
      if ((val < num_ids) &&
          (c->enable[id_word] & id_mask)) {
        c->claimed[id_word] &= ~id_mask;
        update = true;
      }
      break;
    }
    default:
      ret = false;
      break;
  };

  if (update) {
    context_update(c);
  }

  return ret;
}

void plic_t::set_interrupt_level(uint32_t id, int lvl)
{
  if (id <= 0 || num_ids <= id) {
    return;
  }

  uint8_t id_prio = priority[id];
  uint32_t id_word = id / 32;
  uint32_t id_mask = 1 << (id % 32);

  if (lvl) {
    level[id_word] |= id_mask;
  } else {
    level[id_word] &= ~id_mask;
  }

  /*
   * Note: PLIC interrupts are level-triggered. As of now,
   * there is no notion of edge-triggered interrupts. To
   * handle this we auto-clear edge-triggered interrupts
   * when PLIC context CLAIM register is read.
   */
  for (size_t i = 0; i < contexts.size(); i++) {
    plic_context_t* c = &contexts[i];

    if (c->enable[id_word] & id_mask) {
      if (lvl) {
        c->pending[id_word] |= id_mask;
        c->pending_priority[id] = id_prio;
      } else {
        c->pending[id_word] &= ~id_mask;
        c->pending_priority[id] = 0;
        c->claimed[id_word] &= ~id_mask;
      }
      context_update(c);
      break;
    }
  }
}

bool plic_t::load(reg_t addr, size_t len, uint8_t* bytes)
{
  bool ret = false;
  uint32_t val = 0;

  switch (len) {
    case 4:
      break;
    case 8:
      // Implement double-word loads as a pair of word loads
      return load(addr, 4, bytes) && load(addr + 4, 4, bytes + 4);
    default:
      // Subword loads are not supported
      return false;
  }

  if (PRIORITY_BASE <= addr && addr < PENDING_BASE) {
    ret = priority_read(addr, &val);
  } else if (PENDING_BASE <= addr && addr < ENABLE_BASE) {
    ret = pending_read(addr - PENDING_BASE, &val);
  } else if (ENABLE_BASE <= addr && addr < CONTEXT_BASE) {
    uint32_t cntx = (addr - ENABLE_BASE) / ENABLE_PER_HART;
    addr -= cntx * ENABLE_PER_HART + ENABLE_BASE;
    if (cntx < contexts.size()) {
      ret = context_enable_read(&contexts[cntx], addr, &val);
    }
  } else if (CONTEXT_BASE <= addr && addr < REG_SIZE) {
    uint32_t cntx = (addr - CONTEXT_BASE) / CONTEXT_PER_HART;
    addr -= cntx * CONTEXT_PER_HART + CONTEXT_BASE;
    if (cntx < contexts.size()) {
      ret = context_read(&contexts[cntx], addr, &val);
    }
  }

  read_little_endian_reg(val, addr, len, bytes);

  return ret;
}

bool plic_t::store(reg_t addr, size_t len, const uint8_t* bytes)
{
  bool ret = false;
  uint32_t val = 0;

  switch (len) {
    case 4:
      break;
    case 8:
      // Implement double-word stores as a pair of word stores
      return store(addr, 4, bytes) && store(addr + 4, 4, bytes + 4);
    default:
      // Subword stores are not supported
      return false;
  }

  write_little_endian_reg(&val, addr, len, bytes);

  if (PRIORITY_BASE <= addr && addr < ENABLE_BASE) {
    ret = priority_write(addr, val);
  } else if (ENABLE_BASE <= addr && addr < CONTEXT_BASE) {
    uint32_t cntx = (addr - ENABLE_BASE) / ENABLE_PER_HART;
    addr -= cntx * ENABLE_PER_HART + ENABLE_BASE;
    if (cntx < contexts.size())
      ret = context_enable_write(&contexts[cntx], addr, val);
  } else if (CONTEXT_BASE <= addr && addr < REG_SIZE) {
    uint32_t cntx = (addr - CONTEXT_BASE) / CONTEXT_PER_HART;
    addr -= cntx * CONTEXT_PER_HART + CONTEXT_BASE;
    if (cntx < contexts.size())
      ret = context_write(&contexts[cntx], addr, val);
  }

  return ret;
}

std::string plic_generate_dts(const sim_t* sim)
{
  std::stringstream s;
  s << std::hex
    << "    PLIC: plic@" << PLIC_BASE << " {\n"
       "      compatible = \"riscv,plic0\";\n"
       "      #address-cells = <2>;\n"
       "      interrupts-extended = <" << std::dec;
  for (size_t i = 0; i < sim->get_cfg().nprocs(); i++)
    s << "&CPU" << i << "_intc 11 &CPU" << i << "_intc 9 ";
  reg_t plicbs = PLIC_BASE;
  reg_t plicsz = PLIC_SIZE;
  s << std::hex << ">;\n"
      "      reg = <0x" << (plicbs >> 32) << " 0x" << (plicbs & (uint32_t)-1) <<
      " 0x" << (plicsz >> 32) << " 0x" << (plicsz & (uint32_t)-1) << ">;\n"
      "      riscv,ndev = <0x" << PLIC_NDEV << ">;\n"
      "      riscv,max-priority = <0x" << ((1U << PLIC_PRIO_BITS) - 1) << ">;\n"
      "      #interrupt-cells = <1>;\n"
      "      interrupt-controller;\n"
      "    };\n";
  return s.str();
}

plic_t* plic_parse_from_fdt(const void* fdt, const sim_t* sim, reg_t* base, const std::vector<std::string>& sargs)
{
  uint32_t plic_ndev;
  if (fdt_parse_plic(fdt, base, &plic_ndev, "riscv,plic0") == 0 ||
      fdt_parse_plic(fdt, base, &plic_ndev, "sifive,plic-1.0.0") == 0)
    return new plic_t(sim, plic_ndev);
  else
    return nullptr;
}

REGISTER_DEVICE(plic, plic_parse_from_fdt, plic_generate_dts)

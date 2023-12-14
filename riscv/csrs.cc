// See LICENSE for license details.

// For std::any_of
#include <algorithm>

#include "csrs.h"
// For processor_t:
#include "processor.h"
#include "mmu.h"
// For get_field():
#include "decode_macros.h"
// For trap_virtual_instruction and trap_illegal_instruction:
#include "trap.h"
// For require():
#include "insn_macros.h"
// For CSR_DCSR_V:
#include "debug_defines.h"

// STATE macro used by require_privilege() macro:
#undef STATE
#define STATE (*p->get_state())

// implement class csr_t
csr_t::csr_t(const reg_t addr):
  address(addr),
  csr_priv(get_field(addr, 0x300)),
  csr_read_only(get_field(addr, 0xC00) == 3) {
}

void csr_t::verify_permissions(insn_t insn, bool write, processor_t* p) const {
  // Check permissions. Raise virtual-instruction exception if V=1,
  // privileges are insufficient, and the CSR belongs to supervisor or
  // hypervisor. Raise illegal-instruction exception otherwise.
  state_t* s = p->get_state();
  unsigned priv = s->prv == PRV_S && !s->v ? PRV_HS : s->prv;

  if ((csr_priv == PRV_S  && !p->extension_enabled('S')) ||
      (csr_priv == PRV_HS && !p->extension_enabled('H')))
    throw trap_illegal_instruction(insn.bits());

  if (write && csr_read_only)
    throw trap_illegal_instruction(insn.bits());
  if (priv < csr_priv) {
    if (s->v && csr_priv <= PRV_HS)
      throw trap_virtual_instruction(insn.bits());
    throw trap_illegal_instruction(insn.bits());
  }
}

csr_t::~csr_t() {
}

void csr_t::write(const reg_t val, processor_t* p) noexcept {
  const bool success = unlogged_write(val, p);
  if (success) {
    log_write(p);
  }
}

void csr_t::log_write(processor_t* p) const noexcept {
  log_special_write(address, written_value(p), p);
}

void csr_t::log_special_write(const reg_t UNUSED address, const reg_t UNUSED val, processor_t* p) const noexcept {
  if (p->get_log_commits_enabled())
    p->get_state()->log_reg_write[((address) << 4) | 4] = {val, 0};
}

reg_t csr_t::written_value(processor_t* p) const noexcept {
  return read(p);
}

// implement class basic_csr_t
basic_csr_t::basic_csr_t(const reg_t addr, const reg_t init):
  csr_t(addr),
  val(init) {
}

bool basic_csr_t::unlogged_write(const reg_t val, processor_t* p) noexcept {
  this->val = val;
  return true;
}

// implement class pmpaddr_csr_t
pmpaddr_csr_t::pmpaddr_csr_t(const reg_t addr):
  csr_t(addr),
  val(0),
  cfg(0),
  pmpidx(address - CSR_PMPADDR0) {
}

void pmpaddr_csr_t::verify_permissions(insn_t insn, bool write, processor_t* p) const {
  csr_t::verify_permissions(insn, write, p);
  // If n_pmp is zero, that means pmp is not implemented hence raise
  // trap if it tries to access the csr. I would prefer to implement
  // this by not instantiating any pmpaddr_csr_t for these regs, but
  // n_pmp can change after reset() is run.
  if (p->n_pmp == 0)
    throw trap_illegal_instruction(insn.bits());
}

reg_t pmpaddr_csr_t::read(processor_t* p) const noexcept {
  if ((cfg & PMP_A) >= PMP_NAPOT)
    return val | (~p->pmp_tor_mask() >> 1);
  return val & p->pmp_tor_mask();
}

bool pmpaddr_csr_t::unlogged_write(const reg_t val, processor_t* p) noexcept {
  // If no PMPs are configured, disallow access to all. Otherwise,
  // allow access to all, but unimplemented ones are hardwired to
  // zero. Note that n_pmp can change after reset(); otherwise I would
  // implement this in state_t::reset() by instantiating the correct
  // number of pmpaddr_csr_t.
  if (p->n_pmp == 0)
    return false;

  state_t* state = p->get_state();

  const bool lock_bypass = state->mseccfg->get_rlb(p);
  const bool locked = !lock_bypass && (cfg & PMP_L);

  if (pmpidx < p->n_pmp && !locked && !next_locked_and_tor(p)) {
    this->val = val & ((reg_t(1) << (MAX_PADDR_BITS - PMP_SHIFT)) - 1);
  }
  else
    return false;
  p->get_mmu()->flush_tlb();
  return true;
}

bool pmpaddr_csr_t::next_locked_and_tor(processor_t* p) const noexcept {
  state_t* state = p->get_state();
  if (pmpidx+1 >= state->max_pmp) return false;  // this is the last entry
  const bool lock_bypass = state->mseccfg->get_rlb(p);
  const bool next_locked = !lock_bypass && (state->pmpaddr[pmpidx+1]->cfg & PMP_L);
  const bool next_tor = (state->pmpaddr[pmpidx+1]->cfg & PMP_A) == PMP_TOR;
  return next_locked && next_tor;
}

reg_t pmpaddr_csr_t::tor_paddr(processor_t* p) const noexcept {
  return (val & p->pmp_tor_mask()) << PMP_SHIFT;
}

reg_t pmpaddr_csr_t::tor_base_paddr(processor_t* p) const noexcept {
  state_t* state = p->get_state();
  if (pmpidx == 0) return 0;  // entry 0 always uses 0 as base
  return state->pmpaddr[pmpidx-1]->tor_paddr(p);
}

reg_t pmpaddr_csr_t::napot_mask(processor_t* p) const noexcept {
  bool is_na4 = (cfg & PMP_A) == PMP_NA4;
  reg_t mask = (val << 1) | (!is_na4) | ~p->pmp_tor_mask();
  return ~(mask & ~(mask + 1)) << PMP_SHIFT;
}

bool pmpaddr_csr_t::match4(reg_t addr, processor_t* p) const noexcept {
  if ((cfg & PMP_A) == 0) return false;
  bool is_tor = (cfg & PMP_A) == PMP_TOR;
  if (is_tor) return tor_base_paddr(p) <= addr && addr < tor_paddr(p);
  // NAPOT or NA4:
  return ((addr ^ tor_paddr(p)) & napot_mask(p)) == 0;
}

bool pmpaddr_csr_t::subset_match(reg_t addr, reg_t len, processor_t* p) const noexcept {
  if ((addr | len) & (len - 1))
    abort();
  reg_t base = tor_base_paddr(p);
  reg_t tor = tor_paddr(p);

  if ((cfg & PMP_A) == 0) return false;

  bool is_tor = (cfg & PMP_A) == PMP_TOR;
  bool begins_after_lower = addr >= base;
  bool begins_after_upper = addr >= tor;
  bool ends_before_lower = (addr & -len) < (base & -len);
  bool ends_before_upper = (addr & -len) < (tor & -len);
  bool tor_homogeneous = ends_before_lower || begins_after_upper ||
    (begins_after_lower && ends_before_upper);

  bool mask_homogeneous = ~(napot_mask(p) << 1) & len;
  bool napot_homogeneous = mask_homogeneous || ((addr ^ tor) / len) != 0;

  return !(is_tor ? tor_homogeneous : napot_homogeneous);
}

bool pmpaddr_csr_t::access_ok(access_type type, reg_t mode, processor_t* p) const noexcept {
  state_t* state = p->get_state();

  const bool cfgx = cfg & PMP_X;
  const bool cfgw = cfg & PMP_W;
  const bool cfgr = cfg & PMP_R;
  const bool cfgl = cfg & PMP_L;

  const bool prvm = mode == PRV_M;

  const bool typer = type == LOAD;
  const bool typex = type == FETCH;
  const bool typew = type == STORE;
  const bool normal_rwx = (typer && cfgr) || (typew && cfgw) || (typex && cfgx);
  const bool mseccfg_mml = state->mseccfg->get_mml(p);

  if (mseccfg_mml) {
    if (cfgx && cfgw && cfgr && cfgl) {
      // Locked Shared data region: Read only on both M and S/U mode.
      return typer;
    } else {
      const bool mml_shared_region = !cfgr && cfgw;
      const bool mml_chk_normal = (prvm == cfgl) && normal_rwx;
      const bool mml_chk_shared =
              (!cfgl && cfgx && (typer || typew)) ||
              (!cfgl && !cfgx && (typer || (typew && prvm))) ||
              (cfgl && typex) ||
              (cfgl && typer && cfgx && prvm);
      return mml_shared_region ? mml_chk_shared : mml_chk_normal;
    }
  } else {
    const bool m_bypass = (prvm && !cfgl);
    return m_bypass || normal_rwx;
  }
}

// implement class pmpcfg_csr_t
pmpcfg_csr_t::pmpcfg_csr_t(const reg_t addr):
  csr_t(addr) {
}

void pmpcfg_csr_t::verify_permissions(insn_t insn, bool write, processor_t* p) const {
  csr_t::verify_permissions(insn, write, p);
  // If n_pmp is zero, that means pmp is not implemented hence raise
  // trap if it tries to access the csr. I would prefer to implement
  // this by not instantiating any pmpcfg_csr_t for these regs, but
  // n_pmp can change after reset() is run.
  if (p->n_pmp == 0)
    throw trap_illegal_instruction(insn.bits());
}

reg_t pmpcfg_csr_t::read(processor_t* p) const noexcept {
  reg_t cfg_res = 0;
  state_t* state = p->get_state();
  for (size_t i0 = (address - CSR_PMPCFG0) * 4, i = i0; i < i0 + p->get_xlen() / 8 && i < state->max_pmp; i++)
    cfg_res |= reg_t(state->pmpaddr[i]->cfg) << (8 * (i - i0));
  return cfg_res;
}

bool pmpcfg_csr_t::unlogged_write(const reg_t val, processor_t* p) noexcept {
  if (p->n_pmp == 0)
    return false;

  bool write_success = false;
  state_t* state = p->get_state();
  const bool rlb = state->mseccfg->get_rlb(p);
  const bool mml = state->mseccfg->get_mml(p);
  for (size_t i0 = (address - CSR_PMPCFG0) * 4, i = i0; i < i0 + p->get_xlen() / 8; i++) {
    if (i < p->n_pmp) {
      const bool locked = (state->pmpaddr[i]->cfg & PMP_L);
      if (rlb || !locked) {
        uint8_t cfg = (val >> (8 * (i - i0))) & (PMP_R | PMP_W | PMP_X | PMP_A | PMP_L);
        // Drop R=0 W=1 when MML = 0
        // Remove the restriction when MML = 1
        if (!mml) {
          cfg &= ~PMP_W | ((cfg & PMP_R) ? PMP_W : 0);
        }
        // Disallow A=NA4 when granularity > 4
        if (p->lg_pmp_granularity != PMP_SHIFT && (cfg & PMP_A) == PMP_NA4)
          cfg |= PMP_NAPOT;
        /*
         * Adding a rule with executable privileges that either is M-mode-only or a locked Shared-Region
         * is not possible and such pmpcfg writes are ignored, leaving pmpcfg unchanged.
         * This restriction can be temporarily lifted e.g. during the boot process, by setting mseccfg.RLB.
         */
        const bool cfgx = cfg & PMP_X;
        const bool cfgw = cfg & PMP_W;
        const bool cfgr = cfg & PMP_R;
        if (rlb || !(mml && ((cfg & PMP_L)      // M-mode-only or a locked Shared-Region
                && !(cfgx && cfgw && cfgr)      // RWX = 111 is allowed
                && (cfgx || (cfgw && !cfgr))    // X=1 or RW=01 is not allowed
        ))) {
          state->pmpaddr[i]->cfg = cfg;
        }
      }
      write_success = true;
    }
  }
  p->get_mmu()->flush_tlb();
  return write_success;
}

// implement class mseccfg_csr_t
mseccfg_csr_t::mseccfg_csr_t(const reg_t addr):
    basic_csr_t(addr, 0) {
}

void mseccfg_csr_t::verify_permissions(insn_t insn, bool write, processor_t* p) const {
  basic_csr_t::verify_permissions(insn, write, p);
  if (!p->extension_enabled(EXT_SMEPMP))
    throw trap_illegal_instruction(insn.bits());
}

bool mseccfg_csr_t::get_mml(processor_t* p) const noexcept {
  return (read(p) & MSECCFG_MML);
}

bool mseccfg_csr_t::get_mmwp(processor_t* p) const noexcept {
  return (read(p) & MSECCFG_MMWP);
}

bool mseccfg_csr_t::get_rlb(processor_t* p) const noexcept {
  return (read(p) & MSECCFG_RLB);
}

bool mseccfg_csr_t::unlogged_write(const reg_t val, processor_t* p) noexcept {
  if (p->n_pmp == 0)
    return false;

  state_t* state = p->get_state();

  // pmpcfg.L is 1 in any rule or entry (including disabled entries)
  const bool pmplock_recorded = std::any_of(state->pmpaddr, state->pmpaddr + p->n_pmp,
          [](const pmpaddr_csr_t_p & c) { return c->is_locked(); } );
  reg_t new_val = read(p);

  // When RLB is 0 and pmplock_recorded, RLB is locked to 0.
  // Otherwise set the RLB bit according val
  if (!(pmplock_recorded && (read(p) & MSECCFG_RLB) == 0)) {
    new_val &= ~MSECCFG_RLB;
    new_val |= (val & MSECCFG_RLB);
  }

  new_val |= (val & MSECCFG_MMWP);  //MMWP is sticky
  new_val |= (val & MSECCFG_MML);   //MML is sticky

  p->get_mmu()->flush_tlb();

  return basic_csr_t::unlogged_write(new_val, p);
}

// implement class virtualized_csr_t
virtualized_csr_t::virtualized_csr_t(csr_t_p orig, csr_t_p virt):
  csr_t(orig->address),
  orig_csr(orig),
  virt_csr(virt) {
}

reg_t virtualized_csr_t::read(processor_t* p) const noexcept {
  state_t* state = p->get_state();
  return readvirt(state->v, p);
}

reg_t virtualized_csr_t::readvirt(bool virt, processor_t* p) const noexcept {
  return virt ? virt_csr->read(p) : orig_csr->read(p);
}

bool virtualized_csr_t::unlogged_write(const reg_t val, processor_t* p) noexcept {
  state_t* state = p->get_state();
  if (state->v)
    virt_csr->write(val, p);
  else
    orig_csr->write(val, p);
  return false; // virt_csr or orig_csr has already logged
}

// implement class epc_csr_t
epc_csr_t::epc_csr_t(const reg_t addr):
  csr_t(addr),
  val(0) {
}

reg_t epc_csr_t::read(processor_t* p) const noexcept {
  return val & p->pc_alignment_mask();
}

bool epc_csr_t::unlogged_write(const reg_t val, processor_t* p) noexcept {
  this->val = val & ~(reg_t)1;
  return true;
}

// implement class tvec_csr_t
tvec_csr_t::tvec_csr_t(const reg_t addr):
  csr_t(addr),
  val(0) {
}

reg_t tvec_csr_t::read(processor_t* p) const noexcept {
  return val;
}

bool tvec_csr_t::unlogged_write(const reg_t val, processor_t* p) noexcept {
  this->val = val & ~(reg_t)2;
  return true;
}

// implement class cause_csr_t
cause_csr_t::cause_csr_t(const reg_t addr):
  basic_csr_t(addr, 0) {
}

reg_t cause_csr_t::read(processor_t* p) const noexcept {
  reg_t val = basic_csr_t::read(p);
  // When reading, the interrupt bit needs to adjust to xlen. Spike does
  // not generally support dynamic xlen, but this code was (partly)
  // there since at least 2015 (ea58df8 and c4350ef).
  if (p->get_isa().get_max_xlen() > p->get_xlen()) // Move interrupt bit to top of xlen
    return val | ((val >> (p->get_isa().get_max_xlen()-1)) << (p->get_xlen()-1));
  return val;
}

// implement class base_status_csr_t
base_status_csr_t::base_status_csr_t(processor_t* const p, const reg_t addr):
  csr_t(addr),
  has_page(p->extension_enabled_const('S') && p->supports_impl(IMPL_MMU)),
  sstatus_write_mask(compute_sstatus_write_mask(p)),
  sstatus_read_mask(sstatus_write_mask | SSTATUS_UBE | SSTATUS_UXL
                    | (p->get_const_xlen() == 32 ? SSTATUS32_SD : SSTATUS64_SD)) {
}

reg_t base_status_csr_t::compute_sstatus_write_mask(processor_t* p) const noexcept {
  // If a configuration has FS bits, they will always be accessible no
  // matter the state of misa.
  const bool has_fs = (p->extension_enabled('S') || p->extension_enabled('F')
              || p->extension_enabled('V')) && !p->extension_enabled(EXT_ZFINX);
  const bool has_vs = p->extension_enabled('V');
  return 0
    | (p->extension_enabled('S') ? (SSTATUS_SIE | SSTATUS_SPIE | SSTATUS_SPP) : 0)
    | (has_page ? (SSTATUS_SUM | SSTATUS_MXR) : 0)
    | (has_fs ? SSTATUS_FS : 0)
    | (p->any_custom_extensions() ? SSTATUS_XS : 0)
    | (has_vs ? SSTATUS_VS : 0)
    ;
}

reg_t base_status_csr_t::adjust_sd(const reg_t val, processor_t* p) const noexcept {
  // This uses get_const_xlen() instead of get_xlen() not only because
  // the variable is static, so it's only called once, but also
  // because the SD bit moves when XLEN changes, which means we would
  // need to call adjust_sd() on every read, instead of on every
  // write.
  static const reg_t sd_bit = p->get_const_xlen() == 64 ? SSTATUS64_SD : SSTATUS32_SD;
  if (((val & SSTATUS_FS) == SSTATUS_FS) ||
      ((val & SSTATUS_VS) == SSTATUS_VS) ||
      ((val & SSTATUS_XS) == SSTATUS_XS)) {
    return val | sd_bit;
  }
  return val & ~sd_bit;
}

void base_status_csr_t::maybe_flush_tlb(const reg_t newval, processor_t* p) noexcept {
  if ((newval ^ read(p)) &
      (MSTATUS_MPP | MSTATUS_MPRV
       | (has_page ? (MSTATUS_MXR | MSTATUS_SUM) : 0)
      ))
    p->get_mmu()->flush_tlb();
}

namespace {
  int xlen_to_uxl(int xlen) {
    if (xlen == 32)
      return 1;
    if (xlen == 64)
      return 2;
    abort();
  }
}

// implement class vsstatus_csr_t
vsstatus_csr_t::vsstatus_csr_t(processor_t* const p, const reg_t addr):
  base_status_csr_t(p, addr),
  val(p->get_state()->mstatus->read(p) & sstatus_read_mask) {
}

bool vsstatus_csr_t::unlogged_write(const reg_t val, processor_t* p) noexcept {
  state_t* state = p->get_state();
  const reg_t newval = (this->val & ~sstatus_write_mask) | (val & sstatus_write_mask);
  if (state->v) maybe_flush_tlb(newval, p);
  this->val = adjust_sd(newval, p);
  return true;
}

// implement class sstatus_proxy_csr_t
sstatus_proxy_csr_t::sstatus_proxy_csr_t(processor_t* const p, const reg_t addr, mstatus_csr_t_p mstatus):
  base_status_csr_t(p, addr),
  mstatus(mstatus) {
}

bool sstatus_proxy_csr_t::unlogged_write(const reg_t val, processor_t* p) noexcept {
  const reg_t new_mstatus = (mstatus->read(p) & ~sstatus_write_mask) | (val & sstatus_write_mask);

  // On RV32 this will only log the low 32 bits, so make sure we're
  // not modifying anything in the upper 32 bits.
  assert((sstatus_write_mask & 0xffffffffU) == sstatus_write_mask);

  mstatus->write(new_mstatus, p);
  return false; // avoid double logging: already logged by mstatus->write()
}

// implement class mstatus_csr_t
mstatus_csr_t::mstatus_csr_t(processor_t* const p, const reg_t addr):
  base_status_csr_t(p, addr),
  val(compute_mstatus_initial_value(p)) {
}

bool mstatus_csr_t::unlogged_write(const reg_t val, processor_t* p) noexcept {
  const bool has_mpv = p->extension_enabled('H');
  const bool has_gva = has_mpv;

  const reg_t mask = sstatus_write_mask
                   | MSTATUS_MIE | MSTATUS_MPIE
                   | (p->extension_enabled('U') ? MSTATUS_MPRV : 0)
                   | MSTATUS_MPP | MSTATUS_TW
                   | (p->extension_enabled('S') ? MSTATUS_TSR : 0)
                   | (has_page ? MSTATUS_TVM : 0)
                   | (has_gva ? MSTATUS_GVA : 0)
                   | (has_mpv ? MSTATUS_MPV : 0);

  const reg_t requested_mpp = p->legalize_privilege(get_field(val, MSTATUS_MPP));
  const reg_t adjusted_val = set_field(val, MSTATUS_MPP, requested_mpp);
  const reg_t new_mstatus = (read(p) & ~mask) | (adjusted_val & mask);
  maybe_flush_tlb(new_mstatus, p);
  this->val = adjust_sd(new_mstatus, p);
  return true;
}

reg_t mstatus_csr_t::compute_mstatus_initial_value(processor_t* const p) const noexcept {
  const reg_t big_endian_bits = (p->extension_enabled_const('U') ? MSTATUS_UBE : 0)
                              | (p->extension_enabled_const('S') ? MSTATUS_SBE : 0)
                              | MSTATUS_MBE;
  return 0
         | set_field((reg_t)0, MSTATUS_MPP, p->extension_enabled_const('U') ? PRV_U : PRV_M)
         | (p->extension_enabled_const('U') && (p->get_const_xlen() != 32) ? set_field((reg_t)0, MSTATUS_UXL, xlen_to_uxl(p->get_const_xlen())) : 0)
         | (p->extension_enabled_const('S') && (p->get_const_xlen() != 32) ? set_field((reg_t)0, MSTATUS_SXL, xlen_to_uxl(p->get_const_xlen())) : 0)
         | (p->get_mmu()->is_target_big_endian() ? big_endian_bits : 0)
         | 0;  // initial value for mstatus
}

// implement class mnstatus_csr_t
mnstatus_csr_t::mnstatus_csr_t(const reg_t addr):
  basic_csr_t(addr, 0) {
}

bool mnstatus_csr_t::unlogged_write(const reg_t val, processor_t* p) noexcept {
  // NMIE can be set but not cleared
  const reg_t mask = (~read(p) & MNSTATUS_NMIE)
                   | (p->extension_enabled('H') ? MNSTATUS_MNPV : 0)
                   | MNSTATUS_MNPP;

  const reg_t requested_mnpp = p->legalize_privilege(get_field(val, MNSTATUS_MNPP));
  const reg_t adjusted_val = set_field(val, MNSTATUS_MNPP, requested_mnpp);
  const reg_t new_mnstatus = (read(p) & ~mask) | (adjusted_val & mask);

  return basic_csr_t::unlogged_write(new_mnstatus, p);
}

// implement class rv32_low_csr_t
rv32_low_csr_t::rv32_low_csr_t(const reg_t addr, csr_t_p orig):
  csr_t(addr),
  orig(orig) {
}

reg_t rv32_low_csr_t::read(processor_t* p) const noexcept {
  return orig->read(p) & 0xffffffffU;
}

void rv32_low_csr_t::verify_permissions(insn_t insn, bool write, processor_t* p) const {
  orig->verify_permissions(insn, write, p);
}

bool rv32_low_csr_t::unlogged_write(const reg_t val, processor_t* p) noexcept {
  return orig->unlogged_write((orig->written_value(p) >> 32 << 32) | (val & 0xffffffffU), p);
}

reg_t rv32_low_csr_t::written_value(processor_t* p) const noexcept {
  return orig->written_value(p) & 0xffffffffU;
}

// implement class rv32_high_csr_t
rv32_high_csr_t::rv32_high_csr_t(const reg_t addr, csr_t_p orig):
  csr_t(addr),
  orig(orig) {
}

reg_t rv32_high_csr_t::read(processor_t* p) const noexcept {
  return (orig->read(p) >> 32) & 0xffffffffU;
}

void rv32_high_csr_t::verify_permissions(insn_t insn, bool write, processor_t* p) const {
  orig->verify_permissions(insn, write, p);
}

bool rv32_high_csr_t::unlogged_write(const reg_t val, processor_t* p) noexcept {
  return orig->unlogged_write((orig->written_value(p) << 32 >> 32) | ((val & 0xffffffffU) << 32), p);
}

reg_t rv32_high_csr_t::written_value(processor_t* p) const noexcept {
  return (orig->written_value(p) >> 32) & 0xffffffffU;
}

// implement class sstatus_csr_t
sstatus_csr_t::sstatus_csr_t(sstatus_proxy_csr_t_p orig, vsstatus_csr_t_p virt):
  virtualized_csr_t(orig, virt),
  orig_sstatus(orig),
  virt_sstatus(virt) {
}

void sstatus_csr_t::dirty(const reg_t dirties, processor_t* p) {
  state_t* state = p->get_state();

  // As an optimization, return early if already dirty.
  if ((orig_sstatus->read(p) & dirties) == dirties) {
    if (likely(!state->v || (virt_sstatus->read(p) & dirties) == dirties))
      return;
  }

  // Catch problems like #823 where P-extension instructions were not
  // checking for mstatus.VS!=Off:
  if (!enabled(dirties, p)) abort();

  orig_sstatus->write(orig_sstatus->read(p) | dirties, p);
  if (state->v) {
    virt_sstatus->write(virt_sstatus->read(p) | dirties, p);
  }
}

bool sstatus_csr_t::enabled(const reg_t which, processor_t* p) {
  state_t* state = p->get_state();

  if ((orig_sstatus->read(p) & which) != 0) {
    if (!state->v || (virt_sstatus->read(p) & which) != 0)
      return true;
  }

  // If the field doesn't exist, it is always enabled. See #823.
  if (!orig_sstatus->field_exists(which))
    return true;

  return false;
}

// implement class misa_csr_t
misa_csr_t::misa_csr_t(const reg_t addr, const reg_t max_isa):
  basic_csr_t(addr, max_isa),
  max_isa(max_isa),
  write_mask(max_isa & (0  // allow MAFDQCHV bits in MISA to be modified
                        | (1L << ('M' - 'A'))
                        | (1L << ('A' - 'A'))
                        | (1L << ('F' - 'A'))
                        | (1L << ('D' - 'A'))
                        | (1L << ('Q' - 'A'))
                        | (1L << ('C' - 'A'))
                        | (1L << ('H' - 'A'))
                        | (1L << ('V' - 'A'))
                        )
             ) {
}

reg_t misa_csr_t::dependency(const reg_t val, const char feature, const char depends_on) const noexcept {
  return (val & (1L << (depends_on - 'A'))) ? val : (val & ~(1L << (feature - 'A')));
}

bool misa_csr_t::unlogged_write(const reg_t val, processor_t* p) noexcept {
  state_t* state = p->get_state();
  const reg_t old_misa = read(p);

  // the write is ignored if increasing IALIGN would misalign the PC
  if (!(val & (1L << ('C' - 'A'))) && (old_misa & (1L << ('C' - 'A'))) && (state->pc & 2))
    return false;

  reg_t adjusted_val = val;
  adjusted_val = dependency(adjusted_val, 'D', 'F');
  adjusted_val = dependency(adjusted_val, 'Q', 'D');
  adjusted_val = dependency(adjusted_val, 'V', 'D');

  const bool prev_h = old_misa & (1L << ('H' - 'A'));
  const reg_t new_misa = (adjusted_val & write_mask) | (old_misa & ~write_mask);
  const bool new_h = new_misa & (1L << ('H' - 'A'));

  p->set_extension_enable(EXT_ZCA, (new_misa & (1L << ('C' - 'A'))) || !p->get_isa().extension_enabled('C'));
  p->set_extension_enable(EXT_ZCF, (new_misa & (1L << ('F' - 'A'))) && p->extension_enabled(EXT_ZCA));
  p->set_extension_enable(EXT_ZCD, (new_misa & (1L << ('D' - 'A'))) && p->extension_enabled(EXT_ZCA));
  p->set_extension_enable(EXT_ZCB, p->extension_enabled(EXT_ZCA));
  p->set_extension_enable(EXT_ZCMP, p->extension_enabled(EXT_ZCA));
  p->set_extension_enable(EXT_ZCMT, p->extension_enabled(EXT_ZCA));
  p->set_extension_enable(EXT_ZFH, new_misa & (1L << ('F' - 'A')));
  p->set_extension_enable(EXT_ZFHMIN, new_misa & (1L << ('F' - 'A')));
  p->set_extension_enable(EXT_ZVFH, (new_misa & (1L << ('V' - 'A'))) && p->extension_enabled(EXT_ZFHMIN));
  p->set_extension_enable(EXT_ZVFHMIN, new_misa & (1L << ('V' - 'A')));

  // update the hypervisor-only bits in MEDELEG and other CSRs
  if (!new_h && prev_h) {
    reg_t hypervisor_exceptions = 0
      | (1 << CAUSE_VIRTUAL_SUPERVISOR_ECALL)
      | (1 << CAUSE_FETCH_GUEST_PAGE_FAULT)
      | (1 << CAUSE_LOAD_GUEST_PAGE_FAULT)
      | (1 << CAUSE_VIRTUAL_INSTRUCTION)
      | (1 << CAUSE_STORE_GUEST_PAGE_FAULT)
      ;

    state->medeleg->write(state->medeleg->read(p) & ~hypervisor_exceptions, p);
    if (state->mnstatus) state->mnstatus->write(state->mnstatus->read(p) & ~MNSTATUS_MNPV, p);
    const reg_t new_mstatus = state->mstatus->read(p) & ~(MSTATUS_GVA | MSTATUS_MPV);
    state->mstatus->write(new_mstatus, p);
    if (state->mstatush) state->mstatush->write(new_mstatus >> 32, p);  // log mstatush change
    state->mie->write_with_mask(MIP_HS_MASK, 0, p);  // also takes care of hie, sie
    state->mip->write_with_mask(MIP_HS_MASK, 0, p);  // also takes care of hip, sip, hvip
    state->hstatus->write(0, p);
    for (reg_t i = 3; i < N_HPMCOUNTERS + 3; ++i) {
      const reg_t new_mevent = state->mevent[i - 3]->read(p) & ~(MHPMEVENT_VUINH | MHPMEVENT_VSINH);
      state->mevent[i - 3]->write(new_mevent, p);
    }
  }

  return basic_csr_t::unlogged_write(new_misa, p);
}

bool misa_csr_t::extension_enabled_const(unsigned char ext, processor_t* p) const noexcept {
  assert(!(1 & (write_mask >> (ext - 'A'))));
  return extension_enabled(ext, p);
}

// implement class mip_or_mie_csr_t
mip_or_mie_csr_t::mip_or_mie_csr_t(const reg_t addr):
  csr_t(addr),
  val(0) {
}

reg_t mip_or_mie_csr_t::read(processor_t* p) const noexcept {
  return val;
}

void mip_or_mie_csr_t::write_with_mask(const reg_t mask, const reg_t val, processor_t* p) noexcept {
  this->val = (this->val & ~mask) | (val & mask);
  log_write(p);
}

bool mip_or_mie_csr_t::unlogged_write(const reg_t val, processor_t* p) noexcept {
  write_with_mask(write_mask(p), val, p);
  return false; // avoid double logging: already logged by write_with_mask()
}

mip_csr_t::mip_csr_t(const reg_t addr):
  mip_or_mie_csr_t(addr) {
}

void mip_csr_t::backdoor_write_with_mask(const reg_t mask, const reg_t val) noexcept {
  this->val = (this->val & ~mask) | (val & mask);
}

reg_t mip_csr_t::write_mask(processor_t* p) const noexcept {
  state_t* state = p->get_state();

  // MIP_STIP is writable unless SSTC exists and STCE is set in MENVCFG
  const reg_t supervisor_ints = p->extension_enabled('S') ? MIP_SSIP | ((state->menvcfg->read(p) &  MENVCFG_STCE) ? 0 : MIP_STIP) | MIP_SEIP : 0;
  const reg_t lscof_int = p->extension_enabled(EXT_SSCOFPMF) ? MIP_LCOFIP : 0;
  const reg_t vssip_int = p->extension_enabled('H') ? MIP_VSSIP : 0;
  const reg_t hypervisor_ints = p->extension_enabled('H') ? MIP_HS_MASK : 0;
  // We must mask off sgeip, vstip, and vseip. All three of these
  // bits are aliases for the same bits in hip. The hip spec says:
  //  * sgeip is read-only -- write hgeip instead
  //  * vseip is read-only -- write hvip instead
  //  * vstip is read-only -- write hvip instead
  return (supervisor_ints | hypervisor_ints | lscof_int) &
         (MIP_SEIP | MIP_SSIP | MIP_STIP | MIP_LCOFIP | vssip_int);
}

mie_csr_t::mie_csr_t(const reg_t addr):
  mip_or_mie_csr_t(addr) {
}

reg_t mie_csr_t::write_mask(processor_t* p) const noexcept {
  const reg_t supervisor_ints = p->extension_enabled('S') ? MIP_SSIP | MIP_STIP | MIP_SEIP : 0;
  const reg_t lscof_int = p->extension_enabled(EXT_SSCOFPMF) ? MIP_LCOFIP : 0;
  const reg_t hypervisor_ints = p->extension_enabled('H') ? MIP_HS_MASK : 0;
  const reg_t coprocessor_ints = (reg_t)p->any_custom_extensions() << IRQ_COP;
  const reg_t delegable_ints = supervisor_ints | coprocessor_ints | lscof_int;
  const reg_t all_ints = delegable_ints | hypervisor_ints | MIP_MSIP | MIP_MTIP | MIP_MEIP;
  return all_ints;
}

// implement class generic_int_accessor_t
generic_int_accessor_t::generic_int_accessor_t(state_t* const state,
                                               const reg_t read_mask,
                                               const reg_t ip_write_mask,
                                               const reg_t ie_write_mask,
                                               const mask_mode_t mask_mode,
                                               const int shiftamt):
  state(state),
  read_mask(read_mask),
  ip_write_mask(ip_write_mask),
  ie_write_mask(ie_write_mask),
  mask_mideleg(mask_mode == MIDELEG),
  mask_hideleg(mask_mode == HIDELEG),
  shiftamt(shiftamt) {
}

reg_t generic_int_accessor_t::ip_read(processor_t* p) const noexcept {
  state_t* s = p->get_state();
  return (state->mip->read(p) & deleg_mask(p) & read_mask) >> shiftamt;
}

void generic_int_accessor_t::ip_write(const reg_t val, processor_t* p) noexcept {
  state_t* s = p->get_state();
  const reg_t mask = deleg_mask(p) & ip_write_mask;
  state->mip->write_with_mask(mask, val << shiftamt, p);
}

reg_t generic_int_accessor_t::ie_read(processor_t *p) const noexcept {
  state_t* s = p->get_state();
  return (state->mie->read(p) & deleg_mask(p) & read_mask) >> shiftamt;
}

void generic_int_accessor_t::ie_write(const reg_t val, processor_t* p) noexcept {
  state_t* s = p->get_state();
  const reg_t mask = deleg_mask(p) & ie_write_mask;
  state->mie->write_with_mask(mask, val << shiftamt, p);
}

reg_t generic_int_accessor_t::deleg_mask(processor_t* p) const {
  state_t* s = p->get_state();
  const reg_t hideleg_mask = mask_hideleg ? state->hideleg->read(p) : (reg_t)~0;
  const reg_t mideleg_mask = mask_mideleg ? state->mideleg->read(p) : (reg_t)~0;
  return hideleg_mask & mideleg_mask;
}

// implement class mip_proxy_csr_t
mip_proxy_csr_t::mip_proxy_csr_t(const reg_t addr, generic_int_accessor_t_p accr):
  csr_t(addr),
  accr(accr) {
}

reg_t mip_proxy_csr_t::read(processor_t* p) const noexcept {
  return accr->ip_read(p);
}

bool mip_proxy_csr_t::unlogged_write(const reg_t val, processor_t* p) noexcept {
  accr->ip_write(val, p);
  return false;  // accr has already logged
}

// implement class mie_proxy_csr_t
mie_proxy_csr_t::mie_proxy_csr_t(const reg_t addr, generic_int_accessor_t_p accr):
  csr_t(addr),
  accr(accr) {
}

reg_t mie_proxy_csr_t::read(processor_t* p) const noexcept {
  return accr->ie_read(p);
}

bool mie_proxy_csr_t::unlogged_write(const reg_t val, processor_t* p) noexcept {
  accr->ie_write(val, p);
  return false;  // accr has already logged
}

// implement class mideleg_csr_t
mideleg_csr_t::mideleg_csr_t(const reg_t addr):
  basic_csr_t(addr, 0) {
}

reg_t mideleg_csr_t::read(processor_t* p) const noexcept {
  reg_t val = basic_csr_t::read(p);
  if (p->extension_enabled('H')) return val | MIDELEG_FORCED_MASK;
  // No need to clear MIDELEG_FORCED_MASK because those bits can never
  // get set in val.
  return val;
}

void mideleg_csr_t::verify_permissions(insn_t insn, bool write, processor_t* p) const {
  basic_csr_t::verify_permissions(insn, write, p);
  if (!p->extension_enabled('S'))
    throw trap_illegal_instruction(insn.bits());
}

bool mideleg_csr_t::unlogged_write(const reg_t val, processor_t* p) noexcept {
  const reg_t supervisor_ints = p->extension_enabled('S') ? MIP_SSIP | MIP_STIP | MIP_SEIP : 0;
  const reg_t lscof_int = p->extension_enabled(EXT_SSCOFPMF) ? MIP_LCOFIP : 0;
  const reg_t coprocessor_ints = (reg_t)p->any_custom_extensions() << IRQ_COP;
  const reg_t delegable_ints = supervisor_ints | coprocessor_ints | lscof_int;

  return basic_csr_t::unlogged_write(val & delegable_ints, p);
}

// implement class medeleg_csr_t
medeleg_csr_t::medeleg_csr_t(const reg_t addr):
  basic_csr_t(addr, 0),
  hypervisor_exceptions(0
                        | (1 << CAUSE_VIRTUAL_SUPERVISOR_ECALL)
                        | (1 << CAUSE_FETCH_GUEST_PAGE_FAULT)
                        | (1 << CAUSE_LOAD_GUEST_PAGE_FAULT)
                        | (1 << CAUSE_VIRTUAL_INSTRUCTION)
                        | (1 << CAUSE_STORE_GUEST_PAGE_FAULT)
                        ) {
}

void medeleg_csr_t::verify_permissions(insn_t insn, bool write, processor_t* p) const {
  basic_csr_t::verify_permissions(insn, write, p);
  if (!p->extension_enabled('S'))
    throw trap_illegal_instruction(insn.bits());
}

bool medeleg_csr_t::unlogged_write(const reg_t val, processor_t* p) noexcept {
  const reg_t mask = 0
    | (1 << CAUSE_MISALIGNED_FETCH)
    | (1 << CAUSE_FETCH_ACCESS)
    | (1 << CAUSE_ILLEGAL_INSTRUCTION)
    | (1 << CAUSE_BREAKPOINT)
    | (1 << CAUSE_MISALIGNED_LOAD)
    | (1 << CAUSE_LOAD_ACCESS)
    | (1 << CAUSE_MISALIGNED_STORE) 
    | (1 << CAUSE_STORE_ACCESS)
    | (1 << CAUSE_USER_ECALL)
    | (1 << CAUSE_SUPERVISOR_ECALL)
    | (1 << CAUSE_FETCH_PAGE_FAULT)
    | (1 << CAUSE_LOAD_PAGE_FAULT)
    | (1 << CAUSE_STORE_PAGE_FAULT)
    | (p->extension_enabled('H') ? hypervisor_exceptions : 0)
    ;
  return basic_csr_t::unlogged_write((read(p) & ~mask) | (val & mask), p);
}

// implement class masked_csr_t
masked_csr_t::masked_csr_t(const reg_t addr, const reg_t mask, const reg_t init):
  basic_csr_t(addr, init),
  mask(mask) {
}

bool masked_csr_t::unlogged_write(const reg_t val, processor_t* p) noexcept {
  return basic_csr_t::unlogged_write((read(p) & ~mask) | (val & mask), p);
}

envcfg_csr_t::envcfg_csr_t(const reg_t addr, const reg_t mask,
                             const reg_t init):
  masked_csr_t(addr, mask, init) {
  // In unlogged_write() we WARLize this field for all three of [msh]envcfg
  assert(MENVCFG_CBIE == SENVCFG_CBIE && MENVCFG_CBIE == HENVCFG_CBIE);
}

bool envcfg_csr_t::unlogged_write(const reg_t val, processor_t* p) noexcept {
  const reg_t cbie_reserved = 2; // Reserved value of xenvcfg.CBIE
  const reg_t adjusted_val = get_field(val, MENVCFG_CBIE) != cbie_reserved ? val : set_field(val, MENVCFG_CBIE, 0);
  return masked_csr_t::unlogged_write(adjusted_val, p);
}

// implement class henvcfg_csr_t
henvcfg_csr_t::henvcfg_csr_t(const reg_t addr, const reg_t mask, const reg_t init, csr_t_p menvcfg):
  envcfg_csr_t(addr, mask, init),
  menvcfg(menvcfg) {
}

// implement class base_atp_csr_t and family
base_atp_csr_t::base_atp_csr_t(const reg_t addr):
  basic_csr_t(addr, 0) {
}

bool base_atp_csr_t::unlogged_write(const reg_t val, processor_t* p) noexcept {
  const reg_t newval = p->supports_impl(IMPL_MMU) ? compute_new_satp(val, p) : 0;
  if (newval != read(p))
    p->get_mmu()->flush_tlb();
  return basic_csr_t::unlogged_write(newval, p);
}

bool base_atp_csr_t::satp_valid(reg_t val, processor_t* p) const noexcept {
  if (p->get_xlen() == 32) {
    switch (get_field(val, SATP32_MODE)) {
      case SATP_MODE_SV32: return p->supports_impl(IMPL_MMU_SV32);
      case SATP_MODE_OFF: return true;
      default: return false;
    }
  } else {
    switch (get_field(val, SATP64_MODE)) {
      case SATP_MODE_SV39: return p->supports_impl(IMPL_MMU_SV39);
      case SATP_MODE_SV48: return p->supports_impl(IMPL_MMU_SV48);
      case SATP_MODE_SV57: return p->supports_impl(IMPL_MMU_SV57);
      case SATP_MODE_OFF: return true;
      default: return false;
    }
  }
}

reg_t base_atp_csr_t::compute_new_satp(reg_t val, processor_t* p) const noexcept {
  reg_t rv64_ppn_mask = (reg_t(1) << (MAX_PADDR_BITS - PGSHIFT)) - 1;

  reg_t mode_mask = p->get_xlen() == 32 ? SATP32_MODE : SATP64_MODE;
  reg_t asid_mask_if_enabled = p->get_xlen() == 32 ? SATP32_ASID : SATP64_ASID;
  reg_t asid_mask = p->supports_impl(IMPL_MMU_ASID) ? asid_mask_if_enabled : 0;
  reg_t ppn_mask = p->get_xlen() == 32 ? SATP32_PPN : SATP64_PPN & rv64_ppn_mask;
  reg_t new_mask = (satp_valid(val, p) ? mode_mask : 0) | asid_mask | ppn_mask;
  reg_t old_mask = satp_valid(val, p) ? 0 : mode_mask;

  return (new_mask & val) | (old_mask & read(p));
}

satp_csr_t::satp_csr_t(const reg_t addr):
  base_atp_csr_t(addr) {
}

void satp_csr_t::verify_permissions(insn_t insn, bool write, processor_t* p) const {
  base_atp_csr_t::verify_permissions(insn, write, p);

  state_t* s = p->get_state();
  if (get_field(s->mstatus->read(p), MSTATUS_TVM))
    require(s->prv == PRV_M);
}

virtualized_satp_csr_t::virtualized_satp_csr_t(satp_csr_t_p orig, csr_t_p virt):
  virtualized_csr_t(orig, virt),
  orig_satp(orig) {
}

void virtualized_satp_csr_t::verify_permissions(insn_t insn, bool write, processor_t* p) const {
  virtualized_csr_t::verify_permissions(insn, write, p);

  state_t* s = p->get_state();

  // If satp is accessed from VS mode, it's really accessing vsatp,
  // and the hstatus.VTVM bit controls.
  if (s->v) {
    if (get_field(s->hstatus->read(p), HSTATUS_VTVM))
      throw trap_virtual_instruction(insn.bits());
  }
  else {
    orig_csr->verify_permissions(insn, write, p);
  }
}

bool virtualized_satp_csr_t::unlogged_write(const reg_t val, processor_t* p) noexcept {
  // If unsupported Mode field: no change to contents
  const reg_t newval = orig_satp->satp_valid(val, p) ? val : read(p);
  return virtualized_csr_t::unlogged_write(newval, p);
}

// implement class wide_counter_csr_t
wide_counter_csr_t::wide_counter_csr_t(const reg_t addr, smcntrpmf_csr_t_p config_csr):
  csr_t(addr),
  val(0),
  config_csr(config_csr) {
}

reg_t wide_counter_csr_t::read(processor_t* p) const noexcept {
  return val;
}

void wide_counter_csr_t::bump(const reg_t howmuch, processor_t* p) noexcept {
  if (is_counting_enabled(p)) {
    val += howmuch;  // to keep log reasonable size, don't log every bump
  }
  // Clear cached value
  config_csr->reset_prev();
}

bool wide_counter_csr_t::unlogged_write(const reg_t val, processor_t* p) noexcept {
  this->val = val;
  // The ISA mandates that if an instruction writes instret, the write
  // takes precedence over the increment to instret.  However, Spike
  // unconditionally increments instret after executing an instruction.
  // Correct for this artifact by decrementing instret here.
  // Ensure that Smctrpmf hasn't disabled counting.
  if (is_counting_enabled(p)) {
    this->val--;
  }
  return true;
}

reg_t wide_counter_csr_t::written_value(processor_t* p) const noexcept {
  // Re-adjust for upcoming bump()
  return this->val + 1;
}

// Returns true if counting is not inhibited by Smcntrpmf.
// Note that minstretcfg / mcyclecfg / mhpmevent* share the same inhibit bits.
bool wide_counter_csr_t::is_counting_enabled(processor_t* p) const noexcept {
  state_t* state = p->get_state();

  auto prv = state->prv_changed ? state->prev_prv : state->prv;
  auto v = state->v_changed ? state->v : state->prev_v;
  auto mask = MHPMEVENT_MINH;
  if (prv == PRV_S) {
    mask = v ? MHPMEVENT_VSINH : MHPMEVENT_SINH;
  } else if (prv == PRV_U) {
    mask = v ? MHPMEVENT_VUINH : MHPMEVENT_UINH;
  }
  return (config_csr->read_prev(p) & mask) == 0;
}

// implement class time_counter_csr_t
time_counter_csr_t::time_counter_csr_t(const reg_t addr):
  csr_t(addr),
  shadow_val(0) {
}

reg_t time_counter_csr_t::read(processor_t* p) const noexcept {
  state_t* state = p->get_state();

  // reading the time CSR in VS or VU mode returns the sum of the contents of
  // htimedelta and the actual value of time.
  if (state->v)
    return shadow_val + state->htimedelta->read(p);
  else
    return shadow_val;
}

void time_counter_csr_t::sync(const reg_t val, processor_t* p) noexcept {
  state_t* state = p->get_state();

  shadow_val = val;
  if (p->extension_enabled(EXT_SSTC)) {
    const reg_t mip_val = (shadow_val >= state->stimecmp->read(p) ? MIP_STIP : 0) |
      (shadow_val + state->htimedelta->read(p) >= state->vstimecmp->read(p) ? MIP_VSTIP : 0);
    state->mip->backdoor_write_with_mask(MIP_STIP | MIP_VSTIP, mip_val);
  }
}

proxy_csr_t::proxy_csr_t(const reg_t addr, csr_t_p delegate):
  csr_t(addr),
  delegate(delegate) {
}

reg_t proxy_csr_t::read(processor_t* p) const noexcept {
  return delegate->read(p);
}

bool proxy_csr_t::unlogged_write(const reg_t val, processor_t* p) noexcept {
  delegate->write(val, p);  // log only under the original (delegate's) name
  return false;
}

const_csr_t::const_csr_t(const reg_t addr, reg_t val):
  csr_t(addr),
  val(val) {
}

reg_t const_csr_t::read(processor_t* p) const noexcept {
  return val;
}

bool const_csr_t::unlogged_write(const reg_t UNUSED val, processor_t* p) noexcept {
  return false;
}

counter_proxy_csr_t::counter_proxy_csr_t(const reg_t addr, csr_t_p delegate):
  proxy_csr_t(addr, delegate) {
}

bool counter_proxy_csr_t::myenable(csr_t_p counteren, processor_t* p) const noexcept {
  return 1 & (counteren->read(p) >> (address & 31));
}

void counter_proxy_csr_t::verify_permissions(insn_t insn, bool write, processor_t* p) const {
  proxy_csr_t::verify_permissions(insn, write, p);

  state_t* s = p->get_state();

  const bool mctr_ok = (s->prv < PRV_M) ? myenable(s->mcounteren, p) : true;
  const bool hctr_ok = s->v ? myenable(s->hcounteren, p) : true;
  const bool sctr_ok = (p->extension_enabled('S') && s->prv < PRV_S) ? myenable(s->scounteren, p) : true;

  if (!mctr_ok)
    throw trap_illegal_instruction(insn.bits());
  if (!hctr_ok)
      throw trap_virtual_instruction(insn.bits());
  if (!sctr_ok) {
    if (s->v)
      throw trap_virtual_instruction(insn.bits());
    else
      throw trap_illegal_instruction(insn.bits());
  }
}

mevent_csr_t::mevent_csr_t(const reg_t addr):
  basic_csr_t(addr, 0) {
}

bool mevent_csr_t::unlogged_write(const reg_t val, processor_t* p) noexcept {
  const reg_t mask = p->extension_enabled(EXT_SSCOFPMF) ? MHPMEVENT_OF | MHPMEVENT_MINH
    | (p->extension_enabled_const('U') ? MHPMEVENT_UINH : 0)
    | (p->extension_enabled_const('S') ? MHPMEVENT_SINH : 0)
    | (p->extension_enabled('H') ? MHPMEVENT_VUINH | MHPMEVENT_VSINH : 0) : 0;
  return basic_csr_t::unlogged_write((read(p) & ~mask) | (val & mask), p);
}

hypervisor_csr_t::hypervisor_csr_t(const reg_t addr):
  basic_csr_t(addr, 0) {
}

void hypervisor_csr_t::verify_permissions(insn_t insn, bool write, processor_t* p) const {
  basic_csr_t::verify_permissions(insn, write, p);
  if (!p->extension_enabled('H'))
    throw trap_illegal_instruction(insn.bits());
}

hideleg_csr_t::hideleg_csr_t(const reg_t addr, csr_t_p mideleg):
  masked_csr_t(addr, MIP_VS_MASK, 0),
  mideleg(mideleg) {
}

reg_t hideleg_csr_t::read(processor_t* p) const noexcept {
  return masked_csr_t::read(p) & mideleg->read(p);
};

hgatp_csr_t::hgatp_csr_t(const reg_t addr):
  basic_csr_t(addr, 0) {
}

void hgatp_csr_t::verify_permissions(insn_t insn, bool write, processor_t* p) const {
  basic_csr_t::verify_permissions(insn, write, p);

  state_t* s = p->get_state();
  if (!s->v && get_field(s->mstatus->read(p), MSTATUS_TVM))
     require_privilege(PRV_M);
}

bool hgatp_csr_t::unlogged_write(const reg_t val, processor_t* p) noexcept {
  p->get_mmu()->flush_tlb();

  reg_t mask;
  if (p->get_const_xlen() == 32) {
    mask = HGATP32_PPN |
        HGATP32_MODE |
        (p->supports_impl(IMPL_MMU_VMID) ? HGATP32_VMID : 0);
  } else {
    mask = (HGATP64_PPN & ((reg_t(1) << (MAX_PADDR_BITS - PGSHIFT)) - 1)) |
        (p->supports_impl(IMPL_MMU_VMID) ? HGATP64_VMID : 0);

    if (get_field(val, HGATP64_MODE) == HGATP_MODE_OFF ||
        (p->supports_impl(IMPL_MMU_SV39) && get_field(val, HGATP64_MODE) == HGATP_MODE_SV39X4) ||
        (p->supports_impl(IMPL_MMU_SV48) && get_field(val, HGATP64_MODE) == HGATP_MODE_SV48X4) ||
        (p->supports_impl(IMPL_MMU_SV57) && get_field(val, HGATP64_MODE) == HGATP_MODE_SV57X4))
      mask |= HGATP64_MODE;
  }
  mask &= ~(reg_t)3;
  return basic_csr_t::unlogged_write((read(p) & ~mask) | (val & mask), p);
}

tselect_csr_t::tselect_csr_t(const reg_t addr):
  basic_csr_t(addr, 0) {
}

bool tselect_csr_t::unlogged_write(const reg_t val, processor_t* p) noexcept {
  return basic_csr_t::unlogged_write((val < p->TM.count()) ? val : read(p), p);
}

tdata1_csr_t::tdata1_csr_t(const reg_t addr):
  csr_t(addr) {
}

reg_t tdata1_csr_t::read(processor_t* p) const noexcept {
  state_t* state = p->get_state();
  return p->TM.tdata1_read(state->tselect->read(p));
}

bool tdata1_csr_t::unlogged_write(const reg_t val, processor_t* p) noexcept {
  state_t* state = p->get_state();
  return p->TM.tdata1_write(state->tselect->read(p), val);
}

tdata2_csr_t::tdata2_csr_t(const reg_t addr):
  csr_t(addr) {
}

reg_t tdata2_csr_t::read(processor_t* p) const noexcept {
  state_t* state = p->get_state();
  return p->TM.tdata2_read(state->tselect->read(p));
}

bool tdata2_csr_t::unlogged_write(const reg_t val, processor_t* p) noexcept {
  state_t* state = p->get_state();
  return p->TM.tdata2_write(state->tselect->read(p), val);
}

tdata3_csr_t::tdata3_csr_t(const reg_t addr):
  csr_t(addr) {
}

reg_t tdata3_csr_t::read(processor_t* p) const noexcept {
  state_t* state = p->get_state();
  return p->TM.tdata3_read(state->tselect->read(p));
}

bool tdata3_csr_t::unlogged_write(const reg_t val, processor_t* p) noexcept {
  state_t* state = p->get_state();
  return p->TM.tdata3_write(state->tselect->read(p), val);
}

tinfo_csr_t::tinfo_csr_t(const reg_t addr) :
  csr_t(addr) {
}

reg_t tinfo_csr_t::read(processor_t* p) const noexcept {
  state_t* state = p->get_state();
  return p->TM.tinfo_read(state->tselect->read(p));
}

debug_mode_csr_t::debug_mode_csr_t(const reg_t addr):
  basic_csr_t(addr, 0) {
}

void debug_mode_csr_t::verify_permissions(insn_t insn, bool write, processor_t* p) const {
  basic_csr_t::verify_permissions(insn, write, p);
  state_t* s = p->get_state();
  if (!s->debug_mode)
    throw trap_illegal_instruction(insn.bits());
}

dpc_csr_t::dpc_csr_t(const reg_t addr):
  epc_csr_t(addr) {
}

void dpc_csr_t::verify_permissions(insn_t insn, bool write, processor_t* p) const {
  epc_csr_t::verify_permissions(insn, write, p);
  state_t* s = p->get_state();
  if (!s->debug_mode)
    throw trap_illegal_instruction(insn.bits());
}

dcsr_csr_t::dcsr_csr_t(const reg_t addr):
  csr_t(addr),
  prv(0),
  step(false),
  ebreakm(false),
  ebreaks(false),
  ebreaku(false),
  ebreakvs(false),
  ebreakvu(false),
  halt(false),
  v(false),
  cause(0) {
}

void dcsr_csr_t::verify_permissions(insn_t insn, bool write, processor_t* p) const {
  csr_t::verify_permissions(insn, write, p);
  state_t* s = p->get_state();
  if (!s->debug_mode)
    throw trap_illegal_instruction(insn.bits());
}

reg_t dcsr_csr_t::read(processor_t* p) const noexcept {
  reg_t result = 0;
  result = set_field(result, DCSR_XDEBUGVER, 1);
  result = set_field(result, DCSR_EBREAKM, ebreakm);
  result = set_field(result, DCSR_EBREAKS, ebreaks);
  result = set_field(result, DCSR_EBREAKU, ebreaku);
  result = set_field(result, CSR_DCSR_EBREAKVS, ebreakvs);
  result = set_field(result, CSR_DCSR_EBREAKVU, ebreakvu);
  result = set_field(result, DCSR_STOPCYCLE, 0);
  result = set_field(result, DCSR_STOPTIME, 0);
  result = set_field(result, DCSR_CAUSE, cause);
  result = set_field(result, DCSR_STEP, step);
  result = set_field(result, DCSR_PRV, prv);
  result = set_field(result, CSR_DCSR_V, v);
  return result;
}

bool dcsr_csr_t::unlogged_write(const reg_t val, processor_t* p) noexcept {
  prv = get_field(val, DCSR_PRV);
  step = get_field(val, DCSR_STEP);
  // TODO: ndreset and fullreset
  ebreakm = get_field(val, DCSR_EBREAKM);
  ebreaks = get_field(val, DCSR_EBREAKS);
  ebreaku = get_field(val, DCSR_EBREAKU);
  ebreakvs = get_field(val, CSR_DCSR_EBREAKVS);
  ebreakvu = get_field(val, CSR_DCSR_EBREAKVU);
  halt = get_field(val, DCSR_HALT);
  v = p->extension_enabled('H') ? get_field(val, CSR_DCSR_V) : false;
  return true;
}

void dcsr_csr_t::write_cause_and_prv(uint8_t cause, reg_t prv, bool v, processor_t* p) noexcept {
  this->cause = cause;
  this->prv = prv;
  this->v = v;
  log_write(p);
}

float_csr_t::float_csr_t(const reg_t addr, const reg_t mask, const reg_t init):
  masked_csr_t(addr, mask, init) {
}

void float_csr_t::verify_permissions(insn_t insn, bool write, processor_t* p) const {
  masked_csr_t::verify_permissions(insn, write, p);
  require_fs;
  state_t* s = p->get_state();
  if (!p->extension_enabled('F') && !p->extension_enabled(EXT_ZFINX))
    throw trap_illegal_instruction(insn.bits());

  if (p->extension_enabled(EXT_SMSTATEEN) && p->extension_enabled(EXT_ZFINX)) {
    if ((s->prv < PRV_M) && !(s->mstateen[0]->read(p) & MSTATEEN0_FCSR))
      throw trap_illegal_instruction(insn.bits());

    if (s->v && !(s->hstateen[0]->read(p) & HSTATEEN0_FCSR))
      throw trap_virtual_instruction(insn.bits());

    if ((p->extension_enabled('S') && s->prv < PRV_S) && !(s->sstateen[0]->read(p) & SSTATEEN0_FCSR)) {
      if (s->v)
        throw trap_virtual_instruction(insn.bits());
      else
        throw trap_illegal_instruction(insn.bits());
    }
  }
}

bool float_csr_t::unlogged_write(const reg_t val, processor_t* p) noexcept {
  dirty_fp_state;
  return masked_csr_t::unlogged_write(val, p);
}

composite_csr_t::composite_csr_t(const reg_t addr, csr_t_p upper_csr, csr_t_p lower_csr, const unsigned upper_lsb):
  csr_t(addr),
  upper_csr(upper_csr),
  lower_csr(lower_csr),
  upper_lsb(upper_lsb) {
}

void composite_csr_t::verify_permissions(insn_t insn, bool write, processor_t* p) const {
  // It is reasonable to assume that either underlying CSR will have
  // the same permissions as this composite.
  upper_csr->verify_permissions(insn, write, p);
}

reg_t composite_csr_t::read(processor_t* p) const noexcept {
  return (upper_csr->read(p) << upper_lsb) | lower_csr->read(p);
}

bool composite_csr_t::unlogged_write(const reg_t val, processor_t* p) noexcept {
  upper_csr->write(val >> upper_lsb, p);
  lower_csr->write(val, p);
  return false;  // logging is done only by the underlying CSRs
}

seed_csr_t::seed_csr_t(const reg_t addr):
  csr_t(addr) {
}

void seed_csr_t::verify_permissions(insn_t insn, bool write, processor_t* p) const {
  /* Read-only access disallowed due to wipe-on-read side effect */
  /* XXX mseccfg.sseed and mseccfg.useed should be verified. */
  if (!p->extension_enabled(EXT_ZKR) || !write)
    throw trap_illegal_instruction(insn.bits());
  csr_t::verify_permissions(insn, write, p);
}

reg_t seed_csr_t::read(processor_t* p) const noexcept {
  return p->es.get_seed();
}

bool seed_csr_t::unlogged_write(const reg_t val, processor_t* p) noexcept {
  p->es.set_seed(val);
  return true;
}

vector_csr_t::vector_csr_t(const reg_t addr, const reg_t mask, const reg_t init):
  basic_csr_t(addr, init),
  mask(mask) {
}

void vector_csr_t::verify_permissions(insn_t insn, bool write, processor_t* p) const {
  require_vector_vs;
  if (!p->extension_enabled('V'))
    throw trap_illegal_instruction(insn.bits());
  basic_csr_t::verify_permissions(insn, write, p);
}

void vector_csr_t::write_raw(const reg_t val, processor_t* p) noexcept {
  const bool success = basic_csr_t::unlogged_write(val, p);
  if (success)
    log_write(p);
}

bool vector_csr_t::unlogged_write(const reg_t val, processor_t* p) noexcept {
  if (mask == 0) return false;
  dirty_vs_state;
  return basic_csr_t::unlogged_write(val & mask, p);
}

vxsat_csr_t::vxsat_csr_t(const reg_t addr):
  masked_csr_t(addr, /*mask*/ 1, /*init*/ 0) {
}

void vxsat_csr_t::verify_permissions(insn_t insn, bool write, processor_t* p) const {
  require_vector_vs;
  if (!p->extension_enabled('V') && !p->extension_enabled(EXT_ZPN))
    throw trap_illegal_instruction(insn.bits());
  masked_csr_t::verify_permissions(insn, write, p);
}

bool vxsat_csr_t::unlogged_write(const reg_t val, processor_t* p) noexcept {
  dirty_vs_state;
  return masked_csr_t::unlogged_write(val, p);
}

// implement class hstateen_csr_t
hstateen_csr_t::hstateen_csr_t(const reg_t addr, const reg_t mask,
                               const reg_t init, uint8_t index):
  masked_csr_t(addr, mask, init),
  index(index) {
}

reg_t hstateen_csr_t::read(processor_t* p) const noexcept {
  state_t* state = p->get_state();

  // For every bit in an mstateen CSR that is zero (whether read-only zero or set to zero),
  // the same bit appears as read-only zero in the matching hstateen and sstateen CSRs
  return masked_csr_t::read(p) & state->mstateen[index]->read(p);
}

bool hstateen_csr_t::unlogged_write(const reg_t val, processor_t* p) noexcept {
  state_t* state = p->get_state();

  // For every bit in an mstateen CSR that is zero (whether read-only zero or set to zero),
  // the same bit appears as read-only zero in the matching hstateen and sstateen CSRs
  return masked_csr_t::unlogged_write(val & state->mstateen[index]->read(p), p);
}

void hstateen_csr_t::verify_permissions(insn_t insn, bool write, processor_t* p) const {
  state_t* state = p->get_state();
  if ((state->prv < PRV_M) && !(state->mstateen[index]->read(p) & MSTATEEN_HSTATEEN))
    throw trap_illegal_instruction(insn.bits());
  masked_csr_t::verify_permissions(insn, write, p);
}

// implement class sstateen_csr_t
sstateen_csr_t::sstateen_csr_t(const reg_t addr, const reg_t mask,
                               const reg_t init, uint8_t index):
  hstateen_csr_t(addr, mask, init, index) {
}

reg_t sstateen_csr_t::read(processor_t* p) const noexcept {
  state_t* state = p->get_state();

  // For every bit in an mstateen CSR that is zero (whether read-only zero or set to zero),
  // the same bit appears as read-only zero in the matching hstateen and sstateen CSRs
  // For every bit in an hstateen CSR that is zero (whether read-only zero or set to zero),
  // the same bit appears as read-only zero in sstateen when accessed in VS-mode
  if (state->v)
    return hstateen_csr_t::read(p) & state->hstateen[index]->read(p);
  else
    return hstateen_csr_t::read(p);
}

bool sstateen_csr_t::unlogged_write(const reg_t val, processor_t* p) noexcept {
  state_t* state = p->get_state();

  // For every bit in an mstateen CSR that is zero (whether read-only zero or set to zero),
  // the same bit appears as read-only zero in the matching hstateen and sstateen CSRs
  // For every bit in an hstateen CSR that is zero (whether read-only zero or set to zero),
  // the same bit appears as read-only zero in sstateen when accessed in VS-mode
  if (state->v)
    return hstateen_csr_t::unlogged_write(val & state->hstateen[index]->read(p), p);
  else
    return hstateen_csr_t::unlogged_write(val, p);
}

void sstateen_csr_t::verify_permissions(insn_t insn, bool write, processor_t* p) const {
  hstateen_csr_t::verify_permissions(insn, write, p);

  state_t* s = p->get_state();
  if (s->v && !(s->hstateen[index]->read(p) & HSTATEEN_SSTATEEN))
      throw trap_virtual_instruction(insn.bits());
}

// implement class senvcfg_csr_t
senvcfg_csr_t::senvcfg_csr_t(const reg_t addr, const reg_t mask,
                             const reg_t init):
  envcfg_csr_t(addr, mask, init) {
}

void senvcfg_csr_t::verify_permissions(insn_t insn, bool write, processor_t* p) const {
  state_t* s = p->get_state();
  if (p->extension_enabled(EXT_SMSTATEEN)) {
    if ((s->prv < PRV_M) && !(s->mstateen[0]->read(p) & MSTATEEN0_HENVCFG))
      throw trap_illegal_instruction(insn.bits());

    if (s->v && !(s->hstateen[0]->read(p) & HSTATEEN0_SENVCFG))
      throw trap_virtual_instruction(insn.bits());
  }

  masked_csr_t::verify_permissions(insn, write, p);
}

void henvcfg_csr_t::verify_permissions(insn_t insn, bool write, processor_t* p) const {
  state_t* state = p->get_state();
  if (p->extension_enabled(EXT_SMSTATEEN)) {
    if ((state->prv < PRV_M) && !(state->mstateen[0]->read(p) & MSTATEEN0_HENVCFG))
      throw trap_illegal_instruction(insn.bits());
  }

  masked_csr_t::verify_permissions(insn, write, p);
}

stimecmp_csr_t::stimecmp_csr_t(const reg_t addr, const reg_t imask):
  basic_csr_t(addr, 0), intr_mask(imask) {
}

bool stimecmp_csr_t::unlogged_write(const reg_t val, processor_t* p) noexcept {
  state_t* state = p->get_state();
  state->mip->backdoor_write_with_mask(intr_mask, state->time->read(p) >= val ? intr_mask : 0);
  return basic_csr_t::unlogged_write(val, p);
}

virtualized_stimecmp_csr_t::virtualized_stimecmp_csr_t(csr_t_p orig, csr_t_p virt):
  virtualized_csr_t(orig, virt) {
}

void virtualized_stimecmp_csr_t::verify_permissions(insn_t insn, bool write, processor_t* p) const {
  state_t* s = p->get_state();
  if (!(s->menvcfg->read(p) & MENVCFG_STCE)) {
    // access to (v)stimecmp with MENVCFG.STCE = 0
    if (s->prv < PRV_M)
      throw trap_illegal_instruction(insn.bits());
  }

  s->time_proxy->verify_permissions(insn, false, p);

  if (s->v && !(s->henvcfg->read(p) & HENVCFG_STCE)) {
    // access to vstimecmp with MENVCFG.STCE = 1 and HENVCFG.STCE = 0 when V = 1
    throw trap_virtual_instruction(insn.bits());
  }

  virtualized_csr_t::verify_permissions(insn, write, p);
}

scountovf_csr_t::scountovf_csr_t(const reg_t addr):
  csr_t(addr) {
}

void scountovf_csr_t::verify_permissions(insn_t insn, bool write, processor_t* p) const {
  if (!p->extension_enabled(EXT_SSCOFPMF))
    throw trap_illegal_instruction(insn.bits());
  csr_t::verify_permissions(insn, write, p);
}

reg_t scountovf_csr_t::read(processor_t* p) const noexcept {
  state_t* state = p->get_state();

  reg_t val = 0;
  for (reg_t i = 3; i < N_HPMCOUNTERS + 3; ++i) {
    bool of = state->mevent[i - 3]->read(p) & MHPMEVENT_OF;
    val |= of << i;
  }

  /* In M and S modes, scountovf bit X is readable when mcounteren bit X is set, */
  /* and otherwise reads as zero. Similarly, in VS mode, scountovf bit X is readable */
  /* when mcounteren bit X and hcounteren bit X are both set, and otherwise reads as zero. */
  val &= state->mcounteren->read(p);
  if (state->v)
    val &= state->hcounteren->read(p);
  return val;
}

bool scountovf_csr_t::unlogged_write(const reg_t UNUSED val, processor_t* p) noexcept {
  /* this function is unused */
  return false;
}

// implement class jvt_csr_t
jvt_csr_t::jvt_csr_t(const reg_t addr, const reg_t init):
  basic_csr_t(addr, init) {
}

void jvt_csr_t::verify_permissions(insn_t insn, bool write, processor_t* p) const {
  basic_csr_t::verify_permissions(insn, write, p);
  state_t* s = p->get_state();

  if (!p->extension_enabled(EXT_ZCMT))
    throw trap_illegal_instruction(insn.bits());

  if (p->extension_enabled(EXT_SMSTATEEN)) {
    if ((s->prv < PRV_M) && !(s->mstateen[0]->read(p) & SSTATEEN0_JVT))
      throw trap_illegal_instruction(insn.bits());

    if (s->v && !(s->hstateen[0]->read(p) & SSTATEEN0_JVT))
      throw trap_virtual_instruction(insn.bits());

    if ((p->extension_enabled('S') && s->prv < PRV_S) && !(s->sstateen[0]->read(p) & SSTATEEN0_JVT)) {
      if (s->v)
        throw trap_virtual_instruction(insn.bits());
      else
        throw trap_illegal_instruction(insn.bits());
    }
  }
}

virtualized_indirect_csr_t::virtualized_indirect_csr_t(csr_t_p orig, csr_t_p virt):
  virtualized_csr_t(orig, virt) {
}

void virtualized_indirect_csr_t::verify_permissions(insn_t insn, bool write, processor_t* p) const {
  state_t* state = p->get_state();
  virtualized_csr_t::verify_permissions(insn, write, p);
  if (state->v)
    virt_csr->verify_permissions(insn, write, p);
  else
    orig_csr->verify_permissions(insn, write, p);
}

sscsrind_reg_csr_t::sscsrind_reg_csr_t(const reg_t addr, csr_t_p iselect) :
  csr_t(addr),
  iselect(iselect) {
}

void sscsrind_reg_csr_t::verify_permissions(insn_t insn, bool write, processor_t* p) const {
  state_t* state = p->get_state();

  // Don't call base verify_permission for VS registers remapped to S-mode
  if (insn.csr() == address)
    csr_t::verify_permissions(insn, write, p);

  csr_t_p proxy_csr = get_reg(p);
  if (proxy_csr == nullptr) {
    if (!state->v) {
      throw trap_illegal_instruction(insn.bits());
    } else {
      throw trap_virtual_instruction(insn.bits());
    }
  }
  proxy_csr->verify_permissions(insn, write, p);
}


reg_t sscsrind_reg_csr_t::read(processor_t* p) const noexcept {
  csr_t_p target_csr = get_reg(p);
  if (target_csr != nullptr) {
    return target_csr->read(p);
  }
  return 0;
}

bool sscsrind_reg_csr_t::unlogged_write(const reg_t val, processor_t* p) noexcept {
  csr_t_p proxy_csr = get_reg(p);
  if (proxy_csr != nullptr) {
    proxy_csr->write(val, p);
  }
  return false;
}

// Returns the actual CSR that maps to value in *siselect or nullptr if no mapping exists
csr_t_p sscsrind_reg_csr_t::get_reg(processor_t* p) const noexcept {
  auto proxy = ireg_proxy;
  auto isel = iselect->read(p);
  auto it = proxy.find(isel);
  return it != proxy.end() ? it->second : nullptr;
}

void sscsrind_reg_csr_t::add_ireg_proxy(const reg_t iselect_value, csr_t_p csr) {
  ireg_proxy[iselect_value] = csr;
}

smcntrpmf_csr_t::smcntrpmf_csr_t(const reg_t addr, const reg_t mask, const reg_t init)
  : masked_csr_t(addr, mask, init) 
{
}

reg_t smcntrpmf_csr_t::read_prev(processor_t* p) const noexcept {
  reg_t val = prev_val.value_or(read(p));
  return val;
}

void smcntrpmf_csr_t::reset_prev() noexcept {
  prev_val.reset();
}

bool smcntrpmf_csr_t::unlogged_write(const reg_t val, processor_t* p) noexcept {
  prev_val = read(p);
  return masked_csr_t::unlogged_write(val, p);
}

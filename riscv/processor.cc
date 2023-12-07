// See LICENSE for license details.

#include "arith.h"
#include "processor.h"
#include "extension.h"
#include "common.h"
#include "config.h"
#include "decode_macros.h"
#include "simif.h"
#include "mmu.h"
#include "disasm.h"
#include "platform.h"
#include "vector_unit.h"
#include <cinttypes>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <assert.h>
#include <limits.h>
#include <stdexcept>
#include <string>
#include <algorithm>

#ifdef __GNUC__
# pragma GCC diagnostic ignored "-Wunused-variable"
#endif

#undef STATE
#define STATE state

processor_t::processor_t(const isa_parser_t *isa, const cfg_t *cfg,
                         simif_t* sim, uint32_t id, bool halt_on_reset,
                         FILE* log_file, std::ostream& sout_)
  : debug(false), halt_request(HR_NONE), isa(isa), cfg(cfg), sim(sim), id(id), xlen(0),
  histogram_enabled(false), log_commits_enabled(false),
  log_file(log_file), sout_(sout_.rdbuf()), halt_on_reset(halt_on_reset),
  in_wfi(false), check_triggers_icount(false),
  impl_table(256, false), extension_enable_table(isa->get_extension_table()),
  last_pc(1), executions(1), TM(cfg->trigger_count)
{
  VU.p = this;
  TM.proc = this;

#ifndef HAVE_INT128
  if (isa->extension_enabled('V')) {
    fprintf(stderr, "V extension is not supported on platforms without __int128 type\n");
    abort();
  }

  if (isa->extension_enabled(EXT_ZACAS) && isa->get_max_xlen() == 64) {
    fprintf(stderr, "Zacas extension is not supported on 64-bit platforms without __int128 type\n");
    abort();
  }
#endif

  parse_varch_string(cfg->varch);

  register_base_instructions();
  mmu = new mmu_t(sim, cfg->endianness, this);

  disassembler = new disassembler_t(isa);
  for (auto e : isa->get_extensions())
    register_extension(find_extension(e.c_str())());

  set_pmp_granularity(cfg->pmpgranularity);
  set_pmp_num(cfg->pmpregions);

  if (isa->get_max_xlen() == 32)
    set_mmu_capability(IMPL_MMU_SV32);
  else if (isa->get_max_xlen() == 64)
    set_mmu_capability(IMPL_MMU_SV57);

  set_impl(IMPL_MMU_ASID, true);
  set_impl(IMPL_MMU_VMID, true);

  reset();
}

processor_t::~processor_t()
{
  if (histogram_enabled)
  {
    std::vector<std::pair<reg_t, uint64_t>> ordered_histo(pc_histogram.begin(), pc_histogram.end());
    std::sort(ordered_histo.begin(), ordered_histo.end(),
              [](auto& lhs, auto& rhs) { return lhs.second < rhs.second; });

    fprintf(stderr, "PC Histogram size:%zu\n", ordered_histo.size());
    for (auto it : ordered_histo)
      fprintf(stderr, "%0" PRIx64 " %" PRIu64 "\n", it.first, it.second);
  }

  delete mmu;
  delete disassembler;
}

static void bad_option_string(const char *option, const char *value,
                              const char *msg)
{
  fprintf(stderr, "error: bad %s option '%s'. %s\n", option, value, msg);
  abort();
}

static void bad_varch_string(const char* varch, const char *msg)
{
  bad_option_string("--varch", varch, msg);
}

static std::string get_string_token(std::string str, const char delimiter, size_t& pos)
{
  size_t _pos = pos;
  while (pos < str.length() && str[pos] != delimiter) ++pos;
  return str.substr(_pos, pos - _pos);
}

static int get_int_token(std::string str, const char delimiter, size_t& pos)
{
  size_t _pos = pos;
  while (pos < str.length() && str[pos] != delimiter) {
    if (!isdigit(str[pos]))
      bad_varch_string(str.c_str(), "Unsupported value"); // An integer is expected
    ++pos;
  }
  return (pos == _pos) ? 0 : stoi(str.substr(_pos, pos - _pos));
}

static bool check_pow2(int val)
{
  return ((val & (val - 1))) == 0;
}

static std::string strtolower(const char* str)
{
  std::string res;
  for (const char *r = str; *r; r++)
    res += std::tolower(*r);
  return res;
}

void processor_t::parse_varch_string(const char* s)
{
  std::string str = strtolower(s);
  size_t pos = 0;
  size_t len = str.length();
  int vlen = 0;
  int elen = 0;
  int vstart_alu = 0;

  while (pos < len) {
    std::string attr = get_string_token(str, ':', pos);

    ++pos;

    if (attr == "vlen")
      vlen = get_int_token(str, ',', pos);
    else if (attr == "elen")
      elen = get_int_token(str, ',', pos);
    else if (attr == "vstartalu")
      vstart_alu = get_int_token(str, ',', pos);
    else
      bad_varch_string(s, "Unsupported token");

    ++pos;
  }

  // The integer should be the power of 2
  if (!check_pow2(vlen) || !check_pow2(elen)) {
    bad_varch_string(s, "The integer value should be the power of 2");
  }

  /* Vector spec requirements. */
  if (vlen < elen)
    bad_varch_string(s, "vlen must be >= elen");

  /* spike requirements. */
  if (vlen > 4096)
    bad_varch_string(s, "vlen must be <= 4096");

  VU.VLEN = vlen;
  VU.ELEN = elen;
  VU.vlenb = vlen / 8;
  VU.vstart_alu = vstart_alu;
}

static int xlen_to_uxl(int xlen)
{
  if (xlen == 32)
    return 1;
  if (xlen == 64)
    return 2;
  abort();
}

void state_t::reset(processor_t* const proc, reg_t max_isa)
{
  pc = DEFAULT_RSTVEC;
  XPR.reset();
  FPR.reset();

  // This assumes xlen is always max_xlen, which is true today (see
  // mstatus_csr_t::unlogged_write()):
  auto xlen = proc->get_isa().get_max_xlen();

  prv = prev_prv = PRV_M;
  v = prev_v = false;
  prv_changed = false;
  v_changed = false;
  csrmap[CSR_MISA] = misa = std::make_shared<misa_csr_t>(proc, CSR_MISA, max_isa);
  mstatus = std::make_shared<mstatus_csr_t>(proc, CSR_MSTATUS);

  if (xlen == 32) {
    csrmap[CSR_MSTATUS] = std::make_shared<rv32_low_csr_t>(proc, CSR_MSTATUS, mstatus);
    csrmap[CSR_MSTATUSH] = mstatush = std::make_shared<rv32_high_csr_t>(proc, CSR_MSTATUSH, mstatus);
  } else {
    csrmap[CSR_MSTATUS] = mstatus;
  }
  csrmap[CSR_MEPC] = mepc = std::make_shared<epc_csr_t>(proc, CSR_MEPC);
  csrmap[CSR_MTVAL] = mtval = std::make_shared<basic_csr_t>(proc, CSR_MTVAL, 0);
  csrmap[CSR_MSCRATCH] = std::make_shared<basic_csr_t>(proc, CSR_MSCRATCH, 0);
  csrmap[CSR_MTVEC] = mtvec = std::make_shared<tvec_csr_t>(proc, CSR_MTVEC);
  csrmap[CSR_MCAUSE] = mcause = std::make_shared<cause_csr_t>(proc, CSR_MCAUSE);

  auto smcntrpmf_enabled = proc->extension_enabled_const(EXT_SMCNTRPMF);
  const reg_t mask = smcntrpmf_enabled ? MHPMEVENT_MINH | MHPMEVENT_SINH |
                                         MHPMEVENT_UINH | MHPMEVENT_VSINH | MHPMEVENT_VUINH : 0;
  auto minstretcfg = std::make_shared<smcntrpmf_csr_t>(proc, CSR_MINSTRETCFG, mask, 0);
  auto mcyclecfg = std::make_shared<smcntrpmf_csr_t>(proc, CSR_MCYCLECFG, mask, 0);

  minstret = std::make_shared<wide_counter_csr_t>(proc, CSR_MINSTRET, minstretcfg);
  mcycle = std::make_shared<wide_counter_csr_t>(proc, CSR_MCYCLE, mcyclecfg);
  time = std::make_shared<time_counter_csr_t>(proc, CSR_TIME);
  if (proc->extension_enabled_const(EXT_ZICNTR)) {
    csrmap[CSR_INSTRET] = std::make_shared<counter_proxy_csr_t>(proc, CSR_INSTRET, minstret);
    csrmap[CSR_CYCLE] = std::make_shared<counter_proxy_csr_t>(proc, CSR_CYCLE, mcycle);
    csrmap[CSR_TIME] = time_proxy = std::make_shared<counter_proxy_csr_t>(proc, CSR_TIME, time);
  }
  if (xlen == 32) {
    csr_t_p minstreth, mcycleh;
    csrmap[CSR_MINSTRET] = std::make_shared<rv32_low_csr_t>(proc, CSR_MINSTRET, minstret);
    csrmap[CSR_MINSTRETH] = minstreth = std::make_shared<rv32_high_csr_t>(proc, CSR_MINSTRETH, minstret);
    csrmap[CSR_MCYCLE] = std::make_shared<rv32_low_csr_t>(proc, CSR_MCYCLE, mcycle);
    csrmap[CSR_MCYCLEH] = mcycleh = std::make_shared<rv32_high_csr_t>(proc, CSR_MCYCLEH, mcycle);
    if (proc->extension_enabled_const(EXT_ZICNTR)) {
      auto timeh = std::make_shared<rv32_high_csr_t>(proc, CSR_TIMEH, time);
      csrmap[CSR_INSTRETH] = std::make_shared<counter_proxy_csr_t>(proc, CSR_INSTRETH, minstreth);
      csrmap[CSR_CYCLEH] = std::make_shared<counter_proxy_csr_t>(proc, CSR_CYCLEH, mcycleh);
      csrmap[CSR_TIMEH] = std::make_shared<counter_proxy_csr_t>(proc, CSR_TIMEH, timeh);
    }
  } else {
    csrmap[CSR_MINSTRET] = minstret;
    csrmap[CSR_MCYCLE] = mcycle;
  }
  for (reg_t i = 3; i < N_HPMCOUNTERS + 3; ++i) {
    const reg_t which_mevent = CSR_MHPMEVENT3 + i - 3;
    const reg_t which_meventh = CSR_MHPMEVENT3H + i - 3;
    const reg_t which_mcounter = CSR_MHPMCOUNTER3 + i - 3;
    const reg_t which_mcounterh = CSR_MHPMCOUNTER3H + i - 3;
    const reg_t which_counter = CSR_HPMCOUNTER3 + i - 3;
    const reg_t which_counterh = CSR_HPMCOUNTER3H + i - 3;
    mevent[i - 3] = std::make_shared<mevent_csr_t>(proc, which_mevent);
    auto mcounter = std::make_shared<const_csr_t>(proc, which_mcounter, 0);
    csrmap[which_mcounter] = mcounter;

    if (proc->extension_enabled_const(EXT_ZIHPM)) {
      auto counter = std::make_shared<counter_proxy_csr_t>(proc, which_counter, mcounter);
      csrmap[which_counter] = counter;
    }
    if (xlen == 32) {
      csrmap[which_mevent] = std::make_shared<rv32_low_csr_t>(proc, which_mevent, mevent[i - 3]);;
      auto mcounterh = std::make_shared<const_csr_t>(proc, which_mcounterh, 0);
      csrmap[which_mcounterh] = mcounterh;
      if (proc->extension_enabled_const(EXT_ZIHPM)) {
        auto counterh = std::make_shared<counter_proxy_csr_t>(proc, which_counterh, mcounterh);
        csrmap[which_counterh] = counterh;
      }
      if (proc->extension_enabled_const(EXT_SSCOFPMF)) {
        auto meventh = std::make_shared<rv32_high_csr_t>(proc, which_meventh, mevent[i - 3]);
        csrmap[which_meventh] = meventh;
      }
    } else {
      csrmap[which_mevent] = mevent[i - 3];
    }
  }
  csrmap[CSR_MCOUNTINHIBIT] = std::make_shared<const_csr_t>(proc, CSR_MCOUNTINHIBIT, 0);
  if (proc->extension_enabled_const(EXT_SSCOFPMF))
    csrmap[CSR_SCOUNTOVF] = std::make_shared<scountovf_csr_t>(proc, CSR_SCOUNTOVF);
  csrmap[CSR_MIE] = mie = std::make_shared<mie_csr_t>(proc, CSR_MIE);
  csrmap[CSR_MIP] = mip = std::make_shared<mip_csr_t>(proc, CSR_MIP);
  auto sip_sie_accr = std::make_shared<generic_int_accessor_t>(
    this,
    ~MIP_HS_MASK,  // read_mask
    MIP_SSIP | MIP_LCOFIP,  // ip_write_mask
    ~MIP_HS_MASK,  // ie_write_mask
    generic_int_accessor_t::mask_mode_t::MIDELEG,
    0              // shiftamt
  );

  auto hip_hie_accr = std::make_shared<generic_int_accessor_t>(
    this,
    MIP_HS_MASK,   // read_mask
    MIP_VSSIP,     // ip_write_mask
    MIP_HS_MASK,   // ie_write_mask
    generic_int_accessor_t::mask_mode_t::MIDELEG,
    0              // shiftamt
  );

  auto hvip_accr = std::make_shared<generic_int_accessor_t>(
    this,
    MIP_VS_MASK,   // read_mask
    MIP_VS_MASK,   // ip_write_mask
    MIP_VS_MASK,   // ie_write_mask
    generic_int_accessor_t::mask_mode_t::NONE,
    0              // shiftamt
  );

  auto vsip_vsie_accr = std::make_shared<generic_int_accessor_t>(
    this,
    MIP_VS_MASK,   // read_mask
    MIP_VSSIP,     // ip_write_mask
    MIP_VS_MASK,   // ie_write_mask
    generic_int_accessor_t::mask_mode_t::HIDELEG,
    1              // shiftamt
  );

  auto nonvirtual_sip = std::make_shared<mip_proxy_csr_t>(proc, CSR_SIP, sip_sie_accr);
  auto vsip = std::make_shared<mip_proxy_csr_t>(proc, CSR_VSIP, vsip_vsie_accr);
  csrmap[CSR_VSIP] = vsip;
  csrmap[CSR_SIP] = std::make_shared<virtualized_csr_t>(proc, nonvirtual_sip, vsip);
  csrmap[CSR_HIP] = std::make_shared<mip_proxy_csr_t>(proc, CSR_HIP, hip_hie_accr);
  csrmap[CSR_HVIP] = std::make_shared<mip_proxy_csr_t>(proc, CSR_HVIP, hvip_accr);

  auto nonvirtual_sie = std::make_shared<mie_proxy_csr_t>(proc, CSR_SIE, sip_sie_accr);
  auto vsie = std::make_shared<mie_proxy_csr_t>(proc, CSR_VSIE, vsip_vsie_accr);
  csrmap[CSR_VSIE] = vsie;
  csrmap[CSR_SIE] = std::make_shared<virtualized_csr_t>(proc, nonvirtual_sie, vsie);
  csrmap[CSR_HIE] = std::make_shared<mie_proxy_csr_t>(proc, CSR_HIE, hip_hie_accr);

  csrmap[CSR_MEDELEG] = medeleg = std::make_shared<medeleg_csr_t>(proc, CSR_MEDELEG);
  csrmap[CSR_MIDELEG] = mideleg = std::make_shared<mideleg_csr_t>(proc, CSR_MIDELEG);
  const reg_t counteren_mask = (proc->extension_enabled_const(EXT_ZICNTR) ? 0x7UL : 0x0) | (proc->extension_enabled_const(EXT_ZIHPM) ? 0xfffffff8ULL : 0x0);
  mcounteren = std::make_shared<masked_csr_t>(proc, CSR_MCOUNTEREN, counteren_mask, 0);
  if (proc->extension_enabled_const('U')) csrmap[CSR_MCOUNTEREN] = mcounteren;
  csrmap[CSR_SCOUNTEREN] = scounteren = std::make_shared<masked_csr_t>(proc, CSR_SCOUNTEREN, counteren_mask, 0);
  nonvirtual_sepc = std::make_shared<epc_csr_t>(proc, CSR_SEPC);
  csrmap[CSR_VSEPC] = vsepc = std::make_shared<epc_csr_t>(proc, CSR_VSEPC);
  csrmap[CSR_SEPC] = sepc = std::make_shared<virtualized_csr_t>(proc, nonvirtual_sepc, vsepc);
  nonvirtual_stval = std::make_shared<basic_csr_t>(proc, CSR_STVAL, 0);
  csrmap[CSR_VSTVAL] = vstval = std::make_shared<basic_csr_t>(proc, CSR_VSTVAL, 0);
  csrmap[CSR_STVAL] = stval = std::make_shared<virtualized_csr_t>(proc, nonvirtual_stval, vstval);
  auto sscratch = std::make_shared<basic_csr_t>(proc, CSR_SSCRATCH, 0);
  auto vsscratch = std::make_shared<basic_csr_t>(proc, CSR_VSSCRATCH, 0);
  // Note: if max_isa does not include H, we don't really need this virtualized_csr_t at all (though it doesn't hurt):
  csrmap[CSR_SSCRATCH] = std::make_shared<virtualized_csr_t>(proc, sscratch, vsscratch);
  csrmap[CSR_VSSCRATCH] = vsscratch;
  nonvirtual_stvec = std::make_shared<tvec_csr_t>(proc, CSR_STVEC);
  csrmap[CSR_VSTVEC] = vstvec = std::make_shared<tvec_csr_t>(proc, CSR_VSTVEC);
  csrmap[CSR_STVEC] = stvec = std::make_shared<virtualized_csr_t>(proc, nonvirtual_stvec, vstvec);
  auto nonvirtual_satp = std::make_shared<satp_csr_t>(proc, CSR_SATP);
  csrmap[CSR_VSATP] = vsatp = std::make_shared<base_atp_csr_t>(proc, CSR_VSATP);
  csrmap[CSR_SATP] = satp = std::make_shared<virtualized_satp_csr_t>(proc, nonvirtual_satp, vsatp);
  nonvirtual_scause = std::make_shared<cause_csr_t>(proc, CSR_SCAUSE);
  csrmap[CSR_VSCAUSE] = vscause = std::make_shared<cause_csr_t>(proc, CSR_VSCAUSE);
  csrmap[CSR_SCAUSE] = scause = std::make_shared<virtualized_csr_t>(proc, nonvirtual_scause, vscause);
  csrmap[CSR_MTVAL2] = mtval2 = std::make_shared<hypervisor_csr_t>(proc, CSR_MTVAL2);
  csrmap[CSR_MTINST] = mtinst = std::make_shared<hypervisor_csr_t>(proc, CSR_MTINST);
  const reg_t hstatus_init = set_field((reg_t)0, HSTATUS_VSXL, xlen_to_uxl(proc->get_const_xlen()));
  const reg_t hstatus_mask = HSTATUS_VTSR | HSTATUS_VTW
    | (proc->supports_impl(IMPL_MMU) ? HSTATUS_VTVM : 0)
    | HSTATUS_HU | HSTATUS_SPVP | HSTATUS_SPV | HSTATUS_GVA;
  csrmap[CSR_HSTATUS] = hstatus = std::make_shared<masked_csr_t>(proc, CSR_HSTATUS, hstatus_mask, hstatus_init);
  csrmap[CSR_HGEIE] = std::make_shared<const_csr_t>(proc, CSR_HGEIE, 0);
  csrmap[CSR_HGEIP] = std::make_shared<const_csr_t>(proc, CSR_HGEIP, 0);
  csrmap[CSR_HIDELEG] = hideleg = std::make_shared<hideleg_csr_t>(proc, CSR_HIDELEG, mideleg);
  const reg_t hedeleg_mask =
    (1 << CAUSE_MISALIGNED_FETCH) |
    (1 << CAUSE_FETCH_ACCESS) |
    (1 << CAUSE_ILLEGAL_INSTRUCTION) |
    (1 << CAUSE_BREAKPOINT) |
    (1 << CAUSE_MISALIGNED_LOAD) |
    (1 << CAUSE_LOAD_ACCESS) |
    (1 << CAUSE_MISALIGNED_STORE) |
    (1 << CAUSE_STORE_ACCESS) |
    (1 << CAUSE_USER_ECALL) |
    (1 << CAUSE_FETCH_PAGE_FAULT) |
    (1 << CAUSE_LOAD_PAGE_FAULT) |
    (1 << CAUSE_STORE_PAGE_FAULT);
  csrmap[CSR_HEDELEG] = hedeleg = std::make_shared<masked_csr_t>(proc, CSR_HEDELEG, hedeleg_mask, 0);
  csrmap[CSR_HCOUNTEREN] = hcounteren = std::make_shared<masked_csr_t>(proc, CSR_HCOUNTEREN, counteren_mask, 0);
  htimedelta = std::make_shared<basic_csr_t>(proc, CSR_HTIMEDELTA, 0);
  if (xlen == 32) {
    csrmap[CSR_HTIMEDELTA] = std::make_shared<rv32_low_csr_t>(proc, CSR_HTIMEDELTA, htimedelta);
    csrmap[CSR_HTIMEDELTAH] = std::make_shared<rv32_high_csr_t>(proc, CSR_HTIMEDELTAH, htimedelta);
  } else {
    csrmap[CSR_HTIMEDELTA] = htimedelta;
  }
  csrmap[CSR_HTVAL] = htval = std::make_shared<basic_csr_t>(proc, CSR_HTVAL, 0);
  csrmap[CSR_HTINST] = htinst = std::make_shared<basic_csr_t>(proc, CSR_HTINST, 0);
  csrmap[CSR_HGATP] = hgatp = std::make_shared<hgatp_csr_t>(proc, CSR_HGATP);
  nonvirtual_sstatus = std::make_shared<sstatus_proxy_csr_t>(proc, CSR_SSTATUS, mstatus);
  csrmap[CSR_VSSTATUS] = vsstatus = std::make_shared<vsstatus_csr_t>(proc, CSR_VSSTATUS);
  csrmap[CSR_SSTATUS] = sstatus = std::make_shared<sstatus_csr_t>(proc, nonvirtual_sstatus, vsstatus);

  csrmap[CSR_DPC] = dpc = std::make_shared<dpc_csr_t>(proc, CSR_DPC);
  csrmap[CSR_DSCRATCH0] = std::make_shared<debug_mode_csr_t>(proc, CSR_DSCRATCH0);
  csrmap[CSR_DSCRATCH1] = std::make_shared<debug_mode_csr_t>(proc, CSR_DSCRATCH1);
  csrmap[CSR_DCSR] = dcsr = std::make_shared<dcsr_csr_t>(proc, CSR_DCSR);

  csrmap[CSR_TSELECT] = tselect = std::make_shared<tselect_csr_t>(proc, CSR_TSELECT);
  if (proc->get_cfg().trigger_count > 0) {
    csrmap[CSR_TDATA1] = std::make_shared<tdata1_csr_t>(proc, CSR_TDATA1);
    csrmap[CSR_TDATA2] = tdata2 = std::make_shared<tdata2_csr_t>(proc, CSR_TDATA2);
    csrmap[CSR_TDATA3] = std::make_shared<tdata3_csr_t>(proc, CSR_TDATA3);
    csrmap[CSR_TINFO] = std::make_shared<tinfo_csr_t>(proc, CSR_TINFO);
  } else {
    csrmap[CSR_TDATA1] = std::make_shared<const_csr_t>(proc, CSR_TDATA1, 0);
    csrmap[CSR_TDATA2] = tdata2 = std::make_shared<const_csr_t>(proc, CSR_TDATA2, 0);
    csrmap[CSR_TDATA3] = std::make_shared<const_csr_t>(proc, CSR_TDATA3, 0);
    csrmap[CSR_TINFO] = std::make_shared<const_csr_t>(proc, CSR_TINFO, 0);
  }
  unsigned scontext_length = (xlen == 32 ? 16 : 34); // debug spec suggests 16-bit for RV32 and 34-bit for RV64
  csrmap[CSR_SCONTEXT] = scontext = std::make_shared<masked_csr_t>(proc, CSR_SCONTEXT, (reg_t(1) << scontext_length) - 1, 0);
  unsigned hcontext_length = (xlen == 32 ? 6 : 13) + (proc->extension_enabled('H') ? 1 : 0); // debug spec suggest 7-bit (6-bit) for RV32 and 14-bit (13-bit) for RV64 with (without) H extension
  csrmap[CSR_HCONTEXT] = std::make_shared<masked_csr_t>(proc, CSR_HCONTEXT, (reg_t(1) << hcontext_length) - 1, 0);
  csrmap[CSR_MCONTEXT] = mcontext = std::make_shared<proxy_csr_t>(proc, CSR_MCONTEXT, csrmap[CSR_HCONTEXT]);
  debug_mode = false;
  single_step = STEP_NONE;

  csrmap[CSR_MSECCFG] = mseccfg = std::make_shared<mseccfg_csr_t>(proc, CSR_MSECCFG);

  for (int i = 0; i < max_pmp; ++i) {
    csrmap[CSR_PMPADDR0 + i] = pmpaddr[i] = std::make_shared<pmpaddr_csr_t>(proc, CSR_PMPADDR0 + i);
  }
  for (int i = 0; i < max_pmp; i += xlen / 8) {
    reg_t addr = CSR_PMPCFG0 + i / 4;
    csrmap[addr] = std::make_shared<pmpcfg_csr_t>(proc, addr);
  }

  csrmap[CSR_FFLAGS] = fflags = std::make_shared<float_csr_t>(proc, CSR_FFLAGS, FSR_AEXC >> FSR_AEXC_SHIFT, 0);
  csrmap[CSR_FRM] = frm = std::make_shared<float_csr_t>(proc, CSR_FRM, FSR_RD >> FSR_RD_SHIFT, 0);
  assert(FSR_AEXC_SHIFT == 0);  // composite_csr_t assumes fflags begins at bit 0
  csrmap[CSR_FCSR] = std::make_shared<composite_csr_t>(proc, CSR_FCSR, frm, fflags, FSR_RD_SHIFT);

  csrmap[CSR_SEED] = std::make_shared<seed_csr_t>(proc, CSR_SEED);

  csrmap[CSR_MARCHID] = std::make_shared<const_csr_t>(proc, CSR_MARCHID, 5);
  csrmap[CSR_MIMPID] = std::make_shared<const_csr_t>(proc, CSR_MIMPID, 0);
  csrmap[CSR_MVENDORID] = std::make_shared<const_csr_t>(proc, CSR_MVENDORID, 0);
  csrmap[CSR_MHARTID] = std::make_shared<const_csr_t>(proc, CSR_MHARTID, proc->get_id());
  csrmap[CSR_MCONFIGPTR] = std::make_shared<const_csr_t>(proc, CSR_MCONFIGPTR, 0);
  if (proc->extension_enabled_const('U')) {
    const reg_t menvcfg_mask = (proc->extension_enabled(EXT_ZICBOM) ? MENVCFG_CBCFE | MENVCFG_CBIE : 0) |
                              (proc->extension_enabled(EXT_ZICBOZ) ? MENVCFG_CBZE : 0) |
                              (proc->extension_enabled(EXT_SVADU) ? MENVCFG_ADUE: 0) |
                              (proc->extension_enabled(EXT_SVPBMT) ? MENVCFG_PBMTE : 0) |
                              (proc->extension_enabled(EXT_SSTC) ? MENVCFG_STCE : 0);
    const reg_t menvcfg_init = (proc->extension_enabled(EXT_SVPBMT) ? MENVCFG_PBMTE : 0);
    menvcfg = std::make_shared<envcfg_csr_t>(proc, CSR_MENVCFG, menvcfg_mask, menvcfg_init);
    if (xlen == 32) {
      csrmap[CSR_MENVCFG] = std::make_shared<rv32_low_csr_t>(proc, CSR_MENVCFG, menvcfg);
      csrmap[CSR_MENVCFGH] = std::make_shared<rv32_high_csr_t>(proc, CSR_MENVCFGH, menvcfg);
    } else {
      csrmap[CSR_MENVCFG] = menvcfg;
    }
    const reg_t senvcfg_mask = (proc->extension_enabled(EXT_ZICBOM) ? SENVCFG_CBCFE | SENVCFG_CBIE : 0) |
                              (proc->extension_enabled(EXT_ZICBOZ) ? SENVCFG_CBZE : 0);
    csrmap[CSR_SENVCFG] = senvcfg = std::make_shared<senvcfg_csr_t>(proc, CSR_SENVCFG, senvcfg_mask, 0);
    const reg_t henvcfg_mask = (proc->extension_enabled(EXT_ZICBOM) ? HENVCFG_CBCFE | HENVCFG_CBIE : 0) |
                              (proc->extension_enabled(EXT_ZICBOZ) ? HENVCFG_CBZE : 0) |
                              (proc->extension_enabled(EXT_SVADU) ? HENVCFG_ADUE: 0) |
                              (proc->extension_enabled(EXT_SVPBMT) ? HENVCFG_PBMTE : 0) |
                              (proc->extension_enabled(EXT_SSTC) ? HENVCFG_STCE : 0);
    const reg_t henvcfg_init = (proc->extension_enabled(EXT_SVPBMT) ? HENVCFG_PBMTE : 0);
    henvcfg = std::make_shared<henvcfg_csr_t>(proc, CSR_HENVCFG, henvcfg_mask, henvcfg_init, menvcfg);
    if (xlen == 32) {
      csrmap[CSR_HENVCFG] = std::make_shared<rv32_low_csr_t>(proc, CSR_HENVCFG, henvcfg);
      csrmap[CSR_HENVCFGH] = std::make_shared<rv32_high_csr_t>(proc, CSR_HENVCFGH, henvcfg);
    } else {
      csrmap[CSR_HENVCFG] = henvcfg;
    }
  }
  if (proc->extension_enabled_const(EXT_SMSTATEEN)) {
    const reg_t sstateen0_mask = (proc->extension_enabled(EXT_ZFINX) ? SSTATEEN0_FCSR : 0) |
                                 (proc->extension_enabled(EXT_ZCMT) ? SSTATEEN0_JVT : 0) |
                                 SSTATEEN0_CS;
    const reg_t hstateen0_mask = sstateen0_mask | HSTATEEN0_SENVCFG | HSTATEEN_SSTATEEN;
    const reg_t mstateen0_mask = hstateen0_mask;
    for (int i = 0; i < 4; i++) {
      const reg_t mstateen_mask = i == 0 ? mstateen0_mask : MSTATEEN_HSTATEEN;
      mstateen[i] = std::make_shared<masked_csr_t>(proc, CSR_MSTATEEN0 + i, mstateen_mask, 0);
      if (xlen == 32) {
        csrmap[CSR_MSTATEEN0 + i] = std::make_shared<rv32_low_csr_t>(proc, CSR_MSTATEEN0 + i, mstateen[i]);
        csrmap[CSR_MSTATEEN0H + i] = std::make_shared<rv32_high_csr_t>(proc, CSR_MSTATEEN0H + i, mstateen[i]);
      } else {
        csrmap[CSR_MSTATEEN0 + i] = mstateen[i];
      }

      const reg_t hstateen_mask = i == 0 ? hstateen0_mask : HSTATEEN_SSTATEEN;
      hstateen[i] = std::make_shared<hstateen_csr_t>(proc, CSR_HSTATEEN0 + i, hstateen_mask, 0, i);
      if (xlen == 32) {
        csrmap[CSR_HSTATEEN0 + i] = std::make_shared<rv32_low_csr_t>(proc, CSR_HSTATEEN0 + i, hstateen[i]);
        csrmap[CSR_HSTATEEN0H + i] = std::make_shared<rv32_high_csr_t>(proc, CSR_HSTATEEN0H + i, hstateen[i]);
      } else {
        csrmap[CSR_HSTATEEN0 + i] = hstateen[i];
      }

      const reg_t sstateen_mask = i == 0 ? sstateen0_mask : 0;
      csrmap[CSR_SSTATEEN0 + i] = sstateen[i] = std::make_shared<sstateen_csr_t>(proc, CSR_HSTATEEN0 + i, sstateen_mask, 0, i);
    }
  }

  if (proc->extension_enabled_const(EXT_SMRNMI)) {
    csrmap[CSR_MNSCRATCH] = std::make_shared<basic_csr_t>(proc, CSR_MNSCRATCH, 0);
    csrmap[CSR_MNEPC] = mnepc = std::make_shared<epc_csr_t>(proc, CSR_MNEPC);
    csrmap[CSR_MNCAUSE] = std::make_shared<const_csr_t>(proc, CSR_MNCAUSE, (reg_t)1 << (xlen - 1));
    csrmap[CSR_MNSTATUS] = mnstatus = std::make_shared<mnstatus_csr_t>(proc, CSR_MNSTATUS);
  }

  if (proc->extension_enabled_const(EXT_SSTC)) {
    stimecmp = std::make_shared<stimecmp_csr_t>(proc, CSR_STIMECMP, MIP_STIP);
    vstimecmp = std::make_shared<stimecmp_csr_t>(proc, CSR_VSTIMECMP, MIP_VSTIP);
    auto virtualized_stimecmp = std::make_shared<virtualized_stimecmp_csr_t>(proc, stimecmp, vstimecmp);
    if (xlen == 32) {
      csrmap[CSR_STIMECMP] = std::make_shared<rv32_low_csr_t>(proc, CSR_STIMECMP, virtualized_stimecmp);
      csrmap[CSR_STIMECMPH] = std::make_shared<rv32_high_csr_t>(proc, CSR_STIMECMPH, virtualized_stimecmp);
      csrmap[CSR_VSTIMECMP] = std::make_shared<rv32_low_csr_t>(proc, CSR_VSTIMECMP, vstimecmp);
      csrmap[CSR_VSTIMECMPH] = std::make_shared<rv32_high_csr_t>(proc, CSR_VSTIMECMPH, vstimecmp);
    } else {
      csrmap[CSR_STIMECMP] = virtualized_stimecmp;
      csrmap[CSR_VSTIMECMP] = vstimecmp;
    }
  }

  if (proc->extension_enabled(EXT_ZCMT))
    csrmap[CSR_JVT] = jvt = std::make_shared<jvt_csr_t>(proc, CSR_JVT, 0);


  // Smcsrind / Sscsrind
  sscsrind_reg_csr_t::sscsrind_reg_csr_t_p mireg[6];
  sscsrind_reg_csr_t::sscsrind_reg_csr_t_p sireg[6];
  sscsrind_reg_csr_t::sscsrind_reg_csr_t_p vsireg[6];

  if (proc->extension_enabled_const(EXT_SMCSRIND)) {
    csr_t_p miselect = std::make_shared<basic_csr_t>(proc, CSR_MISELECT, 0);
    csrmap[CSR_MISELECT] = miselect;

    const reg_t mireg_csrs[] = { CSR_MIREG, CSR_MIREG2, CSR_MIREG3, CSR_MIREG4, CSR_MIREG5, CSR_MIREG6 };
    auto i = 0;
    for (auto csr : mireg_csrs) {
      csrmap[csr] = mireg[i] = std::make_shared<sscsrind_reg_csr_t>(proc, csr, miselect);
      i++;
    }
  }

  if (proc->extension_enabled_const(EXT_SSCSRIND)) {
    csr_t_p vsiselect = std::make_shared<basic_csr_t>(proc, CSR_VSISELECT, 0);
    csrmap[CSR_VSISELECT] = vsiselect;
    csr_t_p siselect = std::make_shared<basic_csr_t>(proc, CSR_SISELECT, 0);
    csrmap[CSR_SISELECT] = std::make_shared<virtualized_csr_t>(proc, siselect, vsiselect);

    const reg_t vsireg_csrs[] = { CSR_VSIREG, CSR_VSIREG2, CSR_VSIREG3, CSR_VSIREG4, CSR_VSIREG5, CSR_VSIREG6 };
    auto i = 0;
    for (auto csr : vsireg_csrs) {
      csrmap[csr] = vsireg[i] = std::make_shared<sscsrind_reg_csr_t>(proc, csr, vsiselect);
      i++;
    }

    const reg_t sireg_csrs[] = { CSR_SIREG, CSR_SIREG2, CSR_SIREG3, CSR_SIREG4, CSR_SIREG5, CSR_SIREG6 };
    i = 0;
    for (auto csr : sireg_csrs) {
      sireg[i] = std::make_shared<sscsrind_reg_csr_t>(proc, csr, siselect);
      csrmap[csr] = std::make_shared<virtualized_indirect_csr_t>(proc, sireg[i], vsireg[i]);
      i++;
    }
  }

  if (smcntrpmf_enabled) {
      if (xlen == 32) {
        csrmap[CSR_MCYCLECFG] = std::make_shared<rv32_low_csr_t>(proc, CSR_MCYCLECFG, mcyclecfg);
        csrmap[CSR_MCYCLECFGH] = std::make_shared<rv32_high_csr_t>(proc, CSR_MCYCLECFGH, mcyclecfg);
        csrmap[CSR_MINSTRETCFG] = std::make_shared<rv32_low_csr_t>(proc, CSR_MINSTRETCFG, minstretcfg);
        csrmap[CSR_MINSTRETCFGH] = std::make_shared<rv32_high_csr_t>(proc, CSR_MINSTRETCFGH, minstretcfg);
      } else {
        csrmap[CSR_MCYCLECFG] = mcyclecfg;
        csrmap[CSR_MINSTRETCFG] = minstretcfg;
      }
  }

  serialized = false;

  log_reg_write.clear();
  log_mem_read.clear();
  log_mem_write.clear();
  last_inst_priv = 0;
  last_inst_xlen = 0;
  last_inst_flen = 0;
}

void processor_t::set_debug(bool value)
{
  debug = value;

  for (auto e : custom_extensions)
    e.second->set_debug(value);
}

void processor_t::set_histogram(bool value)
{
  histogram_enabled = value;
}

void processor_t::enable_log_commits()
{
  log_commits_enabled = true;
}

void processor_t::reset()
{
  xlen = isa->get_max_xlen();
  state.reset(this, isa->get_max_isa());
  state.dcsr->halt = halt_on_reset;
  halt_on_reset = false;
  VU.reset();
  in_wfi = false;

  if (n_pmp > 0) {
    // For backwards compatibility with software that is unaware of PMP,
    // initialize PMP to permit unprivileged access to all of memory.
    put_csr(CSR_PMPADDR0, ~reg_t(0));
    put_csr(CSR_PMPCFG0, PMP_R | PMP_W | PMP_X | PMP_NAPOT);
  }

  for (auto e : custom_extensions) // reset any extensions
    e.second->reset();

  if (sim)
    sim->proc_reset(id);
}

extension_t* processor_t::get_extension()
{
  switch (custom_extensions.size()) {
    case 0: return NULL;
    case 1: return custom_extensions.begin()->second;
    default:
      fprintf(stderr, "processor_t::get_extension() is ambiguous when multiple extensions\n");
      fprintf(stderr, "are present!\n");
      abort();
  }
}

extension_t* processor_t::get_extension(const char* name)
{
  auto it = custom_extensions.find(name);
  if (it == custom_extensions.end())
    abort();
  return it->second;
}

void processor_t::set_pmp_num(reg_t n)
{
  // check the number of pmp is in a reasonable range
  if (n > state.max_pmp) {
    fprintf(stderr, "error: number of PMP regions requested (%" PRIu64 ") exceeds maximum (%d)\n", n, state.max_pmp);
    abort();
  }
  n_pmp = n;
}

void processor_t::set_pmp_granularity(reg_t gran)
{
  // check the pmp granularity is set from dtb(!=0) and is power of 2
  unsigned min = 1 << PMP_SHIFT;
  if (gran < min || (gran & (gran - 1)) != 0) {
    fprintf(stderr, "error: PMP granularity (%" PRIu64 ") must be a power of two and at least %u\n", gran, min);
    abort();
  }

  lg_pmp_granularity = ctz(gran);
}

void processor_t::set_mmu_capability(int cap)
{
  switch (cap) {
    case IMPL_MMU_SV32:
      set_impl(IMPL_MMU_SV32, true);
      set_impl(IMPL_MMU, true);
      break;
    case IMPL_MMU_SV57:
      set_impl(IMPL_MMU_SV57, true);
      // Fall through
    case IMPL_MMU_SV48:
      set_impl(IMPL_MMU_SV48, true);
      // Fall through
    case IMPL_MMU_SV39:
      set_impl(IMPL_MMU_SV39, true);
      set_impl(IMPL_MMU, true);
      break;
    default:
      set_impl(IMPL_MMU_SV32, false);
      set_impl(IMPL_MMU_SV39, false);
      set_impl(IMPL_MMU_SV48, false);
      set_impl(IMPL_MMU_SV57, false);
      set_impl(IMPL_MMU, false);
      break;
  }
}

void processor_t::take_interrupt(reg_t pending_interrupts)
{
  // Do nothing if no pending interrupts
  if (!pending_interrupts) {
    return;
  }

  // Exit WFI if there are any pending interrupts
  in_wfi = false;

  // M-ints have higher priority over HS-ints and VS-ints
  const reg_t mie = get_field(state.mstatus->read(), MSTATUS_MIE);
  const reg_t m_enabled = state.prv < PRV_M || (state.prv == PRV_M && mie);
  reg_t enabled_interrupts = pending_interrupts & ~state.mideleg->read() & -m_enabled;
  if (enabled_interrupts == 0) {
    // HS-ints have higher priority over VS-ints
    const reg_t deleg_to_hs = state.mideleg->read() & ~state.hideleg->read();
    const reg_t sie = get_field(state.sstatus->read(), MSTATUS_SIE);
    const reg_t hs_enabled = state.v || state.prv < PRV_S || (state.prv == PRV_S && sie);
    enabled_interrupts = pending_interrupts & deleg_to_hs & -hs_enabled;
    if (state.v && enabled_interrupts == 0) {
      // VS-ints have least priority and can only be taken with virt enabled
      const reg_t deleg_to_vs = state.hideleg->read();
      const reg_t vs_enabled = state.prv < PRV_S || (state.prv == PRV_S && sie);
      enabled_interrupts = pending_interrupts & deleg_to_vs & -vs_enabled;
    }
  }

  const bool nmie = !(state.mnstatus && !get_field(state.mnstatus->read(), MNSTATUS_NMIE));
  if (!state.debug_mode && nmie && enabled_interrupts) {
    // nonstandard interrupts have highest priority
    if (enabled_interrupts >> (IRQ_M_EXT + 1))
      enabled_interrupts = enabled_interrupts >> (IRQ_M_EXT + 1) << (IRQ_M_EXT + 1);
    // standard interrupt priority is MEI, MSI, MTI, SEI, SSI, STI
    else if (enabled_interrupts & MIP_MEIP)
      enabled_interrupts = MIP_MEIP;
    else if (enabled_interrupts & MIP_MSIP)
      enabled_interrupts = MIP_MSIP;
    else if (enabled_interrupts & MIP_MTIP)
      enabled_interrupts = MIP_MTIP;
    else if (enabled_interrupts & MIP_SEIP)
      enabled_interrupts = MIP_SEIP;
    else if (enabled_interrupts & MIP_SSIP)
      enabled_interrupts = MIP_SSIP;
    else if (enabled_interrupts & MIP_STIP)
      enabled_interrupts = MIP_STIP;
    else if (enabled_interrupts & MIP_LCOFIP)
      enabled_interrupts = MIP_LCOFIP;
    else if (enabled_interrupts & MIP_VSEIP)
      enabled_interrupts = MIP_VSEIP;
    else if (enabled_interrupts & MIP_VSSIP)
      enabled_interrupts = MIP_VSSIP;
    else if (enabled_interrupts & MIP_VSTIP)
      enabled_interrupts = MIP_VSTIP;
    else
      abort();

    if (check_triggers_icount) TM.detect_icount_match();
    throw trap_t(((reg_t)1 << (isa->get_max_xlen() - 1)) | ctz(enabled_interrupts));
  }
}

reg_t processor_t::legalize_privilege(reg_t prv)
{
  assert(prv <= PRV_M);

  if (!extension_enabled('U'))
    return PRV_M;

  if (prv == PRV_HS || (prv == PRV_S && !extension_enabled('S')))
    return PRV_U;

  return prv;
}

void processor_t::set_privilege(reg_t prv, bool virt)
{
  mmu->flush_tlb();
  state.prev_prv = state.prv;
  state.prev_v = state.v;
  state.prv = legalize_privilege(prv);
  state.v = virt && state.prv != PRV_M;
  state.prv_changed = state.prv != state.prev_prv;
  state.v_changed = state.v != state.prev_v;
}

const char* processor_t::get_privilege_string()
{
  if (state.debug_mode)
    return "D";
  if (state.v) {
    switch (state.prv) {
    case 0x0: return "VU";
    case 0x1: return "VS";
    }
  } else {
    switch (state.prv) {
    case 0x0: return "U";
    case 0x1: return "S";
    case 0x3: return "M";
    }
  }
  fprintf(stderr, "Invalid prv=%lx v=%x\n", (unsigned long)state.prv, state.v);
  abort();
}

void processor_t::enter_debug_mode(uint8_t cause)
{
  state.debug_mode = true;
  state.dcsr->write_cause_and_prv(cause, state.prv, state.v);
  set_privilege(PRV_M, false);
  state.dpc->write(state.pc);
  state.pc = DEBUG_ROM_ENTRY;
  in_wfi = false;
}

void processor_t::debug_output_log(std::stringstream *s)
{
  if (log_file == stderr) {
    std::ostream out(sout_.rdbuf());
    out << s->str(); // handles command line options -d -s -l
  } else {
    fputs(s->str().c_str(), log_file); // handles command line option --log
  }
}

void processor_t::take_trap(trap_t& t, reg_t epc)
{
  unsigned max_xlen = isa->get_max_xlen();

  if (debug) {
    std::stringstream s; // first put everything in a string, later send it to output
    s << "core " << std::dec << std::setfill(' ') << std::setw(3) << id
      << ": exception " << t.name() << ", epc 0x"
      << std::hex << std::setfill('0') << std::setw(max_xlen/4) << zext(epc, max_xlen) << std::endl;
    if (t.has_tval())
       s << "core " << std::dec << std::setfill(' ') << std::setw(3) << id
         << ":           tval 0x" << std::hex << std::setfill('0') << std::setw(max_xlen / 4)
         << zext(t.get_tval(), max_xlen) << std::endl;
    debug_output_log(&s);
  }

  if (state.debug_mode) {
    if (t.cause() == CAUSE_BREAKPOINT) {
      state.pc = DEBUG_ROM_ENTRY;
    } else {
      state.pc = DEBUG_ROM_TVEC;
    }
    return;
  }

  // By default, trap to M-mode, unless delegated to HS-mode or VS-mode
  reg_t vsdeleg, hsdeleg;
  reg_t bit = t.cause();
  bool curr_virt = state.v;
  bool interrupt = (bit & ((reg_t)1 << (max_xlen - 1))) != 0;
  if (interrupt) {
    vsdeleg = (curr_virt && state.prv <= PRV_S) ? state.hideleg->read() : 0;
    hsdeleg = (state.prv <= PRV_S) ? state.mideleg->read() : 0;
    bit &= ~((reg_t)1 << (max_xlen - 1));
  } else {
    vsdeleg = (curr_virt && state.prv <= PRV_S) ? (state.medeleg->read() & state.hedeleg->read()) : 0;
    hsdeleg = (state.prv <= PRV_S) ? state.medeleg->read() : 0;
  }
  if (state.prv <= PRV_S && bit < max_xlen && ((vsdeleg >> bit) & 1)) {
    // Handle the trap in VS-mode
    reg_t vector = (state.vstvec->read() & 1) && interrupt ? 4 * bit : 0;
    state.pc = (state.vstvec->read() & ~(reg_t)1) + vector;
    state.vscause->write((interrupt) ? (t.cause() - 1) : t.cause());
    state.vsepc->write(epc);
    state.vstval->write(t.get_tval());

    reg_t s = state.sstatus->read();
    s = set_field(s, MSTATUS_SPIE, get_field(s, MSTATUS_SIE));
    s = set_field(s, MSTATUS_SPP, state.prv);
    s = set_field(s, MSTATUS_SIE, 0);
    state.sstatus->write(s);
    set_privilege(PRV_S, true);
  } else if (state.prv <= PRV_S && bit < max_xlen && ((hsdeleg >> bit) & 1)) {
    // Handle the trap in HS-mode
    reg_t vector = (state.nonvirtual_stvec->read() & 1) && interrupt ? 4 * bit : 0;
    state.pc = (state.nonvirtual_stvec->read() & ~(reg_t)1) + vector;
    state.nonvirtual_scause->write(t.cause());
    state.nonvirtual_sepc->write(epc);
    state.nonvirtual_stval->write(t.get_tval());
    state.htval->write(t.get_tval2());
    state.htinst->write(t.get_tinst());

    reg_t s = state.nonvirtual_sstatus->read();
    s = set_field(s, MSTATUS_SPIE, get_field(s, MSTATUS_SIE));
    s = set_field(s, MSTATUS_SPP, state.prv);
    s = set_field(s, MSTATUS_SIE, 0);
    state.nonvirtual_sstatus->write(s);
    if (extension_enabled('H')) {
      s = state.hstatus->read();
      if (curr_virt)
        s = set_field(s, HSTATUS_SPVP, state.prv);
      s = set_field(s, HSTATUS_SPV, curr_virt);
      s = set_field(s, HSTATUS_GVA, t.has_gva());
      state.hstatus->write(s);
    }
    set_privilege(PRV_S, false);
  } else {
    // Handle the trap in M-mode
    const reg_t vector = (state.mtvec->read() & 1) && interrupt ? 4 * bit : 0;
    const reg_t trap_handler_address = (state.mtvec->read() & ~(reg_t)1) + vector;
    // RNMI exception vector is implementation-defined.  Since we don't model
    // RNMI sources, the feature isn't very useful, so pick an invalid address.
    const reg_t rnmi_trap_handler_address = 0;
    const bool nmie = !(state.mnstatus && !get_field(state.mnstatus->read(), MNSTATUS_NMIE));
    state.pc = !nmie ? rnmi_trap_handler_address : trap_handler_address;
    state.mepc->write(epc);
    state.mcause->write(t.cause());
    state.mtval->write(t.get_tval());
    state.mtval2->write(t.get_tval2());
    state.mtinst->write(t.get_tinst());

    reg_t s = state.mstatus->read();
    s = set_field(s, MSTATUS_MPIE, get_field(s, MSTATUS_MIE));
    s = set_field(s, MSTATUS_MPP, state.prv);
    s = set_field(s, MSTATUS_MIE, 0);
    s = set_field(s, MSTATUS_MPV, curr_virt);
    s = set_field(s, MSTATUS_GVA, t.has_gva());
    state.mstatus->write(s);
    if (state.mstatush) state.mstatush->write(s >> 32);  // log mstatush change
    set_privilege(PRV_M, false);
  }
}

void processor_t::take_trigger_action(triggers::action_t action, reg_t breakpoint_tval, reg_t epc, bool virt)
{
  if (debug) {
    std::stringstream s; // first put everything in a string, later send it to output
    s << "core " << std::dec << std::setfill(' ') << std::setw(3) << id
      << ": trigger action " << (int)action << std::endl;
    debug_output_log(&s);
  }

  switch (action) {
    case triggers::ACTION_DEBUG_MODE:
      enter_debug_mode(DCSR_CAUSE_HWBP);
      break;
    case triggers::ACTION_DEBUG_EXCEPTION: {
      trap_breakpoint trap(virt, breakpoint_tval);
      take_trap(trap, epc);
      break;
    }
    default:
      abort();
  }
}

const char* processor_t::get_symbol(uint64_t addr)
{
  return sim->get_symbol(addr);
}

void processor_t::disasm(insn_t insn)
{
  uint64_t bits = insn.bits();
  if (last_pc != state.pc || last_bits != bits) {
    std::stringstream s;  // first put everything in a string, later send it to output

    const char* sym = get_symbol(state.pc);
    if (sym != nullptr)
    {
      s << "core " << std::dec << std::setfill(' ') << std::setw(3) << id
        << ": >>>>  " << sym << std::endl;
    }

    if (executions != 1) {
      s << "core " << std::dec << std::setfill(' ') << std::setw(3) << id
        << ": Executed " << executions << " times" << std::endl;
    }

    unsigned max_xlen = isa->get_max_xlen();

    s << "core " << std::dec << std::setfill(' ') << std::setw(3) << id
      << std::hex << ": 0x" << std::setfill('0') << std::setw(max_xlen / 4)
      << zext(state.pc, max_xlen) << " (0x" << std::setw(8) << bits << ") "
      << disassembler->disassemble(insn) << std::endl;

    debug_output_log(&s);

    last_pc = state.pc;
    last_bits = bits;
    executions = 1;
  } else {
    executions++;
  }
}

int processor_t::paddr_bits()
{
  unsigned max_xlen = isa->get_max_xlen();
  assert(xlen == max_xlen);
  return max_xlen == 64 ? 50 : 34;
}

void processor_t::put_csr(int which, reg_t val)
{
  val = zext_xlen(val);
  auto search = state.csrmap.find(which);
  if (search != state.csrmap.end()) {
    search->second->write(val);
    return;
  }
}

// Note that get_csr is sometimes called when read side-effects should not
// be actioned.  In other words, Spike cannot currently support CSRs with
// side effects on reads.
reg_t processor_t::get_csr(int which, insn_t insn, bool write, bool peek)
{
  auto search = state.csrmap.find(which);
  if (search != state.csrmap.end()) {
    if (!peek)
      search->second->verify_permissions(insn, write);
    return search->second->read();
  }
  // If we get here, the CSR doesn't exist.  Unimplemented CSRs always throw
  // illegal-instruction exceptions, not virtual-instruction exceptions.
  throw trap_illegal_instruction(insn.bits());
}

reg_t illegal_instruction(processor_t UNUSED *p, insn_t insn, reg_t UNUSED pc)
{
  // The illegal instruction can be longer than ILEN bits, where the tval will
  // contain the first ILEN bits of the faulting instruction. We hard-code the
  // ILEN to 32 bits since all official instructions have at most 32 bits.
  throw trap_illegal_instruction(insn.bits() & 0xffffffffULL);
}

insn_func_t processor_t::decode_insn(insn_t insn)
{
  // look up opcode in hash table
  size_t idx = insn.bits() % OPCODE_CACHE_SIZE;
  insn_desc_t desc = opcode_cache[idx];

  bool rve = extension_enabled('E');

  if (unlikely(insn.bits() != desc.match)) {
    // fall back to linear search
    int cnt = 0;
    insn_desc_t* p = &instructions[0];
    while ((insn.bits() & p->mask) != p->match)
      p++, cnt++;
    desc = *p;

    if (p->mask != 0 && p > &instructions[0]) {
      if (p->match != (p - 1)->match && p->match != (p + 1)->match) {
        // move to front of opcode list to reduce miss penalty
        while (--p >= &instructions[0])
          *(p + 1) = *p;
        instructions[0] = desc;
      }
    }

    opcode_cache[idx] = desc;
    opcode_cache[idx].match = insn.bits();
  }

  return desc.func(xlen, rve, log_commits_enabled);
}

void processor_t::register_insn(insn_desc_t desc)
{
  assert(desc.fast_rv32i && desc.fast_rv64i && desc.fast_rv32e && desc.fast_rv64e &&
         desc.logged_rv32i && desc.logged_rv64i && desc.logged_rv32e && desc.logged_rv64e);

  instructions.push_back(desc);
}

void processor_t::build_opcode_map()
{
  struct cmp {
    bool operator()(const insn_desc_t& lhs, const insn_desc_t& rhs) {
      if (lhs.match == rhs.match)
        return lhs.mask > rhs.mask;
      return lhs.match > rhs.match;
    }
  };
  std::sort(instructions.begin(), instructions.end(), cmp());

  for (size_t i = 0; i < OPCODE_CACHE_SIZE; i++)
    opcode_cache[i] = insn_desc_t::illegal();
}

void processor_t::register_extension(extension_t* x)
{
  for (auto insn : x->get_instructions())
    register_insn(insn);
  build_opcode_map();

  for (auto disasm_insn : x->get_disasms())
    disassembler->add_insn(disasm_insn);

  if (!custom_extensions.insert(std::make_pair(x->name(), x)).second) {
    fprintf(stderr, "extensions must have unique names (got two named \"%s\"!)\n", x->name());
    abort();
  }
  x->set_processor(this);
}

void processor_t::register_base_instructions()
{
  #define DECLARE_INSN(name, match, mask) \
    insn_bits_t name##_match = (match), name##_mask = (mask); \
    bool name##_supported = true;

  #include "encoding.h"
  #undef DECLARE_INSN

  #define DECLARE_OVERLAP_INSN(name, ext) { name##_supported = isa->extension_enabled(ext); }
  #include "overlap_list.h"
  #undef DECLARE_OVERLAP_INSN

  #define DEFINE_INSN(name) \
    extern reg_t fast_rv32i_##name(processor_t*, insn_t, reg_t); \
    extern reg_t fast_rv64i_##name(processor_t*, insn_t, reg_t); \
    extern reg_t fast_rv32e_##name(processor_t*, insn_t, reg_t); \
    extern reg_t fast_rv64e_##name(processor_t*, insn_t, reg_t); \
    extern reg_t logged_rv32i_##name(processor_t*, insn_t, reg_t); \
    extern reg_t logged_rv64i_##name(processor_t*, insn_t, reg_t); \
    extern reg_t logged_rv32e_##name(processor_t*, insn_t, reg_t); \
    extern reg_t logged_rv64e_##name(processor_t*, insn_t, reg_t); \
    if (name##_supported) { \
      register_insn((insn_desc_t) { \
        name##_match, \
        name##_mask, \
        fast_rv32i_##name, \
        fast_rv64i_##name, \
        fast_rv32e_##name, \
        fast_rv64e_##name, \
        logged_rv32i_##name, \
        logged_rv64i_##name, \
        logged_rv32e_##name, \
        logged_rv64e_##name}); \
    }
  #include "insn_list.h"
  #undef DEFINE_INSN

  // terminate instruction list with a catch-all
  register_insn(insn_desc_t::illegal());

  build_opcode_map();
}

bool processor_t::load(reg_t addr, size_t len, uint8_t* bytes)
{
  switch (addr)
  {
    case 0:
      if (len <= 4) {
        memset(bytes, 0, len);
        bytes[0] = get_field(state.mip->read(), MIP_MSIP);
        return true;
      }
      break;
  }

  return false;
}

bool processor_t::store(reg_t addr, size_t len, const uint8_t* bytes)
{
  switch (addr)
  {
    case 0:
      if (len <= 4) {
        state.mip->write_with_mask(MIP_MSIP, bytes[0] << IRQ_M_SOFT);
        return true;
      }
      break;
  }

  return false;
}

void processor_t::trigger_updated(const std::vector<triggers::trigger_t *> &triggers)
{
  mmu->flush_tlb();
  mmu->check_triggers_fetch = false;
  mmu->check_triggers_load = false;
  mmu->check_triggers_store = false;
  check_triggers_icount = false;

  for (auto trigger : triggers) {
    if (trigger->get_execute()) {
      mmu->check_triggers_fetch = true;
    }
    if (trigger->get_load()) {
      mmu->check_triggers_load = true;
    }
    if (trigger->get_store()) {
      mmu->check_triggers_store = true;
    }
    if (trigger->icount_check_needed()) {
      check_triggers_icount = true;
    }
  }
}

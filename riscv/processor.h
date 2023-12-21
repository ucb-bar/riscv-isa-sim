// See LICENSE for license details.
#ifndef _RISCV_PROCESSOR_H
#define _RISCV_PROCESSOR_H

#include "decode.h"
#include "trap.h"
#include "abstract_device.h"
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <map>
#include <cassert>
#include "debug_rom_defines.h"
#include "entropy_source.h"
#include "csrs.h"
#include "isa_parser.h"
#include "triggers.h"
#include "../fesvr/memif.h"
#include "vector_unit.h"

#include "arch-state.pb.h"
#include <google/protobuf/arena.h>

#define N_HPMCOUNTERS 29

class processor_t;
class mmu_t;
typedef reg_t (*insn_func_t)(processor_t*, insn_t, reg_t);
class simif_t;
class trap_t;
class extension_t;
class disassembler_t;

reg_t illegal_instruction(processor_t* p, insn_t insn, reg_t pc);

struct insn_desc_t
{
  insn_bits_t match;
  insn_bits_t mask;
  insn_func_t fast_rv32i;
  insn_func_t fast_rv64i;
  insn_func_t fast_rv32e;
  insn_func_t fast_rv64e;
  insn_func_t logged_rv32i;
  insn_func_t logged_rv64i;
  insn_func_t logged_rv32e;
  insn_func_t logged_rv64e;

  insn_func_t func(int xlen, bool rve, bool logged)
  {
    if (logged)
      if (rve)
        return xlen == 64 ? logged_rv64e : logged_rv32e;
      else
        return xlen == 64 ? logged_rv64i : logged_rv32i;
    else
      if (rve)
        return xlen == 64 ? fast_rv64e : fast_rv32e;
      else
        return xlen == 64 ? fast_rv64i : fast_rv32i;
  }

  static insn_desc_t illegal()
  {
    return {0, 0,
            &illegal_instruction, &illegal_instruction, &illegal_instruction, &illegal_instruction,
            &illegal_instruction, &illegal_instruction, &illegal_instruction, &illegal_instruction};
  }
};

// regnum, data
typedef std::unordered_map<reg_t, freg_t> commit_log_reg_t;

// addr, value, size
typedef std::vector<std::tuple<reg_t, uint64_t, uint8_t>> commit_log_mem_t;

// architectural state of a RISC-V hart
struct state_t
{
  void reset(processor_t* const proc, reg_t max_isa);
  bool operator == (state_t& state);

  reg_t pc;
  regfile_t<reg_t, NXPR, true> XPR;
  regfile_t<freg_t, NFPR, false> FPR;

  // control and status registers
  std::unordered_map<reg_t, csr_t_p> csrmap;
  reg_t prv;    // TODO: Can this be an enum instead?
  reg_t prev_prv;
  bool prv_changed;
  bool v_changed;
  bool v;
  bool prev_v;
  misa_csr_t_p misa;
  mstatus_csr_t_p mstatus;
  csr_t_p mstatush;
  csr_t_p mepc;
  csr_t_p mtval;
  csr_t_p mtvec;
  csr_t_p mcause;
  wide_counter_csr_t_p minstret;
  wide_counter_csr_t_p mcycle;
  mie_csr_t_p mie;
  mip_csr_t_p mip;
  csr_t_p medeleg;
  csr_t_p mideleg;
  csr_t_p mcounteren;
  csr_t_p mevent[N_HPMCOUNTERS];
  csr_t_p mnstatus;
  csr_t_p mnepc;
  csr_t_p scounteren;
  csr_t_p sepc;
  csr_t_p stval;
  csr_t_p stvec;
  virtualized_csr_t_p satp;
  csr_t_p scause;

  // When taking a trap into HS-mode, we must access the nonvirtualized HS-mode CSRs directly:
  csr_t_p nonvirtual_stvec;
  csr_t_p nonvirtual_scause;
  csr_t_p nonvirtual_sepc;
  csr_t_p nonvirtual_stval;
  sstatus_proxy_csr_t_p nonvirtual_sstatus;

  csr_t_p mtval2;
  csr_t_p mtinst;
  csr_t_p hstatus;
  csr_t_p hideleg;
  csr_t_p hedeleg;
  csr_t_p hcounteren;
  csr_t_p htval;
  csr_t_p htinst;
  csr_t_p hgatp;
  sstatus_csr_t_p sstatus;
  vsstatus_csr_t_p vsstatus;
  csr_t_p vstvec;
  csr_t_p vsepc;
  csr_t_p vscause;
  csr_t_p vstval;
  csr_t_p vsatp;

  csr_t_p dpc;
  dcsr_csr_t_p dcsr;
  csr_t_p tselect;
  csr_t_p tdata2;
  csr_t_p scontext;
  csr_t_p mcontext;

  csr_t_p jvt;

  bool debug_mode;

  mseccfg_csr_t_p mseccfg;

  static const int max_pmp = 64;
  pmpaddr_csr_t_p pmpaddr[max_pmp];

  float_csr_t_p fflags;
  float_csr_t_p frm;

  csr_t_p menvcfg;
  csr_t_p senvcfg;
  csr_t_p henvcfg;

  csr_t_p mstateen[4];
  csr_t_p sstateen[4];
  csr_t_p hstateen[4];

  csr_t_p htimedelta;
  time_counter_csr_t_p time;
  csr_t_p time_proxy;

  csr_t_p stimecmp;
  csr_t_p vstimecmp;

  bool serialized; // whether timer CSRs are in a well-defined state

  // When true, execute a single instruction and then enter debug mode.  This
  // can only be set by executing dret.
  enum {
      STEP_NONE,
      STEP_STEPPING,
      STEP_STEPPED
  } single_step;

  commit_log_reg_t log_reg_write;
  commit_log_mem_t log_mem_read;
  commit_log_mem_t log_mem_write;
  reg_t last_inst_priv;
  int last_inst_xlen;
  int last_inst_flen;
};

// this class represents one processor in a RISC-V machine.
class processor_t : public abstract_device_t
{
public:
  processor_t(const isa_parser_t *isa, const cfg_t* cfg,
              simif_t* sim, uint32_t id, bool halt_on_reset,
              FILE *log_file, std::ostream& sout_); // because of command line option --log and -s we need both
  ~processor_t();

  const isa_parser_t &get_isa() { return *isa; }
  const cfg_t &get_cfg() { return *cfg; }

  void set_debug(bool value);
  void set_histogram(bool value);
  void enable_log_commits();
  bool get_log_commits_enabled() const { return log_commits_enabled; }
  void reset();
  void step(size_t n); // run for n cycles
  void put_csr(int which, reg_t val);
  uint32_t get_id() const { return id; }
  reg_t get_csr(int which, insn_t insn, bool write, bool peek = 0);
  reg_t get_csr(int which) { return get_csr(which, insn_t(0), false, true); }
  mmu_t* get_mmu() { return mmu; }
  state_t* get_state() { return &state; }
  unsigned get_xlen() const { return xlen; }
  unsigned get_const_xlen() const {
    // Any code that assumes a const xlen should use this method to
    // document that assumption. If Spike ever changes to allow
    // variable xlen, this method should be removed.
    return xlen;
  }
  unsigned get_flen() const {
    return extension_enabled('Q') ? 128 :
           extension_enabled('D') ? 64 :
           extension_enabled('F') ? 32 : 0;
  }
  extension_t* get_extension();
  extension_t* get_extension(const char* name);
  bool any_custom_extensions() const {
    return !custom_extensions.empty();
  }
  bool extension_enabled(unsigned char ext) const {
    return extension_enabled(isa_extension_t(ext));
  }
  bool extension_enabled(isa_extension_t ext) const {
    if (ext >= 'A' && ext <= 'Z')
      return state.misa->extension_enabled(ext);
    else
      return extension_enable_table[ext];
  }
  // Is this extension enabled? and abort if this extension can
  // possibly be disabled dynamically. Useful for documenting
  // assumptions about writable misa bits.
  bool extension_enabled_const(unsigned char ext) const {
    return extension_enabled_const(isa_extension_t(ext));
  }
  bool extension_enabled_const(isa_extension_t ext) const {
    if (ext >= 'A' && ext <= 'Z') {
      return state.misa->extension_enabled_const(ext);
    } else {
      assert(!extension_dynamic[ext]);
      extension_assumed_const[ext] = true;
      return extension_enabled(ext);
    }
  }
  void set_extension_enable(unsigned char ext, bool enable) {
    assert(!extension_assumed_const[ext]);
    extension_dynamic[ext] = true;
    extension_enable_table[ext] = enable && isa->extension_enabled(ext);
  }
  void set_impl(uint8_t impl, bool val) { impl_table[impl] = val; }
  bool supports_impl(uint8_t impl) const {
    return impl_table[impl];
  }
  reg_t pc_alignment_mask() {
    const int ialign = extension_enabled(EXT_ZCA) ? 16 : 32;
    return ~(reg_t)(ialign == 16 ? 0 : 2);
  }
  void check_pc_alignment(reg_t pc) {
    if (unlikely(pc & ~pc_alignment_mask()))
      throw trap_instruction_address_misaligned(state.v, pc, 0, 0);
  }
  reg_t legalize_privilege(reg_t);
  void set_privilege(reg_t, bool);
  const char* get_privilege_string();
  void update_histogram(reg_t pc);
  const disassembler_t* get_disassembler() { return disassembler; }

  FILE *get_log_file() { return log_file; }

  void register_insn(insn_desc_t);
  void register_extension(extension_t*);

  // MMIO slave interface
  bool load(reg_t addr, size_t len, uint8_t* bytes);
  bool store(reg_t addr, size_t len, const uint8_t* bytes);

  // When true, display disassembly of each instruction that's executed.
  bool debug;
  // When true, take the slow simulation path.
  bool slow_path();
  bool halted() { return state.debug_mode; }
  enum {
    HR_NONE,    /* Halt request is inactive. */
    HR_REGULAR, /* Regular halt request/debug interrupt. */
    HR_GROUP    /* Halt requested due to halt group. */
  } halt_request;

  void trigger_updated(const std::vector<triggers::trigger_t *> &triggers);

  void set_pmp_num(reg_t pmp_num);
  void set_pmp_granularity(reg_t pmp_granularity);
  void set_mmu_capability(int cap);

  const char* get_symbol(uint64_t addr);

  void clear_waiting_for_interrupt() { in_wfi = false; };
  bool is_waiting_for_interrupt() { return in_wfi; };

private:
  const isa_parser_t * const isa;
  const cfg_t * const cfg;

  simif_t* sim;
  mmu_t* mmu; // main memory is always accessed via the mmu
  std::unordered_map<std::string, extension_t*> custom_extensions;
  disassembler_t* disassembler;
  state_t state;
  uint32_t id;
  unsigned xlen;
  bool histogram_enabled;
  bool log_commits_enabled;
  FILE *log_file;
  std::ostream sout_; // needed for socket command interface -s, also used for -d and -l, but not for --log
  bool halt_on_reset;
  bool in_wfi;
  bool check_triggers_icount;
  std::vector<bool> impl_table;

  // Note: does not include single-letter extensions in misa
  std::bitset<NUM_ISA_EXTENSIONS> extension_enable_table;
  std::bitset<NUM_ISA_EXTENSIONS> extension_dynamic;
  mutable std::bitset<NUM_ISA_EXTENSIONS> extension_assumed_const;

  std::vector<insn_desc_t> instructions;
  std::unordered_map<reg_t,uint64_t> pc_histogram;

  static const size_t OPCODE_CACHE_SIZE = 8191;
  insn_desc_t opcode_cache[OPCODE_CACHE_SIZE];

  void take_pending_interrupt() { take_interrupt(state.mip->read() & state.mie->read()); }
  void take_interrupt(reg_t mask); // take first enabled interrupt in mask
  void take_trap(trap_t& t, reg_t epc); // take an exception
  void take_trigger_action(triggers::action_t action, reg_t breakpoint_tval, reg_t epc, bool virt);
  void disasm(insn_t insn); // disassemble and print an instruction
  int paddr_bits();

  void enter_debug_mode(uint8_t cause);

  void debug_output_log(std::stringstream *s); // either output to interactive user or write to log file

  friend class mmu_t;
  friend class clint_t;
  friend class plic_t;
  friend class extension_t;

  void parse_varch_string(const char*);
  void parse_priv_string(const char*);
  void build_opcode_map();
  void register_base_instructions();
  insn_func_t decode_insn(insn_t insn);

  // Track repeated executions for processor_t::disasm()
  uint64_t last_pc, last_bits, executions;
public:
  entropy_source es; // Crypto ISE Entropy source.

  reg_t n_pmp;
  reg_t lg_pmp_granularity;
  reg_t pmp_tor_mask() { return -(reg_t(1) << (lg_pmp_granularity - PMP_SHIFT)); }

  vectorUnit_t VU;
  triggers::module_t TM;

  uint64_t tot_instret = 0;

public:
  google::protobuf::Arena* arena;

  BasicCSR* gen_basic_csr_proto(reg_t init) {
    BasicCSR* b = google::protobuf::Arena::Create<BasicCSR>(arena);
    b->set_msg_val(init);
    return b;
  }

  MisaCSR* gen_misa_csr_proto(misa_csr_t_p ptr) {
    MisaCSR*  misa = google::protobuf::Arena::Create<MisaCSR>(arena);
    BasicCSR* b = gen_basic_csr_proto(ptr->val);
    misa->set_allocated_msg_basic_csr(b);
    misa->set_msg_max_isa(ptr->max_isa);
    misa->set_msg_write_mask(ptr->write_mask);
    return misa;
  }

  BaseStatusCSR* gen_base_status_csr_proto(bool has_page, reg_t wm, reg_t rm) {
    BaseStatusCSR* b = google::protobuf::Arena::Create<BaseStatusCSR>(arena);
    b->set_msg_has_page(has_page);
    b->set_msg_sstatus_write_mask(wm);
    b->set_msg_sstatus_read_mask(rm);
    return b;
  }

  MstatusCSR* gen_mstatus_csr_proto(mstatus_csr_t_p csr) {
    MstatusCSR* m = google::protobuf::Arena::Create<MstatusCSR>(arena);
    BaseStatusCSR* base = gen_base_status_csr_proto(csr->has_page,
                                                    csr->sstatus_write_mask,
                                                    csr->sstatus_read_mask);
    m->set_allocated_msg_base_status_csr(base);
    m->set_msg_val(csr->val);
    return m;
  }

  SstatusProxyCSR* gen_sstatus_proxy_csr_proto(sstatus_proxy_csr_t_p csr) {
    SstatusProxyCSR* sp = google::protobuf::Arena::Create<SstatusProxyCSR>(arena);
    MstatusCSR*    m = gen_mstatus_csr_proto(csr->mstatus);
    BaseStatusCSR* b = gen_base_status_csr_proto(csr->has_page,
                                                 csr->sstatus_write_mask,
                                                 csr->sstatus_read_mask);
    sp->set_allocated_msg_base_status_csr(b);
    sp->set_allocated_msg_mstatus_csr(m);
    return sp;
  }

  VsstatusCSR* gen_vsstatus_csr_proto(vsstatus_csr_t_p csr) {
    VsstatusCSR* v = google::protobuf::Arena::Create<VsstatusCSR>(arena);
    BaseStatusCSR* b = gen_base_status_csr_proto(csr->has_page,
                                                 csr->sstatus_write_mask,
                                                 csr->sstatus_read_mask);
    v->set_allocated_msg_base_status_csr(b);
    v->set_msg_val(csr->val);
    return v;
  }

  SstatusCSR* gen_sstatus_csr_proto(sstatus_csr_t_p csr) {
    SstatusCSR*      s  = google::protobuf::Arena::Create<SstatusCSR>(arena);
    SstatusProxyCSR* sp = gen_sstatus_proxy_csr_proto(csr->orig_sstatus);
    VsstatusCSR*     v  = gen_vsstatus_csr_proto(csr->virt_sstatus);

    s->set_allocated_msg_orig_sstatus(sp);
    s->set_allocated_msg_virt_sstatus(v);
    return s;
  }

  MaskedCSR* gen_masked_csr_proto(reg_t val, reg_t mask) {
    MaskedCSR* m = google::protobuf::Arena::Create<MaskedCSR>(arena);
    BasicCSR*  b = gen_basic_csr_proto(val);
    m->set_allocated_msg_basic_csr(b);
    m->set_msg_mask(mask);
    return m;
  }

  SmcntrpmfCSR* gen_smcntrpmf_csr_proto(smcntrpmf_csr_t_p csr) {
    SmcntrpmfCSR* s = google::protobuf::Arena::Create<SmcntrpmfCSR>(arena);
    MaskedCSR*    m = gen_masked_csr_proto(csr->val, csr->mask);

    s->set_allocated_msg_masked_csr(m);
    if (csr->prev_val.has_value()) {
      OptionalUInt64* o = google::protobuf::Arena::Create<OptionalUInt64>(arena);
      o->set_msg_val(csr->prev_val.value());
      s->set_allocated_msg_prev_val(o);
    }
    return s;
  }

  WideCntrCSR* gen_wide_cntr_csr_proto(wide_counter_csr_t_p csr) {
    WideCntrCSR*  w = google::protobuf::Arena::Create<WideCntrCSR>(arena);
    SmcntrpmfCSR* s = gen_smcntrpmf_csr_proto(csr->config_csr);

    w->set_msg_val(csr->val);
    w->set_allocated_msg_config_csr(s);
    return w;
  }

  MedelegCSR* gen_medeleg_csr_proto(csr_t_p csr) {
    MedelegCSR* m = google::protobuf::Arena::Create<MedelegCSR>(arena);
    auto medeleg = std::dynamic_pointer_cast<medeleg_csr_t>(csr);
    BasicCSR* b = gen_basic_csr_proto(medeleg->val);

    m->set_allocated_msg_basic_csr(b);
    m->set_msg_hypervisor_exceptions(medeleg->hypervisor_exceptions);
    return m;
  }

  template <class CSR_T>
  VirtBasicCSR* gen_virt_basic_csr_proto(virtualized_csr_t_p csr) {
    auto vcsr = std::dynamic_pointer_cast<CSR_T>(csr->virt_csr);
    auto ocsr = std::dynamic_pointer_cast<CSR_T>(csr->orig_csr);
    BasicCSR* vproto = gen_basic_csr_proto(vcsr->val);
    BasicCSR* oproto = gen_basic_csr_proto(ocsr->val);

    VirtBasicCSR* vb_proto = google::protobuf::Arena::Create<VirtBasicCSR>(arena);
    vb_proto->set_allocated_msg_nonvirt_csr(oproto);
    vb_proto->set_allocated_msg_virt_csr(vproto);
    return vb_proto;
  }

  HidelegCSR* gen_hideleg_csr_proto(csr_t_p csr) {
    auto hideleg = std::dynamic_pointer_cast<hideleg_csr_t>(csr);
    auto mideleg = std::dynamic_pointer_cast<basic_csr_t>(hideleg->mideleg);

    MaskedCSR* hideleg_proto = gen_masked_csr_proto(hideleg->val, hideleg->mask);
    BasicCSR*  mideleg_proto = gen_basic_csr_proto(mideleg->val);

    HidelegCSR* proto = google::protobuf::Arena::Create<HidelegCSR>(arena);
    proto->set_allocated_msg_hideleg_csr(hideleg_proto);
    proto->set_allocated_msg_mideleg_csr(mideleg_proto);
    return proto;
  }

  DCSR* gen_dcsr_csr_proto(dcsr_csr_t_p csr) {
    DCSR* d = google::protobuf::Arena::Create<DCSR>(arena);
    d->set_msg_prv     (csr->prv);
    d->set_msg_step    (csr->step);
    d->set_msg_ebreakm (csr->ebreakm);
    d->set_msg_ebreaks (csr->ebreaks);
    d->set_msg_ebreaku (csr->ebreaku);
    d->set_msg_ebreakvs(csr->ebreakvs);
    d->set_msg_ebreakvu(csr->ebreakvu);
    d->set_msg_halt    (csr->halt);
    d->set_msg_v       (csr->v);
    d->set_msg_cause   (csr->cause);
    return d;
  }

  McontextCSR* gen_mcontext_csr_proto(std::shared_ptr<proxy_csr_t> csr) {
    McontextCSR* mcp = google::protobuf::Arena::Create<McontextCSR>(arena);
    auto mc = std::dynamic_pointer_cast<masked_csr_t>(csr->delegate);
    MaskedCSR* mp = gen_masked_csr_proto(mc->val, mc->mask);

    mcp->set_allocated_msg_delegate(mp);
    return mcp;
  }

  HenvcfgCSR* gen_henvcfg_csr_proto(std::shared_ptr<henvcfg_csr_t> csr) {
    HenvcfgCSR* henvproto = google::protobuf::Arena::Create<HenvcfgCSR>(arena);

    auto menvcfg = std::dynamic_pointer_cast<masked_csr_t>(csr->menvcfg);
    MaskedCSR* mproto = gen_masked_csr_proto(menvcfg->val, menvcfg->mask);
    MaskedCSR* hproto = gen_masked_csr_proto(csr->val,     csr->mask);

    henvproto->set_allocated_msg_henvcfg(hproto);
    henvproto->set_allocated_msg_menvcfg(mproto);
    return henvproto;
  }

  StimecmpCSR* gen_stimecmp_csr_proto(std::shared_ptr<stimecmp_csr_t> csr) {
    StimecmpCSR* sp = google::protobuf::Arena::Create<StimecmpCSR>(arena);
    BasicCSR*    bp = gen_basic_csr_proto(csr->val);

    sp->set_allocated_msg_basic_csr(bp);
    sp->set_msg_intr_mask(csr->intr_mask);
    return sp;
  }

  void serialize_proto(ArchState* aproto, google::protobuf::Arena* arena) {
    std::cout << "serialize" << std::endl;
    assert(xlen == 64);

    auto csrmap = state.csrmap;

    this->arena = arena;

    aproto->set_msg_pc(state.pc);

    for (int i = 0; i < NXPR; i++) {
      aproto->add_msg_xpr(state.XPR[i]);
    }

    for (int i = 0; i < NFPR; i++) {
      Float128* fp = aproto->add_msg_fpr();
      fp->set_msg_0(state.FPR[i].v[0]);
      fp->set_msg_1(state.FPR[i].v[1]);
    }

    aproto->set_msg_prv(state.prv);
    aproto->set_msg_prev_prv(state.prev_prv);
    aproto->set_msg_prv_changed(state.prv_changed);
    aproto->set_msg_v_changed(state.v_changed);
    aproto->set_msg_v(state.v);
    aproto->set_msg_prev_v(state.prev_v);

    std::cout << " pc: " << state.pc
              << " prv: " << state.prv
              << " prev_prv: " << state.prev_prv
              << " prv_changed: " << state.prv_changed 
              << " v_changed: " << state.v_changed 
              << " v: " << state.v 
              << " prev_v: " << state.prev_v << std::endl;

    if (state.misa) {
      MisaCSR* misa_proto = gen_misa_csr_proto(state.misa);
      aproto->set_allocated_msg_misa(misa_proto);
    }

    if (state.mstatus) {
      MstatusCSR* mstatus_proto = gen_mstatus_csr_proto(state.mstatus);
      aproto->set_allocated_msg_mstatus(mstatus_proto);
    }

    if (state.mepc) {
      auto mepc = std::dynamic_pointer_cast<epc_csr_t>(state.mepc);
      BasicCSR* mepc_proto = gen_basic_csr_proto(mepc->val);
      aproto->set_allocated_msg_mepc(mepc_proto);
    }

    if (state.mtval) {
      auto mtval = std::dynamic_pointer_cast<basic_csr_t>(state.mtval);
      BasicCSR* mtval_proto = gen_basic_csr_proto(mtval->val);
      aproto->set_allocated_msg_mtval(mtval_proto);
    }

    auto it = csrmap.find(CSR_MSCRATCH);
    if (it != csrmap.end()) {
      auto csr = std::dynamic_pointer_cast<basic_csr_t>(it->second);
      BasicCSR* proto = gen_basic_csr_proto(csr->val);
      aproto->set_allocated_msg_mscratch(proto);
    }

    if (state.mtvec) {
      auto mtvec = std::dynamic_pointer_cast<tvec_csr_t>(state.mtvec);
      BasicCSR* mtvec_proto = gen_basic_csr_proto(mtvec->val);
      aproto->set_allocated_msg_mtvec(mtvec_proto);
    }

    if (state.mcause) {
      auto mcause = std::dynamic_pointer_cast<cause_csr_t>(state.mcause);
      BasicCSR* mcause_proto = gen_basic_csr_proto(mcause->val);
      aproto->set_allocated_msg_mcause(mcause_proto);
    }

    if (state.minstret) {
      WideCntrCSR* minstret_proto = gen_wide_cntr_csr_proto(state.minstret);
      aproto->set_allocated_msg_minstret(minstret_proto);
    }

    if (state.mcycle) {
      WideCntrCSR* mcycle_proto = gen_wide_cntr_csr_proto(state.mcycle);
      aproto->set_allocated_msg_mcycle(mcycle_proto);
    }

    if (state.time) {
      auto t = state.time;
      BasicCSR* b = gen_basic_csr_proto(t->shadow_val);
      aproto->set_allocated_msg_time(b);
    }

    for (int i = 0; i < N_HPMCOUNTERS; i++) {
      if (state.mevent[i]) {
        BasicCSR* b_proto = aproto->add_msg_mevent();
        auto mevent = std::dynamic_pointer_cast<basic_csr_t>(state.mevent[i]);
        b_proto->set_msg_val(mevent->val);
      }
    }

    if (state.mie) {
      BasicCSR* mie_proto = gen_basic_csr_proto(state.mie->val);
      aproto->set_allocated_msg_mie(mie_proto);
    }

    if (state.mip) {
      BasicCSR* mip_proto = gen_basic_csr_proto(state.mip->val);
      aproto->set_allocated_msg_mip(mip_proto);
    }

    if (state.medeleg) {
      MedelegCSR* medeleg_proto = gen_medeleg_csr_proto(state.medeleg);
      aproto->set_allocated_msg_medeleg(medeleg_proto);
    }

    if (state.mcounteren) {
      auto csr = std::dynamic_pointer_cast<masked_csr_t>(state.mcounteren);
      MaskedCSR* proto = gen_masked_csr_proto(csr->val, csr->mask);
      aproto->set_allocated_msg_mcounteren(proto);
    }

    if (state.scounteren) {
      auto sen = std::dynamic_pointer_cast<masked_csr_t>(state.scounteren);
      MaskedCSR* m_proto = gen_masked_csr_proto(sen->val, sen->mask);
      aproto->set_allocated_msg_scounteren(m_proto);
    }

    if (state.sepc) {
      auto sepc = std::dynamic_pointer_cast<virtualized_csr_t>(state.sepc);
      VirtBasicCSR* sepc_proto = gen_virt_basic_csr_proto<epc_csr_t>(sepc);
      aproto->set_allocated_msg_sepc(sepc_proto);
    }

    if (state.stval) {
      auto stval = std::dynamic_pointer_cast<virtualized_csr_t>(state.stval);
      VirtBasicCSR* stval_proto = gen_virt_basic_csr_proto<basic_csr_t>(stval);
      aproto->set_allocated_msg_stval(stval_proto);
    }

    it = csrmap.find(CSR_SSCRATCH);
    if (it != csrmap.end()) {
      auto csr = std::dynamic_pointer_cast<virtualized_csr_t>(it->second);
      VirtBasicCSR* proto = gen_virt_basic_csr_proto<basic_csr_t>(csr);
      aproto->set_allocated_msg_sscratch(proto);
    }

    if (state.stvec) {
      auto stvec = std::dynamic_pointer_cast<virtualized_csr_t>(state.stvec);
      VirtBasicCSR* stvec_proto = gen_virt_basic_csr_proto<tvec_csr_t>(stvec);
      aproto->set_allocated_msg_stvec(stvec_proto);
    }

    if (state.satp) {
      VirtBasicCSR* satp_proto = gen_virt_basic_csr_proto<basic_csr_t>(state.satp);
      aproto->set_allocated_msg_satp(satp_proto);
    }

    if (state.scause) {
      auto scause = std::dynamic_pointer_cast<virtualized_csr_t>(state.scause);
      VirtBasicCSR* scause_proto = gen_virt_basic_csr_proto<basic_csr_t>(scause);
      aproto->set_allocated_msg_scause(scause_proto);
    }

    if (state.mtval2) {
      auto mtval2 = std::dynamic_pointer_cast<basic_csr_t>(state.mtval2);
      BasicCSR* mtval2_proto = gen_basic_csr_proto(mtval2->val);
      aproto->set_allocated_msg_mtval2(mtval2_proto);
    }

    if (state.mtinst) {
      auto mtinst = std::dynamic_pointer_cast<basic_csr_t>(state.mtinst);
      BasicCSR* mtinst_proto = gen_basic_csr_proto(mtinst->val);;
      aproto->set_allocated_msg_mtinst(mtinst_proto);
    }

    if (state.hstatus) {
      auto csr = std::dynamic_pointer_cast<masked_csr_t>(state.hstatus);
      MaskedCSR* proto = gen_masked_csr_proto(csr->val, csr->mask);
      aproto->set_allocated_msg_hstatus(proto);
    }

    if (state.hideleg) {
      HidelegCSR* hideleg_proto = gen_hideleg_csr_proto(state.hideleg);
      aproto->set_allocated_msg_hideleg(hideleg_proto);
    }

    if (state.hedeleg) {
      auto csr = std::dynamic_pointer_cast<masked_csr_t>(state.hedeleg);
      MaskedCSR* proto = gen_masked_csr_proto(csr->val, csr->mask);
      aproto->set_allocated_msg_hedeleg(proto);
    }

    if (state.hcounteren) {
      auto csr = std::dynamic_pointer_cast<masked_csr_t>(state.hcounteren);
      MaskedCSR* proto = gen_masked_csr_proto(csr->val, csr->mask);
      aproto->set_allocated_msg_hcounteren(proto);
    }

    if (state.htimedelta) {
      auto ht = std::dynamic_pointer_cast<basic_csr_t>(state.htimedelta);
      BasicCSR* b = gen_basic_csr_proto(ht->val);
      aproto->set_allocated_msg_htimedelta(b);
    }

    if (state.htval) {
      auto htval = std::dynamic_pointer_cast<basic_csr_t>(state.htval);
      BasicCSR* b_proto = gen_basic_csr_proto(htval->val);
      aproto->set_allocated_msg_htval(b_proto);
    }

    if (state.htinst) {
      auto htinst = std::dynamic_pointer_cast<basic_csr_t>(state.htinst);
      BasicCSR* b_proto = gen_basic_csr_proto(htinst->val);
      aproto->set_allocated_msg_htinst(b_proto);
    }

    if (state.hgatp) {
      auto hgatp = std::dynamic_pointer_cast<basic_csr_t>(state.hgatp);
      BasicCSR* b_proto = gen_basic_csr_proto(hgatp->val);
      aproto->set_allocated_msg_hgatp(b_proto);
    }

    if (state.sstatus) {
      SstatusCSR* s_proto = gen_sstatus_csr_proto(state.sstatus);
      aproto->set_allocated_msg_sstatus(s_proto);
    }

    if (state.dpc) {
      auto dpc = std::dynamic_pointer_cast<epc_csr_t>(state.dpc);
      BasicCSR* b = gen_basic_csr_proto(dpc->val);
      aproto->set_allocated_msg_dpc(b);
    }

    it = csrmap.find(CSR_DSCRATCH0);
    if (it != csrmap.end()) {
      auto csr = std::dynamic_pointer_cast<basic_csr_t>(it->second);
      BasicCSR* b = gen_basic_csr_proto(csr->val);
      aproto->set_allocated_msg_dscratch0(b);
    }

    it = csrmap.find(CSR_DSCRATCH1);
    if (it != csrmap.end()) {
      auto csr = std::dynamic_pointer_cast<basic_csr_t>(it->second);
      BasicCSR* b = gen_basic_csr_proto(csr->val);
      aproto->set_allocated_msg_dscratch1(b);
    }

    if (state.dcsr) {
      auto dcsr = std::dynamic_pointer_cast<dcsr_csr_t>(state.dcsr);
      DCSR* d = gen_dcsr_csr_proto(dcsr);
      aproto->set_allocated_msg_dcsr(d);
    }

    if (state.tselect) {
      auto tsel = std::dynamic_pointer_cast<basic_csr_t>(state.tselect);
      BasicCSR* b = gen_basic_csr_proto(tsel->val);
      aproto->set_allocated_msg_tselect(b);
    }

    if (state.scontext) {
      auto sc = std::dynamic_pointer_cast<masked_csr_t>(state.scontext);
      MaskedCSR* c = gen_masked_csr_proto(sc->val, sc->mask);
      aproto->set_allocated_msg_scontext(c);
    }

    it = csrmap.find(CSR_HCONTEXT);
    if (it != csrmap.end()) {
      auto csr = std::dynamic_pointer_cast<masked_csr_t>(it->second);
      MaskedCSR* proto = gen_masked_csr_proto(csr->val, csr->mask);
      aproto->set_allocated_msg_hcontext(proto);
    }

    if (state.mseccfg) {
      auto mseccfg = std::dynamic_pointer_cast<basic_csr_t>(state.mseccfg);
      BasicCSR* b = gen_basic_csr_proto(mseccfg->val);
      aproto->set_allocated_msg_mseccfg(b);
    }

    for (int i = 0; i < state.max_pmp; i++) {
      if (state.pmpaddr[i]) {
        PmpCSR* p_proto = aproto->add_msg_pmpaddr();

        auto pmpaddr = state.pmpaddr[i];
        BasicCSR* c_proto = gen_basic_csr_proto(pmpaddr->val);
        p_proto->set_allocated_msg_basic_csr(c_proto);
        p_proto->set_msg_cfg(pmpaddr->cfg);
        p_proto->set_msg_pmpidx(pmpaddr->pmpidx);
      }
    }

    if (state.fflags) {
      auto fflags = state.fflags;
      MaskedCSR* m_proto = gen_masked_csr_proto(fflags->val, fflags->mask);
      aproto->set_allocated_msg_fflags(m_proto);
    }

    if (state.frm) {
      auto frm = state.frm;
      MaskedCSR* m_proto = gen_masked_csr_proto(frm->val, frm->mask);
      aproto->set_allocated_msg_frm(m_proto);
    }

    if (state.senvcfg) {
      auto senv = std::dynamic_pointer_cast<masked_csr_t>(state.senvcfg);
      MaskedCSR* m_proto = gen_masked_csr_proto(senv->val, senv->mask);
      aproto->set_allocated_msg_senvcfg(m_proto);
    }

    if (state.henvcfg) {
      auto henv = std::dynamic_pointer_cast<henvcfg_csr_t>(state.henvcfg);
      HenvcfgCSR* h_proto = gen_henvcfg_csr_proto(henv);
      aproto->set_allocated_msg_henvcfg(h_proto);
    }

    for (int i = 0; i < 4; i++) {
      if (state.mstateen[i]) {
        auto csr = std::dynamic_pointer_cast<masked_csr_t>(state.mstateen[i]);
        MaskedCSR* m_proto = aproto->add_msg_mstateen();
        BasicCSR*  b_proto = gen_basic_csr_proto(csr->val);
        m_proto->set_allocated_msg_basic_csr(b_proto);
        m_proto->set_msg_mask(csr->mask);
      }
    }

    for (int i = 0; i < 4; i++) {
      if (state.sstateen[i]) {
        auto csr = std::dynamic_pointer_cast<hstateen_csr_t>(state.sstateen[i]);
        HstateenCSR* h_proto = aproto->add_msg_sstateen();
        MaskedCSR*   m_proto = gen_masked_csr_proto(csr->val, csr->mask);
        h_proto->set_allocated_msg_masked_csr(m_proto);
        h_proto->set_msg_index(csr->index);
      }
    }

    for (int i = 0; i < 4; i++) {
      if (state.hstateen[i]) {
        auto csr = std::dynamic_pointer_cast<hstateen_csr_t>(state.hstateen[i]);
        HstateenCSR* h_proto = aproto->add_msg_hstateen();
        MaskedCSR*   m_proto = gen_masked_csr_proto(csr->val, csr->mask);
        h_proto->set_msg_index(csr->index);
      }
    }

    it = csrmap.find(CSR_MNSCRATCH);
    if (it != csrmap.end()) {
      auto csr = std::dynamic_pointer_cast<basic_csr_t>(it->second);
      BasicCSR* proto = gen_basic_csr_proto(csr->val);
      aproto->set_allocated_msg_mnscratch(proto);
    }

    if (state.mnepc) {
      auto csr = std::dynamic_pointer_cast<basic_csr_t>(state.mnepc);
      BasicCSR* proto = gen_basic_csr_proto(csr->val);
      aproto->set_allocated_msg_mnepc(proto);
    }

    if (state.mnstatus) {
      auto csr = std::dynamic_pointer_cast<basic_csr_t>(state.mnstatus);
      BasicCSR* proto = gen_basic_csr_proto(csr->val);
      aproto->set_allocated_msg_mnstatus(proto);
    }

    if (state.stimecmp) {
      auto st = std::dynamic_pointer_cast<stimecmp_csr_t>(state.stimecmp);
      StimecmpCSR* sp = gen_stimecmp_csr_proto(st);
      aproto->set_allocated_msg_stimecmp(sp);
    }

    if (state.vstimecmp) {
      auto st = std::dynamic_pointer_cast<stimecmp_csr_t>(state.vstimecmp);
      StimecmpCSR* sp = gen_stimecmp_csr_proto(st);
      aproto->set_allocated_msg_vstimecmp(sp);
    }

    if (state.jvt) {
      auto jvt = std::dynamic_pointer_cast<basic_csr_t>(state.jvt);
      BasicCSR* b = gen_basic_csr_proto(jvt->val);
      aproto->set_allocated_msg_jvt(b);
    }

    it = csrmap.find(CSR_MISELECT);
    if (it != csrmap.end()) {
      auto csr = std::dynamic_pointer_cast<basic_csr_t>(it->second);
      BasicCSR* proto = gen_basic_csr_proto(csr->val);
      aproto->set_allocated_msg_miselect(proto);
    }

    it = csrmap.find(CSR_SISELECT);
    if (it != csrmap.end()) {
      auto csr = std::dynamic_pointer_cast<virtualized_csr_t>(it->second);
      VirtBasicCSR* proto = gen_virt_basic_csr_proto<basic_csr_t>(csr);
      aproto->set_allocated_msg_siselect(proto);
    }

    aproto->set_msg_debug_mode(state.debug_mode);
    aproto->set_msg_serialized(state.serialized);
    aproto->set_msg_single_step(state.single_step);
    aproto->set_msg_last_inst_priv(state.last_inst_priv);
    aproto->set_msg_last_inst_xlen(state.last_inst_xlen);
    aproto->set_msg_last_inst_flen(state.last_inst_flen);
  }

  template <class T>
  void set_basic_csr_from_proto(T& csr, const BasicCSR& proto) {
    csr.val = proto.msg_val();
  }

  void set_medeleg_csr_from_proto(medeleg_csr_t& csr, const MedelegCSR& proto) {
    set_basic_csr_from_proto<basic_csr_t>(csr, proto.msg_basic_csr());
    csr.hypervisor_exceptions = proto.msg_hypervisor_exceptions();
  }

  void set_misa_csr_from_proto(misa_csr_t& csr, const MisaCSR& proto) {
    set_basic_csr_from_proto<basic_csr_t>(csr, proto.msg_basic_csr());
    csr.max_isa    = proto.msg_max_isa();
    csr.write_mask = proto.msg_write_mask();
  }

  void set_basestatus_csr_from_proto(base_status_csr_t& csr, const BaseStatusCSR& proto) {
    csr.has_page = proto.msg_has_page();
    csr.sstatus_write_mask = proto.msg_sstatus_write_mask();
    csr.sstatus_read_mask  = proto.msg_sstatus_read_mask();
  }

  void set_mstatus_csr_from_proto(mstatus_csr_t& csr, const MstatusCSR& proto) {
    set_basestatus_csr_from_proto(csr, proto.msg_base_status_csr());
    csr.val = proto.msg_val();
  }

  void set_sstatus_proxy_csr_from_proto(sstatus_proxy_csr_t& csr, const SstatusProxyCSR& proto) {
    set_mstatus_csr_from_proto(*(csr.mstatus), proto.msg_mstatus_csr());
    set_basestatus_csr_from_proto(csr, proto.msg_base_status_csr());
  }

  void set_vsstatus_csr_from_proto(vsstatus_csr_t& csr, const VsstatusCSR& proto) {
    set_basestatus_csr_from_proto(csr, proto.msg_base_status_csr());
    csr.val = proto.msg_val();
  }

  void set_sstatus_csr_from_proto(sstatus_csr_t& csr, const SstatusCSR& proto) {
    set_sstatus_proxy_csr_from_proto(*(csr.orig_sstatus), proto.msg_orig_sstatus());
    set_vsstatus_csr_from_proto     (*(csr.virt_sstatus), proto.msg_virt_sstatus());
  }

  void set_mcause_csr_from_proto(cause_csr_t& csr, const BasicCSR& proto) {
    set_basic_csr_from_proto<basic_csr_t>(csr, proto);
  }

  void set_masked_csr_from_proto(masked_csr_t& csr, const MaskedCSR& proto) {
    set_basic_csr_from_proto<basic_csr_t>(csr, proto.msg_basic_csr());
    csr.mask = proto.msg_mask();
  }

  void set_smcntrpmf_csr_from_proto(smcntrpmf_csr_t& csr, const SmcntrpmfCSR& proto) {
    set_masked_csr_from_proto(csr, proto.msg_masked_csr());
    if (proto.has_msg_prev_val()) {
      auto opt = proto.msg_prev_val();
      csr.prev_val = opt.msg_val();
    }
  }

  void set_widecntr_csr_from_proto(wide_counter_csr_t& csr, const WideCntrCSR& proto) {
    csr.val = proto.msg_val();
    set_smcntrpmf_csr_from_proto(*(csr.config_csr), proto.msg_config_csr());
  }

  template <class T>
  void set_virt_basic_csr_from_proto(virtualized_csr_t& csr, T& vcsr, const VirtBasicCSR& proto) {
    set_basic_csr_from_proto<T>(*std::dynamic_pointer_cast<T>(csr.orig_csr), proto.msg_nonvirt_csr());
    set_basic_csr_from_proto<T>(*std::dynamic_pointer_cast<T>(csr.virt_csr), proto.msg_virt_csr());
    set_basic_csr_from_proto<T>(vcsr, proto.msg_virt_csr());
  }

  void set_hideleg_csr_from_proto(hideleg_csr_t& csr, const HidelegCSR& proto) {
    set_masked_csr_from_proto(csr, proto.msg_hideleg_csr());

    auto mideleg = std::dynamic_pointer_cast<basic_csr_t>(csr.mideleg);
    set_basic_csr_from_proto(*mideleg, proto.msg_mideleg_csr());
  }

  void set_dcsr_csr_from_proto(dcsr_csr_t& csr, const DCSR& proto) {
    csr.prv      = proto.msg_prv();
    csr.step     = proto.msg_step();
    csr.ebreakm  = proto.msg_ebreakm();
    csr.ebreaks  = proto.msg_ebreaks();
    csr.ebreaku  = proto.msg_ebreaku();
    csr.ebreakvs = proto.msg_ebreakvs();
    csr.ebreakvu = proto.msg_ebreakvu();
    csr.halt     = proto.msg_halt();
    csr.v        = proto.msg_v();
    csr.cause    = proto.msg_cause();
  }

  void set_mcontext_csr_from_proto(proxy_csr_t& csr, const McontextCSR& proto) {
    auto delegate = std::dynamic_pointer_cast<masked_csr_t>(csr.delegate);
    set_masked_csr_from_proto(*delegate, proto.msg_delegate());
  }

  void set_pmpaddr_csr_from_proto(pmpaddr_csr_t& csr, const PmpCSR& proto) {
    csr.val = proto.msg_basic_csr().msg_val();
    csr.cfg = proto.msg_cfg();
    csr.pmpidx = proto.msg_pmpidx();
  }

  void set_henvcfg_csr_from_proto(henvcfg_csr_t& csr, const HenvcfgCSR& proto) {
    set_masked_csr_from_proto(csr, proto.msg_henvcfg());
    auto menv = std::dynamic_pointer_cast<masked_csr_t>(csr.menvcfg);
    set_masked_csr_from_proto(*menv, proto.msg_menvcfg());
  }

  void set_hstateen_csr_from_proto(hstateen_csr_t& csr, const HstateenCSR& proto) {
    set_masked_csr_from_proto(csr, proto.msg_masked_csr());
    csr.index = proto.msg_index();
  }

  void set_time_counter_csr_from_proto(time_counter_csr_t& csr, const BasicCSR& proto) {
    csr.shadow_val = proto.msg_val();
  }

  void set_stimecmp_csr_from_proto(stimecmp_csr_t& csr, const StimecmpCSR& proto) {
    set_basic_csr_from_proto<basic_csr_t>(csr, proto.msg_basic_csr());
    csr.intr_mask = proto.msg_intr_mask();
  }

  void deserialize_proto(ArchState* aproto) {
    std::cout << "deserialize" << std::endl;
    assert(xlen == 64);

    auto csrmap = state.csrmap;

    state.pc = aproto->msg_pc();

    for (int i = 0, cnt = aproto->msg_xpr_size(); i < cnt; i++) {
      state.XPR.write(i, aproto->msg_xpr(i));
    }

    for (int i = 0, cnt = aproto->msg_fpr_size(); i < cnt; i++) {
      auto fpr_msg = aproto->msg_fpr(i);

      float128_t fp;
      fp.v[0] = fpr_msg.msg_0();
      fp.v[1] = fpr_msg.msg_1();
      state.FPR.write(i, fp);
    }

    state.prv         = aproto->msg_prv();
    state.prev_prv    = aproto->msg_prev_prv();
    state.prv_changed = aproto->msg_prv_changed();
    state.v_changed   = aproto->msg_v_changed();
    state.v           = aproto->msg_v();
    state.prev_v      = aproto->msg_prev_v();

    std::cout << " pc: " << state.pc
              << " prv: " << state.prv
              << " prev_prv: " << state.prev_prv
              << " prv_changed: " << state.prv_changed 
              << " v_changed: " << state.v_changed 
              << " v: " << state.v 
              << " prev_v: " << state.prev_v << std::endl;

    if (aproto->has_msg_misa()) {
      set_misa_csr_from_proto(*(state.misa), aproto->msg_misa());
    }

    if (aproto->has_msg_mstatus()) {
      set_mstatus_csr_from_proto(*(state.mstatus), aproto->msg_mstatus());
    }

    if (aproto->has_msg_mepc()) {
      auto mepc = std::dynamic_pointer_cast<epc_csr_t>(state.mepc);
      set_basic_csr_from_proto<epc_csr_t>(*mepc, aproto->msg_mepc());
    }

    if (aproto->has_msg_mtval()) {
      auto mtval = std::dynamic_pointer_cast<basic_csr_t>(state.mtval);
      set_basic_csr_from_proto<basic_csr_t>(*mtval, aproto->msg_mtval());
    }

    if (aproto->has_msg_mscratch()) {
      auto it = csrmap.find(CSR_MSCRATCH);
      assert(it != csrmap.end());
      auto csr = std::dynamic_pointer_cast<basic_csr_t>(it->second);
      set_basic_csr_from_proto<basic_csr_t>(*csr, aproto->msg_mscratch());
    }

    if (aproto->has_msg_mtvec()) {
      auto mtvec = std::dynamic_pointer_cast<tvec_csr_t>(state.mtvec);
      set_basic_csr_from_proto<tvec_csr_t>(*mtvec, aproto->msg_mtvec());
    }

    if (aproto->has_msg_mcause()) {
      auto mcause = std::dynamic_pointer_cast<cause_csr_t>(state.mcause);
      set_basic_csr_from_proto<basic_csr_t>(*mcause, aproto->msg_mcause());
    }

    if (aproto->has_msg_minstret()) {
      auto minstret = state.minstret;
      set_widecntr_csr_from_proto(*minstret, aproto->msg_minstret());
    }

    if (aproto->has_msg_mcycle()) {
      auto mcycle = state.mcycle;
      set_widecntr_csr_from_proto(*mcycle, aproto->msg_mcycle());
    }

    if (aproto->has_msg_time()) {
      set_time_counter_csr_from_proto(*(state.time), aproto->msg_time());
    }


    if (int cnt = aproto->msg_mevent_size() > 0) {
      assert(cnt <= N_HPMCOUNTERS);
      for (int i = 0; i < cnt; i++) {
        auto mevent = std::dynamic_pointer_cast<basic_csr_t>(state.mevent[i]);
        set_basic_csr_from_proto<basic_csr_t>(*mevent, aproto->msg_mevent(i));
      }
    }

    if (aproto->has_msg_mie()) {
      auto mie = state.mie;
      set_basic_csr_from_proto<mip_or_mie_csr_t>(*mie, aproto->msg_mie());
    }

    if (aproto->has_msg_mip()) {
      auto mip = state.mip;
      set_basic_csr_from_proto<mip_or_mie_csr_t>(*mip, aproto->msg_mip());
    }

    if (aproto->has_msg_medeleg()) {
      auto csr  = std::dynamic_pointer_cast<medeleg_csr_t>(state.medeleg);
      set_medeleg_csr_from_proto(*csr, aproto->msg_medeleg());
    }

    if (aproto->has_msg_mcounteren()) {
      auto mcounteren = std::dynamic_pointer_cast<masked_csr_t>(state.mcounteren);
      set_masked_csr_from_proto(*mcounteren, aproto->msg_mcounteren());
    }

    if (aproto->has_msg_scounteren()) {
      auto sen = std::dynamic_pointer_cast<masked_csr_t>(state.scounteren);
      set_masked_csr_from_proto(*sen, aproto->msg_scounteren());
    }

    if (aproto->has_msg_sepc()) {
      auto vsepc = std::dynamic_pointer_cast<epc_csr_t>(state.vsepc);
      auto sepc  = std::dynamic_pointer_cast<virtualized_csr_t>(state.sepc);
      set_virt_basic_csr_from_proto<epc_csr_t>(*sepc, *vsepc, aproto->msg_sepc());
    }

    if (aproto->has_msg_stval()) {
      auto vstval = std::dynamic_pointer_cast<basic_csr_t>(state.vstval);
      auto stval  = std::dynamic_pointer_cast<virtualized_csr_t>(state.stval);
      set_virt_basic_csr_from_proto<basic_csr_t>(*stval, *vstval, aproto->msg_stval());
    }

    if (aproto->has_msg_sscratch()) {
      auto it = csrmap.find(CSR_SSCRATCH);
      assert(it != csrmap.end());
      auto ss = std::dynamic_pointer_cast<virtualized_csr_t>(it->second);

      it = csrmap.find(CSR_VSSCRATCH);
      assert(it != csrmap.end());
      auto vs = std::dynamic_pointer_cast<basic_csr_t>(it->second);

      set_virt_basic_csr_from_proto<basic_csr_t>(*ss, *vs, aproto->msg_sscratch());
    }

    if (aproto->has_msg_stvec()) {
      auto vstvec = std::dynamic_pointer_cast<tvec_csr_t>(state.vstvec);
      auto stvec  = std::dynamic_pointer_cast<virtualized_csr_t>(state.stvec);
      set_virt_basic_csr_from_proto<tvec_csr_t>(*stvec, *vstvec, aproto->msg_stvec());
    }

    if (aproto->has_msg_satp()) {
      auto vsatp = std::dynamic_pointer_cast<basic_csr_t>(state.vsatp);
      auto satp  = state.satp;
      set_virt_basic_csr_from_proto<basic_csr_t>(*satp, *vsatp, aproto->msg_satp());
    }

    if (aproto->has_msg_scause()) {
      auto vscause = std::dynamic_pointer_cast<basic_csr_t>(state.vscause);
      auto scause = std::dynamic_pointer_cast<virtualized_csr_t>(state.scause);
      set_virt_basic_csr_from_proto<basic_csr_t>(*scause, *vscause, aproto->msg_scause());
    }

    if (aproto->has_msg_mtval2()) {
      auto mtval2 = std::dynamic_pointer_cast<basic_csr_t>(state.mtval2);
      set_basic_csr_from_proto<basic_csr_t>(*mtval2, aproto->msg_mtval2());
    }

    if (aproto->has_msg_mtinst()) {
      auto mtinst = std::dynamic_pointer_cast<basic_csr_t>(state.mtinst);
      set_basic_csr_from_proto<basic_csr_t>(*mtinst, aproto->msg_mtinst());
    }

    if (aproto->has_msg_hstatus()) {
      auto hstatus = std::dynamic_pointer_cast<masked_csr_t>(state.hstatus);
      set_masked_csr_from_proto(*hstatus, aproto->msg_hstatus());
    }

    if (aproto->has_msg_hideleg()) {
      auto hideleg = std::dynamic_pointer_cast<hideleg_csr_t>(state.hideleg);
      auto mideleg = std::dynamic_pointer_cast<mideleg_csr_t>(hideleg->mideleg);
      set_hideleg_csr_from_proto(*hideleg, aproto->msg_hideleg());
    }

    if (aproto->has_msg_hedeleg()) {
      auto hedeleg = std::dynamic_pointer_cast<masked_csr_t>(state.hedeleg);
      set_masked_csr_from_proto(*hedeleg, aproto->msg_hedeleg());
    }

    if (aproto->has_msg_hcounteren()) {
      auto hcounteren = std::dynamic_pointer_cast<masked_csr_t>(state.hcounteren);
      set_masked_csr_from_proto(*hcounteren, aproto->msg_hcounteren());
    }

    if (aproto->has_msg_htimedelta()) {
      auto ht = std::dynamic_pointer_cast<basic_csr_t>(state.htimedelta);
      set_basic_csr_from_proto<basic_csr_t>(*ht, aproto->msg_htimedelta());
    }

    if (aproto->has_msg_htval()) {
      auto htval = std::dynamic_pointer_cast<basic_csr_t>(state.htval);
      set_basic_csr_from_proto<basic_csr_t>(*htval, aproto->msg_htval());
    }

    if (aproto->has_msg_htinst()) {
      auto htinst = std::dynamic_pointer_cast<basic_csr_t>(state.htinst);
      set_basic_csr_from_proto<basic_csr_t>(*htinst, aproto->msg_htinst());
    }

    if (aproto->has_msg_hgatp()) {
      auto hgatp = std::dynamic_pointer_cast<basic_csr_t>(state.hgatp);
      set_basic_csr_from_proto<basic_csr_t>(*hgatp, aproto->msg_hgatp());
    }

    if (aproto->has_msg_sstatus()) {
      set_sstatus_csr_from_proto(*(state.sstatus), aproto->msg_sstatus());
    }

    if (aproto->has_msg_dpc()) {
      auto dpc = std::dynamic_pointer_cast<epc_csr_t>(state.dpc);
      set_basic_csr_from_proto<epc_csr_t>(*dpc, aproto->msg_dpc());
    }

    if (aproto->has_msg_dscratch0()) {
      auto it = csrmap.find(CSR_DSCRATCH0);
      assert(it != csrmap.end());
      auto csr = std::dynamic_pointer_cast<basic_csr_t>(it->second);
      set_basic_csr_from_proto(*csr, aproto->msg_dscratch0());
    }

    if (aproto->has_msg_dscratch1()) {
      auto it = csrmap.find(CSR_DSCRATCH1);
      assert(it != csrmap.end());
      auto csr = std::dynamic_pointer_cast<basic_csr_t>(it->second);
      set_basic_csr_from_proto(*csr, aproto->msg_dscratch1());
    }

    if (aproto->has_msg_dcsr()) {
      auto dcsr = std::dynamic_pointer_cast<dcsr_csr_t>(state.dcsr);
      set_dcsr_csr_from_proto(*dcsr, aproto->msg_dcsr());
    }

    if (aproto->has_msg_tselect()) {
      auto tsel = std::dynamic_pointer_cast<tselect_csr_t>(state.tselect);
      set_basic_csr_from_proto(*tsel, aproto->msg_tselect());
    }

    if (aproto->has_msg_scontext()) {
      auto sc = std::dynamic_pointer_cast<masked_csr_t>(state.scontext);
      set_masked_csr_from_proto(*sc, aproto->msg_scontext());
    }

    if (aproto->has_msg_hcontext()) {
      auto it = csrmap.find(CSR_HCONTEXT);
      assert(it != csrmap.end());
      auto csr = std::dynamic_pointer_cast<masked_csr_t>(it->second);
      set_masked_csr_from_proto(*csr, aproto->msg_hcontext());
    }

    if (aproto->has_msg_mseccfg()) {
      auto mseccfg = std::dynamic_pointer_cast<basic_csr_t>(state.mseccfg);
      set_basic_csr_from_proto<basic_csr_t>(*mseccfg, aproto->msg_mseccfg());
    }

    if (int cnt = aproto->msg_pmpaddr_size() > 0) {
      assert(cnt <= state.max_pmp);
      for (int i = 0; i < cnt; i++) {
        auto pmpaddr = state.pmpaddr[i];
        set_pmpaddr_csr_from_proto(*pmpaddr, aproto->msg_pmpaddr(i));
      }
    }

    if (aproto->has_msg_fflags()) {
      set_masked_csr_from_proto(*(state.fflags), aproto->msg_fflags());
    }

    if (aproto->has_msg_frm()) {
      set_masked_csr_from_proto(*(state.frm), aproto->msg_frm());
    }

    if (aproto->has_msg_senvcfg()) {
      auto senv = std::dynamic_pointer_cast<masked_csr_t>(state.senvcfg);
      set_masked_csr_from_proto(*senv, aproto->msg_senvcfg());
    }

    if (aproto->has_msg_henvcfg()) {
      auto henv = std::dynamic_pointer_cast<henvcfg_csr_t>(state.henvcfg);
      set_henvcfg_csr_from_proto(*henv, aproto->msg_henvcfg());
    }

    if (int cnt = aproto->msg_mstateen_size() > 0) {
      assert(cnt <= 4);
      for (int i = 0; i < cnt; i++) {
        auto mstateen = std::dynamic_pointer_cast<masked_csr_t>(state.mstateen[i]);
        set_masked_csr_from_proto(*mstateen, aproto->msg_mstateen(i));
      }
    }

    if (int cnt = aproto->msg_sstateen_size() > 0) {
      assert(cnt <= 4);
      for (int i = 0; i < cnt; i++) {
        auto sstateen = std::dynamic_pointer_cast<hstateen_csr_t>(state.sstateen[i]);
        set_hstateen_csr_from_proto(*sstateen, aproto->msg_sstateen(i));
      }
    }

    if (int cnt = aproto->msg_hstateen_size() > 0) {
      assert(cnt <= 4);
      for (int i = 0; i < cnt; i++) {
        auto hstateen = std::dynamic_pointer_cast<hstateen_csr_t>(state.hstateen[i]);
        set_hstateen_csr_from_proto(*hstateen, aproto->msg_hstateen(i));
      }
    }

    if (aproto->has_msg_mnscratch()) {
      auto it = csrmap.find(CSR_MNSCRATCH);
      assert(it != csrmap.end());
      auto csr = std::dynamic_pointer_cast<basic_csr_t>(it->second);
      set_basic_csr_from_proto(*csr, aproto->msg_mnscratch());
    }

    if (aproto->has_msg_mnepc()) {
      auto mnepc = std::dynamic_pointer_cast<epc_csr_t>(state.mnepc);
      set_basic_csr_from_proto<epc_csr_t>(*mnepc, aproto->msg_mnepc());
    }

    if (aproto->has_msg_mnstatus()) {
      auto mnstatus = std::dynamic_pointer_cast<basic_csr_t>(state.mnstatus);
      set_basic_csr_from_proto<basic_csr_t>(*mnstatus, aproto->msg_mnstatus());
    }

    if (aproto->has_msg_stimecmp()) {
      auto st = std::dynamic_pointer_cast<stimecmp_csr_t>(state.stimecmp);
      set_stimecmp_csr_from_proto(*st, aproto->msg_stimecmp());
    }

    if (aproto->has_msg_vstimecmp()) {
      auto st = std::dynamic_pointer_cast<stimecmp_csr_t>(state.vstimecmp);
      set_stimecmp_csr_from_proto(*st, aproto->msg_vstimecmp());
    }

    if (aproto->has_msg_jvt()) {
      auto jvt = std::dynamic_pointer_cast<basic_csr_t>(state.jvt);
      set_basic_csr_from_proto<basic_csr_t>(*jvt, aproto->msg_jvt());
    }

    if (aproto->has_msg_miselect()) {
      auto it = csrmap.find(CSR_MISELECT);
      assert(it != csrmap.end());
      auto csr = std::dynamic_pointer_cast<basic_csr_t>(it->second);
      set_basic_csr_from_proto(*csr, aproto->msg_miselect());
    }

    if (aproto->has_msg_siselect()) {
      auto it = csrmap.find(CSR_SISELECT);
      assert(it != csrmap.end());
      auto ss = std::dynamic_pointer_cast<virtualized_csr_t>(it->second);

      it = csrmap.find(CSR_VSISELECT);
      assert(it != csrmap.end());
      auto vs = std::dynamic_pointer_cast<basic_csr_t>(it->second);

      set_virt_basic_csr_from_proto<basic_csr_t>(*ss, *vs, aproto->msg_siselect());
    }

    state.debug_mode = aproto->msg_debug_mode();

    state.serialized     = aproto->msg_serialized();

    auto ss = aproto->msg_single_step();
    if (ss == 0) {
      state.single_step = state_t::STEP_NONE;
    } else if (ss == 1) {
      state.single_step = state_t::STEP_STEPPING;
    } else if (ss == 2) {
      state.single_step = state_t::STEP_STEPPED;
    } else {
      assert(false);
    }
    state.last_inst_priv = aproto->msg_last_inst_priv();
    state.last_inst_xlen = aproto->msg_last_inst_xlen();
    state.last_inst_flen = aproto->msg_last_inst_flen();

    google::protobuf::ShutdownProtobufLibrary();
  }
};

#endif

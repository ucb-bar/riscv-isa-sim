// See LICENSE for license details.
#ifndef _RISCV_PROCESSOR_H
#define _RISCV_PROCESSOR_H

#include "decode.h"
#include "trap.h"
#include "abstract_device.h"
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
    processor_t* p = const_cast<processor_t*>(this);
    if (ext >= 'A' && ext <= 'Z')
      return state.misa->extension_enabled(ext, p);
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
    processor_t* p = const_cast<processor_t*>(this);
    if (ext >= 'A' && ext <= 'Z') {
      return state.misa->extension_enabled_const(ext, p);
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

  void take_pending_interrupt() { take_interrupt(state.mip->read(this) & state.mie->read(this)); }
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
  reg_t pmp_tor_mask() const { return -(reg_t(1) << (lg_pmp_granularity - PMP_SHIFT)); }

  vectorUnit_t VU;
  triggers::module_t TM;

public:
  ArchState aproto;
  google::protobuf::Arena* arena;

  CSR* gen_csr_proto(reg_t addr) {
    CSR* csr = google::protobuf::Arena::Create<CSR>(arena);;
    csr->set_msg_addr(addr);
    csr->set_msg_csr_priv(get_field(addr, 0x300));
    csr->set_msg_csr_read_only(get_field(addr, 0xC00) == 3);
    return csr;
  }

  BasicCSR* gen_basic_csr_proto(reg_t addr, reg_t init) {
    CSR* c = gen_csr_proto(addr);
    BasicCSR* b = google::protobuf::Arena::Create<BasicCSR>(arena);
    b->set_allocated_msg_csr(c);
    b->set_msg_val(init);
    return b;
  }

  MisaCSR* gen_misa_csr_proto(misa_csr_t_p ptr) {
    BasicCSR* b = gen_basic_csr_proto(ptr->address, ptr->val);
    MisaCSR* misa = google::protobuf::Arena::Create<MisaCSR>(arena);
    misa->set_allocated_msg_basic_csr(b);
    misa->set_msg_max_isa(ptr->max_isa);
    misa->set_msg_write_mask(ptr->write_mask);
    return misa;
  }

  SatpCSR* gen_satp_csr_proto(virtualized_csr_t_p csr) {
    auto vsatp = std::dynamic_pointer_cast<basic_csr_t>(csr->virt_csr);
    auto osatp = std::dynamic_pointer_cast<basic_csr_t>(csr->orig_csr);
    BasicCSR* virt = gen_basic_csr_proto(vsatp->address, vsatp->val);
    BasicCSR* orig = gen_basic_csr_proto(osatp->address, osatp->val);

    SatpCSR* satp = google::protobuf::Arena::Create<SatpCSR>(arena);
    satp->set_allocated_msg_nonvirt_satp_csr(orig);
    satp->set_allocated_msg_virt_satp_csr(virt);
    return satp;
  }

  BaseStatusCSR* gen_base_status_csr_proto(reg_t addr, bool has_page, reg_t wm, reg_t rm) {
    CSR* c = gen_csr_proto(addr);

    BaseStatusCSR* b = google::protobuf::Arena::Create<BaseStatusCSR>(arena);
    b->set_allocated_msg_csr(c);
    b->set_msg_has_page(has_page);
    b->set_msg_sstatus_write_mask(wm);
    b->set_msg_sstatus_read_mask(rm);
    return b;
  }

  MstatusCSR* gen_mstatus_csr_proto(mstatus_csr_t_p csr) {
    BaseStatusCSR* base = gen_base_status_csr_proto(csr->address,
                                                    csr->has_page,
                                                    csr->sstatus_write_mask,
                                                    csr->sstatus_read_mask);
    MstatusCSR* m = google::protobuf::Arena::Create<MstatusCSR>(arena);
    m->set_allocated_msg_base_status_csr(base);
    m->set_msg_val(csr->val);
    return m;
  }

  MaskedCSR* gen_masked_csr_proto(reg_t addr, reg_t val, reg_t mask) {
    BasicCSR* b = gen_basic_csr_proto(addr, val);

    MaskedCSR* m = google::protobuf::Arena::Create<MaskedCSR>(arena);
    m->set_allocated_msg_basic_csr(b);
    m->set_msg_mask(mask);
    return m;
  }

  SmcntrpmfCSR* gen_smcntrpmf_csr_proto(smcntrpmf_csr_t_p csr) {
    MaskedCSR* m = gen_masked_csr_proto(csr->address, csr->val, csr->mask);

    SmcntrpmfCSR* s = google::protobuf::Arena::Create<SmcntrpmfCSR>(arena);
    s->set_allocated_msg_masked_csr(m);
    if (csr->prev_val.has_value()) {
      OptionalUInt64* o = google::protobuf::Arena::Create<OptionalUInt64>(arena);
      o->set_msg_val(csr->prev_val.value());
      s->set_allocated_msg_prev_val(o);
    }
    return s;
  }

  WideCntrCSR* gen_wide_cntr_csr_proto(wide_counter_csr_t_p csr) {
    CSR* c = gen_csr_proto(csr->address);
    SmcntrpmfCSR* s = gen_smcntrpmf_csr_proto(csr->config_csr);

    WideCntrCSR* w = google::protobuf::Arena::Create<WideCntrCSR>(arena);
    w->set_allocated_msg_csr(c);
    w->set_msg_val(csr->val);
    w->set_allocated_msg_config_csr(s);
    return w;
  }

  MedelegCSR* get_medeleg_csr_proto(csr_t_p csr) {
    auto medeleg = std::dynamic_pointer_cast<medeleg_csr_t>(csr);
    BasicCSR* b = gen_basic_csr_proto(medeleg->address, medeleg->val);

    MedelegCSR* m = google::protobuf::Arena::Create<MedelegCSR>(arena);
    m->set_allocated_msg_basic_csr(b);
    m->set_msg_hypervisor_exceptions(medeleg->hypervisor_exceptions);
    return m;
  }

  void serialize_proto(std::string& os) {
    std::cout << "serialize" << std::endl;

    aproto.set_msg_pc(state.pc);
    aproto.set_msg_prv(state.prv);
    aproto.set_msg_prev_prv(state.prev_prv);
    aproto.set_msg_prv_changed(state.prv_changed);
    aproto.set_msg_v_changed(state.v_changed);
    aproto.set_msg_v(state.v);
    aproto.set_msg_prev_v(state.prev_v);

    std::cout << " pc: " << state.pc
              << " prv: " << state.prv
              << " prev_prv: " << state.prev_prv
              << " prv_changed: " << state.prv_changed 
              << " v_changed: " << state.v_changed 
              << " v: " << state.v 
              << " prev_v: " << state.prev_v << std::endl;

    if (state.misa) {
      MisaCSR* misa_proto = gen_misa_csr_proto(state.misa);
      aproto.set_allocated_msg_misa(misa_proto);
      state.misa->print();
    } else {
      std::cout << "state.misa empty: " << state.misa << "/" << std::endl;
    }

    if (state.mstatus) {
      MstatusCSR* mstatus_proto = gen_mstatus_csr_proto(state.mstatus);
      aproto.set_allocated_msg_mstatus(mstatus_proto);
      state.mstatus->print();
    } else {
      std::cout << "state.mstatus empty: " << state.mstatus << "/" << std::endl;
    }

    // FIXME
    if (state.mstatush) {
      CSR* mstatush_proto = gen_csr_proto(state.mstatush->address);
      aproto.set_allocated_msg_mstatush(mstatush_proto);
      state.mstatush->print();
    } else {
      std::cout << "state.mstatush empty: " << state.mstatush << "/" << std::endl;
    }

    if (state.mepc) {
      auto mepc = std::dynamic_pointer_cast<epc_csr_t>(state.mepc);
      BasicCSR* mepc_proto = gen_basic_csr_proto(mepc->address, mepc->val);
      aproto.set_allocated_msg_mepc(mepc_proto);
      mepc->print();
    } else {
      std::cout << "state.mepc empty: " << state.mepc << "/" << std::endl;
    }

    if (state.mtval) {
      auto mtval = std::dynamic_pointer_cast<basic_csr_t>(state.mtval);
      BasicCSR* mtval_proto = gen_basic_csr_proto(mtval->address, mtval->val);
      aproto.set_allocated_msg_mtval(mtval_proto);
      mtval->print();
    } else {
      std::cout << "state.mtval empty: " << state.mtval << "/" << std::endl;
    }

    if (state.mtvec) {
      auto mtvec = std::dynamic_pointer_cast<tvec_csr_t>(state.mtvec);
      BasicCSR* mtvec_proto = gen_basic_csr_proto(mtvec->address, mtvec->val);
      aproto.set_allocated_msg_mtvec(mtvec_proto);
      mtvec->print();
    } else {
      std::cout << "state.mtvec empty: " << state.mtvec << "/" << std::endl;
    }

    if (state.mcause) {
      auto mcause = std::dynamic_pointer_cast<cause_csr_t>(state.mcause);
      BasicCSR* mcause_proto = gen_basic_csr_proto(mcause->address, mcause->val);
      aproto.set_allocated_msg_mcause(mcause_proto);
      mcause->print();
    } else {
      std::cout << "state.mcause empty: " << state.mcause << "/" << std::endl;
    }

    if (state.minstret) {
      WideCntrCSR* minstret_proto = gen_wide_cntr_csr_proto(state.minstret);
      aproto.set_allocated_msg_minstret(minstret_proto);
      state.minstret->print();
    } else {
      std::cout << "state.minstret empty: " << state.minstret << "/" << std::endl;
    }

    if (state.mcycle) {
      WideCntrCSR* mcycle_proto = gen_wide_cntr_csr_proto(state.mcycle);
      aproto.set_allocated_msg_mcycle(mcycle_proto);
      state.mcycle->print();
    } else {
      std::cout << "state.mcycle empty: " << state.mcycle << "/" << std::endl;
    }

    if (state.mie) {
      BasicCSR* mie_proto = gen_basic_csr_proto(state.mie->address, state.mie->val);
      aproto.set_allocated_msg_mie(mie_proto);
      state.mie->print();
    } else {
      std::cout << "state.mie empty: " << state.mie << "/" << std::endl;
    }

    if (state.mip) {
      BasicCSR* mip_proto = gen_basic_csr_proto(state.mip->address, state.mip->val);
      aproto.set_allocated_msg_mip(mip_proto);
      state.mip->print();
    } else {
      std::cout << "state.mip empty: " << state.mip << "/" << std::endl;
    }

    if (state.medeleg) {
      MedelegCSR* medeleg_proto = get_medeleg_csr_proto(state.medeleg);
      aproto.set_allocated_msg_medeleg(medeleg_proto);

      auto medeleg = std::dynamic_pointer_cast<medeleg_csr_t>(state.medeleg);
      medeleg->print();
    } else {
      std::cout << "state.medeleg empty: " << state.medeleg << "/" << std::endl;
    }

    if (state.mideleg) {
      auto mideleg = std::dynamic_pointer_cast<mideleg_csr_t>(state.mideleg);
      BasicCSR* mideleg_proto = gen_basic_csr_proto(mideleg->address, mideleg->val);
      aproto.set_allocated_msg_mideleg(mideleg_proto);
      mideleg->print();
    } else {
      std::cout << "state.mideleg empty: " << state.mideleg << "/" << std::endl;
    }

    if (state.satp) {
      SatpCSR* satp = gen_satp_csr_proto(state.satp);
      aproto.set_allocated_msg_satp(satp);
      state.satp->print();
      state.vsatp->print();
    } else {
      std::cout << "state.satpempty: " << state.satp << "/" << std::endl;
    }

    aproto.SerializeToString(&os);

    aproto.release_msg_misa();
    aproto.release_msg_mstatus();
    aproto.release_msg_mepc();
    aproto.release_msg_mtval();
    aproto.release_msg_mtvec();
    aproto.release_msg_mcause();
    aproto.release_msg_minstret();
    aproto.release_msg_mcycle();
    aproto.release_msg_mie();
    aproto.release_msg_mip();
    aproto.release_msg_medeleg();
    aproto.release_msg_mideleg();
    aproto.release_msg_satp();
  }

  void set_csr_from_proto(csr_t& csr, const CSR& proto) {
    csr.address       = proto.msg_addr();
    csr.csr_priv      = proto.msg_csr_priv();
    csr.csr_read_only = proto.msg_csr_read_only();
  }

  void set_basic_csr_from_proto(basic_csr_t& csr, const BasicCSR& proto) {
    set_csr_from_proto(csr, proto.msg_csr());
    csr.val           = proto.msg_val();
  }

  void set_misa_csr_from_proto(misa_csr_t& csr, const MisaCSR& proto) {
    set_basic_csr_from_proto(csr, proto.msg_basic_csr());
    csr.max_isa    = proto.msg_max_isa();
    csr.write_mask = proto.msg_write_mask();
  }

  void set_basestatus_csr_from_proto(base_status_csr_t& csr, const BaseStatusCSR& proto) {
    set_csr_from_proto(csr, proto.msg_csr());
    csr.has_page = proto.msg_has_page();
    csr.sstatus_write_mask = proto.msg_sstatus_write_mask();
    csr.sstatus_read_mask  = proto.msg_sstatus_read_mask();
  }

  void set_mstatus_csr_from_proto(mstatus_csr_t& csr, const MstatusCSR& proto) {
    set_basestatus_csr_from_proto(csr, proto.msg_base_status_csr());
    csr.val = proto.msg_val();
  }

  void set_mepc_csr_from_proto(epc_csr_t& csr, const BasicCSR& proto) {
    set_csr_from_proto(csr, proto.msg_csr());
    csr.val = proto.msg_val();
  }

  void set_mtval_csr_from_proto(basic_csr_t& csr, const BasicCSR& proto) {
    set_basic_csr_from_proto(csr, proto);
  }

  void set_mtvec_csr_from_proto(tvec_csr_t& csr, const BasicCSR& proto) {
    set_csr_from_proto(csr, proto.msg_csr());
    csr.val = proto.msg_val();
  }

  void set_mcause_csr_from_proto(cause_csr_t& csr, const BasicCSR& proto) {
    set_basic_csr_from_proto(csr, proto);
  }

  void set_masked_csr_from_proto(masked_csr_t& csr, const MaskedCSR& proto) {
    set_basic_csr_from_proto(csr, proto.msg_basic_csr());
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
    set_csr_from_proto(csr, proto.msg_csr());
    csr.val = proto.msg_val();
    set_smcntrpmf_csr_from_proto(*(csr.config_csr), proto.msg_config_csr());
  }

  void set_satp_csr_from_proto(virtualized_csr_t& satp, base_atp_csr_t& vsatp, const SatpCSR& proto) {
    set_basic_csr_from_proto(*std::dynamic_pointer_cast<basic_csr_t>(satp.orig_csr), proto.msg_nonvirt_satp_csr());
    set_basic_csr_from_proto(*std::dynamic_pointer_cast<basic_csr_t>(satp.virt_csr), proto.msg_virt_satp_csr());
    set_basic_csr_from_proto(vsatp, proto.msg_virt_satp_csr());
  }

  void deserialize_proto(std::string& is) {
    std::cout << "deserialize" << std::endl;
    aproto.ParseFromString(is);

    state.pc          = aproto.msg_pc();
    state.prv         = aproto.msg_prv();
    state.prev_prv    = aproto.msg_prev_prv();
    state.prv_changed = aproto.msg_prv_changed();
    state.v_changed   = aproto.msg_v_changed();
    state.v           = aproto.msg_v();
    state.prev_v      = aproto.msg_prev_v();

    std::cout << " pc: " << state.pc
              << " prv: " << state.prv
              << " prev_prv: " << state.prev_prv
              << " prv_changed: " << state.prv_changed 
              << " v_changed: " << state.v_changed 
              << " v: " << state.v 
              << " prev_v: " << state.prev_v << std::endl;

    if (aproto.has_msg_misa()) {
      set_misa_csr_from_proto(*(state.misa), aproto.msg_misa());
      state.misa->print();
    } else {
      std::cout << "state.misa empty: " << state.misa << "/" << std::endl;
    }

    if (aproto.has_msg_mstatus()) {
      set_mstatus_csr_from_proto(*(state.mstatus), aproto.msg_mstatus());
      state.mstatus->print();
    } else {
      std::cout << "state.mstatus empty: " << state.mstatus << "/" << std::endl;
    }

    // FIXME
    if (aproto.has_msg_mstatush()) {
      set_csr_from_proto(*(state.mstatush), aproto.msg_mstatush());
    } else {
      std::cout << "state.mstatush empty: " << state.mstatush << "/" << std::endl;
    }

    if (aproto.has_msg_mepc()) {
      auto mepc = std::dynamic_pointer_cast<epc_csr_t>(state.mepc);
      set_mepc_csr_from_proto(*mepc, aproto.msg_mepc());
      mepc->print();
    } else {
      std::cout << "state.mepc empty: " << state.mepc << "/" << std::endl;
    }

    if (aproto.has_msg_mtval()) {
      auto mtval = std::dynamic_pointer_cast<basic_csr_t>(state.mtval);
      set_mtval_csr_from_proto(*mtval, aproto.msg_mtval());
      mtval->print();
    } else {
      std::cout << "state.mtval empty: " << state.mtval << "/" << std::endl;
    }

    if (aproto.has_msg_mtvec()) {
      auto mtvec = std::dynamic_pointer_cast<tvec_csr_t>(state.mtvec);
      set_mtvec_csr_from_proto(*mtvec, aproto.msg_mtvec());
      mtvec->print();
    } else {
      std::cout << "state.mtvec empty: " << state.mtvec << "/" << std::endl;
    }

    if (aproto.has_msg_mcause()) {
      auto mcause = std::dynamic_pointer_cast<cause_csr_t>(state.mcause);
      set_mcause_csr_from_proto(*mcause, aproto.msg_mcause());
      mcause->print();
    } else {
      std::cout << "state.mcause empty: " << state.mcause << "/" << std::endl;
    }

    if (aproto.has_msg_minstret()) {
      auto minstret = state.minstret;
      set_widecntr_csr_from_proto(*minstret, aproto.msg_minstret());
      minstret->print();
    } else {
      std::cout << "state.minstret empty: " << state.minstret << "/" << std::endl;
    }

    if (aproto.has_msg_satp()) {
      auto vsatp = std::dynamic_pointer_cast<base_atp_csr_t>(state.vsatp);
      auto satp  = state.satp;
      set_satp_csr_from_proto(*satp, *vsatp, aproto.msg_satp());
      satp->print();
      vsatp->print();
    } else {
      std::cout << "state.satp empty: " << state.satp << "/" << std::endl;
    }
  }
};

#endif

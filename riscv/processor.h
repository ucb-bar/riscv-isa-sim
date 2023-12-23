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

public:
  google::protobuf::Arena* arena;

  template <class T>
  inline T* create_protobuf(google::protobuf::Arena* arena) {
    return google::protobuf::Arena::Create<T>(arena);
  }

  // Protobuf serialization
  BasicCSR*        gen_basic_csr_proto(reg_t init);
  MisaCSR*         gen_misa_csr_proto(misa_csr_t_p ptr);
  BaseStatusCSR*   gen_base_status_csr_proto(bool has_page, reg_t wm, reg_t rm);
  MstatusCSR*      gen_mstatus_csr_proto(mstatus_csr_t_p csr);

  SstatusProxyCSR* gen_sstatus_proxy_csr_proto(sstatus_proxy_csr_t_p csr);
  VsstatusCSR*     gen_vsstatus_csr_proto(vsstatus_csr_t_p csr);
  SstatusCSR*      gen_sstatus_csr_proto(sstatus_csr_t_p csr);
  MaskedCSR*       gen_masked_csr_proto(reg_t val, reg_t mask);

  SmcntrpmfCSR*    gen_smcntrpmf_csr_proto(smcntrpmf_csr_t_p csr);
  WideCntrCSR*     gen_wide_cntr_csr_proto(wide_counter_csr_t_p csr);
  MedelegCSR*      gen_medeleg_csr_proto(csr_t_p csr);

  template <class CSR_T>
  VirtBasicCSR*    gen_virt_basic_csr_proto(virtualized_csr_t_p csr);

  HidelegCSR*      gen_hideleg_csr_proto(csr_t_p csr);
  DCSR*            gen_dcsr_csr_proto(dcsr_csr_t_p csr);
  McontextCSR*     gen_mcontext_csr_proto(std::shared_ptr<proxy_csr_t> csr);
  HenvcfgCSR*      gen_henvcfg_csr_proto(std::shared_ptr<henvcfg_csr_t> csr);
  StimecmpCSR*     gen_stimecmp_csr_proto(std::shared_ptr<stimecmp_csr_t> csr);

  void serialize_proto(ArchState* aproto, google::protobuf::Arena* arena);


  // Protobuf deserialization
  template <class T>
  void set_basic_csr_from_proto(T& csr, const BasicCSR& proto);

  void set_medeleg_csr_from_proto(medeleg_csr_t& csr, const MedelegCSR& proto);
  void set_misa_csr_from_proto(misa_csr_t& csr, const MisaCSR& proto);
  void set_basestatus_csr_from_proto(base_status_csr_t& csr, const BaseStatusCSR& proto);
  void set_mstatus_csr_from_proto(mstatus_csr_t& csr, const MstatusCSR& proto);

  void set_sstatus_proxy_csr_from_proto(sstatus_proxy_csr_t& csr, const SstatusProxyCSR& proto);
  void set_vsstatus_csr_from_proto(vsstatus_csr_t& csr, const VsstatusCSR& proto);
  void set_sstatus_csr_from_proto(sstatus_csr_t& csr, const SstatusCSR& proto);
  void set_mcause_csr_from_proto(cause_csr_t& csr, const BasicCSR& proto);

  void set_masked_csr_from_proto(masked_csr_t& csr, const MaskedCSR& proto);
  void set_smcntrpmf_csr_from_proto(smcntrpmf_csr_t& csr, const SmcntrpmfCSR& proto);
  void set_widecntr_csr_from_proto(wide_counter_csr_t& csr, const WideCntrCSR& proto);

  template <class T>
  void set_virt_basic_csr_from_proto(virtualized_csr_t& csr, T& vcsr, const VirtBasicCSR& proto);

  void set_hideleg_csr_from_proto(hideleg_csr_t& csr, const HidelegCSR& proto);
  void set_dcsr_csr_from_proto(dcsr_csr_t& csr, const DCSR& proto);
  void set_mcontext_csr_from_proto(proxy_csr_t& csr, const McontextCSR& proto);
  void set_pmpaddr_csr_from_proto(pmpaddr_csr_t& csr, const PmpCSR& proto);

  void set_henvcfg_csr_from_proto(henvcfg_csr_t& csr, const HenvcfgCSR& proto);
  void set_hstateen_csr_from_proto(hstateen_csr_t& csr, const HstateenCSR& proto);
  void set_time_counter_csr_from_proto(time_counter_csr_t& csr, const BasicCSR& proto);
  void set_stimecmp_csr_from_proto(stimecmp_csr_t& csr, const StimecmpCSR& proto);

  void deserialize_proto(ArchState* aproto);

  // Profiler specific APIs
public:
  reg_t get_asid();
  reg_t get_ppn();

  std::vector<reg_t>& step_pctrace() { return step_PC; }

private:
  std::vector<reg_t> step_PC;
};

#endif

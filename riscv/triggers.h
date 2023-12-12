#ifndef _RISCV_TRIGGERS_H
#define _RISCV_TRIGGERS_H

#include <vector>
#include <optional>

#include "decode.h"

namespace triggers {

typedef enum {
  OPERATION_EXECUTE,
  OPERATION_STORE,
  OPERATION_LOAD,
} operation_t;

typedef enum
{
  ACTION_DEBUG_EXCEPTION = MCONTROL_ACTION_DEBUG_EXCEPTION,
  ACTION_DEBUG_MODE = MCONTROL_ACTION_DEBUG_MODE,
  ACTION_TRACE_START = MCONTROL_ACTION_TRACE_START,
  ACTION_TRACE_STOP = MCONTROL_ACTION_TRACE_STOP,
  ACTION_TRACE_EMIT = MCONTROL_ACTION_TRACE_EMIT,
  ACTION_MAXVAL = MCONTROL_ACTION_TRACE_EMIT
} action_t;

typedef enum {
  TIMING_BEFORE = 0,
  TIMING_AFTER = 1
} timing_t;

typedef enum {
  SSELECT_IGNORE = 0,
  SSELECT_SCONTEXT = 1,
  SSELECT_ASID = 2,
  SSELECT_MAXVAL = 2
} sselect_t;

typedef enum {
  MHSELECT_MODE_IGNORE,
  MHSELECT_MODE_MCONTEXT,
  MHSELECT_MODE_VMID,
} mhselect_mode_t;

struct match_result_t {
  match_result_t(const timing_t t=TIMING_BEFORE, const action_t a=ACTION_DEBUG_EXCEPTION) {
    timing = t;
    action = a;
  }
  timing_t timing;
  action_t action;
};

class matched_t
{
  public:
    matched_t(triggers::operation_t operation, reg_t address, action_t action, bool gva) :
      operation(operation), address(address), action(action), gva(gva) {}

    triggers::operation_t operation;
    reg_t address;
    action_t action;
    bool gva;
};

class trigger_t {
public:
  virtual ~trigger_t() {};

  virtual reg_t tdata1_read(const processor_t * const proc) const noexcept = 0;
  virtual void tdata1_write(processor_t * const proc, const reg_t val, const bool allow_chain) noexcept = 0;
  reg_t tdata2_read(const processor_t * const proc) const noexcept;
  void tdata2_write(processor_t * const proc, const reg_t val) noexcept;
  reg_t tdata3_read(const processor_t * const proc) const noexcept;
  void tdata3_write(processor_t * const proc, const reg_t val) noexcept;

  virtual bool get_dmode() const = 0;
  virtual bool get_chain() const { return false; }
  virtual bool get_execute() const { return false; }
  virtual bool get_store() const { return false; }
  virtual bool get_load() const { return false; }
  virtual action_t get_action() const { return ACTION_DEBUG_EXCEPTION; }
  virtual bool icount_check_needed() const { return false; }
  virtual void stash_read_values() {}

  virtual std::optional<match_result_t> detect_memory_access_match(processor_t UNUSED * const proc,
      operation_t UNUSED operation, reg_t UNUSED address, std::optional<reg_t> UNUSED data) noexcept { return std::nullopt; }
  virtual std::optional<match_result_t> detect_icount_fire(processor_t UNUSED * const proc) { return std::nullopt; }
  virtual void detect_icount_decrement(processor_t UNUSED * const proc) {}
  virtual std::optional<match_result_t> detect_trap_match(processor_t UNUSED * const proc, const trap_t UNUSED & t) noexcept { return std::nullopt; }

protected:
  static action_t legalize_action(reg_t val, reg_t action_mask, reg_t dmode_mask) noexcept;
  bool common_match(processor_t * const proc, bool use_prev_prv = false) const noexcept;
  bool allow_action(processor_t * proc) const;
  reg_t tdata2;

  bool vs = false;
  bool vu = false;
  bool m = false;
  bool s = false;
  bool u = false;

private:
  unsigned legalize_mhselect(bool h_enabled) const noexcept;
  bool mode_match(reg_t prv, bool v) const noexcept;
  bool textra_match(processor_t * const proc) const noexcept;

  struct mhselect_interpretation {
    const unsigned mhselect;
    const mhselect_mode_t mode;
    const std::optional<bool> shift_mhvalue;
    unsigned compare_val(const unsigned mhvalue) const {
      return shift_mhvalue.value() ? (mhvalue << 1 | mhselect >> 2) : mhvalue;
    };
  };

  mhselect_interpretation interpret_mhselect(bool h_enabled) const noexcept {
    static unsigned warlize_if_h[8] = { 0, 1, 2, 0, 4, 5, 6, 4 };  // 3,7 downgrade
    static unsigned warlize_no_h[8] = { 0, 0, 0, 0, 4, 4, 4, 4 };  // only 0,4 legal
    static std::optional<mhselect_interpretation> table[8] = {
      mhselect_interpretation{ 0, MHSELECT_MODE_IGNORE, std::nullopt },
      mhselect_interpretation{ 1, MHSELECT_MODE_MCONTEXT, true },
      mhselect_interpretation{ 2, MHSELECT_MODE_VMID, true },
      std::nullopt,
      mhselect_interpretation{ 4, MHSELECT_MODE_MCONTEXT, false },
      mhselect_interpretation{ 5, MHSELECT_MODE_MCONTEXT, true },
      mhselect_interpretation{ 6, MHSELECT_MODE_VMID, true },
      std::nullopt
    };
    assert(mhselect < 8);
    unsigned legal = h_enabled ? warlize_if_h[mhselect] : warlize_no_h[mhselect];
    assert(legal < 8);
    return table[legal].value();
  }

  sselect_t sselect;
  unsigned svalue;
  unsigned sbytemask;
  unsigned mhselect;
  unsigned mhvalue;
};

class disabled_trigger_t : public trigger_t {
public:
  virtual reg_t tdata1_read(const processor_t * const proc) const noexcept override;
  virtual void tdata1_write(processor_t * const proc, const reg_t val, const bool allow_chain) noexcept override;

  virtual bool get_dmode() const override { return dmode; }

private:
  bool dmode;
};

class trap_common_t : public trigger_t {
public:
  bool get_dmode() const override { return dmode; }
  virtual action_t get_action() const override { return action; }

  virtual std::optional<match_result_t> detect_trap_match(processor_t * const proc, const trap_t& t) noexcept override;

private:
  virtual bool simple_match(bool interrupt, reg_t bit) const = 0;

protected:
  bool dmode;
  bool hit;
  action_t action;
};

class itrigger_t : public trap_common_t {
public:
  virtual reg_t tdata1_read(const processor_t * const proc) const noexcept override;
  virtual void tdata1_write(processor_t * const proc, const reg_t val, const bool allow_chain) noexcept override;

private:
  virtual bool simple_match(bool interrupt, reg_t bit) const override;
  bool nmi;
};

class etrigger_t : public trap_common_t {
public:
  virtual reg_t tdata1_read(const processor_t * const proc) const noexcept override;
  virtual void tdata1_write(processor_t * const proc, const reg_t val, const bool allow_chain) noexcept override;

private:
  virtual bool simple_match(bool interrupt, reg_t bit) const override;
};

class mcontrol_common_t : public trigger_t {
public:
  typedef enum
  {
    MATCH_EQUAL = MCONTROL_MATCH_EQUAL,
    MATCH_NAPOT = MCONTROL_MATCH_NAPOT,
    MATCH_GE = MCONTROL_MATCH_GE,
    MATCH_LT = MCONTROL_MATCH_LT,
    MATCH_MASK_LOW = MCONTROL_MATCH_MASK_LOW,
    MATCH_MASK_HIGH = MCONTROL_MATCH_MASK_HIGH
  } match_t;

  virtual bool get_dmode() const override { return dmode; }
  virtual bool get_chain() const override { return chain; }
  virtual bool get_execute() const override { return execute; }
  virtual bool get_store() const override { return store; }
  virtual bool get_load() const override { return load; }
  virtual action_t get_action() const override { return action; }

  virtual std::optional<match_result_t> detect_memory_access_match(processor_t * const proc,
      operation_t operation, reg_t address, std::optional<reg_t> data) noexcept override;

private:
  bool simple_match(unsigned xlen, reg_t value) const;

protected:
  static match_t legalize_match(reg_t val) noexcept;
  static bool legalize_timing(reg_t val, reg_t timing_mask, reg_t select_mask, reg_t execute_mask, reg_t load_mask) noexcept;
  bool dmode = false;
  action_t action = ACTION_DEBUG_EXCEPTION;
  bool hit = false;
  bool select = false;
  bool timing = false;
  bool chain = false;
  match_t match = MATCH_EQUAL;
  bool execute = false;
  bool store = false;
  bool load = false;
};

class mcontrol_t : public mcontrol_common_t {
public:
  virtual reg_t tdata1_read(const processor_t * const proc) const noexcept override;
  virtual void tdata1_write(processor_t * const proc, const reg_t val, const bool allow_chain) noexcept override;
};

class mcontrol6_t : public mcontrol_common_t {
public:
  virtual reg_t tdata1_read(const processor_t * const proc) const noexcept override;
  virtual void tdata1_write(processor_t * const proc, const reg_t val, const bool allow_chain) noexcept override;
};

class icount_t : public trigger_t {
public:
  virtual reg_t tdata1_read(const processor_t * const proc) const noexcept override;
  virtual void tdata1_write(processor_t * const proc, const reg_t val, const bool allow_chain) noexcept override;

  bool get_dmode() const override { return dmode; }
  virtual action_t get_action() const override { return action; }
  virtual bool icount_check_needed() const override { return count > 0 || pending; }
  virtual void stash_read_values() override;

  virtual std::optional<match_result_t> detect_icount_fire(processor_t * const proc) noexcept override;
  virtual void detect_icount_decrement(processor_t * const proc) noexcept override;

private:
  bool dmode = false;
  bool hit = false;
  unsigned count = 1, count_read_value = 1;
  bool pending = false, pending_read_value = false;
  action_t action = (action_t)0;
};

class module_t {
public:
  module_t(unsigned count);
  ~module_t();

  reg_t tdata1_read(unsigned index) const noexcept;
  bool tdata1_write(unsigned index, const reg_t val) noexcept;
  reg_t tdata2_read(unsigned index) const noexcept;
  bool tdata2_write(unsigned index, const reg_t val) noexcept;
  reg_t tdata3_read(unsigned index) const noexcept;
  bool tdata3_write(unsigned index, const reg_t val) noexcept;
  reg_t tinfo_read(unsigned index) const noexcept;

  unsigned count() const { return triggers.size(); }

  std::optional<match_result_t> detect_memory_access_match(operation_t operation, reg_t address, std::optional<reg_t> data) noexcept;
  std::optional<match_result_t> detect_icount_match() noexcept;
  std::optional<match_result_t> detect_trap_match(const trap_t& t) noexcept;

  processor_t *proc;
private:
  std::vector<trigger_t *> triggers;
};

};

#endif

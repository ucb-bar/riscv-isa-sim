// See LICENSE for license details.
#ifndef _RISCV_DEBUG_MODULE_H
#define _RISCV_DEBUG_MODULE_H

#include <set>
#include <vector>

#include "abstract_device.h"

class simif_t;
class bus_t;
class processor_t;

typedef struct {
  // Size of program_buffer in 32-bit words, as exposed to the rest of the
  // world.
  unsigned progbufsize = 2;
  unsigned max_sba_data_width = 0;
  bool require_authentication = false;
  unsigned abstract_rti = 0;
  bool support_hasel = true;
  bool support_abstract_csr_access = true;
  bool support_abstract_fpr_access = true;
  bool support_haltgroups = true;
  bool support_impebreak = true;
} debug_module_config_t;

typedef struct {
  bool haltreq;
  bool resumereq;
  bool hasel;
  unsigned hartsel;
  bool hartreset;
  bool dmactive;
  bool ndmreset;
} dmcontrol_t;

typedef struct {
  bool impebreak;
  bool allhavereset;
  bool anyhavereset;
  bool allnonexistant;
  bool anynonexistant;
  bool allunavail;
  bool anyunavail;
  bool allrunning;
  bool anyrunning;
  bool allhalted;
  bool anyhalted;
  bool allresumeack;
  bool anyresumeack;
  bool authenticated;
  bool authbusy;
  bool cfgstrvalid;
  unsigned version;
} dmstatus_t;

typedef enum cmderr {
  CMDERR_NONE = 0,
  CMDERR_BUSY = 1,
  CMDERR_NOTSUP = 2,
  CMDERR_EXCEPTION = 3,
  CMDERR_HALTRESUME = 4,
  CMDERR_OTHER = 7
} cmderr_t;

typedef struct {
  bool busy;
  unsigned datacount;
  unsigned progbufsize;
  cmderr_t cmderr;
} abstractcs_t;

typedef struct {
  unsigned autoexecprogbuf;
  unsigned autoexecdata;
} abstractauto_t;

typedef struct {
  unsigned version;
  bool readonaddr;
  unsigned sbaccess;
  bool autoincrement;
  bool readondata;
  unsigned error;
  unsigned asize;
  bool access128;
  bool access64;
  bool access32;
  bool access16;
  bool access8;
  bool sbbusyerror;
} sbcs_t;

typedef struct {
  bool halted;
  bool resumeack;
  bool havereset;
  uint8_t haltgroup;
} hart_debug_state_t;

class debug_module_t : public abstract_device_t
{
  public:
    /*
     * If require_authentication is true, then a debugger must authenticate as
     * follows:
     * 1. Read a 32-bit value from authdata:
     * 2. Write the value that was read back, plus one, to authdata.
     *
     * abstract_rti is extra run-test/idle cycles that each abstract command
     * takes to execute. Useful for testing OpenOCD.
     */
    debug_module_t(simif_t *sim, const debug_module_config_t &config);
    ~debug_module_t();

    bool load(reg_t addr, size_t len, uint8_t* bytes);
    bool store(reg_t addr, size_t len, const uint8_t* bytes);

    // Debug Module Interface that the debugger (in our case through JTAG DTM)
    // uses to access the DM.
    // Return true for success, false for failure.
    bool dmi_read(unsigned address, uint32_t *value);
    bool dmi_write(unsigned address, uint32_t value);

    // Called for every cycle the JTAG TAP spends in Run-Test/Idle.
    void run_test_idle();

    // Called when one of the attached harts was reset.
    void proc_reset(unsigned id);

  private:
    static const unsigned datasize = 2;
    debug_module_config_t config;
    // Actual size of the program buffer, which is 1 word bigger than we let on
    // to implement the implicit ebreak at the end.
    unsigned program_buffer_bytes;
    static const unsigned debug_data_start = 0x380;
    unsigned debug_progbuf_start;

    static const unsigned debug_abstract_size = 12;
    unsigned debug_abstract_start;
    // R/W this through custom registers, to allow debuggers to test that
    // functionality.
    unsigned custom_base;

    simif_t *sim;

    uint8_t debug_rom_whereto[4];
    uint8_t debug_abstract[debug_abstract_size * 4];
    uint8_t *program_buffer;
    uint8_t dmdata[datasize * 4];

    std::vector<hart_debug_state_t> hart_state;
    uint8_t debug_rom_flags[1024];

    void write32(uint8_t *rom, unsigned int index, uint32_t value);
    uint32_t read32(uint8_t *rom, unsigned int index);

    void sb_autoincrement();

    /* Start a system bus access. (It could be instantaneous, but to help test
     * OpenOCD a delay can be added.) */
    void sb_read_start();
    void sb_write_start();

    /* Actually read/write. */
    void sb_read();
    void sb_write();

    /* Return true iff a system bus access is in progress. */
    bool sb_busy() const;

    unsigned sb_access_bits();

    dmcontrol_t dmcontrol;
    dmstatus_t dmstatus;
    abstractcs_t abstractcs;
    abstractauto_t abstractauto;
    uint32_t command;
    uint16_t hawindowsel;
    std::vector<bool> hart_array_mask;

    sbcs_t sbcs;
    uint32_t sbaddress[4];
    uint32_t sbdata[4];

    uint32_t challenge;
    const uint32_t secret = 1;

    bool hart_selected(unsigned hartid) const;
    void reset();
    bool perform_abstract_command();

    bool abstract_command_completed;
    unsigned rti_remaining;

    size_t selected_hart_id() const;
    hart_debug_state_t& selected_hart_state();

    /* Whether the first 2 harts are available is controllable through DMCUSTOM,
     * where bit 0 corresponds to hart 0, etc. When a bit is one the hart
     * available.  Otherwise it is unavailable. */
    bool hart_available_state[2];
    bool hart_available(unsigned hart_id) const;

    unsigned sb_read_wait, sb_write_wait;
};

#endif

// See LICENSE for license details.

#ifndef _RISCV_SIM_LIB_H
#define _RISCV_SIM_LIB_H

#include "cfg.h"
#include "debug_module.h"
#include "devices.h"
#include "log_file.h"
#include "processor.h"
#include "simif.h"
#include "sim.h"

#include <cstdint>
#include <fesvr/htif.h>
#include <vector>
#include <queue>
#include <map>
#include <string>
#include <memory>
#include <sys/types.h>

class mmu_t;
class remote_bitbang_t;
class socketif_t;


class sim_lib_t : public sim_t
{
public:
  sim_lib_t(const cfg_t *cfg, bool halted,
        std::vector<std::pair<reg_t, abstract_mem_t*>> mems,
        std::vector<device_factory_t*> plugin_device_factories,
        const std::vector<std::string>& args,
        const debug_module_config_t &dm_config, const char *log_path,
        bool dtb_enabled, const char *dtb_file,
        bool socket_enabled,
        FILE *cmd_file,
        bool checkpoint); // needed for command line option --cmd
  ~sim_lib_t();

  void run_for(uint64_t steps);
  void take_ckpt(std::string& os);
  void load_ckpt(std::string& is, bool is_json);

  virtual int run() override;
  void init();
  bool target_running();
  int  stop_sim();

  void step_proc(size_t n, unsigned int idx);
  void yield_load_rsrv(unsigned int idx); // for atomics
  void step_devs(size_t n);
  void step_target(size_t proc_step, size_t dev_step);

  // htif apis
  uint64_t check_tohost_req();
  void handle_tohost_req(uint64_t req);
  void send_fromhost_req();

private:
  friend class processor_t;
  friend class mmu_t;
  friend class sim_t;

  std::queue<reg_t> fromhost_queue;
  std::function<void(reg_t)> fromhost_callback;


public:
  // Returns the pc trace fore the current "run_for"
  std::vector<reg_t>& pctrace() { return target_PC; }

private:
  std::vector<reg_t> target_PC;
};


#endif //_RISCV_SIM_LIB_H

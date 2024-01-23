// See LICENSE for license details.

#include "config.h"
#include "sim.h"
#include "sim_lib.h"
#include "mmu.h"
#include "dts.h"
#include "byteorder.h"
#include "platform.h"
#include "libfdt.h"
#include "htif.h"
#include "socketif.h"
#include <cstdint>
#include <fstream>
#include <map>
#include <iostream>
#include <memory>
#include <sstream>
#include <climits>
#include <cstdlib>
#include <cassert>
#include <signal.h>
<<<<<<< HEAD
=======
#include <inttypes.h>
#include <string>
>>>>>>> f051e001... Working state... cleanup later
#include <unistd.h>
#include <sys/wait.h>
#include <sys/types.h>


extern device_factory_t* clint_factory;
extern device_factory_t* plic_factory;
extern device_factory_t* ns16550_factory;

sim_lib_t::sim_lib_t(const cfg_t *cfg, bool halted,
        std::vector<std::pair<reg_t, abstract_mem_t*>> mems,
        std::vector<device_factory_t*> plugin_device_factories,
        const std::vector<std::string>& args,
        const debug_module_config_t &dm_config, const char *log_path,
        bool dtb_enabled, const char *dtb_file,
        bool socket_enabled,
        FILE *cmd_file,
        bool checkpoint)
  : sim_t(cfg, halted, mems, plugin_device_factories, args, dm_config,
          log_path, dtb_enabled, dtb_file, socket_enabled, cmd_file, checkpoint)
{
  auto enq_func = [](std::queue<reg_t>* q, uint64_t x) { q->push(x); };
  fromhost_callback = std::bind(enq_func, &fromhost_queue, std::placeholders::_1);
}

sim_lib_t::~sim_lib_t() {
}

void sim_lib_t::run_for(uint64_t steps) {
  uint64_t tot_step = 0;
  target_trace.clear();

  while (target_running() && tot_step < steps) {
    uint64_t tohost_req = check_tohost_req();
    if (tohost_req) {
      handle_tohost_req(tohost_req);
    } else {
      uint64_t cur_step = std::min(steps - tot_step, INTERLEAVE);
      uint64_t dev_step = std::max((uint64_t)1, cur_step / INSNS_PER_RTC_TICK);
      step_target(cur_step, dev_step);
      for (int i = 0, nprocs = (int)procs.size(); i < nprocs; i++) {
        auto pst = procs[i]->step_trace();
        target_trace.insert(target_trace.end(), pst.begin(), pst.end());
        tot_step += pst.size();
      }
    }
    send_fromhost_req();
  }

  if (!target_running()) {
    fprintf(stderr, "target finished before %" PRIu64 " steps\n", steps);
/* exit(1); */
  }
}

void sim_lib_t::take_ckpt(std::string& os) {
  serialize_proto(os);
}

void sim_lib_t::load_ckpt(std::string& is, bool is_json) {
  deserialize_proto(is, is_json);
}

// Example usage of the APIs.
// We can decompose the below loop to have fine-grained control over the
// fesver polling loop.
int sim_lib_t::run() {
  while (target_running()) {
    uint64_t tohost_req = check_tohost_req();
    if (tohost_req) {
      handle_tohost_req(tohost_req);
    } else {
      if (debug || ctrlc_pressed)
        interactive();
      else
        step_target(INTERLEAVE, INTERLEAVE / INSNS_PER_RTC_TICK);
    }
    send_fromhost_req();
  }
  return stop_sim();
}

void sim_lib_t::init() {
  if (!debug && log)
    set_procs_debug(true);

  htif_t::set_expected_xlen(isa.get_max_xlen());

  // load the binary
  start();
}

bool sim_lib_t::target_running() {
  return (!signal_exit && exitcode) == 0;
}

int sim_lib_t::stop_sim() {
  stop();
  return exit_code();
}

void sim_lib_t::step_proc(size_t n, unsigned int idx) {
  procs[idx]->step(n);
}

void sim_lib_t::yield_load_rsrv(unsigned int idx) {
  procs[idx]->get_mmu()->yield_load_reservation();
}

void sim_lib_t::step_devs(size_t n) {
  for (auto &dev : devices)
    dev->tick(n);
}

void sim_lib_t::step_target(size_t proc_step, size_t dev_step) {
  unsigned int nprocs = (unsigned int)procs.size();

  // TODO : yield_load_rsrv in multicore. currently, spike may loop infinitely
  // when the single processor yields every time it executes a single step
  assert(nprocs == 1);
  for (unsigned int pidx = 0; pidx < nprocs; pidx++) {
    step_proc(proc_step, pidx);
/* yield_load_rsrv(pidx); */
  }
  step_devs(dev_step);
}


// htif stuff

uint64_t sim_lib_t::check_tohost_req() {
  uint64_t tohost;
  try {
    if ((tohost = from_target(mem.read_uint64(tohost_addr))) != 0)
      mem.write_uint64(tohost_addr, target_endian<uint64_t>::zero);
  } catch (mem_trap_t& t) {
    bad_address("accessing tohost", t.get_tval());
  }
  return tohost;
}

void sim_lib_t::handle_tohost_req(uint64_t req) {
  try {
    command_t cmd(mem, req, fromhost_callback);
    device_list.handle_command(cmd);
    device_list.tick();
  } catch (mem_trap_t& t) {
    std::stringstream tohost_hex;
    tohost_hex << std::hex << req;
    bad_address("host was accessing memory on behalf of target (tohost = 0x" + tohost_hex.str() + ")", t.get_tval());
  }
}

void sim_lib_t::send_fromhost_req() {
  try {
    if (!fromhost_queue.empty() && !mem.read_uint64(fromhost_addr)) {
      mem.write_uint64(fromhost_addr, to_target(fromhost_queue.front()));
      fromhost_queue.pop();
    }
  } catch (mem_trap_t& t) {
    bad_address("accessing fromhost", t.get_tval());
  }
}

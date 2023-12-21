// See LICENSE for license details.

#ifndef _RISCV_SIM_H
#define _RISCV_SIM_H

#include "cfg.h"
#include "debug_module.h"
#include "devices.h"
#include "log_file.h"
#include "mmu.h"
#include "processor.h"
#include "simif.h"

#include <cstdint>
#include <fesvr/htif.h>
#include <vector>
#include <map>
#include <string>
#include <memory>
#include <sys/types.h>

#include "arch-state.pb.h"
#include <google/protobuf/arena.h>
#include <google/protobuf/util/json_util.h>

class mmu_t;
class remote_bitbang_t;
class socketif_t;

// this class encapsulates the processors and memory in a RISC-V machine.
class sim_t : public htif_t, public simif_t
{
public:
  sim_t(const cfg_t *cfg, bool halted,
        std::vector<std::pair<reg_t, abstract_mem_t*>> mems,
        std::vector<device_factory_t*> plugin_device_factories,
        const std::vector<std::string>& args,
        const debug_module_config_t &dm_config, const char *log_path,
        bool dtb_enabled, const char *dtb_file,
        bool socket_enabled,
        FILE *cmd_file); // needed for command line option --cmd
  ~sim_t();

  // run the simulation to completion
  virtual int run();
  void set_debug(bool value);
  void set_histogram(bool value);
  void add_device(reg_t addr, std::shared_ptr<abstract_device_t> dev);

  // Configure logging
  //
  // If enable_log is true, an instruction trace will be generated. If
  // enable_commitlog is true, so will the commit results
  void configure_log(bool enable_log, bool enable_commitlog);

  void set_procs_debug(bool value);
  void set_remote_bitbang(remote_bitbang_t* remote_bitbang) {
    this->remote_bitbang = remote_bitbang;
  }
  const char* get_dts() { return dts.c_str(); }
  processor_t* get_core(size_t i) { return procs.at(i); }
  abstract_interrupt_controller_t* get_intctrl() const { assert(plic.get()); return plic.get(); }
  virtual const cfg_t &get_cfg() const override { return *cfg; }

  virtual const std::map<size_t, processor_t*>& get_harts() const override { return harts; }

  // Callback for processors to let the simulation know they were reset.
  virtual void proc_reset(unsigned id) override;

  static const size_t INTERLEAVE = 5000;
  static const size_t INSNS_PER_RTC_TICK = 100; // 10 MHz clock for 1 BIPS core
  static const size_t CPU_HZ = 1000000000; // 1GHz CPU
  
  std::vector<std::pair<reg_t, abstract_mem_t*>> get_mems() { return mems; }

protected:
  isa_parser_t isa;
  const cfg_t * const cfg;
  std::vector<std::pair<reg_t, abstract_mem_t*>> mems;
  std::vector<processor_t*> procs;
  std::map<size_t, processor_t*> harts;
  std::pair<reg_t, reg_t> initrd_range;
  std::string dts;
  std::string dtb;
  bool dtb_enabled;
  std::vector<std::shared_ptr<abstract_device_t>> devices;
  std::shared_ptr<clint_t> clint;
  std::shared_ptr<plic_t> plic;
  bus_t bus;
  log_file_t log_file;

  FILE *cmd_file; // pointer to debug command input file

  socketif_t *socketif;
  std::ostream sout_; // used for socket and terminal interface

  processor_t* get_core(const std::string& i);
  void step(size_t n); // step through simulation
  size_t current_step;
  size_t current_proc;
  bool debug;
  bool histogram_enabled; // provide a histogram of PCs
  bool log;
  remote_bitbang_t* remote_bitbang;
  std::optional<std::function<void()>> next_interactive_action;

  // memory-mapped I/O routines
  virtual char* addr_to_mem(reg_t paddr) override;
  virtual bool mmio_load(reg_t paddr, size_t len, uint8_t* bytes) override;
  virtual bool mmio_store(reg_t paddr, size_t len, const uint8_t* bytes) override;
  void set_rom();

  virtual const char* get_symbol(uint64_t paddr) override;

  // presents a prompt for introspection into the simulation
  void interactive();

  // functions that help implement interactive()
  void interactive_help(const std::string& cmd, const std::vector<std::string>& args);
  void interactive_quit(const std::string& cmd, const std::vector<std::string>& args);
  void interactive_run(const std::string& cmd, const std::vector<std::string>& args, bool noisy);
  void interactive_run_noisy(const std::string& cmd, const std::vector<std::string>& args);
  void interactive_run_silent(const std::string& cmd, const std::vector<std::string>& args);
  void interactive_vreg(const std::string& cmd, const std::vector<std::string>& args);
  void interactive_reg(const std::string& cmd, const std::vector<std::string>& args);
  void interactive_freg(const std::string& cmd, const std::vector<std::string>& args);
  void interactive_fregh(const std::string& cmd, const std::vector<std::string>& args);
  void interactive_fregs(const std::string& cmd, const std::vector<std::string>& args);
  void interactive_fregd(const std::string& cmd, const std::vector<std::string>& args);
  void interactive_pc(const std::string& cmd, const std::vector<std::string>& args);
  void interactive_priv(const std::string& cmd, const std::vector<std::string>& args);
  void interactive_mem(const std::string& cmd, const std::vector<std::string>& args);
  void interactive_str(const std::string& cmd, const std::vector<std::string>& args);
  void interactive_dumpmems(const std::string& cmd, const std::vector<std::string>& args);
  void interactive_mtime(const std::string& cmd, const std::vector<std::string>& args);
  void interactive_mtimecmp(const std::string& cmd, const std::vector<std::string>& args);
  void interactive_until(const std::string& cmd, const std::vector<std::string>& args, bool noisy);
  void interactive_until_silent(const std::string& cmd, const std::vector<std::string>& args);
  void interactive_until_noisy(const std::string& cmd, const std::vector<std::string>& args);
  reg_t get_reg(const std::vector<std::string>& args);
  freg_t get_freg(const std::vector<std::string>& args, int size);
  reg_t get_mem(const std::vector<std::string>& args);
  reg_t get_pc(const std::vector<std::string>& args);

  friend class processor_t;
  friend class mmu_t;

  // htif
  virtual void reset() override;
  virtual void idle() override;
  virtual void read_chunk(addr_t taddr, size_t len, void* dst) override;
  virtual void write_chunk(addr_t taddr, size_t len, const void* src) override;
  virtual size_t chunk_align() override { return 8; }
  virtual size_t chunk_max_size() override { return 8; }
  virtual endianness_t get_target_endianness() const override;

public:
  // Initialize this after procs, because in debug_module_t::reset() we
  // enumerate processors, which segfaults if procs hasn't been initialized
  // yet.
  debug_module_t debug_module;

public:
  google::protobuf::Arena* arena;

  void serialize_proto(std::string& os) {
    arena = new google::protobuf::Arena();
    SimState* sim_proto = google::protobuf::Arena::Create<SimState>(arena);

    for (int i = 0, cnt = (int)procs.size(); i < cnt; i++) {
      ArchState* arch_proto = sim_proto->add_msg_arch_state();
      procs[i]->serialize_proto(arch_proto, arena);
      printf("proc instret: %" PRIu64 "\n", procs[i]->tot_instret);
    }

    // only one dram device for now
    assert((int)mems.size() == 1);

    for (auto& addr_mem : mems) {
      auto mem = (mem_t*)addr_mem.second;
      std::map<reg_t, char*>& spm = mem->sparse_memory_map;
      for (auto& page: spm) {
        Page* page_proto = sim_proto->add_msg_sparse_mm();
        page_proto->set_msg_ppn(page.first);
        page_proto->set_msg_bytes((const void*)page.second, PGSIZE);
      }
    }

    sim_proto->SerializeToString(&os);

    // Create a json_string from sr.
    std::string json_string;
    google::protobuf::util::JsonPrintOptions options;
    options.add_whitespace = true;
    options.always_print_primitive_fields = true;
    options.preserve_proto_field_names = true;
    google::protobuf::util::MessageToJsonString(*sim_proto, &json_string, options);

    std::fstream json_file;
    json_file.open("arch-state.json", std::ios::out);
    json_file << json_string << std::endl;
    json_file.close();

    google::protobuf::ShutdownProtobufLibrary();
  }

  void deserialize_proto(std::string& is, bool is_json) {
    arena = new google::protobuf::Arena();
    SimState* sim_proto = google::protobuf::Arena::Create<SimState>(arena);

    if (is_json) {
        google::protobuf::util::JsonParseOptions options;
        google::protobuf::util::JsonStringToMessage(is, sim_proto, options);
    } else {
      sim_proto->ParseFromString(is);
    }

    for (int i = 0, cnt = sim_proto->msg_arch_state_size(); i < cnt; i++) {
      auto arch_proto = sim_proto->msg_arch_state(i);
      procs[i]->deserialize_proto(&arch_proto);
    }

    // only one dram device for now
    assert((int)mems.size() == 1);

    for (auto& addr_mem : mems) {
      auto mem = (mem_t*)addr_mem.second;
      std::map<reg_t, char*>& spm = mem->sparse_memory_map;

      // clear memory
      for (auto& page: spm) {
        free(page.second);
      }
      spm.clear();

      // insert new dram contents
      for (int i = 0, cnt = sim_proto->msg_sparse_mm_size(); i < cnt; i++) {
        const Page& page_proto = sim_proto->msg_sparse_mm(i);

        reg_t ppn = page_proto.msg_ppn();
        const std::string& bytes = page_proto.msg_bytes();
        assert(bytes.size() == PGSIZE);

        char* res = (char*)calloc(PGSIZE, 1);
        if (res == nullptr)
          throw std::bad_alloc();
        memcpy((void*)res, bytes.data(), bytes.size());
        spm[ppn] = res;
      }
    }


    google::protobuf::ShutdownProtobufLibrary();
  }

  bool compare(processor_t* proc) {
    auto state0 = *(procs[0]->get_state());
    auto state1 = *(proc->get_state());
    return state0 == state1;
  }

  bool compare_mem(mem_t* mem) {
    auto my_mem = (mem_t*)(mems[0].second);
    auto my_mm  = my_mem->sparse_memory_map;
    auto ref_mm = mem->sparse_memory_map;
    if (ref_mm.size() != my_mm.size()) {
      std::cerr << "Page count mismatch ref vs my "
                << std::dec << ref_mm.size() << my_mm.size() << std::endl;
      return false;
    }

    for (auto& a2p : ref_mm) {
      reg_t ppn = a2p.first;
      uint64_t* ref_page_by_8B = (uint64_t*)a2p.second;

      auto it = my_mm.find(ppn);
      if (it == my_mm.end()) {
        std::cerr << "PPN: " << ppn << " not found" << std::endl;
        return false;
      }
      uint64_t*  my_page_by_8B = (uint64_t*)it->second;

      for (int i = 0; i < PGSIZE / 8; i++) {
        uint64_t ref_data = ref_page_by_8B[i];
        uint64_t my_data  = my_page_by_8B[i];
        if (ref_data != my_data) {
          std::cerr << "page[" << 8 * i << "]: expect "
                    << std::hex << ref_data << " got " << my_data << std::endl;
          return false;
        }
      }
    }
    return true;
  }
};

extern volatile bool ctrlc_pressed;

#endif

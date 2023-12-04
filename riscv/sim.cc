// See LICENSE for license details.

#include "config.h"
#include "sim.h"
#include "mmu.h"
#include "dts.h"
#include "remote_bitbang.h"
#include "byteorder.h"
#include "platform.h"
#include "libfdt.h"
#include "socketif.h"
#include <fstream>
#include <map>
#include <iostream>
#include <sstream>
#include <climits>
#include <cstdlib>
#include <cassert>
#include <signal.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/types.h>

volatile bool ctrlc_pressed = false;
static void handle_signal(int sig)
{
  if (ctrlc_pressed)
    exit(-1);
  ctrlc_pressed = true;
  signal(sig, &handle_signal);
}

const size_t sim_t::INTERLEAVE;

extern device_factory_t* clint_factory;
extern device_factory_t* plic_factory;
extern device_factory_t* ns16550_factory;

sim_t::sim_t(const cfg_t *cfg, bool halted,
             std::vector<std::pair<reg_t, abstract_mem_t*>> mems,
             std::vector<device_factory_t*> plugin_device_factories,
             const std::vector<std::string>& args,
             const debug_module_config_t &dm_config,
             const char *log_path,
             bool dtb_enabled, const char *dtb_file,
             bool socket_enabled,
             FILE *cmd_file) // needed for command line option --cmd
  : htif_t(args),
    isa(cfg->isa(), cfg->priv()),
    cfg(cfg),
    mems(mems),
    procs(std::max(cfg->nprocs(), size_t(1))),
    dtb_enabled(dtb_enabled),
    log_file(log_path),
    cmd_file(cmd_file),
    sout_(nullptr),
    current_step(0),
    current_proc(0),
    debug(false),
    histogram_enabled(false),
    log(false),
    remote_bitbang(NULL),
    debug_module(this, dm_config)
{
  signal(SIGINT, &handle_signal);

  sout_.rdbuf(std::cerr.rdbuf()); // debug output goes to stderr by default

  for (auto& x : mems)
    bus.add_device(x.first, x.second);

  bus.add_device(DEBUG_START, &debug_module);

  socketif = NULL;
#ifdef HAVE_BOOST_ASIO
  if (socket_enabled) {
    socketif = new socketif_t();
  }
#else
  if (socket_enabled) {
    fputs("Socket support requires compilation with boost asio; "
          "please rebuild the riscv-isa-sim project using "
          "\"configure --with-boost-asio\".\n",
          stderr);
    abort();
  }
#endif

#ifndef RISCV_ENABLE_DUAL_ENDIAN
  if (cfg->endianness != endianness_little) {
    fputs("Big-endian support has not been prroperly enabled; "
          "please rebuild the riscv-isa-sim project using "
          "\"configure --enable-dual-endian\".\n",
          stderr);
    abort();
  }
#endif

  debug_mmu = new mmu_t(this, cfg->endianness, NULL);

  for (size_t i = 0; i < cfg->nprocs(); i++) {
    procs[i] = new processor_t(&isa, cfg, this, cfg->hartids()[i], halted,
                               log_file.get(), sout_);
    harts[cfg->hartids()[i]] = procs[i];
  }

  // When running without using a dtb, skip the fdt-based configuration steps
  if (!dtb_enabled) return;

  // Only make a CLINT (Core-Local INTerrupt controller) and PLIC (Platform-
  // Level-Interrupt-Controller) if they are specified in the device tree
  // configuration.
  //
  // This isn't *quite* as general as we could get (because you might have one
  // that's not bus-accessible), but it should handle the normal use cases. In
  // particular, the default device tree configuration that you get without
  // setting the dtb_file argument has one.
  std::vector<const device_factory_t*> device_factories = {
    clint_factory, // clint must be element 0
    plic_factory, // plic must be element 1
    ns16550_factory};
  device_factories.insert(device_factories.end(),
                          plugin_device_factories.begin(),
                          plugin_device_factories.end());

  // Load dtb_file if provided, otherwise self-generate a dts/dtb
  if (dtb_file) {
    std::ifstream fin(dtb_file, std::ios::binary);
    if (!fin.good()) {
      std::cerr << "can't find dtb file: " << dtb_file << std::endl;
      exit(-1);
    }
    std::stringstream strstream;
    strstream << fin.rdbuf();
    dtb = strstream.str();
  } else {
    std::pair<reg_t, reg_t> initrd_bounds = cfg->initrd_bounds();
    std::string device_nodes;
    for (const device_factory_t *factory : device_factories)
      device_nodes.append(factory->generate_dts(this));
    dts = make_dts(INSNS_PER_RTC_TICK, CPU_HZ,
                   initrd_bounds.first, initrd_bounds.second,
                   cfg->bootargs(), cfg->pmpregions, cfg->pmpgranularity,
                   procs, mems, device_nodes);
    dtb = dts_compile(dts);
  }

  int fdt_code = fdt_check_header(dtb.c_str());
  if (fdt_code) {
    std::cerr << "Failed to read DTB from ";
    if (!dtb_file) {
      std::cerr << "auto-generated DTS string";
    } else {
      std::cerr << "`" << dtb_file << "'";
    }
    std::cerr << ": " << fdt_strerror(fdt_code) << ".\n";
    exit(-1);
  }

  void *fdt = (void *)dtb.c_str();

  for (size_t i = 0; i < device_factories.size(); i++) {
    const device_factory_t *factory = device_factories[i];
    reg_t device_base = 0;
    abstract_device_t* device = factory->parse_from_fdt(fdt, this, &device_base);
    if (device) {
      assert(device_base);
      std::shared_ptr<abstract_device_t> dev_ptr(device);
      add_device(device_base, dev_ptr);

      if (i == 0) // clint_factory
        clint = std::static_pointer_cast<clint_t>(dev_ptr);
      else if (i == 1) // plic_factory
        plic = std::static_pointer_cast<plic_t>(dev_ptr);
    }
  }

  //per core attribute
  int cpu_offset = 0, rc;
  size_t cpu_idx = 0;
  cpu_offset = fdt_get_offset(fdt, "/cpus");
  if (cpu_offset < 0)
    return;

  for (cpu_offset = fdt_get_first_subnode(fdt, cpu_offset); cpu_offset >= 0;
       cpu_offset = fdt_get_next_subnode(fdt, cpu_offset)) {

    if (cpu_idx >= nprocs())
      break;

    //handle pmp
    reg_t pmp_num, pmp_granularity;
    if (fdt_parse_pmp_num(fdt, cpu_offset, &pmp_num) != 0)
      pmp_num = 0;
    procs[cpu_idx]->set_pmp_num(pmp_num);

    if (fdt_parse_pmp_alignment(fdt, cpu_offset, &pmp_granularity) == 0) {
      procs[cpu_idx]->set_pmp_granularity(pmp_granularity);
    }

    //handle mmu-type
    const char *mmu_type;
    rc = fdt_parse_mmu_type(fdt, cpu_offset, &mmu_type);
    if (rc == 0) {
      procs[cpu_idx]->set_mmu_capability(IMPL_MMU_SBARE);
      if (strncmp(mmu_type, "riscv,sv32", strlen("riscv,sv32")) == 0) {
        procs[cpu_idx]->set_mmu_capability(IMPL_MMU_SV32);
      } else if (strncmp(mmu_type, "riscv,sv39", strlen("riscv,sv39")) == 0) {
        procs[cpu_idx]->set_mmu_capability(IMPL_MMU_SV39);
      } else if (strncmp(mmu_type, "riscv,sv48", strlen("riscv,sv48")) == 0) {
        procs[cpu_idx]->set_mmu_capability(IMPL_MMU_SV48);
      } else if (strncmp(mmu_type, "riscv,sv57", strlen("riscv,sv57")) == 0) {
        procs[cpu_idx]->set_mmu_capability(IMPL_MMU_SV57);
      } else if (strncmp(mmu_type, "riscv,sbare", strlen("riscv,sbare")) == 0) {
        //has been set in the beginning
      } else {
        std::cerr << "core ("
                  << cpu_idx
                  << ") has an invalid 'mmu-type': "
                  << mmu_type << ").\n";
        exit(1);
      }
    } else {
      procs[cpu_idx]->set_mmu_capability(IMPL_MMU_SBARE);
    }

    cpu_idx++;
  }

  if (cpu_idx != nprocs()) {
      std::cerr << "core number in dts ("
                <<  cpu_idx
                << ") doesn't match it in command line ("
                << nprocs() << ").\n";
      exit(1);
  }
}

sim_t::~sim_t()
{
  for (size_t i = 0; i < procs.size(); i++)
    delete procs[i];
  delete debug_mmu;
}

int sim_t::run()
{
  if (!debug && log)
    set_procs_debug(true);

  htif_t::set_expected_xlen(isa.get_max_xlen());

  // htif_t::run() will repeatedly call back into sim_t::idle(), each
  // invocation of which will advance target time
  return htif_t::run();
}

void sim_t::step(size_t n)
{
  for (size_t i = 0, steps = 0; i < n; i += steps)
  {
    steps = std::min(n - i, INTERLEAVE - current_step);
    procs[current_proc]->step(steps);

    current_step += steps;
    if (current_step == INTERLEAVE)
    {
      current_step = 0;
      procs[current_proc]->get_mmu()->yield_load_reservation();
      if (++current_proc == procs.size()) {
        current_proc = 0;
        reg_t rtc_ticks = INTERLEAVE / INSNS_PER_RTC_TICK;
        for (auto &dev : devices) dev->tick(rtc_ticks);
      }
    }
  }
}

void sim_t::add_device(reg_t addr, std::shared_ptr<abstract_device_t> dev) {
  bus.add_device(addr, dev.get());
  devices.push_back(dev);
}

void sim_t::set_debug(bool value)
{
  debug = value;
}

void sim_t::set_histogram(bool value)
{
  histogram_enabled = value;
  for (size_t i = 0; i < procs.size(); i++) {
    procs[i]->set_histogram(histogram_enabled);
  }
}

void sim_t::configure_log(bool enable_log, bool enable_commitlog)
{
  log = enable_log;

  if (!enable_commitlog)
    return;

  for (processor_t *proc : procs) {
    proc->enable_log_commits();
  }
}

void sim_t::set_procs_debug(bool value)
{
  for (size_t i=0; i< procs.size(); i++)
    procs[i]->set_debug(value);
}

static bool paddr_ok(reg_t addr)
{
  return (addr >> MAX_PADDR_BITS) == 0;
}

bool sim_t::mmio_load(reg_t paddr, size_t len, uint8_t* bytes)
{
  if (paddr + len < paddr || !paddr_ok(paddr + len - 1))
    return false;
  return bus.load(paddr, len, bytes);
}

bool sim_t::mmio_store(reg_t paddr, size_t len, const uint8_t* bytes)
{
  if (paddr + len < paddr || !paddr_ok(paddr + len - 1))
    return false;
  return bus.store(paddr, len, bytes);
}

void sim_t::set_rom()
{
  const int reset_vec_size = 8;

  reg_t start_pc = cfg->start_pc.value_or(get_entry_point());

  uint32_t reset_vec[reset_vec_size] = {
    0x297,                                      // auipc  t0,0x0
    0x28593 + (reset_vec_size * 4 << 20),       // addi   a1, t0, &dtb
    0xf1402573,                                 // csrr   a0, mhartid
    get_core(0)->get_xlen() == 32 ?
      0x0182a283u :                             // lw     t0,24(t0)
      0x0182b283u,                              // ld     t0,24(t0)
    0x28067,                                    // jr     t0
    0,
    (uint32_t) (start_pc & 0xffffffff),
    (uint32_t) (start_pc >> 32)
  };
  if (get_target_endianness() == endianness_big) {
    int i;
    // Instuctions are little endian
    for (i = 0; reset_vec[i] != 0; i++)
      reset_vec[i] = to_le(reset_vec[i]);
    // Data is big endian
    for (; i < reset_vec_size; i++)
      reset_vec[i] = to_be(reset_vec[i]);

    // Correct the high/low order of 64-bit start PC
    if (get_core(0)->get_xlen() != 32)
      std::swap(reset_vec[reset_vec_size-2], reset_vec[reset_vec_size-1]);
  } else {
    for (int i = 0; i < reset_vec_size; i++)
      reset_vec[i] = to_le(reset_vec[i]);
  }

  std::vector<char> rom((char*)reset_vec, (char*)reset_vec + sizeof(reset_vec));

  rom.insert(rom.end(), dtb.begin(), dtb.end());
  const int align = 0x1000;
  rom.resize((rom.size() + align - 1) / align * align);

  std::shared_ptr<rom_device_t> boot_rom(new rom_device_t(rom));
  add_device(DEFAULT_RSTVEC, boot_rom);
}

char* sim_t::addr_to_mem(reg_t paddr) {
  if (!paddr_ok(paddr))
    return NULL;
  auto desc = bus.find_device(paddr);
  if (auto mem = dynamic_cast<abstract_mem_t*>(desc.second))
    if (paddr - desc.first < mem->size())
      return mem->contents(paddr - desc.first);
  return NULL;
}

const char* sim_t::get_symbol(uint64_t paddr)
{
  return htif_t::get_symbol(paddr);
}

// htif

void sim_t::reset()
{
  if (dtb_enabled)
    set_rom();
}

void sim_t::idle()
{
  if (done())
    return;

  if (debug || ctrlc_pressed)
    interactive();
  else
    step(INTERLEAVE);

  if (remote_bitbang)
    remote_bitbang->tick();
}

void sim_t::read_chunk(addr_t taddr, size_t len, void* dst)
{
  assert(len == 8);
  auto data = debug_mmu->to_target(debug_mmu->load<uint64_t>(taddr));
  memcpy(dst, &data, sizeof data);
}

void sim_t::write_chunk(addr_t taddr, size_t len, const void* src)
{
  assert(len == 8);
  target_endian<uint64_t> data;
  memcpy(&data, src, sizeof data);
  debug_mmu->store<uint64_t>(taddr, debug_mmu->from_target(data));
}

endianness_t sim_t::get_target_endianness() const
{
  return debug_mmu->is_target_big_endian()? endianness_big : endianness_little;
}

void sim_t::proc_reset(unsigned id)
{
  debug_module.proc_reset(id);
}

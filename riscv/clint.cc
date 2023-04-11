#include <sys/time.h>
#include <sstream>
#include "devices.h"
#include "processor.h"
#include "simif.h"
#include "sim.h"
#include "dts.h"

clint_t::clint_t(const simif_t* sim, uint64_t freq_hz, bool real_time)
  : sim(sim), freq_hz(freq_hz), real_time(real_time), mtime(0)
{
  struct timeval base;

  gettimeofday(&base, NULL);

  real_time_ref_secs = base.tv_sec;
  real_time_ref_usecs = base.tv_usec;
  tick(0);
}

/* 0000 msip hart 0
 * 0004 msip hart 1
 * 4000 mtimecmp hart 0 lo
 * 4004 mtimecmp hart 0 hi
 * 4008 mtimecmp hart 1 lo
 * 400c mtimecmp hart 1 hi
 * bff8 mtime lo
 * bffc mtime hi
 */

#define MSIP_BASE	0x0
#define MTIMECMP_BASE	0x4000
#define MTIME_BASE	0xbff8

bool clint_t::load(reg_t addr, size_t len, uint8_t* bytes)
{
  if (len > 8)
    return false;

  tick(0);

  if (addr >= MSIP_BASE && addr < MTIMECMP_BASE) {
    if (len == 8) {
      // Implement double-word loads as a pair of word loads
      return load(addr, 4, bytes) && load(addr + 4, 4, bytes + 4);
    }

    const auto hart_id = (addr - MSIP_BASE) / sizeof(msip_t);
    const msip_t res = sim->get_harts().count(hart_id) && (sim->get_harts().at(hart_id)->state.mip->read() & MIP_MSIP);
    read_little_endian_reg(res, addr, len, bytes);
    return true;
  } else if (addr >= MTIMECMP_BASE && addr < MTIME_BASE) {
    const auto hart_id = (addr - MTIMECMP_BASE) / sizeof(mtimecmp_t);
    const mtime_t res = sim->get_harts().count(hart_id) ? mtimecmp[hart_id] : 0;
    read_little_endian_reg(res, addr, len, bytes);
  } else if (addr >= MTIME_BASE && addr < MTIME_BASE + sizeof(mtime_t)) {
    read_little_endian_reg(mtime, addr, len, bytes);
  } else if (addr + len <= CLINT_SIZE) {
    memset(bytes, 0, len);
  } else {
    return false;
  }
  return true;
}

bool clint_t::store(reg_t addr, size_t len, const uint8_t* bytes)
{
  if (len > 8)
    return false;

  if (addr >= MSIP_BASE && addr < MTIMECMP_BASE) {
    if (len == 8) {
      // Implement double-word stores as a pair of word stores
      return store(addr, 4, bytes) && store(addr + 4, 4, bytes + 4);
    }

    if (addr % sizeof(msip_t) == 0) {  // ignore in-between bytes
      msip_t msip = 0;
      write_little_endian_reg(&msip, addr, len, bytes);

      const auto hart_id = (addr - MSIP_BASE) / sizeof(msip_t);
      if (sim->get_harts().count(hart_id))
        sim->get_harts().at(hart_id)->state.mip->backdoor_write_with_mask(MIP_MSIP, msip & 1 ? MIP_MSIP : 0);
    }
  } else if (addr >= MTIMECMP_BASE && addr < MTIME_BASE) {
    const auto hart_id = (addr - MTIMECMP_BASE) / sizeof(mtimecmp_t);
    if (sim->get_harts().count(hart_id))
      write_little_endian_reg(&mtimecmp[hart_id], addr, len, bytes);
  } else if (addr >= MTIME_BASE && addr < MTIME_BASE + sizeof(mtime_t)) {
    write_little_endian_reg(&mtime, addr, len, bytes);
  } else if (addr + len <= CLINT_SIZE) {
    // Do nothing
  } else {
    return false;
  }
  tick(0);
  return true;
}

void clint_t::tick(reg_t rtc_ticks)
{
  if (real_time) {
   struct timeval now;
   uint64_t diff_usecs;

   gettimeofday(&now, NULL);
   diff_usecs = ((now.tv_sec - real_time_ref_secs) * 1000000) + (now.tv_usec - real_time_ref_usecs);
   mtime = diff_usecs * freq_hz / 1000000;
  } else {
    mtime += rtc_ticks;
  }

  for (const auto& [hart_id, hart] : sim->get_harts()) {
    hart->state.time->sync(mtime);
    hart->state.mip->backdoor_write_with_mask(MIP_MTIP, mtime >= mtimecmp[hart_id] ? MIP_MTIP : 0);
  }
}

clint_t* clint_parse_from_fdt(const void* fdt, const sim_t* sim, reg_t* base,
    const std::vector<std::string>& sargs) {
  if (fdt_parse_clint(fdt, base, "riscv,clint0") == 0 || fdt_parse_clint(fdt, base, "sifive,clint0") == 0)
    return new clint_t(sim,
                       sim->CPU_HZ / sim->INSNS_PER_RTC_TICK,
                       sim->get_cfg().real_time_clint);
  else
    return nullptr;
}

std::string clint_generate_dts(const sim_t* sim) {
  std::stringstream s;
  s << std::hex
    << "    clint@" << CLINT_BASE << " {\n"
       "      compatible = \"riscv,clint0\";\n"
       "      interrupts-extended = <" << std::dec;
  for (size_t i = 0; i < sim->get_cfg().nprocs(); i++)
    s << "&CPU" << i << "_intc 3 &CPU" << i << "_intc 7 ";
  reg_t clintbs = CLINT_BASE;
  reg_t clintsz = CLINT_SIZE;
  s << std::hex << ">;\n"
    "      reg = <0x" << (clintbs >> 32) << " 0x" << (clintbs & (uint32_t)-1) <<
    " 0x" << (clintsz >> 32) << " 0x" << (clintsz & (uint32_t)-1) << ">;\n"
    "    };\n";
  return s.str();
}

REGISTER_DEVICE(clint, clint_parse_from_fdt, clint_generate_dts)

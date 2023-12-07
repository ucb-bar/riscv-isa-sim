// See LICENSE for license details.

#include "cfg.h"
#include "mmu.h"
#include "decode.h"
#include "encoding.h"
#include "platform.h"

mem_cfg_t::mem_cfg_t(reg_t base, reg_t size) : base(base), size(size)
{
  assert(mem_cfg_t::check_if_supported(base, size));
}

bool mem_cfg_t::check_if_supported(reg_t base, reg_t size)
{
  // The truth of these conditions should be ensured by whatever is creating
  // the regions in the first place, but we have them here to make sure that
  // we can't end up describing memory regions that don't make sense. They
  // ask that the page size is a multiple of the minimum page size, that the
  // page is aligned to the minimum page size, that the page is non-empty and
  // that the top address is still representable in a reg_t.
  //
  // Note: (base + size == 0) part of the assertion is to handle cases like
  //   { base = 0xffff_ffff_ffff_f000, size: 0x1000 }
  return (size % PGSIZE == 0) &&
         (base % PGSIZE == 0) &&
         (size > 0) &&
         ((base + size > base) || (base + size == 0));
}

cfg_t::cfg_t()
{
  // The default system configuration
  initrd_bounds    = std::make_pair((reg_t)0, (reg_t)0);
  bootargs         = nullptr;
  isa              = DEFAULT_ISA;
  priv             = DEFAULT_PRIV;
  varch            = DEFAULT_VARCH;
  misaligned       = false;
  endianness       = endianness_little;
  pmpregions       = 16;
  pmpgranularity   = (1 << PMP_SHIFT);
  mem_layout       = std::vector<mem_cfg_t>({mem_cfg_t(reg_t(DRAM_BASE), (size_t)2048 << 20)});
  hartids          = std::vector<size_t>({0});
  explicit_hartids = false;
  real_time_clint  = false;
  trigger_count    = 4;
}

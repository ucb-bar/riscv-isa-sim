// See LICENSE for license details

#include "config.h"
#include "vector_unit.h"
#include "processor.h"
#include "arith.h"

void vectorUnit_t::vectorUnit_t::reset()
{
  free(reg_file);
  VLEN = get_vlen();
  ELEN = get_elen();
  reg_file = malloc(NVPR * vlenb);
  memset(reg_file, 0, NVPR * vlenb);

  auto& csrmap = p->get_state()->csrmap;
  csrmap[CSR_VXSAT] = vxsat = std::make_shared<vxsat_csr_t>(CSR_VXSAT);
  csrmap[CSR_VSTART] = vstart = std::make_shared<vector_csr_t>(CSR_VSTART, /*mask*/ VLEN - 1);
  csrmap[CSR_VXRM] = vxrm = std::make_shared<vector_csr_t>(CSR_VXRM, /*mask*/ 0x3ul);
  csrmap[CSR_VL] = vl = std::make_shared<vector_csr_t>(CSR_VL, /*mask*/ 0);
  csrmap[CSR_VTYPE] = vtype = std::make_shared<vector_csr_t>(CSR_VTYPE, /*mask*/ 0);
  csrmap[CSR_VLENB] = std::make_shared<vector_csr_t>(CSR_VLENB, /*mask*/ 0, /*init*/ vlenb);
  assert(VCSR_VXSAT_SHIFT == 0);  // composite_csr_t assumes vxsat begins at bit 0
  csrmap[CSR_VCSR] = std::make_shared<composite_csr_t>(CSR_VCSR, vxrm, vxsat, VCSR_VXRM_SHIFT);

  vtype->write_raw(0, p);
  set_vl(0, 0, 0, -1); // default to illegal configuration
}

reg_t vectorUnit_t::vectorUnit_t::set_vl(int rd, int rs1, reg_t reqVL, reg_t newType)
{
  int new_vlmul = 0;
  if (vtype->read(p) != newType) {
    vsew = 1 << (extract64(newType, 3, 3) + 3);
    new_vlmul = int8_t(extract64(newType, 0, 3) << 5) >> 5;
    vflmul = new_vlmul >= 0 ? 1 << new_vlmul : 1.0 / (1 << -new_vlmul);
    vlmax = (VLEN/vsew) * vflmul;
    vta = extract64(newType, 6, 1);
    vma = extract64(newType, 7, 1);

    vill = !(vflmul >= 0.125 && vflmul <= 8)
           || vsew > std::min(vflmul, 1.0f) * ELEN
           || (newType >> 8) != 0;

    if (vill) {
      vlmax = 0;
      vtype->write_raw(UINT64_MAX << (p->get_xlen() - 1), p);
    } else {
      vtype->write_raw(newType, p);
    }
  }

  // set vl
  if (vlmax == 0) {
    vl->write_raw(0, p);
  } else if (rd == 0 && rs1 == 0) {
    vl->write_raw(std::min(vl->read(p), vlmax), p);
  } else if (rd != 0 && rs1 == 0) {
    vl->write_raw(vlmax, p);
  } else if (rs1 != 0) {
    vl->write_raw(std::min(reqVL, vlmax), p);
  }

  vstart->write_raw(0, p);
  setvl_count++;
  return vl->read(p);
}

template<class T> T& vectorUnit_t::elt(reg_t vReg, reg_t n, bool UNUSED is_write) {
  assert(vsew != 0);
  assert((VLEN >> 3)/sizeof(T) > 0);
  reg_t elts_per_reg = (VLEN >> 3) / (sizeof(T));
  vReg += n / elts_per_reg;
  n = n % elts_per_reg;
#ifdef WORDS_BIGENDIAN
  // "V" spec 0.7.1 requires lower indices to map to lower significant
  // bits when changing SEW, thus we need to index from the end on BE.
  n ^= elts_per_reg - 1;
#endif
  reg_referenced[vReg] = 1;

  if (unlikely(p->get_log_commits_enabled() && is_write))
    p->get_state()->log_reg_write[((vReg) << 4) | 2] = {0, 0};

  T *regStart = (T*)((char*)reg_file + vReg * (VLEN >> 3));
  return regStart[n];
}

// The logic differences between 'elt()' and 'elt_group()' come from
// the fact that, while 'elt()' requires that the element is fully
// contained in a single vector register, the element group may span
// multiple registers in a single register group (LMUL>1).
//
// Notes:
// - We do NOT check that a single element - i.e., the T in the element
//   group type std::array<T, N> - fits within a single register, or that
//   T is smaller or equal to VSEW. Implementations of the instructions
//   sometimes use a different T than what the specification suggests.
//   Instructon implementations should 'require()' what the specification
//   dictates.
// - We do NOT check that 'vReg' is a valid register group, or that
//   'n+1' element groups fit in the register group 'vReg'. It is
//   the responsibility of the caller to validate those preconditions.
template<typename EG> EG&
vectorUnit_t::elt_group(reg_t vReg, reg_t n, bool UNUSED is_write) {
#ifdef WORDS_BIGENDIAN
  fputs("vectorUnit_t::elt_group is not compatible with WORDS_BIGENDIAN setup.\n",
          stderr);
  abort();
#endif
  using T = typename EG::value_type;
  constexpr std::size_t N = std::tuple_size<EG>::value;
  assert(N > 0);

  assert(vsew != 0);
  constexpr reg_t elt_group_size = N * sizeof(T);
  const reg_t reg_group_size = (VLEN >> 3) * vflmul;
  assert(((n + 1) * elt_group_size) <= reg_group_size);

  const reg_t start_byte = n * elt_group_size;
  const reg_t bytes_per_reg = VLEN >> 3;

  // Inclusive first/last register indices.
  const reg_t reg_first = vReg + start_byte / bytes_per_reg;
  const reg_t reg_last = vReg + (start_byte + elt_group_size - 1) / bytes_per_reg;

  // Element groups per register groups
  for (reg_t vidx = reg_first; vidx <= reg_last; ++vidx) {
      reg_referenced[vidx] = 1;

      if (unlikely(p->get_log_commits_enabled() && is_write)) {
          p->get_state()->log_reg_write[(vidx << 4) | 2] = {0, 0};
      }
  }

  return *(EG*)((char*)reg_file + vReg * (VLEN >> 3) + start_byte);
}

template signed char& vectorUnit_t::elt<signed char>(reg_t, reg_t, bool);
template short& vectorUnit_t::elt<short>(reg_t, reg_t, bool);
template int& vectorUnit_t::elt<int>(reg_t, reg_t, bool);
template long& vectorUnit_t::elt<long>(reg_t, reg_t, bool);
template long long& vectorUnit_t::elt<long long>(reg_t, reg_t, bool);
template uint8_t& vectorUnit_t::elt<uint8_t>(reg_t, reg_t, bool);
template uint16_t& vectorUnit_t::elt<uint16_t>(reg_t, reg_t, bool);
template uint32_t& vectorUnit_t::elt<uint32_t>(reg_t, reg_t, bool);
template uint64_t& vectorUnit_t::elt<uint64_t>(reg_t, reg_t, bool);
template float16_t& vectorUnit_t::elt<float16_t>(reg_t, reg_t, bool);
template float32_t& vectorUnit_t::elt<float32_t>(reg_t, reg_t, bool);
template float64_t& vectorUnit_t::elt<float64_t>(reg_t, reg_t, bool);

template EGU32x4_t& vectorUnit_t::elt_group<EGU32x4_t>(reg_t, reg_t, bool);
template EGU32x8_t& vectorUnit_t::elt_group<EGU32x8_t>(reg_t, reg_t, bool);
template EGU64x4_t& vectorUnit_t::elt_group<EGU64x4_t>(reg_t, reg_t, bool);
template EGU8x16_t& vectorUnit_t::elt_group<EGU8x16_t>(reg_t, reg_t, bool);

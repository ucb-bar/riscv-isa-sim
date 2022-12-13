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
  csrmap[CSR_VXSAT] = vxsat = std::make_shared<vxsat_csr_t>(p, CSR_VXSAT);
  csrmap[CSR_VSTART] = vstart = std::make_shared<vector_csr_t>(p, CSR_VSTART, /*mask*/ VLEN - 1);
  csrmap[CSR_VXRM] = vxrm = std::make_shared<vector_csr_t>(p, CSR_VXRM, /*mask*/ 0x3ul);
  csrmap[CSR_VL] = vl = std::make_shared<vector_csr_t>(p, CSR_VL, /*mask*/ 0);
  csrmap[CSR_VTYPE] = vtype = std::make_shared<vector_csr_t>(p, CSR_VTYPE, /*mask*/ 0);
  csrmap[CSR_VLENB] = std::make_shared<vector_csr_t>(p, CSR_VLENB, /*mask*/ 0, /*init*/ vlenb);
  assert(VCSR_VXSAT_SHIFT == 0);  // composite_csr_t assumes vxsat begins at bit 0
  csrmap[CSR_VCSR] = std::make_shared<composite_csr_t>(p, CSR_VCSR, vxrm, vxsat, VCSR_VXRM_SHIFT);

  vtype->write_raw(0);
  set_vl(0, 0, 0, -1); // default to illegal configuration
}

reg_t vectorUnit_t::vectorUnit_t::set_vl(int rd, int rs1, reg_t reqVL, reg_t newType)
{
  int new_vlmul = 0;
  if (vtype->read() != newType) {
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
      vtype->write_raw(UINT64_MAX << (p->get_xlen() - 1));
    } else {
      vtype->write_raw(newType);
    }
  }

  // set vl
  if (vlmax == 0) {
    vl->write_raw(0);
  } else if (rd == 0 && rs1 == 0) {
    vl->write_raw(vl->read() > vlmax ? vlmax : vl->read());
  } else if (rd != 0 && rs1 == 0) {
    vl->write_raw(vlmax);
  } else if (rs1 != 0) {
    vl->write_raw(reqVL > vlmax ? vlmax : reqVL);
  }

  vstart->write_raw(0);
  setvl_count++;
  return vl->read();
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

#ifdef RISCV_ENABLE_COMMITLOG
  if (is_write)
    p->get_state()->log_reg_write[((vReg) << 4) | 2] = {0, 0};
#endif

  T *regStart = (T*)((char*)reg_file + vReg * (VLEN >> 3));
  return regStart[n];
}

template signed char& vectorUnit_t::elt<signed char>(reg_t, reg_t, bool);
template short& vectorUnit_t::elt<short>(reg_t, reg_t, bool);
template int& vectorUnit_t::elt<int>(reg_t, reg_t, bool);
template long& vectorUnit_t::elt<long>(reg_t, reg_t, bool);
template uint8_t& vectorUnit_t::elt<uint8_t>(reg_t, reg_t, bool);
template uint16_t& vectorUnit_t::elt<uint16_t>(reg_t, reg_t, bool);
template uint32_t& vectorUnit_t::elt<uint32_t>(reg_t, reg_t, bool);
template uint64_t& vectorUnit_t::elt<uint64_t>(reg_t, reg_t, bool);
template float16_t& vectorUnit_t::elt<float16_t>(reg_t, reg_t, bool);
template float32_t& vectorUnit_t::elt<float32_t>(reg_t, reg_t, bool);
template float64_t& vectorUnit_t::elt<float64_t>(reg_t, reg_t, bool);

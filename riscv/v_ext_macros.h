// See LICENSE for license details.

#ifndef _RISCV_V_EXT_MACROS_H
#define _RISCV_V_EXT_MACROS_H

#include "vector_unit.h"

//
// vector: masking skip helper
//
#define VI_MASK_VARS \
  const int midx = i / 64; \
  const int mpos = i % 64;

#define VI_LOOP_ELEMENT_SKIP(BODY) \
  VI_MASK_VARS \
  if (insn.v_vm() == 0) { \
    BODY; \
    bool skip = ((P.VU.elt<uint64_t>(0, midx) >> mpos) & 0x1) == 0; \
    if (skip) { \
        continue; \
    } \
  }

#define VI_ELEMENT_SKIP \
  if (i >= vl) { \
    continue; \
  } else if (i < P.VU.vstart->read()) { \
    continue; \
  } else { \
    VI_LOOP_ELEMENT_SKIP(); \
  }

//
// vector: operation and register acccess check helper
//
static inline bool is_overlapped(const int astart, int asize,
                                 const int bstart, int bsize)
{
  asize = asize == 0 ? 1 : asize;
  bsize = bsize == 0 ? 1 : bsize;

  const int aend = astart + asize;
  const int bend = bstart + bsize;

  return std::max(aend, bend) - std::min(astart, bstart) < asize + bsize;
}

static inline bool is_overlapped_widen(const int astart, int asize,
                                       const int bstart, int bsize)
{
  asize = asize == 0 ? 1 : asize;
  bsize = bsize == 0 ? 1 : bsize;

  const int aend = astart + asize;
  const int bend = bstart + bsize;

  if (astart < bstart &&
      is_overlapped(astart, asize, bstart, bsize) &&
      !is_overlapped(astart, asize, bstart + bsize, bsize)) {
    return false;
  } else  {
    return std::max(aend, bend) - std::min(astart, bstart) < asize + bsize;
  }
}

static inline bool is_aligned(const unsigned val, const unsigned pos)
{
  return pos ? (val & (pos - 1)) == 0 : true;
}

#define VI_NARROW_CHECK_COMMON \
  require_vector(true); \
  require(P.VU.vflmul <= 4); \
  require(P.VU.vsew * 2 <= P.VU.ELEN); \
  require_align(insn.rs2(), P.VU.vflmul * 2); \
  require_align(insn.rd(), P.VU.vflmul); \
  require_vm; \

#define VI_WIDE_CHECK_COMMON \
  require_vector(true); \
  require(P.VU.vflmul <= 4); \
  require(P.VU.vsew * 2 <= P.VU.ELEN); \
  require_align(insn.rd(), P.VU.vflmul * 2); \
  require_vm; \

#define VI_CHECK_ST_INDEX(elt_width) \
  require_vector(false); \
  require(elt_width <= P.VU.ELEN); \
  float vemul = ((float)elt_width / P.VU.vsew * P.VU.vflmul); \
  require(vemul >= 0.125 && vemul <= 8); \
  reg_t UNUSED emul = vemul < 1 ? 1 : vemul; \
  reg_t flmul = P.VU.vflmul < 1 ? 1 : P.VU.vflmul; \
  require_align(insn.rd(), P.VU.vflmul); \
  require_align(insn.rs2(), vemul); \
  require((nf * flmul) <= (NVPR / 4) && \
          (insn.rd() + nf * flmul) <= NVPR); \

#define VI_CHECK_LD_INDEX(elt_width) \
  VI_CHECK_ST_INDEX(elt_width); \
  for (reg_t idx = 0; idx < nf; ++idx) { \
    reg_t flmul = P.VU.vflmul < 1 ? 1 : P.VU.vflmul; \
    reg_t seg_vd = insn.rd() + flmul * idx; \
    if (elt_width > P.VU.vsew) { \
      if (seg_vd != insn.rs2()) \
        require_noover(seg_vd, P.VU.vflmul, insn.rs2(), vemul); \
    } else if (elt_width < P.VU.vsew) { \
      if (vemul < 1) { \
        require_noover(seg_vd, P.VU.vflmul, insn.rs2(), vemul); \
      } else { \
        require_noover_widen(seg_vd, P.VU.vflmul, insn.rs2(), vemul); \
      } \
    } \
    if (nf >= 2) { \
      require_noover(seg_vd, P.VU.vflmul, insn.rs2(), vemul); \
    } \
  } \
  require_vm; \

#define VI_CHECK_MSS(is_vs1) \
  if (insn.rd() != insn.rs2()) \
    require_noover(insn.rd(), 1, insn.rs2(), P.VU.vflmul); \
  require_align(insn.rs2(), P.VU.vflmul); \
  if (is_vs1) { \
    if (insn.rd() != insn.rs1()) \
      require_noover(insn.rd(), 1, insn.rs1(), P.VU.vflmul); \
    require_align(insn.rs1(), P.VU.vflmul); \
  } \

#define VI_CHECK_SSS(is_vs1) \
  require_vm; \
  if (P.VU.vflmul > 1) { \
    require_align(insn.rd(), P.VU.vflmul); \
    require_align(insn.rs2(), P.VU.vflmul); \
    if (is_vs1) { \
      require_align(insn.rs1(), P.VU.vflmul); \
    } \
  }

#define VI_CHECK_STORE(elt_width, is_mask_ldst) \
  require_vector(false); \
  reg_t veew = is_mask_ldst ? 1 : sizeof(elt_width##_t) * 8; \
  float vemul = is_mask_ldst ? 1 : ((float)veew / P.VU.vsew * P.VU.vflmul); \
  reg_t emul = vemul < 1 ? 1 : vemul; \
  require(vemul >= 0.125 && vemul <= 8); \
  require_align(insn.rd(), vemul); \
  require((nf * emul) <= (NVPR / 4) && \
          (insn.rd() + nf * emul) <= NVPR); \
  require(veew <= P.VU.ELEN); \

#define VI_CHECK_LOAD(elt_width, is_mask_ldst) \
  VI_CHECK_STORE(elt_width, is_mask_ldst); \
  require_vm; \

#define VI_CHECK_DSS(is_vs1) \
  VI_WIDE_CHECK_COMMON; \
  require_align(insn.rs2(), P.VU.vflmul); \
  if (P.VU.vflmul < 1) { \
    require_noover(insn.rd(), P.VU.vflmul * 2, insn.rs2(), P.VU.vflmul); \
  } else { \
    require_noover_widen(insn.rd(), P.VU.vflmul * 2, insn.rs2(), P.VU.vflmul); \
  } \
  if (is_vs1) { \
    require_align(insn.rs1(), P.VU.vflmul); \
    if (P.VU.vflmul < 1) { \
      require_noover(insn.rd(), P.VU.vflmul * 2, insn.rs1(), P.VU.vflmul); \
    } else { \
      require_noover_widen(insn.rd(), P.VU.vflmul * 2, insn.rs1(), P.VU.vflmul); \
    } \
  }

#define VI_CHECK_DDS(is_rs) \
  VI_WIDE_CHECK_COMMON; \
  require_align(insn.rs2(), P.VU.vflmul * 2); \
  if (is_rs) { \
     require_align(insn.rs1(), P.VU.vflmul); \
    if (P.VU.vflmul < 1) { \
      require_noover(insn.rd(), P.VU.vflmul * 2, insn.rs1(), P.VU.vflmul); \
    } else { \
      require_noover_widen(insn.rd(), P.VU.vflmul * 2, insn.rs1(), P.VU.vflmul); \
    } \
  }

#define VI_CHECK_SDS(is_vs1) \
  VI_NARROW_CHECK_COMMON; \
  if (insn.rd() != insn.rs2()) \
    require_noover(insn.rd(), P.VU.vflmul, insn.rs2(), P.VU.vflmul * 2); \
  if (is_vs1) \
    require_align(insn.rs1(), P.VU.vflmul); \

#define VI_CHECK_REDUCTION(is_wide) \
  require_vector(true); \
  if (is_wide) { \
    require(P.VU.vsew * 2 <= P.VU.ELEN); \
  } \
  require_align(insn.rs2(), P.VU.vflmul); \
  require(P.VU.vstart->read() == 0); \

#define VI_CHECK_SLIDE(is_over) \
  require_align(insn.rs2(), P.VU.vflmul); \
  require_align(insn.rd(), P.VU.vflmul); \
  require_vm; \
  if (is_over) \
    require(insn.rd() != insn.rs2()); \

//
// vector: loop header and end helper
//
#define VI_GENERAL_LOOP_BASE \
  require(P.VU.vsew >= e8 && P.VU.vsew <= e64); \
  require_vector(true); \
  reg_t vl = P.VU.vl->read(); \
  reg_t UNUSED sew = P.VU.vsew; \
  reg_t rd_num = insn.rd(); \
  reg_t UNUSED rs1_num = insn.rs1(); \
  reg_t rs2_num = insn.rs2(); \
  for (reg_t i = P.VU.vstart->read(); i < vl; ++i) {

#define VI_LOOP_BASE \
    VI_GENERAL_LOOP_BASE \
    VI_LOOP_ELEMENT_SKIP();

#define VI_LOOP_END \
  } \
  P.VU.vstart->write(0);

#define VI_LOOP_REDUCTION_END(x) \
  } \
  if (vl > 0) { \
    vd_0_des = vd_0_res; \
  } \
  P.VU.vstart->write(0);

#define VI_LOOP_CARRY_BASE \
  VI_GENERAL_LOOP_BASE \
  VI_MASK_VARS \
  auto v0 = P.VU.elt<uint64_t>(0, midx); \
  const uint64_t mmask = UINT64_C(1) << mpos; \
  const uint128_t op_mask = (UINT64_MAX >> (64 - sew)); \
  uint64_t carry = insn.v_vm() == 0 ? (v0 >> mpos) & 0x1 : 0; \
  uint128_t res = 0; \
  auto &vd = P.VU.elt<uint64_t>(rd_num, midx, true);

#define VI_LOOP_CARRY_END \
    vd = (vd & ~mmask) | (((res) << mpos) & mmask); \
  } \
  P.VU.vstart->write(0);
#define VI_LOOP_WITH_CARRY_BASE \
  VI_GENERAL_LOOP_BASE \
  VI_MASK_VARS \
  auto &v0 = P.VU.elt<uint64_t>(0, midx); \
  const uint128_t op_mask = (UINT64_MAX >> (64 - sew)); \
  uint64_t carry = (v0 >> mpos) & 0x1;

#define VI_LOOP_CMP_BASE \
  require(P.VU.vsew >= e8 && P.VU.vsew <= e64); \
  require_vector(true); \
  reg_t vl = P.VU.vl->read(); \
  reg_t sew = P.VU.vsew; \
  reg_t UNUSED rd_num = insn.rd(); \
  reg_t UNUSED rs1_num = insn.rs1(); \
  reg_t rs2_num = insn.rs2(); \
  for (reg_t i = P.VU.vstart->read(); i < vl; ++i) { \
    VI_LOOP_ELEMENT_SKIP(); \
    uint64_t mmask = UINT64_C(1) << mpos; \
    uint64_t &vdi = P.VU.elt<uint64_t>(insn.rd(), midx, true); \
    uint64_t res = 0;

#define VI_LOOP_CMP_END \
    vdi = (vdi & ~mmask) | (((res) << mpos) & mmask); \
  } \
  P.VU.vstart->write(0);

#define VI_LOOP_MASK(op) \
  require(P.VU.vsew <= e64); \
  require_vector(true); \
  reg_t vl = P.VU.vl->read(); \
  for (reg_t i = P.VU.vstart->read(); i < vl; ++i) { \
    int midx = i / 64; \
    int mpos = i % 64; \
    uint64_t mmask = UINT64_C(1) << mpos; \
    uint64_t vs2 = P.VU.elt<uint64_t>(insn.rs2(), midx); \
    uint64_t vs1 = P.VU.elt<uint64_t>(insn.rs1(), midx); \
    uint64_t &res = P.VU.elt<uint64_t>(insn.rd(), midx, true); \
    res = (res & ~mmask) | ((op) & (1ULL << mpos)); \
  } \
  P.VU.vstart->write(0);

#define VI_LOOP_NSHIFT_BASE \
  VI_GENERAL_LOOP_BASE; \
  VI_LOOP_ELEMENT_SKIP({ \
    require(!(insn.rd() == 0 && P.VU.vflmul > 1)); \
  });

#define INT_ROUNDING(result, xrm, gb) \
  do { \
    const uint64_t lsb = 1UL << (gb); \
    const uint64_t lsb_half = lsb >> 1; \
    switch (xrm) { \
      case VRM::RNU: \
        result += lsb_half; \
        break; \
      case VRM::RNE: \
        if ((result & lsb_half) && ((result & (lsb_half - 1)) || (result & lsb))) \
          result += lsb; \
        break; \
      case VRM::RDN: \
        break; \
      case VRM::ROD: \
        if (result & (lsb - 1)) \
          result |= lsb; \
        break; \
      case VRM::INVALID_RM: \
        assert(true); \
    } \
  } while (0)

//
// vector: integer and masking operand access helper
//
#define VXI_PARAMS(x) \
  type_sew_t<x>::type &vd = P.VU.elt<type_sew_t<x>::type>(rd_num, i, true); \
  type_sew_t<x>::type vs1 = P.VU.elt<type_sew_t<x>::type>(rs1_num, i); \
  type_sew_t<x>::type vs2 = P.VU.elt<type_sew_t<x>::type>(rs2_num, i); \
  type_sew_t<x>::type rs1 = (type_sew_t<x>::type)RS1; \
  type_sew_t<x>::type simm5 = (type_sew_t<x>::type)insn.v_simm5();

#define VV_U_PARAMS(x) \
  type_usew_t<x>::type &vd = P.VU.elt<type_usew_t<x>::type>(rd_num, i, true); \
  type_usew_t<x>::type vs1 = P.VU.elt<type_usew_t<x>::type>(rs1_num, i); \
  type_usew_t<x>::type vs2 = P.VU.elt<type_usew_t<x>::type>(rs2_num, i);

#define VX_U_PARAMS(x) \
  type_usew_t<x>::type &vd = P.VU.elt<type_usew_t<x>::type>(rd_num, i, true); \
  type_usew_t<x>::type rs1 = (type_usew_t<x>::type)RS1; \
  type_usew_t<x>::type vs2 = P.VU.elt<type_usew_t<x>::type>(rs2_num, i);

#define VI_U_PARAMS(x) \
  type_usew_t<x>::type &vd = P.VU.elt<type_usew_t<x>::type>(rd_num, i, true); \
  type_usew_t<x>::type UNUSED zimm5 = (type_usew_t<x>::type)insn.v_zimm5(); \
  type_usew_t<x>::type vs2 = P.VU.elt<type_usew_t<x>::type>(rs2_num, i);

#define VV_PARAMS(x) \
  type_sew_t<x>::type UNUSED &vd = P.VU.elt<type_sew_t<x>::type>(rd_num, i, true); \
  type_sew_t<x>::type vs1 = P.VU.elt<type_sew_t<x>::type>(rs1_num, i); \
  type_sew_t<x>::type UNUSED vs2 = P.VU.elt<type_sew_t<x>::type>(rs2_num, i);

#define VX_PARAMS(x) \
  type_sew_t<x>::type UNUSED &vd = P.VU.elt<type_sew_t<x>::type>(rd_num, i, true); \
  type_sew_t<x>::type rs1 = (type_sew_t<x>::type)RS1; \
  type_sew_t<x>::type UNUSED vs2 = P.VU.elt<type_sew_t<x>::type>(rs2_num, i);

#define VI_PARAMS(x) \
  type_sew_t<x>::type &vd = P.VU.elt<type_sew_t<x>::type>(rd_num, i, true); \
  type_sew_t<x>::type simm5 = (type_sew_t<x>::type)insn.v_simm5(); \
  type_sew_t<x>::type UNUSED vs2 = P.VU.elt<type_sew_t<x>::type>(rs2_num, i);

#define XV_PARAMS(x) \
  type_sew_t<x>::type &vd = P.VU.elt<type_sew_t<x>::type>(rd_num, i, true); \
  type_usew_t<x>::type vs2 = P.VU.elt<type_usew_t<x>::type>(rs2_num, RS1);

#define VV_SU_PARAMS(x) \
  type_sew_t<x>::type &vd = P.VU.elt<type_sew_t<x>::type>(rd_num, i, true); \
  type_usew_t<x>::type vs1 = P.VU.elt<type_usew_t<x>::type>(rs1_num, i); \
  type_sew_t<x>::type vs2 = P.VU.elt<type_sew_t<x>::type>(rs2_num, i);

#define VX_SU_PARAMS(x) \
  type_sew_t<x>::type &vd = P.VU.elt<type_sew_t<x>::type>(rd_num, i, true); \
  type_usew_t<x>::type rs1 = (type_usew_t<x>::type)RS1; \
  type_sew_t<x>::type vs2 = P.VU.elt<type_sew_t<x>::type>(rs2_num, i);

#define VV_UCMP_PARAMS(x) \
  type_usew_t<x>::type vs1 = P.VU.elt<type_usew_t<x>::type>(rs1_num, i); \
  type_usew_t<x>::type vs2 = P.VU.elt<type_usew_t<x>::type>(rs2_num, i);

#define VX_UCMP_PARAMS(x) \
  type_usew_t<x>::type rs1 = (type_usew_t<x>::type)RS1; \
  type_usew_t<x>::type vs2 = P.VU.elt<type_usew_t<x>::type>(rs2_num, i);

#define VI_UCMP_PARAMS(x) \
  type_usew_t<x>::type vs2 = P.VU.elt<type_usew_t<x>::type>(rs2_num, i);

#define VV_CMP_PARAMS(x) \
  type_sew_t<x>::type vs1 = P.VU.elt<type_sew_t<x>::type>(rs1_num, i); \
  type_sew_t<x>::type vs2 = P.VU.elt<type_sew_t<x>::type>(rs2_num, i);

#define VX_CMP_PARAMS(x) \
  type_sew_t<x>::type rs1 = (type_sew_t<x>::type)RS1; \
  type_sew_t<x>::type vs2 = P.VU.elt<type_sew_t<x>::type>(rs2_num, i);

#define VI_CMP_PARAMS(x) \
  type_sew_t<x>::type simm5 = (type_sew_t<x>::type)insn.v_simm5(); \
  type_sew_t<x>::type vs2 = P.VU.elt<type_sew_t<x>::type>(rs2_num, i);

#define VI_XI_SLIDEDOWN_PARAMS(x, off) \
  auto &vd = P.VU.elt<type_sew_t<x>::type>(rd_num, i, true); \
  auto vs2 = P.VU.elt<type_sew_t<x>::type>(rs2_num, i + off);

#define VI_XI_SLIDEUP_PARAMS(x, offset) \
  auto &vd = P.VU.elt<type_sew_t<x>::type>(rd_num, i, true); \
  auto vs2 = P.VU.elt<type_sew_t<x>::type>(rs2_num, i - offset);

#define VI_NARROW_PARAMS(sew1, sew2) \
  auto &vd = P.VU.elt<type_usew_t<sew1>::type>(rd_num, i, true); \
  auto UNUSED vs2_u = P.VU.elt<type_usew_t<sew2>::type>(rs2_num, i); \
  auto UNUSED vs2 = P.VU.elt<type_sew_t<sew2>::type>(rs2_num, i); \
  auto zimm5 = (type_usew_t<sew1>::type)insn.v_zimm5();

#define VX_NARROW_PARAMS(sew1, sew2) \
  auto &vd = P.VU.elt<type_usew_t<sew1>::type>(rd_num, i, true); \
  auto UNUSED vs2_u = P.VU.elt<type_usew_t<sew2>::type>(rs2_num, i); \
  auto UNUSED vs2 = P.VU.elt<type_sew_t<sew2>::type>(rs2_num, i); \
  auto rs1 = (type_sew_t<sew1>::type)RS1;

#define VV_NARROW_PARAMS(sew1, sew2) \
  auto &vd = P.VU.elt<type_usew_t<sew1>::type>(rd_num, i, true); \
  auto UNUSED vs2_u = P.VU.elt<type_usew_t<sew2>::type>(rs2_num, i); \
  auto UNUSED vs2 = P.VU.elt<type_sew_t<sew2>::type>(rs2_num, i); \
  auto vs1 = P.VU.elt<type_sew_t<sew1>::type>(rs1_num, i);

#define XI_CARRY_PARAMS(x) \
  auto vs2 = P.VU.elt<type_sew_t<x>::type>(rs2_num, i); \
  auto UNUSED rs1 = (type_sew_t<x>::type)RS1; \
  auto UNUSED simm5 = (type_sew_t<x>::type)insn.v_simm5(); \

#define VV_CARRY_PARAMS(x) \
  auto vs2 = P.VU.elt<type_sew_t<x>::type>(rs2_num, i); \
  auto vs1 = P.VU.elt<type_sew_t<x>::type>(rs1_num, i); \

#define XI_WITH_CARRY_PARAMS(x) \
  auto vs2 = P.VU.elt<type_sew_t<x>::type>(rs2_num, i); \
  auto UNUSED rs1 = (type_sew_t<x>::type)RS1; \
  auto UNUSED simm5 = (type_sew_t<x>::type)insn.v_simm5(); \
  auto &vd = P.VU.elt<type_sew_t<x>::type>(rd_num, i, true);

#define VV_WITH_CARRY_PARAMS(x) \
  auto vs2 = P.VU.elt<type_sew_t<x>::type>(rs2_num, i); \
  auto vs1 = P.VU.elt<type_sew_t<x>::type>(rs1_num, i); \
  auto &vd = P.VU.elt<type_sew_t<x>::type>(rd_num, i, true);

#define VFP_V_PARAMS(width) \
  float##width##_t &vd = P.VU.elt<float##width##_t>(rd_num, i, true); \
  float##width##_t vs2 = P.VU.elt<float##width##_t>(rs2_num, i);

#define VFP_VV_PARAMS(width) \
  float##width##_t UNUSED &vd = P.VU.elt<float##width##_t>(rd_num, i, true); \
  float##width##_t vs1 = P.VU.elt<float##width##_t>(rs1_num, i); \
  float##width##_t vs2 = P.VU.elt<float##width##_t>(rs2_num, i);

#define VFP_VF_PARAMS(width) \
  float##width##_t UNUSED &vd = P.VU.elt<float##width##_t>(rd_num, i, true); \
  float##width##_t rs1 = f##width(READ_FREG(rs1_num)); \
  float##width##_t vs2 = P.VU.elt<float##width##_t>(rs2_num, i);

#define CVT_FP_TO_FP_PARAMS(from_width, to_width) \
  auto vs2 = P.VU.elt<float##from_width##_t>(rs2_num, i); \
  auto &vd = P.VU.elt<float##to_width##_t>(rd_num, i, true);

#define CVT_INT_TO_FP_PARAMS(from_width, to_width, sign) \
  auto vs2 = P.VU.elt<sign##from_width##_t>(rs2_num, i); \
  auto &vd = P.VU.elt<float##to_width##_t>(rd_num, i, true);

#define CVT_FP_TO_INT_PARAMS(from_width, to_width, sign) \
  auto vs2 = P.VU.elt<float##from_width##_t>(rs2_num, i); \
  auto &vd = P.VU.elt<sign##to_width##_t>(rd_num, i, true);

//
// vector: integer and masking operation loop
//

#define INSNS_BASE(PARAMS, BODY) \
  if (sew == e8) { \
    PARAMS(e8); \
    BODY; \
  } else if (sew == e16) { \
    PARAMS(e16); \
    BODY; \
  } else if (sew == e32) { \
    PARAMS(e32); \
    BODY; \
  } else if (sew == e64) { \
    PARAMS(e64); \
    BODY; \
  }

// comparision result to masking register
#define VI_LOOP_CMP_BODY(PARAMS, BODY) \
  VI_LOOP_CMP_BASE \
  INSNS_BASE(PARAMS, BODY) \
  VI_LOOP_CMP_END

#define VI_VV_LOOP_CMP(BODY) \
  VI_CHECK_MSS(true); \
  VI_LOOP_CMP_BODY(VV_CMP_PARAMS, BODY)

#define VI_VX_LOOP_CMP(BODY) \
  VI_CHECK_MSS(false); \
  VI_LOOP_CMP_BODY(VX_CMP_PARAMS, BODY)

#define VI_VI_LOOP_CMP(BODY) \
  VI_CHECK_MSS(false); \
  VI_LOOP_CMP_BODY(VI_CMP_PARAMS, BODY)

#define VI_VV_ULOOP_CMP(BODY) \
  VI_CHECK_MSS(true); \
  VI_LOOP_CMP_BODY(VV_UCMP_PARAMS, BODY)

#define VI_VX_ULOOP_CMP(BODY) \
  VI_CHECK_MSS(false); \
  VI_LOOP_CMP_BODY(VX_UCMP_PARAMS, BODY)

#define VI_VI_ULOOP_CMP(BODY) \
  VI_CHECK_MSS(false); \
  VI_LOOP_CMP_BODY(VI_UCMP_PARAMS, BODY)

// merge and copy loop
#define VI_MERGE_VARS \
  VI_MASK_VARS \
  bool UNUSED use_first = (P.VU.elt<uint64_t>(0, midx) >> mpos) & 0x1;

#define VI_MERGE_LOOP_BASE \
  VI_GENERAL_LOOP_BASE \
  VI_MERGE_VARS

#define VI_VV_MERGE_LOOP(BODY) \
  VI_CHECK_SSS(true); \
  VI_MERGE_LOOP_BASE \
  if (sew == e8) { \
    VV_PARAMS(e8); \
    BODY; \
  } else if (sew == e16) { \
    VV_PARAMS(e16); \
    BODY; \
  } else if (sew == e32) { \
    VV_PARAMS(e32); \
    BODY; \
  } else if (sew == e64) { \
    VV_PARAMS(e64); \
    BODY; \
  } \
  VI_LOOP_END

#define VI_VX_MERGE_LOOP(BODY) \
  VI_CHECK_SSS(false); \
  VI_MERGE_LOOP_BASE \
  if (sew == e8) { \
    VX_PARAMS(e8); \
    BODY; \
  } else if (sew == e16) { \
    VX_PARAMS(e16); \
    BODY; \
  } else if (sew == e32) { \
    VX_PARAMS(e32); \
    BODY; \
  } else if (sew == e64) { \
    VX_PARAMS(e64); \
    BODY; \
  } \
  VI_LOOP_END 

#define VI_VI_MERGE_LOOP(BODY) \
  VI_CHECK_SSS(false); \
  VI_MERGE_LOOP_BASE \
  if (sew == e8) { \
    VI_PARAMS(e8); \
    BODY; \
  } else if (sew == e16) { \
    VI_PARAMS(e16); \
    BODY; \
  } else if (sew == e32) { \
    VI_PARAMS(e32); \
    BODY; \
  } else if (sew == e64) { \
    VI_PARAMS(e64); \
    BODY; \
  } \
  VI_LOOP_END

#define VI_VF_MERGE_LOOP(BODY) \
  VI_CHECK_SSS(false); \
  VI_VFP_COMMON \
  for (reg_t i = P.VU.vstart->read(); i < vl; ++i) { \
  VI_MERGE_VARS \
  if (P.VU.vsew == e16) { \
    VFP_VF_PARAMS(16); \
    BODY; \
  } else if (P.VU.vsew == e32) { \
    VFP_VF_PARAMS(32); \
    BODY; \
  } else if (P.VU.vsew == e64) { \
    VFP_VF_PARAMS(64); \
    BODY; \
  } \
  VI_LOOP_END

// reduction loop - signed
#define VI_LOOP_REDUCTION_BASE(x) \
  require(x >= e8 && x <= e64); \
  reg_t vl = P.VU.vl->read(); \
  reg_t rd_num = insn.rd(); \
  reg_t rs1_num = insn.rs1(); \
  reg_t rs2_num = insn.rs2(); \
  auto &vd_0_des = P.VU.elt<type_sew_t<x>::type>(rd_num, 0, true); \
  auto vd_0_res = P.VU.elt<type_sew_t<x>::type>(rs1_num, 0); \
  for (reg_t i = P.VU.vstart->read(); i < vl; ++i) { \
    VI_LOOP_ELEMENT_SKIP(); \
    auto vs2 = P.VU.elt<type_sew_t<x>::type>(rs2_num, i); \

#define REDUCTION_LOOP(x, BODY) \
  VI_LOOP_REDUCTION_BASE(x) \
  BODY; \
  VI_LOOP_REDUCTION_END(x)

#define VI_VV_LOOP_REDUCTION(BODY) \
  VI_CHECK_REDUCTION(false); \
  reg_t sew = P.VU.vsew; \
  if (sew == e8) { \
    REDUCTION_LOOP(e8, BODY) \
  } else if (sew == e16) { \
    REDUCTION_LOOP(e16, BODY) \
  } else if (sew == e32) { \
    REDUCTION_LOOP(e32, BODY) \
  } else if (sew == e64) { \
    REDUCTION_LOOP(e64, BODY) \
  }

// reduction loop - unsigned
#define VI_ULOOP_REDUCTION_BASE(x) \
  require(x >= e8 && x <= e64); \
  reg_t vl = P.VU.vl->read(); \
  reg_t rd_num = insn.rd(); \
  reg_t rs1_num = insn.rs1(); \
  reg_t rs2_num = insn.rs2(); \
  auto &vd_0_des = P.VU.elt<type_usew_t<x>::type>(rd_num, 0, true); \
  auto vd_0_res = P.VU.elt<type_usew_t<x>::type>(rs1_num, 0); \
  for (reg_t i = P.VU.vstart->read(); i < vl; ++i) { \
    VI_LOOP_ELEMENT_SKIP(); \
    auto vs2 = P.VU.elt<type_usew_t<x>::type>(rs2_num, i);

#define REDUCTION_ULOOP(x, BODY) \
  VI_ULOOP_REDUCTION_BASE(x) \
  BODY; \
  VI_LOOP_REDUCTION_END(x)

#define VI_VV_ULOOP_REDUCTION(BODY) \
  VI_CHECK_REDUCTION(false); \
  reg_t sew = P.VU.vsew; \
  if (sew == e8) { \
    REDUCTION_ULOOP(e8, BODY) \
  } else if (sew == e16) { \
    REDUCTION_ULOOP(e16, BODY) \
  } else if (sew == e32) { \
    REDUCTION_ULOOP(e32, BODY) \
  } else if (sew == e64) { \
    REDUCTION_ULOOP(e64, BODY) \
  }

// genearl VXI signed/unsigned loop
#define VI_VV_ULOOP(BODY) \
  VI_CHECK_SSS(true) \
  VI_LOOP_BASE \
  if (sew == e8) { \
    VV_U_PARAMS(e8); \
    BODY; \
  } else if (sew == e16) { \
    VV_U_PARAMS(e16); \
    BODY; \
  } else if (sew == e32) { \
    VV_U_PARAMS(e32); \
    BODY; \
  } else if (sew == e64) { \
    VV_U_PARAMS(e64); \
    BODY; \
  } \
  VI_LOOP_END 

#define VI_VV_LOOP(BODY) \
  VI_CHECK_SSS(true) \
  VI_LOOP_BASE \
  if (sew == e8) { \
    VV_PARAMS(e8); \
    BODY; \
  } else if (sew == e16) { \
    VV_PARAMS(e16); \
    BODY; \
  } else if (sew == e32) { \
    VV_PARAMS(e32); \
    BODY; \
  } else if (sew == e64) { \
    VV_PARAMS(e64); \
    BODY; \
  } \
  VI_LOOP_END 

#define VI_VX_ULOOP(BODY) \
  VI_CHECK_SSS(false) \
  VI_LOOP_BASE \
  if (sew == e8) { \
    VX_U_PARAMS(e8); \
    BODY; \
  } else if (sew == e16) { \
    VX_U_PARAMS(e16); \
    BODY; \
  } else if (sew == e32) { \
    VX_U_PARAMS(e32); \
    BODY; \
  } else if (sew == e64) { \
    VX_U_PARAMS(e64); \
    BODY; \
  } \
  VI_LOOP_END 

#define VI_VX_LOOP(BODY) \
  VI_CHECK_SSS(false) \
  VI_LOOP_BASE \
  if (sew == e8) { \
    VX_PARAMS(e8); \
    BODY; \
  } else if (sew == e16) { \
    VX_PARAMS(e16); \
    BODY; \
  } else if (sew == e32) { \
    VX_PARAMS(e32); \
    BODY; \
  } else if (sew == e64) { \
    VX_PARAMS(e64); \
    BODY; \
  } \
  VI_LOOP_END 

#define VI_VI_ULOOP(BODY) \
  VI_CHECK_SSS(false) \
  VI_LOOP_BASE \
  if (sew == e8) { \
    VI_U_PARAMS(e8); \
    BODY; \
  } else if (sew == e16) { \
    VI_U_PARAMS(e16); \
    BODY; \
  } else if (sew == e32) { \
    VI_U_PARAMS(e32); \
    BODY; \
  } else if (sew == e64) { \
    VI_U_PARAMS(e64); \
    BODY; \
  } \
  VI_LOOP_END 

#define VI_VI_LOOP(BODY) \
  VI_CHECK_SSS(false) \
  VI_LOOP_BASE \
  if (sew == e8) { \
    VI_PARAMS(e8); \
    BODY; \
  } else if (sew == e16) { \
    VI_PARAMS(e16); \
    BODY; \
  } else if (sew == e32) { \
    VI_PARAMS(e32); \
    BODY; \
  } else if (sew == e64) { \
    VI_PARAMS(e64); \
    BODY; \
  } \
  VI_LOOP_END 

// signed unsigned operation loop (e.g. mulhsu)
#define VI_VV_SU_LOOP(BODY) \
  VI_CHECK_SSS(true) \
  VI_LOOP_BASE \
  if (sew == e8) { \
    VV_SU_PARAMS(e8); \
    BODY; \
  } else if (sew == e16) { \
    VV_SU_PARAMS(e16); \
    BODY; \
  } else if (sew == e32) { \
    VV_SU_PARAMS(e32); \
    BODY; \
  } else if (sew == e64) { \
    VV_SU_PARAMS(e64); \
    BODY; \
  } \
  VI_LOOP_END 

#define VI_VX_SU_LOOP(BODY) \
  VI_CHECK_SSS(false) \
  VI_LOOP_BASE \
  if (sew == e8) { \
    VX_SU_PARAMS(e8); \
    BODY; \
  } else if (sew == e16) { \
    VX_SU_PARAMS(e16); \
    BODY; \
  } else if (sew == e32) { \
    VX_SU_PARAMS(e32); \
    BODY; \
  } else if (sew == e64) { \
    VX_SU_PARAMS(e64); \
    BODY; \
  } \
  VI_LOOP_END 

// narrow operation loop
#define VI_VV_LOOP_NARROW(BODY) \
  VI_CHECK_SDS(true); \
  VI_LOOP_BASE \
  if (sew == e8) { \
    VV_NARROW_PARAMS(e8, e16) \
    BODY; \
  } else if (sew == e16) { \
    VV_NARROW_PARAMS(e16, e32) \
    BODY; \
  } else if (sew == e32) { \
    VV_NARROW_PARAMS(e32, e64) \
    BODY; \
  } \
  VI_LOOP_END

#define VI_VX_LOOP_NARROW(BODY) \
  VI_CHECK_SDS(false); \
  VI_LOOP_BASE \
  if (sew == e8) { \
    VX_NARROW_PARAMS(e8, e16) \
    BODY; \
  } else if (sew == e16) { \
    VX_NARROW_PARAMS(e16, e32) \
    BODY; \
  } else if (sew == e32) { \
    VX_NARROW_PARAMS(e32, e64) \
    BODY; \
  } \
  VI_LOOP_END

#define VI_VI_LOOP_NARROW(BODY) \
  VI_CHECK_SDS(false); \
  VI_LOOP_BASE \
  if (sew == e8) { \
    VI_NARROW_PARAMS(e8, e16) \
    BODY; \
  } else if (sew == e16) { \
    VI_NARROW_PARAMS(e16, e32) \
    BODY; \
  } else if (sew == e32) { \
    VI_NARROW_PARAMS(e32, e64) \
    BODY; \
  } \
  VI_LOOP_END

#define VI_VI_LOOP_NSHIFT(BODY) \
  VI_CHECK_SDS(false); \
  VI_LOOP_NSHIFT_BASE \
  if (sew == e8) { \
    VI_NARROW_PARAMS(e8, e16) \
    BODY; \
  } else if (sew == e16) { \
    VI_NARROW_PARAMS(e16, e32) \
    BODY; \
  } else if (sew == e32) { \
    VI_NARROW_PARAMS(e32, e64) \
    BODY; \
  } \
  VI_LOOP_END

#define VI_VX_LOOP_NSHIFT(BODY) \
  VI_CHECK_SDS(false); \
  VI_LOOP_NSHIFT_BASE \
  if (sew == e8) { \
    VX_NARROW_PARAMS(e8, e16) \
    BODY; \
  } else if (sew == e16) { \
    VX_NARROW_PARAMS(e16, e32) \
    BODY; \
  } else if (sew == e32) { \
    VX_NARROW_PARAMS(e32, e64) \
    BODY; \
  } \
  VI_LOOP_END

#define VI_VV_LOOP_NSHIFT(BODY) \
  VI_CHECK_SDS(true); \
  VI_LOOP_NSHIFT_BASE \
  if (sew == e8) { \
    VV_NARROW_PARAMS(e8, e16) \
    BODY; \
  } else if (sew == e16) { \
    VV_NARROW_PARAMS(e16, e32) \
    BODY; \
  } else if (sew == e32) { \
    VV_NARROW_PARAMS(e32, e64) \
    BODY; \
  } \
  VI_LOOP_END

// widen operation loop
#define VI_VV_LOOP_WIDEN(BODY) \
  VI_LOOP_BASE \
  if (sew == e8) { \
    VV_PARAMS(e8); \
    BODY; \
  } else if (sew == e16) { \
    VV_PARAMS(e16); \
    BODY; \
  } else if (sew == e32) { \
    VV_PARAMS(e32); \
    BODY; \
  } \
  VI_LOOP_END

#define VI_VX_LOOP_WIDEN(BODY) \
  VI_LOOP_BASE \
  if (sew == e8) { \
    VX_PARAMS(e8); \
    BODY; \
  } else if (sew == e16) { \
    VX_PARAMS(e16); \
    BODY; \
  } else if (sew == e32) { \
    VX_PARAMS(e32); \
    BODY; \
  } \
  VI_LOOP_END

#define VI_WIDE_OP_AND_ASSIGN(var0, var1, var2, op0, op1, sign) \
  switch (P.VU.vsew) { \
  case e8: { \
    sign##16_t UNUSED vd_w = P.VU.elt<sign##16_t>(rd_num, i); \
    P.VU.elt<uint16_t>(rd_num, i, true) = \
      op1((sign##16_t)(sign##8_t)var0 op0 (sign##16_t)(sign##8_t)var1) + var2; \
    } \
    break; \
  case e16: { \
    sign##32_t UNUSED vd_w = P.VU.elt<sign##32_t>(rd_num, i); \
    P.VU.elt<uint32_t>(rd_num, i, true) = \
      op1((sign##32_t)(sign##16_t)var0 op0 (sign##32_t)(sign##16_t)var1) + var2; \
    } \
    break; \
  default: { \
    sign##64_t UNUSED vd_w = P.VU.elt<sign##64_t>(rd_num, i); \
    P.VU.elt<uint64_t>(rd_num, i, true) = \
      op1((sign##64_t)(sign##32_t)var0 op0 (sign##64_t)(sign##32_t)var1) + var2; \
    } \
    break; \
  }

#define VI_WIDE_OP_AND_ASSIGN_MIX(var0, var1, var2, op0, op1, sign_d, sign_1, sign_2) \
  switch (P.VU.vsew) { \
  case e8: { \
    sign_d##16_t UNUSED vd_w = P.VU.elt<sign_d##16_t>(rd_num, i); \
    P.VU.elt<uint16_t>(rd_num, i, true) = \
      op1((sign_1##16_t)(sign_1##8_t)var0 op0 (sign_2##16_t)(sign_2##8_t)var1) + var2; \
    } \
    break; \
  case e16: { \
    sign_d##32_t UNUSED vd_w = P.VU.elt<sign_d##32_t>(rd_num, i); \
    P.VU.elt<uint32_t>(rd_num, i, true) = \
      op1((sign_1##32_t)(sign_1##16_t)var0 op0 (sign_2##32_t)(sign_2##16_t)var1) + var2; \
    } \
    break; \
  default: { \
    sign_d##64_t UNUSED vd_w = P.VU.elt<sign_d##64_t>(rd_num, i); \
    P.VU.elt<uint64_t>(rd_num, i, true) = \
      op1((sign_1##64_t)(sign_1##32_t)var0 op0 (sign_2##64_t)(sign_2##32_t)var1) + var2; \
    } \
    break; \
  }

#define VI_WIDE_WVX_OP(var0, op0, sign) \
  switch (P.VU.vsew) { \
  case e8: { \
    sign##16_t &vd_w = P.VU.elt<sign##16_t>(rd_num, i, true); \
    sign##16_t vs2_w = P.VU.elt<sign##16_t>(rs2_num, i); \
    vd_w = vs2_w op0 (sign##16_t)(sign##8_t)var0; \
    } \
    break; \
  case e16: { \
    sign##32_t &vd_w = P.VU.elt<sign##32_t>(rd_num, i, true); \
    sign##32_t vs2_w = P.VU.elt<sign##32_t>(rs2_num, i); \
    vd_w = vs2_w op0 (sign##32_t)(sign##16_t)var0; \
    } \
    break; \
  default: { \
    sign##64_t &vd_w = P.VU.elt<sign##64_t>(rd_num, i, true); \
    sign##64_t vs2_w = P.VU.elt<sign##64_t>(rs2_num, i); \
    vd_w = vs2_w op0 (sign##64_t)(sign##32_t)var0; \
    } \
    break; \
  }

// wide reduction loop - signed
#define VI_LOOP_WIDE_REDUCTION_BASE(sew1, sew2) \
  reg_t vl = P.VU.vl->read(); \
  reg_t rd_num = insn.rd(); \
  reg_t rs1_num = insn.rs1(); \
  reg_t rs2_num = insn.rs2(); \
  auto &vd_0_des = P.VU.elt<type_sew_t<sew2>::type>(rd_num, 0, true); \
  auto vd_0_res = P.VU.elt<type_sew_t<sew2>::type>(rs1_num, 0); \
  for (reg_t i = P.VU.vstart->read(); i < vl; ++i) { \
    VI_LOOP_ELEMENT_SKIP(); \
    auto vs2 = P.VU.elt<type_sew_t<sew1>::type>(rs2_num, i);

#define WIDE_REDUCTION_LOOP(sew1, sew2, BODY) \
  VI_LOOP_WIDE_REDUCTION_BASE(sew1, sew2) \
  BODY; \
  VI_LOOP_REDUCTION_END(sew2)

#define VI_VV_LOOP_WIDE_REDUCTION(BODY) \
  VI_CHECK_REDUCTION(true); \
  reg_t sew = P.VU.vsew; \
  if (sew == e8) { \
    WIDE_REDUCTION_LOOP(e8, e16, BODY) \
  } else if (sew == e16) { \
    WIDE_REDUCTION_LOOP(e16, e32, BODY) \
  } else if (sew == e32) { \
    WIDE_REDUCTION_LOOP(e32, e64, BODY) \
  }

// wide reduction loop - unsigned
#define VI_ULOOP_WIDE_REDUCTION_BASE(sew1, sew2) \
  reg_t vl = P.VU.vl->read(); \
  reg_t rd_num = insn.rd(); \
  reg_t rs1_num = insn.rs1(); \
  reg_t rs2_num = insn.rs2(); \
  auto &vd_0_des = P.VU.elt<type_usew_t<sew2>::type>(rd_num, 0, true); \
  auto vd_0_res = P.VU.elt<type_usew_t<sew2>::type>(rs1_num, 0); \
  for (reg_t i = P.VU.vstart->read(); i < vl; ++i) { \
    VI_LOOP_ELEMENT_SKIP(); \
    auto vs2 = P.VU.elt<type_usew_t<sew1>::type>(rs2_num, i);

#define WIDE_REDUCTION_ULOOP(sew1, sew2, BODY) \
  VI_ULOOP_WIDE_REDUCTION_BASE(sew1, sew2) \
  BODY; \
  VI_LOOP_REDUCTION_END(sew2)

#define VI_VV_ULOOP_WIDE_REDUCTION(BODY) \
  VI_CHECK_REDUCTION(true); \
  reg_t sew = P.VU.vsew; \
  if (sew == e8) { \
    WIDE_REDUCTION_ULOOP(e8, e16, BODY) \
  } else if (sew == e16) { \
    WIDE_REDUCTION_ULOOP(e16, e32, BODY) \
  } else if (sew == e32) { \
    WIDE_REDUCTION_ULOOP(e32, e64, BODY) \
  }

// carry/borrow bit loop
#define VI_VV_LOOP_CARRY(BODY) \
  VI_CHECK_MSS(true); \
  VI_LOOP_CARRY_BASE \
    if (sew == e8) { \
      VV_CARRY_PARAMS(e8) \
      BODY; \
    } else if (sew == e16) { \
      VV_CARRY_PARAMS(e16) \
      BODY; \
    } else if (sew == e32) { \
      VV_CARRY_PARAMS(e32) \
      BODY; \
    } else if (sew == e64) { \
      VV_CARRY_PARAMS(e64) \
      BODY; \
    } \
  VI_LOOP_CARRY_END

#define VI_XI_LOOP_CARRY(BODY) \
  VI_CHECK_MSS(false); \
  VI_LOOP_CARRY_BASE \
    if (sew == e8) { \
      XI_CARRY_PARAMS(e8) \
      BODY; \
    } else if (sew == e16) { \
      XI_CARRY_PARAMS(e16) \
      BODY; \
    } else if (sew == e32) { \
      XI_CARRY_PARAMS(e32) \
      BODY; \
    } else if (sew == e64) { \
      XI_CARRY_PARAMS(e64) \
      BODY; \
    } \
  VI_LOOP_CARRY_END

#define VI_VV_LOOP_WITH_CARRY(BODY) \
  VI_CHECK_SSS(true); \
  VI_LOOP_WITH_CARRY_BASE \
    if (sew == e8) { \
      VV_WITH_CARRY_PARAMS(e8) \
      BODY; \
    } else if (sew == e16) { \
      VV_WITH_CARRY_PARAMS(e16) \
      BODY; \
    } else if (sew == e32) { \
      VV_WITH_CARRY_PARAMS(e32) \
      BODY; \
    } else if (sew == e64) { \
      VV_WITH_CARRY_PARAMS(e64) \
      BODY; \
    } \
  VI_LOOP_END

#define VI_XI_LOOP_WITH_CARRY(BODY) \
  VI_CHECK_SSS(false); \
  VI_LOOP_WITH_CARRY_BASE \
    if (sew == e8) { \
      XI_WITH_CARRY_PARAMS(e8) \
      BODY; \
    } else if (sew == e16) { \
      XI_WITH_CARRY_PARAMS(e16) \
      BODY; \
    } else if (sew == e32) { \
      XI_WITH_CARRY_PARAMS(e32) \
      BODY; \
    } else if (sew == e64) { \
      XI_WITH_CARRY_PARAMS(e64) \
      BODY; \
    } \
  VI_LOOP_END

// average loop
#define VI_VV_LOOP_AVG(op) \
VRM xrm = p->VU.get_vround_mode(); \
VI_VV_LOOP({ \
  uint128_t res = ((uint128_t)vs2) op vs1; \
  INT_ROUNDING(res, xrm, 1); \
  vd = res >> 1; \
})

#define VI_VX_LOOP_AVG(op) \
VRM xrm = p->VU.get_vround_mode(); \
VI_VX_LOOP({ \
  uint128_t res = ((uint128_t)vs2) op rs1; \
  INT_ROUNDING(res, xrm, 1); \
  vd = res >> 1; \
})

#define VI_VV_ULOOP_AVG(op) \
VRM xrm = p->VU.get_vround_mode(); \
VI_VV_ULOOP({ \
  uint128_t res = ((uint128_t)vs2) op vs1; \
  INT_ROUNDING(res, xrm, 1); \
  vd = res >> 1; \
})

#define VI_VX_ULOOP_AVG(op) \
VRM xrm = p->VU.get_vround_mode(); \
VI_VX_ULOOP({ \
  uint128_t res = ((uint128_t)vs2) op rs1; \
  INT_ROUNDING(res, xrm, 1); \
  vd = res >> 1; \
})

//
// vector: load/store helper 
//
#define VI_STRIP(inx) \
  reg_t vreg_inx = inx;

#define VI_DUPLICATE_VREG(reg_num, idx_sew) \
reg_t index[P.VU.vlmax]; \
 for (reg_t i = 0; i < P.VU.vlmax && P.VU.vl->read() != 0; ++i) { \
  switch (idx_sew) { \
    case e8: \
      index[i] = P.VU.elt<uint8_t>(reg_num, i); \
      break; \
    case e16: \
      index[i] = P.VU.elt<uint16_t>(reg_num, i); \
      break; \
    case e32: \
      index[i] = P.VU.elt<uint32_t>(reg_num, i); \
      break; \
    case e64: \
      index[i] = P.VU.elt<uint64_t>(reg_num, i); \
      break; \
  } \
}

#define VI_LD(stride, offset, elt_width, is_mask_ldst) \
  const reg_t nf = insn.v_nf() + 1; \
  const reg_t vl = is_mask_ldst ? ((P.VU.vl->read() + 7) / 8) : P.VU.vl->read(); \
  const reg_t baseAddr = RS1; \
  const reg_t vd = insn.rd(); \
  VI_CHECK_LOAD(elt_width, is_mask_ldst); \
  for (reg_t i = 0; i < vl; ++i) { \
    VI_ELEMENT_SKIP; \
    VI_STRIP(i); \
    P.VU.vstart->write(i); \
    for (reg_t fn = 0; fn < nf; ++fn) { \
      elt_width##_t val = MMU.load<elt_width##_t>( \
        baseAddr + (stride) + (offset) * sizeof(elt_width##_t)); \
      P.VU.elt<elt_width##_t>(vd + fn * emul, vreg_inx, true) = val; \
    } \
  } \
  P.VU.vstart->write(0);

#define VI_LD_INDEX(elt_width, is_seg) \
  const reg_t nf = insn.v_nf() + 1; \
  const reg_t vl = P.VU.vl->read(); \
  const reg_t baseAddr = RS1; \
  const reg_t vd = insn.rd(); \
  if (!is_seg) \
    require(nf == 1); \
  VI_CHECK_LD_INDEX(elt_width); \
  VI_DUPLICATE_VREG(insn.rs2(), elt_width); \
  for (reg_t i = 0; i < vl; ++i) { \
    VI_ELEMENT_SKIP; \
    VI_STRIP(i); \
    P.VU.vstart->write(i); \
    for (reg_t fn = 0; fn < nf; ++fn) { \
      switch (P.VU.vsew) { \
        case e8: \
          P.VU.elt<uint8_t>(vd + fn * flmul, vreg_inx, true) = \
            MMU.load<uint8_t>(baseAddr + index[i] + fn * 1); \
          break; \
        case e16: \
          P.VU.elt<uint16_t>(vd + fn * flmul, vreg_inx, true) = \
            MMU.load<uint16_t>(baseAddr + index[i] + fn * 2); \
          break; \
        case e32: \
          P.VU.elt<uint32_t>(vd + fn * flmul, vreg_inx, true) = \
            MMU.load<uint32_t>(baseAddr + index[i] + fn * 4); \
          break; \
        default: \
          P.VU.elt<uint64_t>(vd + fn * flmul, vreg_inx, true) = \
            MMU.load<uint64_t>(baseAddr + index[i] + fn * 8); \
          break; \
      } \
    } \
  } \
  P.VU.vstart->write(0);

#define VI_ST(stride, offset, elt_width, is_mask_ldst) \
  const reg_t nf = insn.v_nf() + 1; \
  const reg_t vl = is_mask_ldst ? ((P.VU.vl->read() + 7) / 8) : P.VU.vl->read(); \
  const reg_t baseAddr = RS1; \
  const reg_t vs3 = insn.rd(); \
  VI_CHECK_STORE(elt_width, is_mask_ldst); \
  for (reg_t i = 0; i < vl; ++i) { \
    VI_STRIP(i) \
    VI_ELEMENT_SKIP; \
    P.VU.vstart->write(i); \
    for (reg_t fn = 0; fn < nf; ++fn) { \
      elt_width##_t val = P.VU.elt<elt_width##_t>(vs3 + fn * emul, vreg_inx); \
      MMU.store<elt_width##_t>( \
        baseAddr + (stride) + (offset) * sizeof(elt_width##_t), val); \
    } \
  } \
  P.VU.vstart->write(0);

#define VI_ST_INDEX(elt_width, is_seg) \
  const reg_t nf = insn.v_nf() + 1; \
  const reg_t vl = P.VU.vl->read(); \
  const reg_t baseAddr = RS1; \
  const reg_t vs3 = insn.rd(); \
  if (!is_seg) \
    require(nf == 1); \
  VI_CHECK_ST_INDEX(elt_width); \
  VI_DUPLICATE_VREG(insn.rs2(), elt_width); \
  for (reg_t i = 0; i < vl; ++i) { \
    VI_STRIP(i) \
    VI_ELEMENT_SKIP; \
    P.VU.vstart->write(i); \
    for (reg_t fn = 0; fn < nf; ++fn) { \
      switch (P.VU.vsew) { \
      case e8: \
        MMU.store<uint8_t>(baseAddr + index[i] + fn * 1, \
          P.VU.elt<uint8_t>(vs3 + fn * flmul, vreg_inx)); \
        break; \
      case e16: \
        MMU.store<uint16_t>(baseAddr + index[i] + fn * 2, \
          P.VU.elt<uint16_t>(vs3 + fn * flmul, vreg_inx)); \
        break; \
      case e32: \
        MMU.store<uint32_t>(baseAddr + index[i] + fn * 4, \
          P.VU.elt<uint32_t>(vs3 + fn * flmul, vreg_inx)); \
        break; \
      default: \
        MMU.store<uint64_t>(baseAddr + index[i] + fn * 8, \
          P.VU.elt<uint64_t>(vs3 + fn * flmul, vreg_inx)); \
        break; \
      } \
    } \
  } \
  P.VU.vstart->write(0);

#define VI_LDST_FF(elt_width) \
  const reg_t nf = insn.v_nf() + 1; \
  const reg_t vl = p->VU.vl->read(); \
  const reg_t baseAddr = RS1; \
  const reg_t rd_num = insn.rd(); \
  VI_CHECK_LOAD(elt_width, false); \
  bool early_stop = false; \
  for (reg_t i = p->VU.vstart->read(); i < vl; ++i) { \
    VI_STRIP(i); \
    VI_ELEMENT_SKIP; \
    \
    for (reg_t fn = 0; fn < nf; ++fn) { \
      uint64_t val; \
      try { \
        val = MMU.load<elt_width##_t>( \
          baseAddr + (i * nf + fn) * sizeof(elt_width##_t)); \
      } catch (trap_t& t) { \
        if (i == 0) \
          throw; /* Only take exception on zeroth element */ \
        /* Reduce VL if an exception occurs on a later element */ \
        early_stop = true; \
        P.VU.vl->write_raw(i); \
        break; \
      } \
      p->VU.elt<elt_width##_t>(rd_num + fn * emul, vreg_inx, true) = val; \
    } \
    \
    if (early_stop) { \
      break; \
    } \
  } \
  p->VU.vstart->write(0);

#define VI_LD_WHOLE(elt_width) \
  require_vector_novtype(true); \
  require(sizeof(elt_width ## _t) * 8 <= P.VU.ELEN); \
  const reg_t baseAddr = RS1; \
  const reg_t vd = insn.rd(); \
  const reg_t len = insn.v_nf() + 1; \
  require_align(vd, len); \
  const reg_t elt_per_reg = P.VU.vlenb / sizeof(elt_width ## _t); \
  const reg_t size = len * elt_per_reg; \
  if (P.VU.vstart->read() < size) { \
    reg_t i = P.VU.vstart->read() / elt_per_reg; \
    reg_t off = P.VU.vstart->read() % elt_per_reg; \
    if (off) { \
      for (reg_t pos = off; pos < elt_per_reg; ++pos) { \
        auto val = MMU.load<elt_width##_t>(baseAddr + \
          P.VU.vstart->read() * sizeof(elt_width ## _t)); \
        P.VU.elt<elt_width ## _t>(vd + i, pos, true) = val; \
        P.VU.vstart->write(P.VU.vstart->read() + 1); \
      } \
      ++i; \
    } \
    for (; i < len; ++i) { \
      for (reg_t pos = 0; pos < elt_per_reg; ++pos) { \
        auto val = MMU.load<elt_width##_t>(baseAddr + \
          P.VU.vstart->read() * sizeof(elt_width ## _t)); \
        P.VU.elt<elt_width ## _t>(vd + i, pos, true) = val; \
        P.VU.vstart->write(P.VU.vstart->read() + 1); \
      } \
    } \
  } \
  P.VU.vstart->write(0);

#define VI_ST_WHOLE \
  require_vector_novtype(true); \
  const reg_t baseAddr = RS1; \
  const reg_t vs3 = insn.rd(); \
  const reg_t len = insn.v_nf() + 1; \
  require_align(vs3, len); \
  const reg_t size = len * P.VU.vlenb; \
  \
  if (P.VU.vstart->read() < size) { \
    reg_t i = P.VU.vstart->read() / P.VU.vlenb; \
    reg_t off = P.VU.vstart->read() % P.VU.vlenb; \
    if (off) { \
      for (reg_t pos = off; pos < P.VU.vlenb; ++pos) { \
        auto val = P.VU.elt<uint8_t>(vs3 + i, pos); \
        MMU.store<uint8_t>(baseAddr + P.VU.vstart->read(), val); \
        P.VU.vstart->write(P.VU.vstart->read() + 1); \
      } \
      i++; \
    } \
    for (; i < len; ++i) { \
      for (reg_t pos = 0; pos < P.VU.vlenb; ++pos) { \
        auto val = P.VU.elt<uint8_t>(vs3 + i, pos); \
        MMU.store<uint8_t>(baseAddr + P.VU.vstart->read(), val); \
        P.VU.vstart->write(P.VU.vstart->read() + 1); \
      } \
    } \
  } \
  P.VU.vstart->write(0);

//
// vector: amo 
//
#define VI_AMO(op, type, idx_type) \
  require_vector(false); \
  require_align(insn.rd(), P.VU.vflmul); \
  require(P.VU.vsew <= P.get_xlen() && P.VU.vsew >= 32); \
  require_align(insn.rd(), P.VU.vflmul); \
  float vemul = ((float)idx_type / P.VU.vsew * P.VU.vflmul); \
  require(vemul >= 0.125 && vemul <= 8); \
  require_align(insn.rs2(), vemul); \
  if (insn.v_wd()) { \
    require_vm; \
    if (idx_type > P.VU.vsew) { \
      if (insn.rd() != insn.rs2()) \
        require_noover(insn.rd(), P.VU.vflmul, insn.rs2(), vemul); \
    } else if (idx_type < P.VU.vsew) { \
      if (vemul < 1) { \
        require_noover(insn.rd(), P.VU.vflmul, insn.rs2(), vemul); \
      } else { \
        require_noover_widen(insn.rd(), P.VU.vflmul, insn.rs2(), vemul); \
      } \
    } \
  } \
  VI_DUPLICATE_VREG(insn.rs2(), idx_type); \
  const reg_t vl = P.VU.vl->read(); \
  const reg_t baseAddr = RS1; \
  const reg_t vd = insn.rd(); \
  for (reg_t i = P.VU.vstart->read(); i < vl; ++i) { \
    VI_ELEMENT_SKIP; \
    VI_STRIP(i); \
    P.VU.vstart->write(i); \
    switch (P.VU.vsew) { \
    case e32: { \
      auto vs3 = P.VU.elt< type ## 32_t>(vd, vreg_inx); \
      auto val = MMU.amo<uint32_t>(baseAddr + index[i], [&](type ## 32_t UNUSED lhs) { op }); \
      if (insn.v_wd()) \
        P.VU.elt< type ## 32_t>(vd, vreg_inx, true) = val; \
      } \
      break; \
    case e64: { \
      auto vs3 = P.VU.elt< type ## 64_t>(vd, vreg_inx); \
      auto val = MMU.amo<uint64_t>(baseAddr + index[i], [&](type ## 64_t UNUSED lhs) { op }); \
      if (insn.v_wd()) \
        P.VU.elt< type ## 64_t>(vd, vreg_inx, true) = val; \
      } \
      break; \
    default: \
      require(0); \
      break; \
    } \
  } \
  P.VU.vstart->write(0);

// vector: sign/unsiged extension
#define VI_VV_EXT(div, type) \
  require(insn.rd() != insn.rs2()); \
  require_vm; \
  reg_t from = P.VU.vsew / div; \
  require(from >= e8 && from <= e64); \
  require(((float)P.VU.vflmul / div) >= 0.125 && ((float)P.VU.vflmul / div) <= 8 ); \
  require_align(insn.rd(), P.VU.vflmul); \
  require_align(insn.rs2(), P.VU.vflmul / div); \
  if ((P.VU.vflmul / div) < 1) { \
    require_noover(insn.rd(), P.VU.vflmul, insn.rs2(), P.VU.vflmul / div); \
  } else { \
    require_noover_widen(insn.rd(), P.VU.vflmul, insn.rs2(), P.VU.vflmul / div); \
  } \
  reg_t pat = (((P.VU.vsew >> 3) << 4) | from >> 3); \
  VI_GENERAL_LOOP_BASE \
  VI_LOOP_ELEMENT_SKIP(); \
    switch (pat) { \
      case 0x21: \
        P.VU.elt<type##16_t>(rd_num, i, true) = P.VU.elt<type##8_t>(rs2_num, i); \
        break; \
      case 0x41: \
        P.VU.elt<type##32_t>(rd_num, i, true) = P.VU.elt<type##8_t>(rs2_num, i); \
        break; \
      case 0x81: \
        P.VU.elt<type##64_t>(rd_num, i, true) = P.VU.elt<type##8_t>(rs2_num, i); \
        break; \
      case 0x42: \
        P.VU.elt<type##32_t>(rd_num, i, true) = P.VU.elt<type##16_t>(rs2_num, i); \
        break; \
      case 0x82: \
        P.VU.elt<type##64_t>(rd_num, i, true) = P.VU.elt<type##16_t>(rs2_num, i); \
        break; \
      case 0x84: \
        P.VU.elt<type##64_t>(rd_num, i, true) = P.VU.elt<type##32_t>(rs2_num, i); \
        break; \
      default: \
        break; \
    } \
  VI_LOOP_END 

//
// vector: vfp helper
//
#define VI_VFP_COMMON \
  require_fp; \
  require((P.VU.vsew == e16 && p->extension_enabled(EXT_ZVFH)) || \
          (P.VU.vsew == e32 && p->extension_enabled('F')) || \
          (P.VU.vsew == e64 && p->extension_enabled('D'))); \
  require_vector(true); \
  require(STATE.frm->read() < 0x5); \
  reg_t UNUSED vl = P.VU.vl->read(); \
  reg_t UNUSED rd_num = insn.rd(); \
  reg_t UNUSED rs1_num = insn.rs1(); \
  reg_t UNUSED rs2_num = insn.rs2(); \
  softfloat_roundingMode = STATE.frm->read();

#define VI_VFP_LOOP_BASE \
  VI_VFP_COMMON \
  for (reg_t i = P.VU.vstart->read(); i < vl; ++i) { \
    VI_LOOP_ELEMENT_SKIP();

#define VI_VFP_LOOP_CMP_BASE \
  VI_VFP_COMMON \
  for (reg_t i = P.VU.vstart->read(); i < vl; ++i) { \
    VI_LOOP_ELEMENT_SKIP(); \
    uint64_t mmask = UINT64_C(1) << mpos; \
    uint64_t &vd = P.VU.elt<uint64_t>(rd_num, midx, true); \
    uint64_t res = 0;

#define VI_VFP_LOOP_REDUCTION_BASE(width) \
  float##width##_t vd_0 = P.VU.elt<float##width##_t>(rd_num, 0); \
  float##width##_t vs1_0 = P.VU.elt<float##width##_t>(rs1_num, 0); \
  vd_0 = vs1_0; \
  bool is_active = false; \
  for (reg_t i = P.VU.vstart->read(); i < vl; ++i) { \
    VI_LOOP_ELEMENT_SKIP(); \
    float##width##_t vs2 = P.VU.elt<float##width##_t>(rs2_num, i); \
    is_active = true; \

#define VI_VFP_LOOP_WIDE_REDUCTION_BASE \
  VI_VFP_COMMON \
  float64_t vd_0 = f64(P.VU.elt<float64_t>(rs1_num, 0).v); \
  for (reg_t i = P.VU.vstart->read(); i < vl; ++i) { \
    VI_LOOP_ELEMENT_SKIP();

#define VI_VFP_LOOP_END \
  } \
  P.VU.vstart->write(0); \

#define VI_VFP_LOOP_REDUCTION_END(x) \
  } \
  P.VU.vstart->write(0); \
  if (vl > 0) { \
    if (is_propagate && !is_active) { \
      switch (x) { \
        case e16: { \
            auto ret = f16_classify(f16(vd_0.v)); \
            if (ret & 0x300) { \
              if (ret & 0x100) { \
                softfloat_exceptionFlags |= softfloat_flag_invalid; \
                set_fp_exceptions; \
              } \
              P.VU.elt<uint16_t>(rd_num, 0, true) = defaultNaNF16UI; \
            } else { \
              P.VU.elt<uint16_t>(rd_num, 0, true) = vd_0.v; \
            } \
          } \
          break; \
        case e32: { \
            auto ret = f32_classify(f32(vd_0.v)); \
            if (ret & 0x300) { \
              if (ret & 0x100) { \
                softfloat_exceptionFlags |= softfloat_flag_invalid; \
                set_fp_exceptions; \
              } \
              P.VU.elt<uint32_t>(rd_num, 0, true) = defaultNaNF32UI; \
            } else { \
              P.VU.elt<uint32_t>(rd_num, 0, true) = vd_0.v; \
            } \
          } \
          break; \
        case e64: { \
            auto ret = f64_classify(f64(vd_0.v)); \
            if (ret & 0x300) { \
              if (ret & 0x100) { \
                softfloat_exceptionFlags |= softfloat_flag_invalid; \
                set_fp_exceptions; \
              } \
              P.VU.elt<uint64_t>(rd_num, 0, true) = defaultNaNF64UI; \
            } else { \
              P.VU.elt<uint64_t>(rd_num, 0, true) = vd_0.v; \
            } \
          } \
          break; \
      } \
    } else { \
      P.VU.elt<type_sew_t<x>::type>(rd_num, 0, true) = vd_0.v; \
    } \
  }

#define VI_VFP_LOOP_CMP_END \
  switch (P.VU.vsew) { \
    case e16: \
    case e32: \
    case e64: { \
      vd = (vd & ~mmask) | (((res) << mpos) & mmask); \
      break; \
    } \
    default: \
      require(0); \
      break; \
    }; \
  } \
  P.VU.vstart->write(0);

#define VI_VFP_VV_LOOP(BODY16, BODY32, BODY64) \
  VI_CHECK_SSS(true); \
  VI_VFP_LOOP_BASE \
  switch (P.VU.vsew) { \
    case e16: { \
      VFP_VV_PARAMS(16); \
      BODY16; \
      set_fp_exceptions; \
      break; \
    } \
    case e32: { \
      VFP_VV_PARAMS(32); \
      BODY32; \
      set_fp_exceptions; \
      break; \
    } \
    case e64: { \
      VFP_VV_PARAMS(64); \
      BODY64; \
      set_fp_exceptions; \
      break; \
    } \
    default: \
      require(0); \
      break; \
  }; \
  DEBUG_RVV_FP_VV; \
  VI_VFP_LOOP_END

#define VI_VFP_V_LOOP(BODY16, BODY32, BODY64) \
  VI_CHECK_SSS(false); \
  VI_VFP_LOOP_BASE \
  switch (P.VU.vsew) { \
    case e16: { \
      VFP_V_PARAMS(16); \
      BODY16; \
      break; \
    } \
    case e32: { \
      VFP_V_PARAMS(32); \
      BODY32; \
      break; \
    } \
    case e64: { \
      VFP_V_PARAMS(64); \
      BODY64; \
      break; \
    } \
    default: \
      require(0); \
      break; \
  }; \
  set_fp_exceptions; \
  VI_VFP_LOOP_END

#define VI_VFP_VV_LOOP_REDUCTION(BODY16, BODY32, BODY64) \
  VI_CHECK_REDUCTION(false) \
  VI_VFP_COMMON \
  switch (P.VU.vsew) { \
    case e16: { \
      VI_VFP_LOOP_REDUCTION_BASE(16) \
        BODY16; \
        set_fp_exceptions; \
      VI_VFP_LOOP_REDUCTION_END(e16) \
      break; \
    } \
    case e32: { \
      VI_VFP_LOOP_REDUCTION_BASE(32) \
        BODY32; \
        set_fp_exceptions; \
      VI_VFP_LOOP_REDUCTION_END(e32) \
      break; \
    } \
    case e64: { \
      VI_VFP_LOOP_REDUCTION_BASE(64) \
        BODY64; \
        set_fp_exceptions; \
      VI_VFP_LOOP_REDUCTION_END(e64) \
      break; \
    } \
    default: \
      require(0); \
      break; \
  }; \

#define VI_VFP_VV_LOOP_WIDE_REDUCTION(BODY16, BODY32) \
  VI_CHECK_REDUCTION(true) \
  VI_VFP_COMMON \
  require((P.VU.vsew == e16 && p->extension_enabled('F')) || \
          (P.VU.vsew == e32 && p->extension_enabled('D'))); \
  bool is_active = false; \
  switch (P.VU.vsew) { \
    case e16: { \
      float32_t vd_0 = P.VU.elt<float32_t>(rs1_num, 0); \
      for (reg_t i = P.VU.vstart->read(); i < vl; ++i) { \
        VI_LOOP_ELEMENT_SKIP(); \
        is_active = true; \
        float32_t vs2 = f16_to_f32(P.VU.elt<float16_t>(rs2_num, i)); \
        BODY16; \
        set_fp_exceptions; \
      VI_VFP_LOOP_REDUCTION_END(e32) \
      break; \
    } \
    case e32: { \
      float64_t vd_0 = P.VU.elt<float64_t>(rs1_num, 0); \
      for (reg_t i = P.VU.vstart->read(); i < vl; ++i) { \
        VI_LOOP_ELEMENT_SKIP(); \
        is_active = true; \
        float64_t vs2 = f32_to_f64(P.VU.elt<float32_t>(rs2_num, i)); \
        BODY32; \
        set_fp_exceptions; \
      VI_VFP_LOOP_REDUCTION_END(e64) \
      break; \
    } \
    default: \
      require(0); \
      break; \
  }; \

#define VI_VFP_VF_LOOP(BODY16, BODY32, BODY64) \
  VI_CHECK_SSS(false); \
  VI_VFP_LOOP_BASE \
  switch (P.VU.vsew) { \
    case e16: { \
      VFP_VF_PARAMS(16); \
      BODY16; \
      set_fp_exceptions; \
      break; \
    } \
    case e32: { \
      VFP_VF_PARAMS(32); \
      BODY32; \
      set_fp_exceptions; \
      break; \
    } \
    case e64: { \
      VFP_VF_PARAMS(64); \
      BODY64; \
      set_fp_exceptions; \
      break; \
    } \
    default: \
      require(0); \
      break; \
  }; \
  DEBUG_RVV_FP_VF; \
  VI_VFP_LOOP_END

#define VI_VFP_VV_LOOP_CMP(BODY16, BODY32, BODY64) \
  VI_CHECK_MSS(true); \
  VI_VFP_LOOP_CMP_BASE \
  switch (P.VU.vsew) { \
    case e16: { \
      VFP_VV_PARAMS(16); \
      BODY16; \
      set_fp_exceptions; \
      break; \
    } \
    case e32: { \
      VFP_VV_PARAMS(32); \
      BODY32; \
      set_fp_exceptions; \
      break; \
    } \
    case e64: { \
      VFP_VV_PARAMS(64); \
      BODY64; \
      set_fp_exceptions; \
      break; \
    } \
    default: \
      require(0); \
      break; \
  }; \
  VI_VFP_LOOP_CMP_END \

#define VI_VFP_VF_LOOP_CMP(BODY16, BODY32, BODY64) \
  VI_CHECK_MSS(false); \
  VI_VFP_LOOP_CMP_BASE \
  switch (P.VU.vsew) { \
    case e16: { \
      VFP_VF_PARAMS(16); \
      BODY16; \
      set_fp_exceptions; \
      break; \
    } \
    case e32: { \
      VFP_VF_PARAMS(32); \
      BODY32; \
      set_fp_exceptions; \
      break; \
    } \
    case e64: { \
      VFP_VF_PARAMS(64); \
      BODY64; \
      set_fp_exceptions; \
      break; \
    } \
    default: \
      require(0); \
      break; \
  }; \
  VI_VFP_LOOP_CMP_END \

#define VI_VFP_VF_LOOP_WIDE(BODY16, BODY32) \
  VI_CHECK_DSS(false); \
  VI_VFP_LOOP_BASE \
  switch (P.VU.vsew) { \
    case e16: { \
      float32_t &vd = P.VU.elt<float32_t>(rd_num, i, true); \
      float32_t vs2 = f16_to_f32(P.VU.elt<float16_t>(rs2_num, i)); \
      float32_t rs1 = f16_to_f32(FRS1_H); \
      BODY16; \
      set_fp_exceptions; \
      break; \
    } \
    case e32: { \
      float64_t &vd = P.VU.elt<float64_t>(rd_num, i, true); \
      float64_t vs2 = f32_to_f64(P.VU.elt<float32_t>(rs2_num, i)); \
      float64_t rs1 = f32_to_f64(FRS1_F); \
      BODY32; \
      set_fp_exceptions; \
      break; \
    } \
    default: \
      require(0); \
      break; \
  }; \
  DEBUG_RVV_FP_VV; \
  VI_VFP_LOOP_END

#define VI_VFP_VV_LOOP_WIDE(BODY16, BODY32) \
  VI_CHECK_DSS(true); \
  VI_VFP_LOOP_BASE \
  switch (P.VU.vsew) { \
    case e16: { \
      float32_t &vd = P.VU.elt<float32_t>(rd_num, i, true); \
      float32_t vs2 = f16_to_f32(P.VU.elt<float16_t>(rs2_num, i)); \
      float32_t vs1 = f16_to_f32(P.VU.elt<float16_t>(rs1_num, i)); \
      BODY16; \
      set_fp_exceptions; \
      break; \
    } \
    case e32: { \
      float64_t &vd = P.VU.elt<float64_t>(rd_num, i, true); \
      float64_t vs2 = f32_to_f64(P.VU.elt<float32_t>(rs2_num, i)); \
      float64_t vs1 = f32_to_f64(P.VU.elt<float32_t>(rs1_num, i)); \
      BODY32; \
      set_fp_exceptions; \
      break; \
    } \
    default: \
      require(0); \
      break; \
  }; \
  DEBUG_RVV_FP_VV; \
  VI_VFP_LOOP_END

#define VI_VFP_WF_LOOP_WIDE(BODY16, BODY32) \
  VI_CHECK_DDS(false); \
  VI_VFP_LOOP_BASE \
  switch (P.VU.vsew) { \
    case e16: { \
      float32_t &vd = P.VU.elt<float32_t>(rd_num, i, true); \
      float32_t vs2 = P.VU.elt<float32_t>(rs2_num, i); \
      float32_t rs1 = f16_to_f32(FRS1_H); \
      BODY16; \
      set_fp_exceptions; \
      break; \
    } \
    case e32: { \
      float64_t &vd = P.VU.elt<float64_t>(rd_num, i, true); \
      float64_t vs2 = P.VU.elt<float64_t>(rs2_num, i); \
      float64_t rs1 = f32_to_f64(FRS1_F); \
      BODY32; \
      set_fp_exceptions; \
      break; \
    } \
    default: \
      require(0); \
  }; \
  DEBUG_RVV_FP_VV; \
  VI_VFP_LOOP_END

#define VI_VFP_WV_LOOP_WIDE(BODY16, BODY32) \
  VI_CHECK_DDS(true); \
  VI_VFP_LOOP_BASE \
  switch (P.VU.vsew) { \
    case e16: { \
      float32_t &vd = P.VU.elt<float32_t>(rd_num, i, true); \
      float32_t vs2 = P.VU.elt<float32_t>(rs2_num, i); \
      float32_t vs1 = f16_to_f32(P.VU.elt<float16_t>(rs1_num, i)); \
      BODY16; \
      set_fp_exceptions; \
      break; \
    } \
    case e32: { \
      float64_t &vd = P.VU.elt<float64_t>(rd_num, i, true); \
      float64_t vs2 = P.VU.elt<float64_t>(rs2_num, i); \
      float64_t vs1 = f32_to_f64(P.VU.elt<float32_t>(rs1_num, i)); \
      BODY32; \
      set_fp_exceptions; \
      break; \
    } \
    default: \
      require(0); \
  }; \
  DEBUG_RVV_FP_VV; \
  VI_VFP_LOOP_END

#define VI_VFP_LOOP_SCALE_BASE \
  require_fp; \
  require_vector(true); \
  require(STATE.frm->read() < 0x5); \
  reg_t vl = P.VU.vl->read(); \
  reg_t rd_num = insn.rd(); \
  reg_t UNUSED rs1_num = insn.rs1(); \
  reg_t rs2_num = insn.rs2(); \
  softfloat_roundingMode = STATE.frm->read(); \
  for (reg_t i = P.VU.vstart->read(); i < vl; ++i) { \
    VI_LOOP_ELEMENT_SKIP();

#define VI_VFP_CVT_LOOP(CVT_PARAMS, CHECK, BODY) \
  CHECK \
  VI_VFP_LOOP_SCALE_BASE \
  CVT_PARAMS \
  BODY \
  set_fp_exceptions; \
  VI_VFP_LOOP_END

#define VI_VFP_CVT_INT_TO_FP(BODY16, BODY32, BODY64, sign) \
  VI_CHECK_SSS(false); \
  VI_VFP_COMMON \
  switch (P.VU.vsew) { \
    case e16: \
      { VI_VFP_CVT_LOOP(CVT_INT_TO_FP_PARAMS(16, 16, sign), \
        { p->extension_enabled(EXT_ZVFH); }, \
        BODY16); } \
      break; \
    case e32: \
      { VI_VFP_CVT_LOOP(CVT_INT_TO_FP_PARAMS(32, 32, sign), \
        { p->extension_enabled('F'); }, \
        BODY32); } \
      break; \
    case e64: \
      { VI_VFP_CVT_LOOP(CVT_INT_TO_FP_PARAMS(64, 64, sign), \
        { p->extension_enabled('D'); }, \
        BODY64); } \
      break; \
    default: \
      require(0); \
      break; \
  }

#define VI_VFP_CVT_FP_TO_INT(BODY16, BODY32, BODY64, sign) \
  VI_CHECK_SSS(false); \
  VI_VFP_COMMON \
  switch (P.VU.vsew) { \
    case e16: \
      { VI_VFP_CVT_LOOP(CVT_FP_TO_INT_PARAMS(16, 16, sign), \
        { p->extension_enabled(EXT_ZVFH); }, \
        BODY16); } \
      break; \
    case e32: \
      { VI_VFP_CVT_LOOP(CVT_FP_TO_INT_PARAMS(32, 32, sign), \
        { p->extension_enabled('F'); }, \
        BODY32); } \
      break; \
    case e64: \
      { VI_VFP_CVT_LOOP(CVT_FP_TO_INT_PARAMS(64, 64, sign), \
        { p->extension_enabled('D'); }, \
        BODY64); } \
      break; \
    default: \
      require(0); \
      break; \
  }

#define VI_VFP_WCVT_FP_TO_FP(BODY8, BODY16, BODY32, \
                             CHECK8, CHECK16, CHECK32) \
  VI_CHECK_DSS(false); \
  switch (P.VU.vsew) { \
    case e16: \
      { VI_VFP_CVT_LOOP(CVT_FP_TO_FP_PARAMS(16, 32), CHECK16, BODY16); } \
      break; \
    case e32: \
      { VI_VFP_CVT_LOOP(CVT_FP_TO_FP_PARAMS(32, 64), CHECK32, BODY32); } \
      break; \
    default: \
      require(0); \
      break; \
  }

#define VI_VFP_WCVT_INT_TO_FP(BODY8, BODY16, BODY32, \
                              CHECK8, CHECK16, CHECK32, \
                              sign) \
  VI_CHECK_DSS(false); \
  switch (P.VU.vsew) { \
    case e8: \
      { VI_VFP_CVT_LOOP(CVT_INT_TO_FP_PARAMS(8, 16, sign), CHECK8, BODY8); } \
      break; \
    case e16: \
      { VI_VFP_CVT_LOOP(CVT_INT_TO_FP_PARAMS(16, 32, sign), CHECK16, BODY16); } \
      break; \
    case e32: \
      { VI_VFP_CVT_LOOP(CVT_INT_TO_FP_PARAMS(32, 64, sign), CHECK32, BODY32); } \
      break; \
    default: \
      require(0); \
      break; \
  }

#define VI_VFP_WCVT_FP_TO_INT(BODY8, BODY16, BODY32, \
                              CHECK8, CHECK16, CHECK32, \
                              sign) \
  VI_CHECK_DSS(false); \
  switch (P.VU.vsew) { \
    case e16: \
      { VI_VFP_CVT_LOOP(CVT_FP_TO_INT_PARAMS(16, 32, sign), CHECK16, BODY16); } \
      break; \
    case e32: \
      { VI_VFP_CVT_LOOP(CVT_FP_TO_INT_PARAMS(32, 64, sign), CHECK32, BODY32); } \
      break; \
    default: \
      require(0); \
      break; \
  }

#define VI_VFP_NCVT_FP_TO_FP(BODY8, BODY16, BODY32, \
                             CHECK8, CHECK16, CHECK32) \
  VI_CHECK_SDS(false); \
  switch (P.VU.vsew) { \
    case e16: \
      { VI_VFP_CVT_LOOP(CVT_FP_TO_FP_PARAMS(32, 16), CHECK16, BODY16); } \
      break; \
    case e32: \
      { VI_VFP_CVT_LOOP(CVT_FP_TO_FP_PARAMS(64, 32), CHECK32, BODY32); } \
      break; \
    default: \
      require(0); \
      break; \
  }

#define VI_VFP_NCVT_INT_TO_FP(BODY8, BODY16, BODY32, \
                              CHECK8, CHECK16, CHECK32, \
                              sign) \
  VI_CHECK_SDS(false); \
  switch (P.VU.vsew) { \
    case e16: \
      { VI_VFP_CVT_LOOP(CVT_INT_TO_FP_PARAMS(32, 16, sign), CHECK16, BODY16); } \
      break; \
    case e32: \
      { VI_VFP_CVT_LOOP(CVT_INT_TO_FP_PARAMS(64, 32, sign), CHECK32, BODY32); } \
      break; \
    default: \
      require(0); \
      break; \
  }

#define VI_VFP_NCVT_FP_TO_INT(BODY8, BODY16, BODY32, \
                              CHECK8, CHECK16, CHECK32, \
                              sign) \
  VI_CHECK_SDS(false); \
  switch (P.VU.vsew) { \
    case e8: \
      { VI_VFP_CVT_LOOP(CVT_FP_TO_INT_PARAMS(16, 8, sign), CHECK8, BODY8); } \
      break; \
    case e16: \
      { VI_VFP_CVT_LOOP(CVT_FP_TO_INT_PARAMS(32, 16, sign), CHECK16, BODY16); } \
      break; \
    case e32: \
      { VI_VFP_CVT_LOOP(CVT_FP_TO_INT_PARAMS(64, 32, sign), CHECK32, BODY32); } \
      break; \
    default: \
      require(0); \
      break; \
  }

#endif

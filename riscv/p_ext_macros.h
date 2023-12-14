// See LICENSE for license details.

#ifndef _RISCV_P_EXT_MACROS_H
#define _RISCV_P_EXT_MACROS_H

// The p-extension support is contributed by
// Programming Langauge Lab, Department of Computer Science, National Tsing-Hua University, Taiwan

#define P_FIELD(R, INDEX, SIZE) \
  (type_sew_t<SIZE>::type)get_field(R, make_mask64(((INDEX) * SIZE), SIZE))

#define P_UFIELD(R, INDEX, SIZE) \
  (type_usew_t<SIZE>::type)get_field(R, make_mask64(((INDEX) * SIZE), SIZE))

#define P_B(R, INDEX) P_UFIELD(R, INDEX, 8)
#define P_H(R, INDEX) P_UFIELD(R, INDEX, 16)
#define P_W(R, INDEX) P_UFIELD(R, INDEX, 32)
#define P_SB(R, INDEX) P_FIELD(R, INDEX, 8)
#define P_SH(R, INDEX) P_FIELD(R, INDEX, 16)
#define P_SW(R, INDEX) P_FIELD(R, INDEX, 32)

#define READ_REG_PAIR(reg) ({ \
  require((reg) % 2 == 0); \
  (reg) == 0 ? reg_t(0) : \
  (READ_REG((reg) + 1) << 32) + zext32(READ_REG(reg)); })

#define RS1_PAIR READ_REG_PAIR(insn.rs1())
#define RS2_PAIR READ_REG_PAIR(insn.rs2())
#define RD_PAIR READ_REG_PAIR(insn.rd())

#define WRITE_PD() \
  rd_tmp = set_field(rd_tmp, make_mask64((i * sizeof(pd) * 8), sizeof(pd) * 8), pd);

#define WRITE_RD_PAIR(value) \
  if (insn.rd() != 0) { \
    require(insn.rd() % 2 == 0); \
    WRITE_REG(insn.rd(), sext32(value)); \
    WRITE_REG(insn.rd() + 1, (sreg_t(value)) >> 32); \
  }

#define P_SET_OV(ov) \
  if (ov) P.VU.vxsat->write(1, p);

#define P_SAT(R, BIT) \
  if (R > INT##BIT##_MAX) { \
    R = INT##BIT##_MAX; \
    P_SET_OV(1); \
  } else if (R < INT##BIT##_MIN) { \
    R = INT##BIT##_MIN; \
    P_SET_OV(1); \
  }

#define P_SATU(R, BIT) \
  if (R > UINT##BIT##_MAX) { \
    R = UINT##BIT##_MAX; \
    P_SET_OV(1); \
  } else if (R < 0) { \
    P_SET_OV(1); \
    R = 0; \
  }

#define P_LOOP_BASE(BIT) \
  require_extension(EXT_ZPN); \
  require(BIT == e8 || BIT == e16 || BIT == e32); \
  reg_t rd_tmp = RD; \
  reg_t rs1 = RS1; \
  reg_t rs2 = RS2; \
  sreg_t len = xlen / BIT; \
  for (sreg_t i = len - 1; i >= 0; --i) {

#define P_ONE_LOOP_BASE(BIT) \
  require_extension(EXT_ZPN); \
  require(BIT == e8 || BIT == e16 || BIT == e32); \
  reg_t rd_tmp = RD; \
  reg_t rs1 = RS1; \
  sreg_t len = xlen / BIT; \
  for (sreg_t i = len - 1; i >= 0; --i) {

#define P_I_LOOP_BASE(BIT, IMMBIT) \
  require_extension(EXT_ZPN); \
  require(BIT == e8 || BIT == e16 || BIT == e32); \
  reg_t rd_tmp = RD; \
  reg_t rs1 = RS1; \
  type_usew_t<BIT>::type imm##IMMBIT##u = insn.p_imm##IMMBIT(); \
  sreg_t len = xlen / BIT; \
  for (sreg_t i = len - 1; i >= 0; --i) {

#define P_X_LOOP_BASE(BIT, LOWBIT) \
  require_extension(EXT_ZPN); \
  require(BIT == e8 || BIT == e16 || BIT == e32); \
  reg_t rd_tmp = RD; \
  reg_t rs1 = RS1; \
  type_usew_t<BIT>::type sa = RS2 & ((uint64_t(1) << LOWBIT) - 1); \
  type_sew_t<BIT>::type UNUSED ssa = int64_t(RS2) << (64 - LOWBIT) >> (64 - LOWBIT); \
  sreg_t len = xlen / BIT; \
  for (sreg_t i = len - 1; i >= 0; --i) {

#define P_MUL_LOOP_BASE(BIT) \
  require_extension(EXT_ZPN); \
  require(BIT == e8 || BIT == e16 || BIT == e32); \
  reg_t rd_tmp = RD; \
  reg_t rs1 = RS1; \
  reg_t rs2 = RS2; \
  sreg_t len = 32 / BIT; \
  for (sreg_t i = len - 1; i >= 0; --i) {

#define P_REDUCTION_LOOP_BASE(BIT, BIT_INNER, USE_RD) \
  require_extension(EXT_ZPN); \
  require(BIT == e16 || BIT == e32 || BIT == e64); \
  reg_t rd_tmp = USE_RD ? zext_xlen(RD) : 0; \
  reg_t rs1 = zext_xlen(RS1); \
  reg_t rs2 = zext_xlen(RS2); \
  sreg_t len = 64 / BIT; \
  sreg_t len_inner = BIT / BIT_INNER; \
  for (sreg_t i = len - 1; i >= 0; --i) { \
    sreg_t pd_res = P_FIELD(rd_tmp, i, BIT); \
    for (sreg_t j = i * len_inner; j < (i + 1) * len_inner; ++j) {

#define P_REDUCTION_ULOOP_BASE(BIT, BIT_INNER, USE_RD) \
  require_extension(EXT_ZPN); \
  require(BIT == e16 || BIT == e32 || BIT == e64); \
  reg_t rd_tmp = USE_RD ? zext_xlen(RD) : 0; \
  reg_t rs1 = zext_xlen(RS1); \
  reg_t rs2 = zext_xlen(RS2); \
  sreg_t len = 64 / BIT; \
  sreg_t len_inner = BIT / BIT_INNER; \
  for (sreg_t i = len - 1; i >=0; --i) { \
    reg_t pd_res = P_UFIELD(rd_tmp, i, BIT); \
    for (sreg_t j = i * len_inner; j < (i + 1) * len_inner; ++j) {

#define P_PARAMS(BIT) \
  auto pd = P_FIELD(rd_tmp, i, BIT); \
  auto ps1 = P_FIELD(rs1, i, BIT); \
  auto ps2 = P_FIELD(rs2, i, BIT);

#define P_UPARAMS(BIT) \
  auto pd = P_UFIELD(rd_tmp, i, BIT); \
  auto ps1 = P_UFIELD(rs1, i, BIT); \
  auto ps2 = P_UFIELD(rs2, i, BIT);

#define P_CORSS_PARAMS(BIT) \
  auto pd = P_FIELD(rd_tmp, i, BIT); \
  auto UNUSED ps1 = P_FIELD(rs1, i, BIT); \
  auto UNUSED ps2 = P_FIELD(rs2, (i ^ 1), BIT);

#define P_CORSS_UPARAMS(BIT) \
  auto pd = P_UFIELD(rd_tmp, i, BIT); \
  auto ps1 = P_UFIELD(rs1, i, BIT); \
  auto ps2 = P_UFIELD(rs2, (i ^ 1), BIT);

#define P_ONE_PARAMS(BIT) \
  auto pd = P_FIELD(rd_tmp, i, BIT); \
  auto ps1 = P_FIELD(rs1, i, BIT);

#define P_ONE_UPARAMS(BIT) \
  auto pd = P_UFIELD(rd_tmp, i, BIT); \
  auto ps1 = P_UFIELD(rs1, i, BIT);

#define P_ONE_SUPARAMS(BIT) \
  auto pd = P_UFIELD(rd_tmp, i, BIT); \
  auto ps1 = P_FIELD(rs1, i, BIT);

#define P_MUL_PARAMS(BIT) \
  auto pd = P_FIELD(rd_tmp, i, BIT * 2); \
  auto ps1 = P_FIELD(rs1, i, BIT); \
  auto ps2 = P_FIELD(rs2, i, BIT);

#define P_MUL_UPARAMS(BIT) \
  auto pd = P_UFIELD(rd_tmp, i, BIT * 2); \
  auto ps1 = P_UFIELD(rs1, i, BIT); \
  auto ps2 = P_UFIELD(rs2, i, BIT);

#define P_MUL_CROSS_PARAMS(BIT) \
  auto pd = P_FIELD(rd_tmp, i, BIT * 2); \
  auto ps1 = P_FIELD(rs1, i, BIT); \
  auto ps2 = P_FIELD(rs2, (i ^ 1), BIT);

#define P_MUL_CROSS_UPARAMS(BIT) \
  auto pd = P_UFIELD(rd_tmp, i, BIT*2); \
  auto ps1 = P_UFIELD(rs1, i, BIT); \
  auto ps2 = P_UFIELD(rs2, (i ^ 1), BIT);

#define P_REDUCTION_PARAMS(BIT_INNER) \
  auto ps1 = P_FIELD(rs1, j, BIT_INNER); \
  auto ps2 = P_FIELD(rs2, j, BIT_INNER);

#define P_REDUCTION_UPARAMS(BIT_INNER) \
  auto ps1 = P_UFIELD(rs1, j, BIT_INNER); \
  auto ps2 = P_UFIELD(rs2, j, BIT_INNER);

#define P_REDUCTION_SUPARAMS(BIT_INNER) \
  auto ps1 = P_FIELD(rs1, j, BIT_INNER); \
  auto ps2 = P_UFIELD(rs2, j, BIT_INNER);

#define P_REDUCTION_CROSS_PARAMS(BIT_INNER) \
  auto ps1 = P_FIELD(rs1, j, BIT_INNER); \
  auto ps2 = P_FIELD(rs2, (j ^ 1), BIT_INNER);

#define P_LOOP_BODY(BIT, BODY) { \
  P_PARAMS(BIT) \
  BODY \
  WRITE_PD(); \
}

#define P_ULOOP_BODY(BIT, BODY) { \
  P_UPARAMS(BIT) \
  BODY \
  WRITE_PD(); \
}

#define P_ONE_LOOP_BODY(BIT, BODY) { \
  P_ONE_PARAMS(BIT) \
  BODY \
  WRITE_PD(); \
}

#define P_CROSS_LOOP_BODY(BIT, BODY) { \
  P_CORSS_PARAMS(BIT) \
  BODY \
  WRITE_PD(); \
}

#define P_CROSS_ULOOP_BODY(BIT, BODY) { \
  P_CORSS_UPARAMS(BIT) \
  BODY \
  WRITE_PD(); \
}

#define P_ONE_ULOOP_BODY(BIT, BODY) { \
  P_ONE_UPARAMS(BIT) \
  BODY \
  WRITE_PD(); \
}

#define P_MUL_LOOP_BODY(BIT, BODY) { \
  P_MUL_PARAMS(BIT) \
  BODY \
  WRITE_PD(); \
}

#define P_MUL_ULOOP_BODY(BIT, BODY) { \
  P_MUL_UPARAMS(BIT) \
  BODY \
  WRITE_PD(); \
}

#define P_MUL_CROSS_LOOP_BODY(BIT, BODY) { \
  P_MUL_CROSS_PARAMS(BIT) \
  BODY \
  WRITE_PD(); \
}

#define P_MUL_CROSS_ULOOP_BODY(BIT, BODY) { \
  P_MUL_CROSS_UPARAMS(BIT) \
  BODY \
  WRITE_PD(); \
}

#define P_LOOP(BIT, BODY) \
  P_LOOP_BASE(BIT) \
  P_LOOP_BODY(BIT, BODY) \
  P_LOOP_END()

#define P_ONE_LOOP(BIT, BODY) \
  P_ONE_LOOP_BASE(BIT) \
  P_ONE_LOOP_BODY(BIT, BODY) \
  P_LOOP_END()

#define P_ULOOP(BIT, BODY) \
  P_LOOP_BASE(BIT) \
  P_ULOOP_BODY(BIT, BODY) \
  P_LOOP_END()

#define P_CROSS_LOOP(BIT, BODY1, BODY2) \
  P_LOOP_BASE(BIT) \
  P_CROSS_LOOP_BODY(BIT, BODY1) \
  --i; \
  if (sizeof(#BODY2) == 1) { \
    P_CROSS_LOOP_BODY(BIT, BODY1) \
  } \
  else { \
    P_CROSS_LOOP_BODY(BIT, BODY2) \
  } \
  P_LOOP_END()

#define P_CROSS_ULOOP(BIT, BODY1, BODY2) \
  P_LOOP_BASE(BIT) \
  P_CROSS_ULOOP_BODY(BIT, BODY1) \
  --i; \
  P_CROSS_ULOOP_BODY(BIT, BODY2) \
  P_LOOP_END()

#define P_STRAIGHT_LOOP(BIT, BODY1, BODY2) \
  P_LOOP_BASE(BIT) \
  P_LOOP_BODY(BIT, BODY1) \
  --i; \
  P_LOOP_BODY(BIT, BODY2) \
  P_LOOP_END()

#define P_STRAIGHT_ULOOP(BIT, BODY1, BODY2) \
  P_LOOP_BASE(BIT) \
  P_ULOOP_BODY(BIT, BODY1) \
  --i; \
  P_ULOOP_BODY(BIT, BODY2) \
  P_LOOP_END()

#define P_X_LOOP(BIT, RS2_LOW_BIT, BODY) \
  P_X_LOOP_BASE(BIT, RS2_LOW_BIT) \
  P_ONE_LOOP_BODY(BIT, BODY) \
  P_LOOP_END()

#define P_X_ULOOP(BIT, RS2_LOW_BIT, BODY) \
  P_X_LOOP_BASE(BIT, RS2_LOW_BIT) \
  P_ONE_ULOOP_BODY(BIT, BODY) \
  P_LOOP_END()

#define P_I_LOOP(BIT, IMMBIT, BODY) \
  P_I_LOOP_BASE(BIT, IMMBIT) \
  P_ONE_LOOP_BODY(BIT, BODY) \
  P_LOOP_END()

#define P_I_ULOOP(BIT, IMMBIT, BODY) \
  P_I_LOOP_BASE(BIT, IMMBIT) \
  P_ONE_ULOOP_BODY(BIT, BODY) \
  P_LOOP_END()

#define P_MUL_LOOP(BIT, BODY) \
  P_MUL_LOOP_BASE(BIT) \
  P_MUL_LOOP_BODY(BIT, BODY) \
  P_PAIR_LOOP_END()

#define P_MUL_ULOOP(BIT, BODY) \
  P_MUL_LOOP_BASE(BIT) \
  P_MUL_ULOOP_BODY(BIT, BODY) \
  P_PAIR_LOOP_END()

#define P_MUL_CROSS_LOOP(BIT, BODY) \
  P_MUL_LOOP_BASE(BIT) \
  P_MUL_CROSS_LOOP_BODY(BIT, BODY) \
  P_PAIR_LOOP_END()

#define P_MUL_CROSS_ULOOP(BIT, BODY) \
  P_MUL_LOOP_BASE(BIT) \
  P_MUL_CROSS_ULOOP_BODY(BIT, BODY) \
  P_PAIR_LOOP_END()

#define P_REDUCTION_LOOP(BIT, BIT_INNER, USE_RD, IS_SAT, BODY) \
  P_REDUCTION_LOOP_BASE(BIT, BIT_INNER, USE_RD) \
  P_REDUCTION_PARAMS(BIT_INNER) \
  BODY \
  P_REDUCTION_LOOP_END(BIT, IS_SAT)

#define P_REDUCTION_ULOOP(BIT, BIT_INNER, USE_RD, IS_SAT, BODY) \
  P_REDUCTION_ULOOP_BASE(BIT, BIT_INNER, USE_RD) \
  P_REDUCTION_UPARAMS(BIT_INNER) \
  BODY \
  P_REDUCTION_ULOOP_END(BIT, IS_SAT)

#define P_REDUCTION_SULOOP(BIT, BIT_INNER, USE_RD, IS_SAT, BODY) \
  P_REDUCTION_LOOP_BASE(BIT, BIT_INNER, USE_RD) \
  P_REDUCTION_SUPARAMS(BIT_INNER) \
  BODY \
  P_REDUCTION_LOOP_END(BIT, IS_SAT)

#define P_REDUCTION_CROSS_LOOP(BIT, BIT_INNER, USE_RD, IS_SAT, BODY) \
  P_REDUCTION_LOOP_BASE(BIT, BIT_INNER, USE_RD) \
  P_REDUCTION_CROSS_PARAMS(BIT_INNER) \
  BODY \
  P_REDUCTION_LOOP_END(BIT, IS_SAT)

#define P_LOOP_END() \
  } \
  WRITE_RD(sext_xlen(rd_tmp));

#define P_PAIR_LOOP_END() \
  } \
  if (xlen == 32) { \
    WRITE_RD_PAIR(rd_tmp); \
  } \
  else { \
    WRITE_RD(sext_xlen(rd_tmp)); \
  }

#define P_REDUCTION_LOOP_END(BIT, IS_SAT) \
    } \
    if (IS_SAT) { \
      P_SAT(pd_res, BIT); \
    } \
    type_usew_t<BIT>::type pd = pd_res; \
    WRITE_PD(); \
  } \
  WRITE_RD(sext_xlen(rd_tmp));

#define P_REDUCTION_ULOOP_END(BIT, IS_SAT) \
    } \
    if (IS_SAT) { \
      P_SATU(pd_res, BIT); \
    } \
    type_usew_t<BIT>::type pd = pd_res; \
    WRITE_PD(); \
  } \
  WRITE_RD(sext_xlen(rd_tmp));

#define P_SUNPKD8(X, Y) \
  require_extension(EXT_ZPN); \
  reg_t rd_tmp = 0; \
  int16_t pd[4] = { \
    P_SB(RS1, Y), \
    P_SB(RS1, X), \
    P_SB(RS1, Y + 4), \
    P_SB(RS1, X + 4), \
  }; \
  if (xlen == 64) { \
    memcpy(&rd_tmp, pd, 8); \
  } else { \
    memcpy(&rd_tmp, pd, 4); \
  } \
  WRITE_RD(sext_xlen(rd_tmp));

#define P_ZUNPKD8(X, Y) \
  require_extension(EXT_ZPN); \
  reg_t rd_tmp = 0; \
  uint16_t pd[4] = { \
    P_B(RS1, Y), \
    P_B(RS1, X), \
    P_B(RS1, Y + 4), \
    P_B(RS1, X + 4), \
  }; \
  if (xlen == 64) { \
    memcpy(&rd_tmp, pd, 8); \
  } else { \
    memcpy(&rd_tmp, pd, 4); \
  } \
  WRITE_RD(sext_xlen(rd_tmp));

#define P_PK(BIT, X, Y) \
  require_extension(EXT_ZPN); \
  require(BIT == e16 || BIT == e32); \
  reg_t rd_tmp = 0, UNUSED rs1 = RS1, UNUSED rs2 = RS2; \
  for (sreg_t i = 0; i < xlen / BIT / 2; i++) { \
    rd_tmp = set_field(rd_tmp, make_mask64(i * 2 * BIT, BIT), \
      P_UFIELD(RS2, i * 2 + Y, BIT)); \
    rd_tmp = set_field(rd_tmp, make_mask64((i * 2 + 1) * BIT, BIT), \
      P_UFIELD(RS1, i * 2 + X, BIT)); \
  } \
  WRITE_RD(sext_xlen(rd_tmp));

#define P_64_PROFILE_BASE() \
  require_extension(EXT_ZPSFOPERAND); \
  sreg_t rd, rs1, rs2;

#define P_64_UPROFILE_BASE() \
  require_extension(EXT_ZPSFOPERAND); \
  reg_t rd, rs1, rs2;

#define P_64_PROFILE_PARAM(USE_RD, INPUT_PAIR) \
  if (xlen == 32) { \
    rs1 = INPUT_PAIR ? RS1_PAIR : RS1; \
    rs2 = INPUT_PAIR ? RS2_PAIR : RS2; \
    rd = USE_RD ? RD_PAIR : 0; \
  } else { \
    rs1 = RS1; \
    rs2 = RS2; \
    rd = USE_RD ? RD : 0; \
  }

#define P_64_PROFILE(BODY) \
  P_64_PROFILE_BASE() \
  P_64_PROFILE_PARAM(false, true) \
  BODY \
  P_64_PROFILE_END() \

#define P_64_UPROFILE(BODY) \
  P_64_UPROFILE_BASE() \
  P_64_PROFILE_PARAM(false, true) \
  BODY \
  P_64_PROFILE_END() \

#define P_64_PROFILE_REDUCTION(BIT, BODY) \
  P_64_PROFILE_BASE() \
  P_64_PROFILE_PARAM(true, false) \
  for (sreg_t i = 0; i < xlen / BIT; i++) { \
    sreg_t ps1 = P_FIELD(rs1, i, BIT); \
    sreg_t ps2 = P_FIELD(rs2, i, BIT); \
    BODY \
  } \
  P_64_PROFILE_END() \

#define P_64_UPROFILE_REDUCTION(BIT, BODY) \
  P_64_UPROFILE_BASE() \
  P_64_PROFILE_PARAM(true, false) \
  for (sreg_t i = 0; i < xlen / BIT; i++) { \
    reg_t ps1 = P_UFIELD(rs1, i, BIT); \
    reg_t ps2 = P_UFIELD(rs2, i, BIT); \
    BODY \
  } \
  P_64_PROFILE_END() \

#define P_64_PROFILE_END() \
  if (xlen == 32) { \
    WRITE_RD_PAIR(rd); \
  } else { \
    WRITE_RD(sext_xlen(rd)); \
  }

#endif

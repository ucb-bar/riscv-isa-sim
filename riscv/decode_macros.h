// See LICENSE for license details.

#ifndef _RISCV_DECODE_H
#define _RISCV_DECODE_H

#if (-1 != ~0) || ((-1 >> 1) != -1)
# error spike requires a two''s-complement c++ implementation
#endif

#include <algorithm>
#include <cstdint>
#include <string.h>
#include <strings.h>
#include "encoding.h"
#include "config.h"
#include "common.h"
#include "softfloat_types.h"
#include "specialize.h"
#include "p_ext_macros.h"
#include "v_ext_macros.h"
#include <cinttypes>
#include <type_traits>

typedef int64_t sreg_t;
typedef uint64_t reg_t;

#ifdef __SIZEOF_INT128__
typedef __int128 int128_t;
typedef unsigned __int128 uint128_t;
#endif

const int NXPR = 32;
const int NFPR = 32;
const int NVPR = 32;
const int NCSR = 4096;

#define X_RA 1
#define X_SP 2
#define X_S0 8
#define X_A0 10
#define X_A1 11
#define X_Sn 16

#define VCSR_VXRM_SHIFT 1
#define VCSR_VXRM  (0x3 << VCSR_VXRM_SHIFT)

#define VCSR_VXSAT_SHIFT 0
#define VCSR_VXSAT  (0x1 << VCSR_VXSAT_SHIFT)

#define FP_RD_NE  0
#define FP_RD_0   1
#define FP_RD_DN  2
#define FP_RD_UP  3
#define FP_RD_NMM 4

#define FSR_RD_SHIFT 5
#define FSR_RD   (0x7 << FSR_RD_SHIFT)

#define FPEXC_NX 0x01
#define FPEXC_UF 0x02
#define FPEXC_OF 0x04
#define FPEXC_DZ 0x08
#define FPEXC_NV 0x10

#define FSR_AEXC_SHIFT 0
#define FSR_NVA  (FPEXC_NV << FSR_AEXC_SHIFT)
#define FSR_OFA  (FPEXC_OF << FSR_AEXC_SHIFT)
#define FSR_UFA  (FPEXC_UF << FSR_AEXC_SHIFT)
#define FSR_DZA  (FPEXC_DZ << FSR_AEXC_SHIFT)
#define FSR_NXA  (FPEXC_NX << FSR_AEXC_SHIFT)
#define FSR_AEXC (FSR_NVA | FSR_OFA | FSR_UFA | FSR_DZA | FSR_NXA)

#define insn_length(x) \
  (((x) & 0x03) < 0x03 ? 2 : \
   ((x) & 0x1f) < 0x1f ? 4 : \
   ((x) & 0x3f) < 0x3f ? 6 : \
   8)
#define MAX_INSN_LENGTH 8
#define PC_ALIGN 2

#define Sn(n) ((n) < 2 ? X_S0 + (n) : X_Sn + (n))

typedef uint64_t insn_bits_t;
class insn_t
{
public:
  insn_t() = default;
  insn_t(insn_bits_t bits) : b(bits) {}
  insn_bits_t bits() { return b; }
  int length() { return insn_length(b); }
  int64_t i_imm() { return xs(20, 12); }
  int64_t shamt() { return x(20, 6); }
  int64_t s_imm() { return x(7, 5) + (xs(25, 7) << 5); }
  int64_t sb_imm() { return (x(8, 4) << 1) + (x(25, 6) << 5) + (x(7, 1) << 11) + (imm_sign() << 12); }
  int64_t u_imm() { return xs(12, 20) << 12; }
  int64_t uj_imm() { return (x(21, 10) << 1) + (x(20, 1) << 11) + (x(12, 8) << 12) + (imm_sign() << 20); }
  uint64_t rd() { return x(7, 5); }
  uint64_t rs1() { return x(15, 5); }
  uint64_t rs2() { return x(20, 5); }
  uint64_t rs3() { return x(27, 5); }
  uint64_t rm() { return x(12, 3); }
  uint64_t csr() { return x(20, 12); }
  uint64_t iorw() { return x(20, 8); }
  uint64_t bs() { return x(30, 2); } // Crypto ISE - SM4/AES32 byte select.
  uint64_t rcon() { return x(20, 4); } // Crypto ISE - AES64 round const.

  int64_t rvc_imm() { return x(2, 5) + (xs(12, 1) << 5); }
  int64_t rvc_zimm() { return x(2, 5) + (x(12, 1) << 5); }
  int64_t rvc_addi4spn_imm() { return (x(6, 1) << 2) + (x(5, 1) << 3) + (x(11, 2) << 4) + (x(7, 4) << 6); }
  int64_t rvc_addi16sp_imm() { return (x(6, 1) << 4) + (x(2, 1) << 5) + (x(5, 1) << 6) + (x(3, 2) << 7) + (xs(12, 1) << 9); }
  int64_t rvc_lwsp_imm() { return (x(4, 3) << 2) + (x(12, 1) << 5) + (x(2, 2) << 6); }
  int64_t rvc_ldsp_imm() { return (x(5, 2) << 3) + (x(12, 1) << 5) + (x(2, 3) << 6); }
  int64_t rvc_swsp_imm() { return (x(9, 4) << 2) + (x(7, 2) << 6); }
  int64_t rvc_sdsp_imm() { return (x(10, 3) << 3) + (x(7, 3) << 6); }
  int64_t rvc_lw_imm() { return (x(6, 1) << 2) + (x(10, 3) << 3) + (x(5, 1) << 6); }
  int64_t rvc_ld_imm() { return (x(10, 3) << 3) + (x(5, 2) << 6); }
  int64_t rvc_j_imm() { return (x(3, 3) << 1) + (x(11, 1) << 4) + (x(2, 1) << 5) + (x(7, 1) << 6) + (x(6, 1) << 7) + (x(9, 2) << 8) + (x(8, 1) << 10) + (xs(12, 1) << 11); }
  int64_t rvc_b_imm() { return (x(3, 2) << 1) + (x(10, 2) << 3) + (x(2, 1) << 5) + (x(5, 2) << 6) + (xs(12, 1) << 8); }
  int64_t rvc_simm3() { return x(10, 3); }
  uint64_t rvc_rd() { return rd(); }
  uint64_t rvc_rs1() { return rd(); }
  uint64_t rvc_rs2() { return x(2, 5); }
  uint64_t rvc_rs1s() { return 8 + x(7, 3); }
  uint64_t rvc_rs2s() { return 8 + x(2, 3); }

  uint64_t rvc_lbimm() { return (x(5, 1) << 1) + x(6, 1); }
  uint64_t rvc_lhimm() { return (x(5, 1) << 1); }

  uint64_t rvc_sreg1() { return x(7, 3); }
  uint64_t rvc_sreg2() { return x(2, 3); }
  uint64_t rvc_rlist() { return x(4, 4); }
  uint64_t rvc_spimm() { return x(2, 2) << 4; }

  uint64_t rvc_index() { return x(2, 8); }

  uint64_t v_vm() { return x(25, 1); }
  uint64_t v_wd() { return x(26, 1); }
  uint64_t v_nf() { return x(29, 3); }
  uint64_t v_simm5() { return xs(15, 5); }
  uint64_t v_zimm5() { return x(15, 5); }
  uint64_t v_zimm10() { return x(20, 10); }
  uint64_t v_zimm11() { return x(20, 11); }
  uint64_t v_lmul() { return x(20, 2); }
  uint64_t v_frac_lmul() { return x(22, 1); }
  uint64_t v_sew() { return 1 << (x(23, 3) + 3); }
  uint64_t v_width() { return x(12, 3); }
  uint64_t v_mop() { return x(26, 2); }
  uint64_t v_lumop() { return x(20, 5); }
  uint64_t v_sumop() { return x(20, 5); }
  uint64_t v_vta() { return x(26, 1); }
  uint64_t v_vma() { return x(27, 1); }
  uint64_t v_mew() { return x(28, 1); }

  uint64_t p_imm2() { return x(20, 2); }
  uint64_t p_imm3() { return x(20, 3); }
  uint64_t p_imm4() { return x(20, 4); }
  uint64_t p_imm5() { return x(20, 5); }
  uint64_t p_imm6() { return x(20, 6); }

  uint64_t zcmp_regmask() {
    unsigned mask = 0;
    uint64_t rlist = rvc_rlist();

    if (rlist >= 4)
      mask |= 1U << X_RA;

    for (reg_t i = 5; i <= rlist; i++)
        mask |= 1U << Sn(i - 5);

    if (rlist == 15)
      mask |= 1U << Sn(11);

    return mask;
  }

  uint64_t zcmp_stack_adjustment(int xlen) {
    reg_t stack_adj_base = 0;
    switch (rvc_rlist()) {
    case 15:
      stack_adj_base += 16;
    case 14:
      if (xlen == 64)
        stack_adj_base += 16;
    case 13:
    case 12:
      stack_adj_base += 16;
    case 11:
    case 10:
      if (xlen == 64)
        stack_adj_base += 16;
    case 9:
    case 8:
      stack_adj_base += 16;
    case 7:
    case 6:
      if (xlen == 64)
        stack_adj_base += 16;
    case 5:
    case 4:
      stack_adj_base += 16;
      break;
    }

    return stack_adj_base + rvc_spimm();
  }

private:
  insn_bits_t b;
  uint64_t x(int lo, int len) { return (b >> lo) & ((insn_bits_t(1) << len) - 1); }
  uint64_t xs(int lo, int len) { return int64_t(b) << (64 - lo - len) >> (64 - len); }
  uint64_t imm_sign() { return xs(31, 1); }
};

template <class T, size_t N, bool zero_reg>
class regfile_t
{
public:
  void write(size_t i, T value)
  {
    if (!zero_reg || i != 0)
      data[i] = value;
  }
  const T& operator [] (size_t i) const
  {
    return data[i];
  }
  regfile_t()
  {
    reset();
  }
  void reset()
  {
    memset(data, 0, sizeof(data));
  }
private:
  T data[N];
};

// helpful macros, etc
#define MMU (*p->get_mmu())
#define STATE (*p->get_state())
#define FLEN (p->get_flen())
#define CHECK_REG(reg) ((void) 0)
#define READ_REG(reg) (CHECK_REG(reg), STATE.XPR[reg])
#define READ_FREG(reg) STATE.FPR[reg]
#define RD READ_REG(insn.rd())
#define RS1 READ_REG(insn.rs1())
#define RS2 READ_REG(insn.rs2())
#define RS3 READ_REG(insn.rs3())
#define WRITE_RD(value) WRITE_REG(insn.rd(), value)

#ifndef RISCV_ENABLE_COMMITLOG
# define WRITE_REG(reg, value) ({ CHECK_REG(reg); STATE.XPR.write(reg, value); })
# define WRITE_FREG(reg, value) DO_WRITE_FREG(reg, freg(value))
# define WRITE_VSTATUS {}
#else
   /* 0 : int
    * 1 : floating
    * 2 : vector reg
    * 3 : vector hint
    * 4 : csr
    */
# define WRITE_REG(reg, value) ({ \
    reg_t wdata = (value); /* value may have side effects */ \
    STATE.log_reg_write[(reg) << 4] = {wdata, 0}; \
    CHECK_REG(reg); \
    STATE.XPR.write(reg, wdata); \
  })
# define WRITE_FREG(reg, value) ({ \
    freg_t wdata = freg(value); /* value may have side effects */ \
    STATE.log_reg_write[((reg) << 4) | 1] = wdata; \
    DO_WRITE_FREG(reg, wdata); \
  })
# define WRITE_VSTATUS STATE.log_reg_write[3] = {0, 0};
#endif

// RVC macros
#define WRITE_RVC_RS1S(value) WRITE_REG(insn.rvc_rs1s(), value)
#define WRITE_RVC_RS2S(value) WRITE_REG(insn.rvc_rs2s(), value)
#define WRITE_RVC_FRS2S(value) WRITE_FREG(insn.rvc_rs2s(), value)
#define RVC_RS1 READ_REG(insn.rvc_rs1())
#define RVC_RS2 READ_REG(insn.rvc_rs2())
#define RVC_RS1S READ_REG(insn.rvc_rs1s())
#define RVC_RS2S READ_REG(insn.rvc_rs2s())
#define RVC_FRS2 READ_FREG(insn.rvc_rs2())
#define RVC_FRS2S READ_FREG(insn.rvc_rs2s())
#define RVC_SP READ_REG(X_SP)

// Zc* macros
#define RVC_SREG1 (Sn(insn.rvc_sreg1()))
#define RVC_SREG2 (Sn(insn.rvc_sreg2()))
#define SP READ_REG(X_SP)
#define RA READ_REG(X_RA)

// FPU macros
#define READ_ZDINX_REG(reg) (xlen == 32 ? f64(READ_REG_PAIR(reg)) : f64(STATE.XPR[reg] & (uint64_t)-1))
#define READ_FREG_H(reg) (p->extension_enabled(EXT_ZFINX) ? f16(STATE.XPR[reg] & (uint16_t)-1) : f16(READ_FREG(reg)))
#define READ_FREG_F(reg) (p->extension_enabled(EXT_ZFINX) ? f32(STATE.XPR[reg] & (uint32_t)-1) : f32(READ_FREG(reg)))
#define READ_FREG_D(reg) (p->extension_enabled(EXT_ZFINX) ? READ_ZDINX_REG(reg) : f64(READ_FREG(reg)))
#define FRS1 READ_FREG(insn.rs1())
#define FRS2 READ_FREG(insn.rs2())
#define FRS3 READ_FREG(insn.rs3())
#define FRS1_H READ_FREG_H(insn.rs1())
#define FRS1_F READ_FREG_F(insn.rs1())
#define FRS1_D READ_FREG_D(insn.rs1())
#define FRS2_H READ_FREG_H(insn.rs2())
#define FRS2_F READ_FREG_F(insn.rs2())
#define FRS2_D READ_FREG_D(insn.rs2())
#define FRS3_H READ_FREG_H(insn.rs3())
#define FRS3_F READ_FREG_F(insn.rs3())
#define FRS3_D READ_FREG_D(insn.rs3())
#define dirty_fp_state  STATE.sstatus->dirty(SSTATUS_FS)
#define dirty_ext_state STATE.sstatus->dirty(SSTATUS_XS)
#define dirty_vs_state  STATE.sstatus->dirty(SSTATUS_VS)
#define DO_WRITE_FREG(reg, value) (STATE.FPR.write(reg, value), dirty_fp_state)
#define WRITE_FRD(value) WRITE_FREG(insn.rd(), value)
#define WRITE_FRD_H(value) \
do { \
  if (p->extension_enabled(EXT_ZFINX)) \
    WRITE_REG(insn.rd(), sext_xlen((int16_t)((value).v))); \
  else { \
    WRITE_FRD(value); \
  } \
} while (0)
#define WRITE_FRD_F(value) \
do { \
  if (p->extension_enabled(EXT_ZFINX)) \
    WRITE_REG(insn.rd(), sext_xlen((value).v)); \
  else { \
    WRITE_FRD(value); \
  } \
} while (0)
#define WRITE_FRD_D(value) \
do { \
  if (p->extension_enabled(EXT_ZFINX)) { \
    if (xlen == 32) { \
      uint64_t val = (value).v; \
      WRITE_RD_PAIR(val); \
    } else { \
      WRITE_REG(insn.rd(), (value).v); \
    } \
  } else { \
    WRITE_FRD(value); \
  } \
} while (0)
 
#define SHAMT (insn.i_imm() & 0x3F)
#define BRANCH_TARGET (pc + insn.sb_imm())
#define JUMP_TARGET (pc + insn.uj_imm())
#define RM ({ int rm = insn.rm(); \
              if (rm == 7) rm = STATE.frm->read(); \
              if (rm > 4) throw trap_illegal_instruction(insn.bits()); \
              rm; })

#define get_field(reg, mask) \
  (((reg) & (std::remove_cv<decltype(reg)>::type)(mask)) / ((mask) & ~((mask) << 1)))

#define set_field(reg, mask, val) \
  (((reg) & ~(std::remove_cv<decltype(reg)>::type)(mask)) | (((std::remove_cv<decltype(reg)>::type)(val) * ((mask) & ~((mask) << 1))) & (std::remove_cv<decltype(reg)>::type)(mask)))

#define require_privilege(p) require(STATE.prv >= (p))
#define require_novirt() (unlikely(STATE.v) ? throw trap_virtual_instruction(insn.bits()) : (void) 0)
#define require_hs_qualified(cond) (STATE.v && !(cond) ? require_novirt() : require(cond))
#define require_privilege_hs_qualified(p) require_hs_qualified(STATE.prv >= (p))
#define require_rv64 require(xlen == 64)
#define require_rv32 require(xlen == 32)
#define require_extension(s) require(p->extension_enabled(s))
#define require_either_extension(A,B) require(p->extension_enabled(A) || p->extension_enabled(B));
#define require_impl(s) require(p->supports_impl(s))
#define require_fs          require(STATE.sstatus->enabled(SSTATUS_FS))
#define require_fp          STATE.fflags->verify_permissions(insn, false)
#define require_accelerator require(STATE.sstatus->enabled(SSTATUS_XS))
#define require_vector_vs   require(STATE.sstatus->enabled(SSTATUS_VS))
#define require_vector(alu) \
  do { \
    require_vector_vs; \
    require_extension('V'); \
    require(!P.VU.vill); \
    if (alu && !P.VU.vstart_alu) \
      require(P.VU.vstart->read() == 0); \
    WRITE_VSTATUS; \
    dirty_vs_state; \
  } while (0);
#define require_vector_novtype(is_log) \
  do { \
    require_vector_vs; \
    require_extension('V'); \
    if (is_log) \
      WRITE_VSTATUS; \
    dirty_vs_state; \
  } while (0);
#define require_align(val, pos) require(is_aligned(val, pos))
#define require_noover(astart, asize, bstart, bsize) \
  require(!is_overlapped(astart, asize, bstart, bsize))
#define require_noover_widen(astart, asize, bstart, bsize) \
  require(!is_overlapped_widen(astart, asize, bstart, bsize))
#define require_vm do { if (insn.v_vm() == 0) require(insn.rd() != 0); } while (0);
#define require_envcfg(field) \
  do { \
    if (((STATE.prv != PRV_M) && (m##field == 0)) || \
        ((STATE.prv == PRV_U && !STATE.v) && (s##field == 0))) \
      throw trap_illegal_instruction(insn.bits()); \
    else if (STATE.v && ((h##field == 0) || \
                        ((STATE.prv == PRV_U) && (s##field == 0)))) \
      throw trap_virtual_instruction(insn.bits()); \
  } while (0);

#define require_zcmp_pushpop \
  do { \
    require_extension(EXT_ZCMP); \
    reg_t rlist = insn.rvc_rlist(); \
    require(rlist >= 4); \
    \
    if (p->extension_enabled('E')) { \
      require(rlist <= 6); \
    } \
  } while (0);

#define set_fp_exceptions ({ if (softfloat_exceptionFlags) { \
                               STATE.fflags->write(STATE.fflags->read() | softfloat_exceptionFlags); \
                             } \
                             softfloat_exceptionFlags = 0; })

#define sext32(x) ((sreg_t)(int32_t)(x))
#define zext32(x) ((reg_t)(uint32_t)(x))
#define sext_xlen(x) (((sreg_t)(x) << (64 - xlen)) >> (64 - xlen))
#define zext(x, pos) (((reg_t)(x) << (64 - (pos))) >> (64 - (pos)))
#define zext_xlen(x) zext(x, xlen)

#define set_pc(x) \
  do { p->check_pc_alignment(x); \
       npc = sext_xlen(x); \
     } while (0)

#define set_pc_and_serialize(x) \
  do { reg_t __npc = (x) & p->pc_alignment_mask(); \
       npc = PC_SERIALIZE_AFTER; \
       STATE.pc = __npc; \
     } while (0)

class wait_for_interrupt_t {};

#define wfi() \
  do { set_pc_and_serialize(npc); \
       throw wait_for_interrupt_t(); \
     } while (0)

#define serialize() set_pc_and_serialize(npc)

/* Sentinel PC values to serialize simulator pipeline */
#define PC_SERIALIZE_BEFORE 3
#define PC_SERIALIZE_AFTER 5
#define invalid_pc(pc) ((pc) & 1)

/* Convenience wrappers to simplify softfloat code sequences */
#define isBoxedF16(r) (isBoxedF32(r) && ((uint64_t)((r.v[0] >> 16) + 1) == ((uint64_t)1 << 48)))
#define unboxF16(r) (isBoxedF16(r) ? (uint16_t)r.v[0] : defaultNaNF16UI)
#define isBoxedF32(r) (isBoxedF64(r) && ((uint32_t)((r.v[0] >> 32) + 1) == 0))
#define unboxF32(r) (isBoxedF32(r) ? (uint32_t)r.v[0] : defaultNaNF32UI)
#define isBoxedF64(r) ((r.v[1] + 1) == 0)
#define unboxF64(r) (isBoxedF64(r) ? r.v[0] : defaultNaNF64UI)
typedef float128_t freg_t;
inline float16_t f16(uint16_t v) { return { v }; }
inline float32_t f32(uint32_t v) { return { v }; }
inline float64_t f64(uint64_t v) { return { v }; }
inline float16_t f16(freg_t r) { return f16(unboxF16(r)); }
inline float32_t f32(freg_t r) { return f32(unboxF32(r)); }
inline float64_t f64(freg_t r) { return f64(unboxF64(r)); }
inline float128_t f128(freg_t r) { return r; }
inline freg_t freg(float16_t f) { return { ((uint64_t)-1 << 16) | f.v, (uint64_t)-1 }; }
inline freg_t freg(float32_t f) { return { ((uint64_t)-1 << 32) | f.v, (uint64_t)-1 }; }
inline freg_t freg(float64_t f) { return { f.v, (uint64_t)-1 }; }
inline freg_t freg(float128_t f) { return f; }
#define F16_SIGN ((uint16_t)1 << 15)
#define F32_SIGN ((uint32_t)1 << 31)
#define F64_SIGN ((uint64_t)1 << 63)
#define fsgnj16(a, b, n, x) \
  f16((f16(a).v & ~F16_SIGN) | ((((x) ? f16(a).v : (n) ? F16_SIGN : 0) ^ f16(b).v) & F16_SIGN))
#define fsgnj32(a, b, n, x) \
  f32((f32(a).v & ~F32_SIGN) | ((((x) ? f32(a).v : (n) ? F32_SIGN : 0) ^ f32(b).v) & F32_SIGN))
#define fsgnj64(a, b, n, x) \
  f64((f64(a).v & ~F64_SIGN) | ((((x) ? f64(a).v : (n) ? F64_SIGN : 0) ^ f64(b).v) & F64_SIGN))

#define isNaNF128(x) isNaNF128UI(x.v[1], x.v[0])
inline float128_t defaultNaNF128()
{
  float128_t nan;
  nan.v[1] = defaultNaNF128UI64;
  nan.v[0] = defaultNaNF128UI0;
  return nan;
}
inline freg_t fsgnj128(freg_t a, freg_t b, bool n, bool x)
{
  a.v[1] = (a.v[1] & ~F64_SIGN) | (((x ? a.v[1] : n ? F64_SIGN : 0) ^ b.v[1]) & F64_SIGN);
  return a;
}
inline freg_t f128_negate(freg_t a)
{
  a.v[1] ^= F64_SIGN;
  return a;
}

#define validate_csr(which, write) ({ \
  if (!STATE.serialized) return PC_SERIALIZE_BEFORE; \
  STATE.serialized = false; \
  /* permissions check occurs in get_csr */ \
  (which); })

/* For debug only. This will fail if the native machine's float types are not IEEE */
inline float to_f(float32_t f) { float r; memcpy(&r, &f, sizeof(r)); return r; }
inline double to_f(float64_t f) { double r; memcpy(&r, &f, sizeof(r)); return r; }
inline long double to_f(float128_t f) { long double r; memcpy(&r, &f, sizeof(r)); return r; }

// Vector macros
#define e8 8      // 8b elements
#define e16 16    // 16b elements
#define e32 32    // 32b elements
#define e64 64    // 64b elements
#define e128 128  // 128b elements
#define e256 256  // 256b elements
#define e512 512  // 512b elements
#define e1024 1024  // 1024b elements

#define vsext(x, sew) (((sreg_t)(x) << (64 - sew)) >> (64 - sew))
#define vzext(x, sew) (((reg_t)(x) << (64 - sew)) >> (64 - sew))

#define DEBUG_RVV 0

#if DEBUG_RVV
#define DEBUG_RVV_FP_VV \
  printf("vfp(%lu) vd=%f vs1=%f vs2=%f\n", i, to_f(vd), to_f(vs1), to_f(vs2));
#define DEBUG_RVV_FP_VF \
  printf("vfp(%lu) vd=%f vs1=%f vs2=%f\n", i, to_f(vd), to_f(rs1), to_f(vs2));
#define DEBUG_RVV_FMA_VV \
  printf("vfma(%lu) vd=%f vs1=%f vs2=%f vd_old=%f\n", i, to_f(vd), to_f(vs1), to_f(vs2), to_f(vd_old));
#define DEBUG_RVV_FMA_VF \
  printf("vfma(%lu) vd=%f vs1=%f vs2=%f vd_old=%f\n", i, to_f(vd), to_f(rs1), to_f(vs2), to_f(vd_old));
#else
#define DEBUG_RVV_FP_VV 0
#define DEBUG_RVV_FP_VF 0
#define DEBUG_RVV_FMA_VV 0
#define DEBUG_RVV_FMA_VF 0
#endif

#define DECLARE_XENVCFG_VARS(field) \
  reg_t m##field = get_field(STATE.menvcfg->read(), MENVCFG_##field); \
  reg_t s##field = get_field(STATE.senvcfg->read(), SENVCFG_##field); \
  reg_t h##field = get_field(STATE.henvcfg->read(), HENVCFG_##field)

#define DEBUG_START             0x0
#define DEBUG_END               (0x1000 - 1)

#endif

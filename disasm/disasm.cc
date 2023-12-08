// See LICENSE for license details.

#include "disasm.h"
#include "decode_macros.h"
#include <cassert>
#include <string>
#include <vector>
#include <cstdarg>
#include <sstream>
#include <stdlib.h>
// For std::reverse:
#include <algorithm>

#ifdef __GNUC__
# pragma GCC diagnostic ignored "-Wunused-variable"
#endif

// Indicates that the next arg (only) is optional.
// If the result of converting the next arg to a string is ""
// then it will not be printed.
static const arg_t* opt = nullptr;

struct : public arg_t {
  std::string to_string(insn_t insn) const {
    return std::to_string((int)insn.i_imm()) + '(' + xpr_name[insn.rs1()] + ')';
  }
} load_address;

struct : public arg_t {
  std::string to_string(insn_t insn) const {
    return std::to_string((int)insn.rvc_lbimm()) + '(' + xpr_name[insn.rvc_rs1s()] + ')';
  }
} rvb_b_address;

struct : public arg_t {
  std::string to_string(insn_t insn) const {
    return std::to_string((int)insn.rvc_lhimm()) + '(' + xpr_name[insn.rvc_rs1s()] + ')';
  }
} rvb_h_address;

struct : public arg_t {
  std::string to_string(insn_t insn) const {
    return std::to_string((int)insn.s_imm()) + '(' + xpr_name[insn.rs1()] + ')';
  }
} store_address;

struct : public arg_t {
  std::string to_string(insn_t insn) const {
    return std::string("(") + xpr_name[insn.rs1()] + ')';
  }
} base_only_address;

struct : public arg_t {
  std::string to_string(insn_t insn) const {
    return xpr_name[insn.rd()];
  }
} xrd;

struct : public arg_t {
  std::string to_string(insn_t insn) const {
    return xpr_name[insn.rs1()];
  }
} xrs1;

struct : public arg_t {
  std::string to_string(insn_t insn) const {
    return std::to_string((uint32_t)insn.rvc_index());
  }
} rvcm_jt_index;

struct : public arg_t {
  std::string to_string(insn_t insn) const {
    int rlist = insn.rvc_rlist();
    if (rlist >= 4) {
      switch(rlist) {
        case 4: return "{ra}";
        case 5: return "{ra, s0}";
        case 15: return "{ra, s0-s11}";
        default: return "{ra, s0-s" + std::to_string(rlist - 5)+'}';
      }
    } else {
      return "unsupport rlist";
    }
  }
} rvcm_pushpop_rlist;

struct : public arg_t {
  std::string to_string(insn_t insn) const {
    return '-' + std::to_string(insn.zcmp_stack_adjustment(32));
  }
} rvcm_push_stack_adj_32;

struct : public arg_t {
  std::string to_string(insn_t insn) const {
    return '-' + std::to_string(insn.zcmp_stack_adjustment(64));
  }
} rvcm_push_stack_adj_64;

struct : public arg_t {
  std::string to_string(insn_t insn) const {
    return std::to_string(insn.zcmp_stack_adjustment(32));
  }
} rvcm_pop_stack_adj_32;

struct : public arg_t {
  std::string to_string(insn_t insn) const {
    return std::to_string(insn.zcmp_stack_adjustment(64));
  }
} rvcm_pop_stack_adj_64;

struct : public arg_t {
  std::string to_string(insn_t insn) const {
    return xpr_name[insn.rs2()];
  }
} xrs2;

struct : public arg_t {
  std::string to_string(insn_t insn) const {
    return xpr_name[insn.rs3()];
  }
} xrs3;

struct : public arg_t {
  std::string to_string(insn_t insn) const {
    return fpr_name[insn.rd()];
  }
} frd;

struct : public arg_t {
  std::string to_string(insn_t insn) const {
    return fpr_name[insn.rs1()];
  }
} frs1;

struct : public arg_t {
  std::string to_string(insn_t insn) const {
    return fpr_name[insn.rs2()];
  }
} frs2;

struct : public arg_t {
  std::string to_string(insn_t insn) const {
    return fpr_name[insn.rs3()];
  }
} frs3;

struct : public arg_t {
  std::string to_string(insn_t insn) const {
    switch (insn.csr())
    {
      #define DECLARE_CSR(name, num) case num: return #name;
      #include "encoding.h"
      #undef DECLARE_CSR
      default:
      {
        char buf[16];
        snprintf(buf, sizeof buf, "unknown_%03" PRIx64, insn.csr());
        return std::string(buf);
      }
    }
  }
} csr;

struct : public arg_t {
  std::string to_string(insn_t insn) const {
    return std::to_string((int)insn.i_imm());
  }
} imm;

struct : public arg_t {
  std::string to_string(insn_t insn) const {
    return std::to_string((int)insn.shamt());
  }
} shamt;

struct : public arg_t {
  std::string to_string(insn_t insn) const {
    std::stringstream s;
    s << std::hex << "0x" << ((uint32_t)insn.u_imm() >> 12);
    return s.str();
  }
} bigimm;

struct : public arg_t {
  std::string to_string(insn_t insn) const {
    return std::to_string(insn.rs1());
  }
} zimm5;

struct : public arg_t {
  std::string to_string(insn_t insn) const {
    return std::to_string(insn.v_zimm6());
  }
} v_zimm6;

struct : public arg_t {
  std::string to_string(insn_t insn) const {
    int32_t target = insn.sb_imm();
    std::string s = target >= 0 ? "pc + " : "pc - ";
    s += std::to_string(abs(target));
    return s;
  }
} branch_target;

struct : public arg_t {
  std::string to_string(insn_t insn) const {
    std::stringstream s;
    int32_t target = insn.uj_imm();
    char sign = target >= 0 ? '+' : '-';
    s << "pc " << sign << std::hex << " 0x" << abs(target);
    return s.str();
  }
} jump_target;

struct : public arg_t {
  std::string to_string(insn_t insn) const {
    return xpr_name[insn.rvc_rs1()];
  }
} rvc_rs1;

struct : public arg_t {
  std::string to_string(insn_t insn) const {
    return xpr_name[insn.rvc_rs2()];
  }
} rvc_rs2;

struct : public arg_t {
  std::string to_string(insn_t insn) const {
    return fpr_name[insn.rvc_rs2()];
  }
} rvc_fp_rs2;

struct : public arg_t {
  std::string to_string(insn_t insn) const {
    return xpr_name[insn.rvc_rs1s()];
  }
} rvc_rs1s;

struct : public arg_t {
  std::string to_string(insn_t insn) const {
    return xpr_name[insn.rvc_rs2s()];
  }
} rvc_rs2s;

struct : public arg_t {
  std::string to_string(insn_t insn) const {
    return xpr_name[RVC_R1S];
  }
} rvc_r1s;

struct : public arg_t {
  std::string to_string(insn_t insn) const {
    return xpr_name[RVC_R2S];
  }
} rvc_r2s;

struct : public arg_t {
  std::string to_string(insn_t insn) const {
    return fpr_name[insn.rvc_rs2s()];
  }
} rvc_fp_rs2s;

struct : public arg_t {
  std::string to_string(insn_t UNUSED insn) const {
    return xpr_name[X_SP];
  }
} rvc_sp;

struct : public arg_t {
  std::string to_string(insn_t insn) const {
    return std::to_string((int)insn.rvc_imm());
  }
} rvc_imm;

struct : public arg_t {
  std::string to_string(insn_t insn) const {
    return std::to_string((int)insn.rvc_addi4spn_imm());
  }
} rvc_addi4spn_imm;

struct : public arg_t {
  std::string to_string(insn_t insn) const {
    return std::to_string((int)insn.rvc_addi16sp_imm());
  }
} rvc_addi16sp_imm;

struct : public arg_t {
  std::string to_string(insn_t insn) const {
    return std::to_string((int)insn.rvc_lwsp_imm());
  }
} rvc_lwsp_imm;

struct : public arg_t {
  std::string to_string(insn_t insn) const {
    return std::to_string((int)(insn.rvc_imm() & 0x3f));
  }
} rvc_shamt;

struct : public arg_t {
  std::string to_string(insn_t insn) const {
    std::stringstream s;
    s << std::hex << "0x" << ((uint32_t)insn.rvc_imm() << 12 >> 12);
    return s.str();
  }
} rvc_uimm;

struct : public arg_t {
  std::string to_string(insn_t insn) const {
    return std::to_string((int)insn.rvc_lwsp_imm()) + '(' + xpr_name[X_SP] + ')';
  }
} rvc_lwsp_address;

struct : public arg_t {
  std::string to_string(insn_t insn) const {
    return std::to_string((int)insn.rvc_ldsp_imm()) + '(' + xpr_name[X_SP] + ')';
  }
} rvc_ldsp_address;

struct : public arg_t {
  std::string to_string(insn_t insn) const {
    return std::to_string((int)insn.rvc_swsp_imm()) + '(' + xpr_name[X_SP] + ')';
  }
} rvc_swsp_address;

struct : public arg_t {
  std::string to_string(insn_t insn) const {
    return std::to_string((int)insn.rvc_sdsp_imm()) + '(' + xpr_name[X_SP] + ')';
  }
} rvc_sdsp_address;

struct : public arg_t {
  std::string to_string(insn_t insn) const {
    return std::to_string((int)insn.rvc_lw_imm()) + '(' + xpr_name[insn.rvc_rs1s()] + ')';
  }
} rvc_lw_address;

struct : public arg_t {
  std::string to_string(insn_t insn) const {
    return std::to_string((int)insn.rvc_ld_imm()) + '(' + xpr_name[insn.rvc_rs1s()] + ')';
  }
} rvc_ld_address;

struct : public arg_t {
  std::string to_string(insn_t insn) const {
    int32_t target = insn.rvc_b_imm();
    std::string s = target >= 0 ? "pc + " : "pc - ";
    s += std::to_string(abs(target));
    return s;
  }
} rvc_branch_target;

struct : public arg_t {
  std::string to_string(insn_t insn) const {
    int32_t target = insn.rvc_j_imm();
    std::string s = target >= 0 ? "pc + " : "pc - ";
    s += std::to_string(abs(target));
    return s;
  }
} rvc_jump_target;

struct : public arg_t {
  std::string to_string(insn_t insn) const {
    return std::string("(") + xpr_name[insn.rs1()] + ')';
  }
} v_address;

struct : public arg_t {
  std::string to_string(insn_t insn) const {
    return vr_name[insn.rd()];
  }
} vd;

struct : public arg_t {
  std::string to_string(insn_t insn) const {
    return vr_name[insn.rs1()];
  }
} vs1;

struct : public arg_t {
  std::string to_string(insn_t insn) const {
    return vr_name[insn.rs2()];
  }
} vs2;

struct : public arg_t {
  std::string to_string(insn_t insn) const {
    return vr_name[insn.rd()];
  }
} vs3;

struct : public arg_t {
  std::string to_string(insn_t insn) const {
    return insn.v_vm() ? "" : "v0.t";
  }
} vm;

struct : public arg_t {
  std::string to_string(insn_t UNUSED insn) const {
    return "v0";
  }
} v0;

struct : public arg_t {
  std::string to_string(insn_t insn) const {
    return std::to_string((int)insn.v_simm5());
  }
} v_simm5;

struct : public arg_t {
  std::string to_string(insn_t insn) const {
    std::stringstream s;
    int sew = insn.v_sew();
    int lmul = insn.v_lmul();
    auto vta = insn.v_vta() == 1 ? "ta" : "tu";
    auto vma = insn.v_vma() == 1 ? "ma" : "mu";
    int newType = (insn.bits() & 0x80000000) ? insn.v_zimm10() : insn.v_zimm11();
    // if bit 31 is set, this is vsetivli and there is a 10-bit vtype, else this is vsetvli and there is an 11-bit vtype
    // If the provided vtype has reserved bits, display the hex version of the vtype instead
    if ((newType >> 8) != 0) {
      s << "0x" << std::hex << newType;
    } else {
      s << "e" << sew;
      if(insn.v_frac_lmul()) {
        std::string lmul_str = "";
        switch(lmul){
          case 3:
            lmul_str = "f2";
            break;
          case 2:
            lmul_str = "f4";
            break;
          case 1:
            lmul_str = "f8";
            break;
          default:
            assert(true && "unsupport fractional LMUL");
        }
        s << ", m" << lmul_str;
      } else {
        s << ", m" << (1 << lmul);
      }
      s << ", " << vta << ", " << vma;
    }

    return s.str();
  }
} v_vtype;

struct : public arg_t {
  std::string to_string(insn_t UNUSED insn) const {
    return "x0";
  }
} x0;

struct : public arg_t {
  std::string to_string(insn_t insn) const {
    std::string s;
    auto iorw = insn.iorw();
    bool has_pre = false;
    static const char type[] = "wroi";
    for (int i = 7; i >= 4; --i) {
      if (iorw & (1ul << i)) {
        s += type[i - 4];
        has_pre = true;
      }
    }

    s += (has_pre ? "," : "");
    for (int i = 3; i >= 0; --i) {
      if (iorw & (1ul << i)) {
        s += type[i];
      }
    }

    return s;
  }
} iorw;

struct : public arg_t {
  std::string to_string(insn_t insn) const {
    return std::to_string((int)insn.p_imm2());
  }
} p_imm2;

struct : public arg_t {
  std::string to_string(insn_t insn) const {
    return std::to_string((int)insn.p_imm3());
  }
} p_imm3;

struct : public arg_t {
  std::string to_string(insn_t insn) const {
    return std::to_string((int)insn.p_imm4());
  }
} p_imm4;

struct : public arg_t {
  std::string to_string(insn_t insn) const {
    return std::to_string((int)insn.p_imm5());
  }
} p_imm5;

struct : public arg_t {
  std::string to_string(insn_t insn) const {
    return std::to_string((int)insn.p_imm6());
  }
} p_imm6;

struct : public arg_t {
  std::string to_string(insn_t insn) const {
    return std::to_string((int)insn.bs());
  }
} bs;

struct : public arg_t {
  std::string to_string(insn_t insn) const {
    return std::to_string((int)insn.rcon());
  }
} rcon;

typedef struct {
  reg_t match;
  reg_t mask;
  const char *fmt;
  std::vector<const arg_t*>& arg;
} custom_fmt_t;

std::string disassembler_t::disassemble(insn_t insn) const
{
  const disasm_insn_t* disasm_insn = lookup(insn);
  return disasm_insn ? disasm_insn->to_string(insn) : "unknown";
}

static void NOINLINE add_noarg_insn(disassembler_t* d, const char* name, uint32_t match, uint32_t mask)
{
  d->add_insn(new disasm_insn_t(name, match, mask, {}));
}

static void NOINLINE add_rtype_insn(disassembler_t* d, const char* name, uint32_t match, uint32_t mask)
{
  d->add_insn(new disasm_insn_t(name, match, mask, {&xrd, &xrs1, &xrs2}));
}

static void NOINLINE add_r1type_insn(disassembler_t* d, const char* name, uint32_t match, uint32_t mask)
{
  d->add_insn(new disasm_insn_t(name, match, mask, {&xrd, &xrs1}));
}

static void NOINLINE add_r3type_insn(disassembler_t* d, const char* name, uint32_t match, uint32_t mask)
{
  d->add_insn(new disasm_insn_t(name, match, mask, {&xrd, &xrs1, &xrs2, &xrs3}));
}

static void NOINLINE add_itype_insn(disassembler_t* d, const char* name, uint32_t match, uint32_t mask)
{
  d->add_insn(new disasm_insn_t(name, match, mask, {&xrd, &xrs1, &imm}));
}

static void NOINLINE add_itype_shift_insn(disassembler_t* d, const char* name, uint32_t match, uint32_t mask)
{
  d->add_insn(new disasm_insn_t(name, match, mask, {&xrd, &xrs1, &shamt}));
}

static void NOINLINE add_xload_insn(disassembler_t* d, const char* name, uint32_t match, uint32_t mask)
{
  d->add_insn(new disasm_insn_t(name, match, mask, {&xrd, &load_address}));
}

static void NOINLINE add_xstore_insn(disassembler_t* d, const char* name, uint32_t match, uint32_t mask)
{
  d->add_insn(new disasm_insn_t(name, match, mask, {&xrs2, &store_address}));
}

static void NOINLINE add_fload_insn(disassembler_t* d, const char* name, uint32_t match, uint32_t mask)
{
  d->add_insn(new disasm_insn_t(name, match, mask, {&frd, &load_address}));
}

static void NOINLINE add_fstore_insn(disassembler_t* d, const char* name, uint32_t match, uint32_t mask)
{
  d->add_insn(new disasm_insn_t(name, match, mask, {&frs2, &store_address}));
}

static void NOINLINE add_xamo_insn(disassembler_t* d, const char* name, uint32_t match, uint32_t mask)
{
  d->add_insn(new disasm_insn_t(name, match, mask, {&xrd, &xrs2, &base_only_address}));
}

static void NOINLINE add_xlr_insn(disassembler_t* d, const char* name, uint32_t match, uint32_t mask)
{
  d->add_insn(new disasm_insn_t(name, match, mask, {&xrd, &base_only_address}));
}

static void NOINLINE add_xst_insn(disassembler_t* d, const char* name, uint32_t match, uint32_t mask)
{
  d->add_insn(new disasm_insn_t(name, match, mask, {&xrs2, &base_only_address}));
}

static void NOINLINE add_btype_insn(disassembler_t* d, const char* name, uint32_t match, uint32_t mask)
{
  d->add_insn(new disasm_insn_t(name, match, mask, {&xrs1, &xrs2, &branch_target}));
}

static void NOINLINE add_b1type_insn(disassembler_t* d, const char* name, uint32_t match, uint32_t mask)
{
  const uint32_t mask_rs2 = 0x1fUL << 20;
  d->add_insn(new disasm_insn_t(name, match, mask | mask_rs2, {&xrs1, &branch_target}));
}

static void NOINLINE add_frtype_insn(disassembler_t* d, const char* name, uint32_t match, uint32_t mask)
{
  d->add_insn(new disasm_insn_t(name, match, mask, {&frd, &frs1, &frs2}));
}

static void NOINLINE add_fr1type_insn(disassembler_t* d, const char* name, uint32_t match, uint32_t mask)
{
  d->add_insn(new disasm_insn_t(name, match, mask, {&frd, &frs1}));
}

static void NOINLINE add_fr3type_insn(disassembler_t* d, const char* name, uint32_t match, uint32_t mask)
{
  d->add_insn(new disasm_insn_t(name, match, mask, {&frd, &frs1, &frs2, &frs3}));
}

static void NOINLINE add_fxtype_insn(disassembler_t* d, const char* name, uint32_t match, uint32_t mask)
{
  d->add_insn(new disasm_insn_t(name, match, mask, {&xrd, &frs1}));
}

static void NOINLINE add_xftype_insn(disassembler_t* d, const char* name, uint32_t match, uint32_t mask)
{
  d->add_insn(new disasm_insn_t(name, match, mask, {&frd, &xrs1}));
}

static void NOINLINE add_fx2type_insn(disassembler_t* d, const char* name, uint32_t match, uint32_t mask)
{
  d->add_insn(new disasm_insn_t(name, match, mask, {&xrd, &frs1, &frs2}));
}

static void NOINLINE add_sfence_insn(disassembler_t* d, const char* name, uint32_t match, uint32_t mask)
{
  d->add_insn(new disasm_insn_t(name, match, mask, {&xrs1, &xrs2}));
}

static void NOINLINE add_pitype3_insn(disassembler_t* d, const char* name, uint32_t match, uint32_t mask)
{
  d->add_insn(new disasm_insn_t(name, match, mask, {&xrd, &xrs1, &p_imm3}));
}

static void NOINLINE add_pitype4_insn(disassembler_t* d, const char* name, uint32_t match, uint32_t mask)
{
  d->add_insn(new disasm_insn_t(name, match, mask, {&xrd, &xrs1, &p_imm4}));
}

static void NOINLINE add_pitype5_insn(disassembler_t* d, const char* name, uint32_t match, uint32_t mask)
{
  d->add_insn(new disasm_insn_t(name, match, mask, {&xrd, &xrs1, &p_imm5}));
}

static void NOINLINE add_pitype6_insn(disassembler_t* d, const char* name, uint32_t match, uint32_t mask)
{
  d->add_insn(new disasm_insn_t(name, match, mask, {&xrd, &xrs1, &p_imm6}));
}

static void NOINLINE add_vector_v_insn(disassembler_t* d, const char* name, uint32_t match, uint32_t mask)
{
  d->add_insn(new disasm_insn_t(name, match, mask, {&vd, &vs2, opt, &vm}));
}

static void NOINLINE add_vector_vv_insn(disassembler_t* d, const char* name, uint32_t match, uint32_t mask)
{
  d->add_insn(new disasm_insn_t(name, match, mask, {&vd, &vs2, &vs1, opt, &vm}));
}

static void NOINLINE add_vector_vx_insn(disassembler_t* d, const char* name, uint32_t match, uint32_t mask)
{
  d->add_insn(new disasm_insn_t(name, match, mask, {&vd, &vs2, &xrs1, opt, &vm}));
}

static void NOINLINE add_vector_vf_insn(disassembler_t* d, const char* name, uint32_t match, uint32_t mask)
{
  d->add_insn(new disasm_insn_t(name, match, mask, {&vd, &vs2, &frs1, opt, &vm}));
}

static void NOINLINE add_vector_vi_insn(disassembler_t* d, const char* name, uint32_t match, uint32_t mask)
{
  d->add_insn(new disasm_insn_t(name, match, mask, {&vd, &vs2, &v_simm5, opt, &vm}));
}

static void NOINLINE add_vector_viu_insn(disassembler_t* d, const char* name, uint32_t match, uint32_t mask)
{
  d->add_insn(new disasm_insn_t(name, match, mask, {&vd, &vs2, &zimm5, opt, &vm}));
}

static void NOINLINE add_vector_viu_z6_insn(disassembler_t* d, const char* name, uint32_t match, uint32_t mask)
{
  d->add_insn(new disasm_insn_t(name, match, mask, {&vd, &vs2, &v_zimm6, opt, &vm}));
}

static void NOINLINE add_vector_vvm_insn(disassembler_t* d, const char* name, uint32_t match, uint32_t mask)
{
  d->add_insn(new disasm_insn_t(name, match, mask, {&vd, &vs2, &vs1, &v0}));
}

static void NOINLINE add_vector_vxm_insn(disassembler_t* d, const char* name, uint32_t match, uint32_t mask)
{
  d->add_insn(new disasm_insn_t(name, match, mask, {&vd, &vs2, &xrs1, &v0}));
}

static void NOINLINE add_vector_vim_insn(disassembler_t* d, const char* name, uint32_t match, uint32_t mask)
{
  d->add_insn(new disasm_insn_t(name, match, mask, {&vd, &vs2, &v_simm5, &v0}));
}

static void NOINLINE add_unknown_insn(disassembler_t* d, const char* name, uint32_t match, uint32_t mask)
{
  std::string s = name;
  s += " (args unknown)";

  d->add_insn(new disasm_insn_t(s.c_str(), match, mask, {}));
}


static void NOINLINE add_unknown_insns(disassembler_t* d)
{
  // provide a default disassembly for all instructions as a fallback
  #define DECLARE_INSN(code, match, mask) \
   add_unknown_insn(d, #code, match, mask);
  #include "encoding.h"
  #undef DECLARE_INSN
}

void disassembler_t::add_instructions(const isa_parser_t* isa)
{
  const uint32_t mask_rd = 0x1fUL << 7;
  const uint32_t match_rd_ra = 1UL << 7;
  const uint32_t mask_rs1 = 0x1fUL << 15;
  const uint32_t match_rs1_ra = 1UL << 15;
  const uint32_t mask_rs2 = 0x1fUL << 20;
  const uint32_t mask_imm = 0xfffUL << 20;
  const uint32_t imm_shift = 20;
  const uint32_t mask_rvc_rs2 = 0x1fUL << 2;
  const uint32_t mask_rvc_imm = mask_rvc_rs2 | 0x1000UL;
  const uint32_t mask_nf = 0x7Ul << 29;
  const uint32_t mask_wd = 0x1Ul << 26;
  const uint32_t mask_vm = 0x1Ul << 25;
  const uint32_t mask_vldst = 0x7Ul << 12 | 0x1UL << 28;
  const uint32_t mask_amoop = 0x1fUl << 27;
  const uint32_t mask_width = 0x7Ul << 12;

  #define DECLARE_INSN(code, match, mask) \
   const uint32_t match_##code = match; \
   const uint32_t mask_##code = mask;
  #include "encoding.h"
  #undef DECLARE_INSN

  // explicit per-instruction disassembly
  #define DISASM_INSN(name, code, extra, ...) \
    add_insn(new disasm_insn_t(name, match_##code, mask_##code | (extra), __VA_ARGS__));
  #define DEFINE_NOARG(code) add_noarg_insn(this, #code, match_##code, mask_##code);
  #define DEFINE_RTYPE(code) add_rtype_insn(this, #code, match_##code, mask_##code);
  #define DEFINE_R1TYPE(code) add_r1type_insn(this, #code, match_##code, mask_##code);
  #define DEFINE_R3TYPE(code) add_r3type_insn(this, #code, match_##code, mask_##code);
  #define DEFINE_ITYPE(code) add_itype_insn(this, #code, match_##code, mask_##code);
  #define DEFINE_ITYPE_SHIFT(code) add_itype_shift_insn(this, #code, match_##code, mask_##code);
  #define DEFINE_I0TYPE(name, code) DISASM_INSN(name, code, mask_rs1, {&xrd, &imm})
  #define DEFINE_I1TYPE(name, code) DISASM_INSN(name, code, mask_imm, {&xrd, &xrs1})
  #define DEFINE_I2TYPE(name, code) DISASM_INSN(name, code, mask_rd | mask_imm, {&xrs1})
  #define DEFINE_PREFETCH(code) DISASM_INSN(#code, code, 0, {&store_address})
  #define DEFINE_LTYPE(code) DISASM_INSN(#code, code, 0, {&xrd, &bigimm})
  #define DEFINE_BTYPE(code) add_btype_insn(this, #code, match_##code, mask_##code);
  #define DEFINE_B1TYPE(name, code) add_b1type_insn(this, name, match_##code, mask_##code);
  #define DEFINE_XLOAD(code) add_xload_insn(this, #code, match_##code, mask_##code);
  #define DEFINE_XSTORE(code) add_xstore_insn(this, #code, match_##code, mask_##code);
  #define DEFINE_XAMO(code) add_xamo_insn(this, #code, match_##code, mask_##code);
  #define DEFINE_XLOAD_BASE(code) add_xlr_insn(this, #code, match_##code, mask_##code);
  #define DEFINE_XSTORE_BASE(code) add_xst_insn(this, #code, match_##code, mask_##code);
  #define DEFINE_FLOAD(code) add_fload_insn(this, #code, match_##code, mask_##code);
  #define DEFINE_FSTORE(code) add_fstore_insn(this, #code, match_##code, mask_##code);
  #define DEFINE_FRTYPE(code) add_frtype_insn(this, #code, match_##code, mask_##code);
  #define DEFINE_FR1TYPE(code) add_fr1type_insn(this, #code, match_##code, mask_##code);
  #define DEFINE_FR3TYPE(code) add_fr3type_insn(this, #code, match_##code, mask_##code);
  #define DEFINE_FXTYPE(code) add_fxtype_insn(this, #code, match_##code, mask_##code);
  #define DEFINE_FX2TYPE(code) add_fx2type_insn(this, #code, match_##code, mask_##code);
  #define DEFINE_XFTYPE(code) add_xftype_insn(this, #code, match_##code, mask_##code);
  #define DEFINE_SFENCE_TYPE(code) add_sfence_insn(this, #code, match_##code, mask_##code);

  add_insn(new disasm_insn_t("unimp", match_csrrw|(CSR_CYCLE<<20), 0xffffffff, {}));
  add_insn(new disasm_insn_t("c.unimp", 0, 0xffff, {}));

  // Following are HINTs, so they must precede their corresponding base-ISA
  // instructions.  We do not condition them on Zicbop/Zihintpause because,
  // definitionally, all implementations provide them.
  DEFINE_PREFETCH(prefetch_r);
  DEFINE_PREFETCH(prefetch_w);
  DEFINE_PREFETCH(prefetch_i);
  DEFINE_NOARG(pause);

  DEFINE_XLOAD(lb)
  DEFINE_XLOAD(lbu)
  DEFINE_XLOAD(lh)
  DEFINE_XLOAD(lhu)
  DEFINE_XLOAD(lw)
  DEFINE_XLOAD(lwu)
  DEFINE_XLOAD(ld)

  DEFINE_XSTORE(sb)
  DEFINE_XSTORE(sh)
  DEFINE_XSTORE(sw)
  DEFINE_XSTORE(sd)

  if (isa->extension_enabled('A')) {
    DEFINE_XAMO(amoadd_w)
    DEFINE_XAMO(amoswap_w)
    DEFINE_XAMO(amoand_w)
    DEFINE_XAMO(amoor_w)
    DEFINE_XAMO(amoxor_w)
    DEFINE_XAMO(amomin_w)
    DEFINE_XAMO(amomax_w)
    DEFINE_XAMO(amominu_w)
    DEFINE_XAMO(amomaxu_w)
    DEFINE_XAMO(amoadd_d)
    DEFINE_XAMO(amoswap_d)
    DEFINE_XAMO(amoand_d)
    DEFINE_XAMO(amoor_d)
    DEFINE_XAMO(amoxor_d)
    DEFINE_XAMO(amomin_d)
    DEFINE_XAMO(amomax_d)
    DEFINE_XAMO(amominu_d)
    DEFINE_XAMO(amomaxu_d)
    DEFINE_XLOAD_BASE(lr_w)
    DEFINE_XAMO(sc_w)
    DEFINE_XLOAD_BASE(lr_d)
    DEFINE_XAMO(sc_d)
  }

  if (isa->extension_enabled(EXT_ZACAS)) {
    DEFINE_XAMO(amocas_w)
    DEFINE_XAMO(amocas_d)
    DEFINE_XAMO(amocas_q)
  }

  if (isa->extension_enabled(EXT_ZABHA)) {
    DEFINE_XAMO(amoadd_b)
    DEFINE_XAMO(amoswap_b)
    DEFINE_XAMO(amoand_b)
    DEFINE_XAMO(amoor_b)
    DEFINE_XAMO(amoxor_b)
    DEFINE_XAMO(amomin_b)
    DEFINE_XAMO(amomax_b)
    DEFINE_XAMO(amominu_b)
    DEFINE_XAMO(amomaxu_b)
    DEFINE_XAMO(amocas_b)
    DEFINE_XAMO(amoadd_h)
    DEFINE_XAMO(amoswap_h)
    DEFINE_XAMO(amoand_h)
    DEFINE_XAMO(amoor_h)
    DEFINE_XAMO(amoxor_h)
    DEFINE_XAMO(amomin_h)
    DEFINE_XAMO(amomax_h)
    DEFINE_XAMO(amominu_h)
    DEFINE_XAMO(amomaxu_h)
    DEFINE_XAMO(amocas_h)
  }

  add_insn(new disasm_insn_t("j", match_jal, mask_jal | mask_rd, {&jump_target}));
  add_insn(new disasm_insn_t("jal", match_jal | match_rd_ra, mask_jal | mask_rd, {&jump_target}));
  add_insn(new disasm_insn_t("jal", match_jal, mask_jal, {&xrd, &jump_target}));

  DEFINE_B1TYPE("beqz", beq);
  DEFINE_B1TYPE("bnez", bne);
  DEFINE_B1TYPE("bltz", blt);
  DEFINE_B1TYPE("bgez", bge);
  DEFINE_BTYPE(beq)
  DEFINE_BTYPE(bne)
  DEFINE_BTYPE(blt)
  DEFINE_BTYPE(bge)
  DEFINE_BTYPE(bltu)
  DEFINE_BTYPE(bgeu)

  DEFINE_LTYPE(lui);
  DEFINE_LTYPE(auipc);

  add_insn(new disasm_insn_t("ret", match_jalr | match_rs1_ra, mask_jalr | mask_rd | mask_rs1 | mask_imm, {}));
  DEFINE_I2TYPE("jr", jalr);
  add_insn(new disasm_insn_t("jalr", match_jalr | match_rd_ra, mask_jalr | mask_rd | mask_imm, {&xrs1}));
  DEFINE_ITYPE(jalr);

  add_noarg_insn(this, "nop", match_addi, mask_addi | mask_rd | mask_rs1 | mask_imm);
  DEFINE_I0TYPE("li", addi);
  DEFINE_I1TYPE("mv", addi);
  DEFINE_ITYPE(addi);
  DEFINE_ITYPE(slti);
  add_insn(new disasm_insn_t("seqz", match_sltiu | (1 << imm_shift), mask_sltiu | mask_imm, {&xrd, &xrs1}));
  DEFINE_ITYPE(sltiu);
  add_insn(new disasm_insn_t("not", match_xori | mask_imm, mask_xori | mask_imm, {&xrd, &xrs1}));
  DEFINE_ITYPE(xori);

  DEFINE_ITYPE_SHIFT(slli);
  DEFINE_ITYPE_SHIFT(srli);
  DEFINE_ITYPE_SHIFT(srai);

  DEFINE_ITYPE(ori);
  DEFINE_ITYPE(andi);
  DEFINE_I1TYPE("sext.w", addiw);
  DEFINE_ITYPE(addiw);

  DEFINE_ITYPE_SHIFT(slliw);
  DEFINE_ITYPE_SHIFT(srliw);
  DEFINE_ITYPE_SHIFT(sraiw);

  DEFINE_RTYPE(add);
  DEFINE_RTYPE(sub);
  DEFINE_RTYPE(sll);
  DEFINE_RTYPE(slt);
  add_insn(new disasm_insn_t("snez", match_sltu, mask_sltu | mask_rs1, {&xrd, &xrs2}));
  DEFINE_RTYPE(sltu);
  DEFINE_RTYPE(xor);
  DEFINE_RTYPE(srl);
  DEFINE_RTYPE(sra);
  DEFINE_RTYPE(or);
  DEFINE_RTYPE(and);
  DEFINE_RTYPE(addw);
  DEFINE_RTYPE(subw);
  DEFINE_RTYPE(sllw);
  DEFINE_RTYPE(srlw);
  DEFINE_RTYPE(sraw);

  DEFINE_NOARG(ecall);
  DEFINE_NOARG(ebreak);
  DEFINE_NOARG(mret);
  DEFINE_NOARG(dret);
  DEFINE_NOARG(wfi);
  add_insn(new disasm_insn_t("fence", match_fence, mask_fence, {&iorw}));
  DEFINE_NOARG(fence_i);

  add_insn(new disasm_insn_t("csrr", match_csrrs, mask_csrrs | mask_rs1, {&xrd, &csr}));
  add_insn(new disasm_insn_t("csrw", match_csrrw, mask_csrrw | mask_rd, {&csr, &xrs1}));
  add_insn(new disasm_insn_t("csrs", match_csrrs, mask_csrrs | mask_rd, {&csr, &xrs1}));
  add_insn(new disasm_insn_t("csrc", match_csrrc, mask_csrrc | mask_rd, {&csr, &xrs1}));
  add_insn(new disasm_insn_t("csrwi", match_csrrwi, mask_csrrwi | mask_rd, {&csr, &zimm5}));
  add_insn(new disasm_insn_t("csrsi", match_csrrsi, mask_csrrsi | mask_rd, {&csr, &zimm5}));
  add_insn(new disasm_insn_t("csrci", match_csrrci, mask_csrrci | mask_rd, {&csr, &zimm5}));
  add_insn(new disasm_insn_t("csrrw", match_csrrw, mask_csrrw, {&xrd, &csr, &xrs1}));
  add_insn(new disasm_insn_t("csrrs", match_csrrs, mask_csrrs, {&xrd, &csr, &xrs1}));
  add_insn(new disasm_insn_t("csrrc", match_csrrc, mask_csrrc, {&xrd, &csr, &xrs1}));
  add_insn(new disasm_insn_t("csrrwi", match_csrrwi, mask_csrrwi, {&xrd, &csr, &zimm5}));
  add_insn(new disasm_insn_t("csrrsi", match_csrrsi, mask_csrrsi, {&xrd, &csr, &zimm5}));
  add_insn(new disasm_insn_t("csrrci", match_csrrci, mask_csrrci, {&xrd, &csr, &zimm5}));

  if (isa->extension_enabled('S')) {
    DEFINE_NOARG(sret);
    DEFINE_SFENCE_TYPE(sfence_vma);
  }

  if (isa->extension_enabled('M')) {
    DEFINE_RTYPE(mul);
    DEFINE_RTYPE(mulh);
    DEFINE_RTYPE(mulhu);
    DEFINE_RTYPE(mulhsu);
    DEFINE_RTYPE(mulw);
    DEFINE_RTYPE(div);
    DEFINE_RTYPE(divu);
    DEFINE_RTYPE(rem);
    DEFINE_RTYPE(remu);
    DEFINE_RTYPE(divw);
    DEFINE_RTYPE(divuw);
    DEFINE_RTYPE(remw);
    DEFINE_RTYPE(remuw);
  }

  if (isa->extension_enabled(EXT_ZBA)) { 
    DEFINE_RTYPE(sh1add);
    DEFINE_RTYPE(sh2add);
    DEFINE_RTYPE(sh3add);
    if (isa->get_max_xlen() == 64) {
      DEFINE_ITYPE_SHIFT(slli_uw);
      add_insn(new disasm_insn_t("zext.w", match_add_uw, mask_add_uw | mask_rs2, {&xrd, &xrs1}));
      DEFINE_RTYPE(add_uw);
      DEFINE_RTYPE(sh1add_uw);
      DEFINE_RTYPE(sh2add_uw);
      DEFINE_RTYPE(sh3add_uw);
    }
  }

  if (isa->extension_enabled(EXT_ZBB)) { 
    DEFINE_RTYPE(ror);
    DEFINE_RTYPE(rol);
    DEFINE_ITYPE_SHIFT(rori);
    DEFINE_R1TYPE(ctz);
    DEFINE_R1TYPE(clz);
    DEFINE_R1TYPE(cpop);
    DEFINE_RTYPE(min);
    DEFINE_RTYPE(minu);
    DEFINE_RTYPE(max);
    DEFINE_RTYPE(maxu);
    DEFINE_RTYPE(andn);
    DEFINE_RTYPE(orn);
    DEFINE_RTYPE(xnor);
    DEFINE_R1TYPE(sext_b);
    DEFINE_R1TYPE(sext_h);
    add_insn(new disasm_insn_t("rev8", match_grevi | ((isa->get_max_xlen() - 8) << imm_shift), mask_grevi | mask_imm, {&xrd, &xrs1}));
    add_insn(new disasm_insn_t("orc.b", match_gorci | (0x7 << imm_shift), mask_grevi | mask_imm, {&xrd, &xrs1}));
    add_insn(new disasm_insn_t("zext.h", (isa->get_max_xlen() == 32 ? match_pack : match_packw), mask_pack | mask_rs2, {&xrd, &xrs1}));
    if (isa->get_max_xlen() == 64) {
      DEFINE_RTYPE(rorw);
      DEFINE_RTYPE(rolw);
      DEFINE_ITYPE_SHIFT(roriw);
      DEFINE_R1TYPE(ctzw);
      DEFINE_R1TYPE(clzw);
      DEFINE_R1TYPE(cpopw);
    }
  }

  if (isa->extension_enabled(EXT_ZBC)) {
    DEFINE_RTYPE(clmul);
    DEFINE_RTYPE(clmulh);
    DEFINE_RTYPE(clmulr);
  }

  if (isa->extension_enabled(EXT_ZBS)) { 
    DEFINE_RTYPE(bclr);
    DEFINE_RTYPE(binv);
    DEFINE_RTYPE(bset);
    DEFINE_RTYPE(bext);
    DEFINE_ITYPE_SHIFT(bclri);
    DEFINE_ITYPE_SHIFT(binvi);
    DEFINE_ITYPE_SHIFT(bseti);
    DEFINE_ITYPE_SHIFT(bexti);
  }

  if (isa->extension_enabled(EXT_ZBKB)) {
    add_insn(new disasm_insn_t("brev8", match_grevi | (0x7 << imm_shift), mask_grevi | mask_imm, {&xrd, &xrs1})); // brev8
    add_insn(new disasm_insn_t("rev8", match_grevi | ((isa->get_max_xlen() - 8) << imm_shift), mask_grevi | mask_imm, {&xrd, &xrs1}));
    DEFINE_RTYPE(pack);
    DEFINE_RTYPE(packh);
    if (isa->get_max_xlen() == 64) {
      DEFINE_RTYPE(packw);
    }
  }

  if (isa->extension_enabled(EXT_SVINVAL)) {
    DEFINE_NOARG(sfence_w_inval);
    DEFINE_NOARG(sfence_inval_ir);
    DEFINE_SFENCE_TYPE(sinval_vma);
    DEFINE_SFENCE_TYPE(hinval_vvma);
    DEFINE_SFENCE_TYPE(hinval_gvma);
  }

  if (isa->extension_enabled('F')) {
    DEFINE_FLOAD(flw)
    DEFINE_FSTORE(fsw)
    DEFINE_FRTYPE(fadd_s);
    DEFINE_FRTYPE(fsub_s);
    DEFINE_FRTYPE(fmul_s);
    DEFINE_FRTYPE(fdiv_s);
    DEFINE_FR1TYPE(fsqrt_s);
    DEFINE_FRTYPE(fmin_s);
    DEFINE_FRTYPE(fmax_s);
    DEFINE_FR3TYPE(fmadd_s);
    DEFINE_FR3TYPE(fmsub_s);
    DEFINE_FR3TYPE(fnmadd_s);
    DEFINE_FR3TYPE(fnmsub_s);
    DEFINE_FRTYPE(fsgnj_s);
    DEFINE_FRTYPE(fsgnjn_s);
    DEFINE_FRTYPE(fsgnjx_s);
    DEFINE_FR1TYPE(fcvt_s_d);
    DEFINE_FR1TYPE(fcvt_s_q);
    DEFINE_XFTYPE(fcvt_s_l);
    DEFINE_XFTYPE(fcvt_s_lu);
    DEFINE_XFTYPE(fcvt_s_w);
    DEFINE_XFTYPE(fcvt_s_wu);
    DEFINE_XFTYPE(fcvt_s_wu);
    DEFINE_XFTYPE(fmv_w_x);
    DEFINE_FXTYPE(fcvt_l_s);
    DEFINE_FXTYPE(fcvt_lu_s);
    DEFINE_FXTYPE(fcvt_w_s);
    DEFINE_FXTYPE(fcvt_wu_s);
    DEFINE_FXTYPE(fclass_s);
    DEFINE_FXTYPE(fmv_x_w);
    DEFINE_FX2TYPE(feq_s);
    DEFINE_FX2TYPE(flt_s);
    DEFINE_FX2TYPE(fle_s);
  }

  if (isa->extension_enabled(EXT_ZFINX)) {
    DEFINE_RTYPE(fadd_s);
    DEFINE_RTYPE(fsub_s);
    DEFINE_RTYPE(fmul_s);
    DEFINE_RTYPE(fdiv_s);
    DEFINE_R1TYPE(fsqrt_s);
    DEFINE_RTYPE(fmin_s);
    DEFINE_RTYPE(fmax_s);
    DEFINE_R3TYPE(fmadd_s);
    DEFINE_R3TYPE(fmsub_s);
    DEFINE_R3TYPE(fnmadd_s);
    DEFINE_R3TYPE(fnmsub_s);
    DEFINE_RTYPE(fsgnj_s);
    DEFINE_RTYPE(fsgnjn_s);
    DEFINE_RTYPE(fsgnjx_s);
    DEFINE_R1TYPE(fcvt_s_d);
    //DEFINE_R1TYPE(fcvt_s_q);
    DEFINE_R1TYPE(fcvt_s_l);
    DEFINE_R1TYPE(fcvt_s_lu);
    DEFINE_R1TYPE(fcvt_s_w);
    DEFINE_R1TYPE(fcvt_s_wu);
    DEFINE_R1TYPE(fcvt_s_wu);
    DEFINE_R1TYPE(fcvt_l_s);
    DEFINE_R1TYPE(fcvt_lu_s);
    DEFINE_R1TYPE(fcvt_w_s);
    DEFINE_R1TYPE(fcvt_wu_s);
    DEFINE_R1TYPE(fclass_s);
    DEFINE_RTYPE(feq_s);
    DEFINE_RTYPE(flt_s);
    DEFINE_RTYPE(fle_s);
  }

  if (isa->extension_enabled('D')) {
    DEFINE_FLOAD(fld)
    DEFINE_FSTORE(fsd)
    DEFINE_FRTYPE(fadd_d);
    DEFINE_FRTYPE(fsub_d);
    DEFINE_FRTYPE(fmul_d);
    DEFINE_FRTYPE(fdiv_d);
    DEFINE_FR1TYPE(fsqrt_d);
    DEFINE_FRTYPE(fmin_d);
    DEFINE_FRTYPE(fmax_d);
    DEFINE_FR3TYPE(fmadd_d);
    DEFINE_FR3TYPE(fmsub_d);
    DEFINE_FR3TYPE(fnmadd_d);
    DEFINE_FR3TYPE(fnmsub_d);
    DEFINE_FRTYPE(fsgnj_d);
    DEFINE_FRTYPE(fsgnjn_d);
    DEFINE_FRTYPE(fsgnjx_d);
    DEFINE_FR1TYPE(fcvt_d_s);
    DEFINE_FR1TYPE(fcvt_d_q);
    DEFINE_XFTYPE(fcvt_d_l);
    DEFINE_XFTYPE(fcvt_d_lu);
    DEFINE_XFTYPE(fcvt_d_w);
    DEFINE_XFTYPE(fcvt_d_wu);
    DEFINE_XFTYPE(fcvt_d_wu);
    DEFINE_XFTYPE(fmv_d_x);
    DEFINE_FXTYPE(fcvt_l_d);
    DEFINE_FXTYPE(fcvt_lu_d);
    DEFINE_FXTYPE(fcvt_w_d);
    DEFINE_FXTYPE(fcvt_wu_d);
    DEFINE_FXTYPE(fclass_d);
    DEFINE_FXTYPE(fmv_x_d);
    DEFINE_FX2TYPE(feq_d);
    DEFINE_FX2TYPE(flt_d);
    DEFINE_FX2TYPE(fle_d);
  }

  if (isa->extension_enabled(EXT_ZDINX)) {
    DEFINE_RTYPE(fadd_d);
    DEFINE_RTYPE(fsub_d);
    DEFINE_RTYPE(fmul_d);
    DEFINE_RTYPE(fdiv_d);
    DEFINE_R1TYPE(fsqrt_d);
    DEFINE_RTYPE(fmin_d);
    DEFINE_RTYPE(fmax_d);
    DEFINE_R3TYPE(fmadd_d);
    DEFINE_R3TYPE(fmsub_d);
    DEFINE_R3TYPE(fnmadd_d);
    DEFINE_R3TYPE(fnmsub_d);
    DEFINE_RTYPE(fsgnj_d);
    DEFINE_RTYPE(fsgnjn_d);
    DEFINE_RTYPE(fsgnjx_d);
    DEFINE_R1TYPE(fcvt_d_s);
    //DEFINE_R1TYPE(fcvt_d_q);
    DEFINE_R1TYPE(fcvt_d_l);
    DEFINE_R1TYPE(fcvt_d_lu);
    DEFINE_R1TYPE(fcvt_d_w);
    DEFINE_R1TYPE(fcvt_d_wu);
    DEFINE_R1TYPE(fcvt_d_wu);
    DEFINE_R1TYPE(fcvt_l_d);
    DEFINE_R1TYPE(fcvt_lu_d);
    DEFINE_R1TYPE(fcvt_w_d);
    DEFINE_R1TYPE(fcvt_wu_d);
    DEFINE_R1TYPE(fclass_d);
    DEFINE_RTYPE(feq_d);
    DEFINE_RTYPE(flt_d);
    DEFINE_RTYPE(fle_d);
  }

  if (isa->extension_enabled(EXT_ZFH)) { 
    DEFINE_FRTYPE(fadd_h);
    DEFINE_FRTYPE(fsub_h);
    DEFINE_FRTYPE(fmul_h);
    DEFINE_FRTYPE(fdiv_h);
    DEFINE_FR1TYPE(fsqrt_h);
    DEFINE_FRTYPE(fmin_h);
    DEFINE_FRTYPE(fmax_h);
    DEFINE_FR3TYPE(fmadd_h);
    DEFINE_FR3TYPE(fmsub_h);
    DEFINE_FR3TYPE(fnmadd_h);
    DEFINE_FR3TYPE(fnmsub_h);
    DEFINE_FRTYPE(fsgnj_h);
    DEFINE_FRTYPE(fsgnjn_h);
    DEFINE_FRTYPE(fsgnjx_h);
    DEFINE_XFTYPE(fcvt_h_l);
    DEFINE_XFTYPE(fcvt_h_lu);
    DEFINE_XFTYPE(fcvt_h_w);
    DEFINE_XFTYPE(fcvt_h_wu);
    DEFINE_XFTYPE(fcvt_h_wu);
    DEFINE_FXTYPE(fcvt_l_h);
    DEFINE_FXTYPE(fcvt_lu_h);
    DEFINE_FXTYPE(fcvt_w_h);
    DEFINE_FXTYPE(fcvt_wu_h);
    DEFINE_FXTYPE(fclass_h);
    DEFINE_FX2TYPE(feq_h);
    DEFINE_FX2TYPE(flt_h);
    DEFINE_FX2TYPE(fle_h);
  }

  if (isa->extension_enabled(EXT_ZHINX)) {
    DEFINE_RTYPE(fadd_h);
    DEFINE_RTYPE(fsub_h);
    DEFINE_RTYPE(fmul_h);
    DEFINE_RTYPE(fdiv_h);
    DEFINE_R1TYPE(fsqrt_h);
    DEFINE_RTYPE(fmin_h);
    DEFINE_RTYPE(fmax_h);
    DEFINE_R3TYPE(fmadd_h);
    DEFINE_R3TYPE(fmsub_h);
    DEFINE_R3TYPE(fnmadd_h);
    DEFINE_R3TYPE(fnmsub_h);
    DEFINE_RTYPE(fsgnj_h);
    DEFINE_RTYPE(fsgnjn_h);
    DEFINE_RTYPE(fsgnjx_h);
    DEFINE_R1TYPE(fcvt_h_l);
    DEFINE_R1TYPE(fcvt_h_lu);
    DEFINE_R1TYPE(fcvt_h_w);
    DEFINE_R1TYPE(fcvt_h_wu);
    DEFINE_R1TYPE(fcvt_h_wu);
    DEFINE_R1TYPE(fcvt_l_h);
    DEFINE_R1TYPE(fcvt_lu_h);
    DEFINE_R1TYPE(fcvt_w_h);
    DEFINE_R1TYPE(fcvt_wu_h);
    DEFINE_R1TYPE(fclass_h);
    DEFINE_RTYPE(feq_h);
    DEFINE_RTYPE(flt_h);
    DEFINE_RTYPE(fle_h);
  }

  if (isa->extension_enabled(EXT_ZFHMIN)) {
    DEFINE_FR1TYPE(fcvt_h_s);
    DEFINE_FR1TYPE(fcvt_h_d);
    DEFINE_FR1TYPE(fcvt_h_q);
    DEFINE_FR1TYPE(fcvt_s_h);
    DEFINE_FR1TYPE(fcvt_d_h);
    DEFINE_FR1TYPE(fcvt_q_h);
  }

  if (isa->extension_enabled(EXT_INTERNAL_ZFH_MOVE)) {
    DEFINE_FLOAD(flh)
    DEFINE_FSTORE(fsh)
    DEFINE_XFTYPE(fmv_h_x);
    DEFINE_FXTYPE(fmv_x_h);
  }

  if (isa->extension_enabled(EXT_ZHINXMIN)) {
    DEFINE_R1TYPE(fcvt_h_s);
    DEFINE_R1TYPE(fcvt_h_d);
    //DEFINE_R1TYPE(fcvt_h_q);
    DEFINE_R1TYPE(fcvt_s_h);
    DEFINE_R1TYPE(fcvt_d_h);
    //DEFINE_R1TYPE(fcvt_q_h);
  }

  if (isa->extension_enabled('Q')) {
    DEFINE_FLOAD(flq)
    DEFINE_FSTORE(fsq)
    DEFINE_FRTYPE(fadd_q);
    DEFINE_FRTYPE(fsub_q);
    DEFINE_FRTYPE(fmul_q);
    DEFINE_FRTYPE(fdiv_q);
    DEFINE_FR1TYPE(fsqrt_q);
    DEFINE_FRTYPE(fmin_q);
    DEFINE_FRTYPE(fmax_q);
    DEFINE_FR3TYPE(fmadd_q);
    DEFINE_FR3TYPE(fmsub_q);
    DEFINE_FR3TYPE(fnmadd_q);
    DEFINE_FR3TYPE(fnmsub_q);
    DEFINE_FRTYPE(fsgnj_q);
    DEFINE_FRTYPE(fsgnjn_q);
    DEFINE_FRTYPE(fsgnjx_q);
    DEFINE_FR1TYPE(fcvt_q_s);
    DEFINE_FR1TYPE(fcvt_q_d);
    DEFINE_XFTYPE(fcvt_q_l);
    DEFINE_XFTYPE(fcvt_q_lu);
    DEFINE_XFTYPE(fcvt_q_w);
    DEFINE_XFTYPE(fcvt_q_wu);
    DEFINE_XFTYPE(fcvt_q_wu);
    DEFINE_FXTYPE(fcvt_l_q);
    DEFINE_FXTYPE(fcvt_lu_q);
    DEFINE_FXTYPE(fcvt_w_q);
    DEFINE_FXTYPE(fcvt_wu_q);
    DEFINE_FXTYPE(fclass_q);
    DEFINE_FX2TYPE(feq_q);
    DEFINE_FX2TYPE(flt_q);
    DEFINE_FX2TYPE(fle_q);
  }

  if (isa->extension_enabled(EXT_ZFBFMIN)) {
    DEFINE_FR1TYPE(fcvt_bf16_s);
    DEFINE_FR1TYPE(fcvt_s_bf16);
  }

  // ext-h
  if (isa->extension_enabled('H')) {
    DEFINE_XLOAD_BASE(hlv_b)
    DEFINE_XLOAD_BASE(hlv_bu)
    DEFINE_XLOAD_BASE(hlv_h)
    DEFINE_XLOAD_BASE(hlv_hu)
    DEFINE_XLOAD_BASE(hlv_w)
    DEFINE_XLOAD_BASE(hlv_wu)
    DEFINE_XLOAD_BASE(hlv_d)

    DEFINE_XLOAD_BASE(hlvx_hu)
    DEFINE_XLOAD_BASE(hlvx_wu)

    DEFINE_XSTORE_BASE(hsv_b)
    DEFINE_XSTORE_BASE(hsv_h)
    DEFINE_XSTORE_BASE(hsv_w)
    DEFINE_XSTORE_BASE(hsv_d)

    DEFINE_SFENCE_TYPE(hfence_gvma);
    DEFINE_SFENCE_TYPE(hfence_vvma);
  }

  // ext-c
  if (isa->extension_enabled(EXT_ZCA)) {
    DISASM_INSN("c.ebreak", c_add, mask_rd | mask_rvc_rs2, {});
    add_insn(new disasm_insn_t("ret", match_c_jr | match_rd_ra, mask_c_jr | mask_rd | mask_rvc_imm, {}));
    DISASM_INSN("c.jr", c_jr, mask_rvc_imm, {&rvc_rs1});
    DISASM_INSN("c.jalr", c_jalr, mask_rvc_imm, {&rvc_rs1});
    DISASM_INSN("c.nop", c_addi, mask_rd | mask_rvc_imm, {});
    DISASM_INSN("c.addi16sp", c_addi16sp, mask_rd, {&rvc_sp, &rvc_addi16sp_imm});
    DISASM_INSN("c.addi4spn", c_addi4spn, 0, {&rvc_rs2s, &rvc_sp, &rvc_addi4spn_imm});
    DISASM_INSN("c.li", c_li, 0, {&xrd, &rvc_imm});
    DISASM_INSN("c.lui", c_lui, 0, {&xrd, &rvc_uimm});
    DISASM_INSN("c.addi", c_addi, 0, {&xrd, &rvc_imm});
    DISASM_INSN("c.slli", c_slli, 0, {&rvc_rs1, &rvc_shamt});
    DISASM_INSN("c.srli", c_srli, 0, {&rvc_rs1s, &rvc_shamt});
    DISASM_INSN("c.srai", c_srai, 0, {&rvc_rs1s, &rvc_shamt});
    DISASM_INSN("c.andi", c_andi, 0, {&rvc_rs1s, &rvc_imm});
    DISASM_INSN("c.mv", c_mv, 0, {&xrd, &rvc_rs2});
    DISASM_INSN("c.add", c_add, 0, {&xrd, &rvc_rs2});
    DISASM_INSN("c.addw", c_addw, 0, {&rvc_rs1s, &rvc_rs2s});
    DISASM_INSN("c.sub", c_sub, 0, {&rvc_rs1s, &rvc_rs2s});
    DISASM_INSN("c.subw", c_subw, 0, {&rvc_rs1s, &rvc_rs2s});
    DISASM_INSN("c.and", c_and, 0, {&rvc_rs1s, &rvc_rs2s});
    DISASM_INSN("c.or", c_or, 0, {&rvc_rs1s, &rvc_rs2s});
    DISASM_INSN("c.xor", c_xor, 0, {&rvc_rs1s, &rvc_rs2s});
    DISASM_INSN("c.lwsp", c_lwsp, 0, {&xrd, &rvc_lwsp_address});
    DISASM_INSN("c.swsp", c_swsp, 0, {&rvc_rs2, &rvc_swsp_address});
    DISASM_INSN("c.lw", c_lw, 0, {&rvc_rs2s, &rvc_lw_address});
    DISASM_INSN("c.sw", c_sw, 0, {&rvc_rs2s, &rvc_lw_address});
    DISASM_INSN("c.beqz", c_beqz, 0, {&rvc_rs1s, &rvc_branch_target});
    DISASM_INSN("c.bnez", c_bnez, 0, {&rvc_rs1s, &rvc_branch_target});
    DISASM_INSN("c.j", c_j, 0, {&rvc_jump_target});
    if (isa->get_max_xlen() == 32) {
      DISASM_INSN("c.jal", c_jal, 0, {&rvc_jump_target});
    } else {
      DISASM_INSN("c.ld", c_ld, 0, {&rvc_rs2s, &rvc_ld_address});
      DISASM_INSN("c.ldsp", c_ldsp, 0, {&xrd, &rvc_ldsp_address});
      DISASM_INSN("c.sd", c_sd, 0, {&rvc_rs2s, &rvc_ld_address});
      DISASM_INSN("c.sdsp", c_sdsp, 0, {&rvc_rs2, &rvc_sdsp_address});
      DISASM_INSN("c.addiw", c_addiw, 0, {&xrd, &rvc_imm});
    }
  }

  if (isa->extension_enabled(EXT_ZCD) && isa->extension_enabled('D')) {
    DISASM_INSN("c.fld", c_fld, 0, {&rvc_fp_rs2s, &rvc_ld_address});
    DISASM_INSN("c.fldsp", c_fldsp, 0, {&frd, &rvc_ldsp_address});
    DISASM_INSN("c.fsd", c_fsd, 0, {&rvc_fp_rs2s, &rvc_ld_address});
    DISASM_INSN("c.fsdsp", c_fsdsp, 0, {&rvc_fp_rs2, &rvc_sdsp_address});
  }

  if (isa->extension_enabled(EXT_ZCF) && isa->extension_enabled('F')) {
    DISASM_INSN("c.flw", c_flw, 0, {&rvc_fp_rs2s, &rvc_lw_address});
    DISASM_INSN("c.flwsp", c_flwsp, 0, {&frd, &rvc_lwsp_address});
    DISASM_INSN("c.fsw", c_fsw, 0, {&rvc_fp_rs2s, &rvc_lw_address});
    DISASM_INSN("c.fswsp", c_fswsp, 0, {&rvc_fp_rs2, &rvc_swsp_address});
  }

  if (isa->extension_enabled(EXT_ZCB)) {
    DISASM_INSN("c.zext.b", c_zext_b, 0, {&rvc_rs1s});
    DISASM_INSN("c.sext.b", c_sext_b, 0, {&rvc_rs1s});
    DISASM_INSN("c.zext.h", c_zext_h, 0, {&rvc_rs1s});
    DISASM_INSN("c.sext.h", c_sext_h, 0, {&rvc_rs1s});
    if (isa->get_max_xlen() == 64) {
      DISASM_INSN("c.zext.w", c_zext_w, 0, {&rvc_rs1s});
    }
    DISASM_INSN("c.not", c_not, 0, {&rvc_rs1s});
    DISASM_INSN("c.mul", c_mul, 0, {&rvc_rs1s, &rvc_rs2s});
    DISASM_INSN("c.lbu", c_lbu, 0, {&rvc_rs2s, &rvb_b_address});
    DISASM_INSN("c.lhu", c_lhu, 0, {&rvc_rs2s, &rvb_h_address});
    DISASM_INSN("c.lh", c_lh, 0, {&rvc_rs2s, &rvb_h_address});
    DISASM_INSN("c.sb", c_sb, 0, {&rvc_rs2s, &rvb_b_address});
    DISASM_INSN("c.sh", c_sh, 0, {&rvc_rs2s, &rvb_h_address});
  }

  if (isa->extension_enabled(EXT_ZCMP)) {
    if (isa->get_max_xlen() == 32) {
      DISASM_INSN("cm.push", cm_push, 0, {&rvcm_pushpop_rlist, &rvcm_push_stack_adj_32});
      DISASM_INSN("cm.pop", cm_pop, 0, {&rvcm_pushpop_rlist, &rvcm_pop_stack_adj_32});
      DISASM_INSN("cm.popret", cm_popret, 0, {&rvcm_pushpop_rlist, &rvcm_pop_stack_adj_32});
      DISASM_INSN("cm.popretz", cm_popretz, 0, {&rvcm_pushpop_rlist, &rvcm_pop_stack_adj_32});
    } else {
      DISASM_INSN("cm.push", cm_push, 0, {&rvcm_pushpop_rlist, &rvcm_push_stack_adj_64});
      DISASM_INSN("cm.pop", cm_pop, 0, {&rvcm_pushpop_rlist, &rvcm_pop_stack_adj_64});
      DISASM_INSN("cm.popret", cm_popret, 0, {&rvcm_pushpop_rlist, &rvcm_pop_stack_adj_64});
      DISASM_INSN("cm.popretz", cm_popretz, 0, {&rvcm_pushpop_rlist, &rvcm_pop_stack_adj_64});
    }

    DISASM_INSN("cm.mva01s", cm_mva01s, 0, {&rvc_r1s, &rvc_r2s});
    DISASM_INSN("cm.mvsa01", cm_mvsa01, 0, {&rvc_r1s, &rvc_r2s});
  }

  if (isa->extension_enabled(EXT_ZCMT)) {
    DISASM_INSN("cm.jt", cm_jalt, 0x380, {&rvcm_jt_index});
    DISASM_INSN("cm.jalt", cm_jalt, 0, {&rvcm_jt_index});
  }

  if (isa->extension_enabled('V')) {
    DISASM_INSN("vsetivli", vsetivli, 0, {&xrd, &zimm5, &v_vtype});
    DISASM_INSN("vsetvli", vsetvli, 0, {&xrd, &xrs1, &v_vtype});
    DEFINE_RTYPE(vsetvl);

    std::vector<const arg_t *> v_ld_unit = {&vd, &v_address, opt, &vm};
    std::vector<const arg_t *> v_st_unit = {&vs3, &v_address, opt, &vm};
    std::vector<const arg_t *> v_ld_stride = {&vd, &v_address, &xrs2, opt, &vm};
    std::vector<const arg_t *> v_st_stride = {&vs3, &v_address, &xrs2, opt, &vm};
    std::vector<const arg_t *> v_ld_index = {&vd, &v_address, &vs2, opt, &vm};
    std::vector<const arg_t *> v_st_index = {&vs3, &v_address, &vs2, opt, &vm};

    add_insn(new disasm_insn_t("vlm.v",  match_vlm_v,     mask_vlm_v, v_ld_unit));
    add_insn(new disasm_insn_t("vsm.v",  match_vsm_v,     mask_vsm_v, v_st_unit));

    // handle vector segment load/store
    for (size_t elt = 0; elt <= 7; ++elt) {
      const custom_fmt_t template_insn[] = {
        {match_vle8_v,   mask_vle8_v,   "vl%se%d.v",   v_ld_unit},
        {match_vse8_v,   mask_vse8_v,   "vs%se%d.v",   v_st_unit},

        {match_vluxei8_v, mask_vluxei8_v, "vlux%sei%d.v", v_ld_index},
        {match_vsuxei8_v, mask_vsuxei8_v, "vsux%sei%d.v", v_st_index},

        {match_vlse8_v,  mask_vlse8_v,  "vls%se%d.v",  v_ld_stride},
        {match_vsse8_v,  mask_vsse8_v,  "vss%se%d.v",  v_st_stride},

        {match_vloxei8_v, mask_vloxei8_v, "vlox%sei%d.v", v_ld_index},
        {match_vsoxei8_v, mask_vsoxei8_v, "vsox%sei%d.v", v_st_index},

        {match_vle8ff_v, mask_vle8ff_v, "vl%se%dff.v", v_ld_unit}
      };

      reg_t elt_map[] = {0x00000000, 0x00005000, 0x00006000, 0x00007000,
                         0x10000000, 0x10005000, 0x10006000, 0x10007000};

      for (unsigned nf = 0; nf <= 7; ++nf) {
        const auto seg_str = nf ? "seg" + std::to_string(nf + 1) : "";

        for (auto item : template_insn) {
          const reg_t match_nf = nf << 29;
          char buf[128];
          snprintf(buf, sizeof(buf), item.fmt, seg_str.c_str(), 8 << elt);
          add_insn(new disasm_insn_t(
            buf,
            ((item.match | match_nf) & ~mask_vldst) | elt_map[elt],
            item.mask | mask_nf,
            item.arg
            ));
        }
      }

      const custom_fmt_t template_insn2[] = {
        {match_vl1re8_v,   mask_vl1re8_v,   "vl%dre%d.v",   v_ld_unit},
      };

      for (reg_t i = 0, nf = 7; i < 4; i++, nf >>= 1) {
        for (auto item : template_insn2) {
          const reg_t match_nf = nf << 29;
          char buf[128];
          snprintf(buf, sizeof(buf), item.fmt, nf + 1, 8 << elt);
          add_insn(new disasm_insn_t(
            buf,
            item.match | match_nf | elt_map[elt],
            item.mask | mask_nf,
            item.arg
          ));
        }
      }
    }

    #define DISASM_ST_WHOLE_INSN(name, nf) \
      add_insn(new disasm_insn_t(#name, match_vs1r_v | (nf << 29), \
                                        mask_vs1r_v | mask_nf, \
                                        {&vs3, &v_address}));
    DISASM_ST_WHOLE_INSN(vs1r.v, 0);
    DISASM_ST_WHOLE_INSN(vs2r.v, 1);
    DISASM_ST_WHOLE_INSN(vs4r.v, 3);
    DISASM_ST_WHOLE_INSN(vs8r.v, 7);

    #undef DISASM_ST_WHOLE_INSN

    #define DEFINE_VECTOR_V(code) add_vector_v_insn(this, #code, match_##code, mask_##code)
    #define DEFINE_VECTOR_VV(code) add_vector_vv_insn(this, #code, match_##code, mask_##code)
    #define DEFINE_VECTOR_VX(code) add_vector_vx_insn(this, #code, match_##code, mask_##code)
    #define DEFINE_VECTOR_VF(code) add_vector_vf_insn(this, #code, match_##code, mask_##code)
    #define DEFINE_VECTOR_VI(code) add_vector_vi_insn(this, #code, match_##code, mask_##code)
    #define DEFINE_VECTOR_VIU(code) add_vector_viu_insn(this, #code, match_##code, mask_##code)

    #define DISASM_OPIV_VXI_INSN(name, sign, suf) \
      DEFINE_VECTOR_VV(name##_##suf##v); \
      DEFINE_VECTOR_VX(name##_##suf##x); \
      if (sign) \
        DEFINE_VECTOR_VI(name##_##suf##i); \
      else \
        DEFINE_VECTOR_VIU(name##_##suf##i)

    #define DISASM_OPIV_VX__INSN(name, sign) \
      DEFINE_VECTOR_VV(name##_vv); \
      DEFINE_VECTOR_VX(name##_vx)

    #define DISASM_OPIV__XI_INSN(name, sign) \
      DEFINE_VECTOR_VX(name##_vx); \
      if (sign) \
        DEFINE_VECTOR_VI(name##_vi); \
      else \
        DEFINE_VECTOR_VIU(name##_vi)

    #define DISASM_OPIV_V___INSN(name, sign) DEFINE_VECTOR_VV(name##_vv)

    #define DISASM_OPIV_S___INSN(name, sign) DEFINE_VECTOR_VV(name##_vs)

    #define DISASM_OPIV_W___INSN(name, sign) \
      DEFINE_VECTOR_VV(name##_wv); \
      DEFINE_VECTOR_VX(name##_wx)

    #define DISASM_OPIV_M___INSN(name, sign) DEFINE_VECTOR_VV(name##_mm)

    #define DISASM_OPIV__X__INSN(name, sign) DEFINE_VECTOR_VX(name##_vx)

    #define DEFINE_VECTOR_VVM(name) \
      add_vector_vvm_insn(this, #name, match_##name, mask_##name | mask_vm)

    #define DEFINE_VECTOR_VXM(name) \
      add_vector_vxm_insn(this, #name, match_##name, mask_##name | mask_vm)

    #define DEFINE_VECTOR_VIM(name) \
      add_vector_vim_insn(this, #name, match_##name, mask_##name | mask_vm)

    #define DISASM_OPIV_VXIM_INSN(name) \
      DEFINE_VECTOR_VVM(name##_vvm); \
      DEFINE_VECTOR_VXM(name##_vxm); \
      DEFINE_VECTOR_VIM(name##_vim)

    #define DISASM_OPIV_VX_M_INSN(name) \
      DEFINE_VECTOR_VVM(name##_vvm); \
      DEFINE_VECTOR_VXM(name##_vxm)

    //OPFVV/OPFVF
    //0b00_0000
    DISASM_OPIV_VXI_INSN(vadd,         1, v);
    DISASM_OPIV_VX__INSN(vsub,         1);
    DISASM_OPIV__XI_INSN(vrsub,        1);
    DISASM_OPIV_VX__INSN(vminu,        0);
    DISASM_OPIV_VX__INSN(vmin,         1);
    DISASM_OPIV_VX__INSN(vmaxu,        1);
    DISASM_OPIV_VX__INSN(vmax,         0);
    DISASM_OPIV_VXI_INSN(vand,         1, v);
    DISASM_OPIV_VXI_INSN(vor,          1, v);
    DISASM_OPIV_VXI_INSN(vxor,         1, v);
    DISASM_OPIV_VXI_INSN(vrgather,     0, v);
    DISASM_OPIV_V___INSN(vrgatherei16, 0);
    DISASM_OPIV__XI_INSN(vslideup,     0);
    DISASM_OPIV__XI_INSN(vslidedown,   0);

    //0b01_0000
    DISASM_OPIV_VXIM_INSN(vadc);
    DISASM_OPIV_VX_M_INSN(vsbc);
    DISASM_OPIV_VXIM_INSN(vmadc);
    DISASM_OPIV_VXI_INSN(vmadc, 1, v);
    DISASM_OPIV_VX_M_INSN(vmsbc);
    DISASM_OPIV_VX__INSN(vmsbc, 1);
    DISASM_OPIV_VXIM_INSN(vmerge);
    DISASM_INSN("vmv.v.i", vmv_v_i, 0, {&vd, &v_simm5});
    DISASM_INSN("vmv.v.v", vmv_v_v, 0, {&vd, &vs1});
    DISASM_INSN("vmv.v.x", vmv_v_x, 0, {&vd, &xrs1});
    DISASM_OPIV_VXI_INSN(vmseq,     1, v);
    DISASM_OPIV_VXI_INSN(vmsne,     1, v);
    DISASM_OPIV_VX__INSN(vmsltu,    0);
    DISASM_OPIV_VX__INSN(vmslt,     1);
    DISASM_OPIV_VXI_INSN(vmsleu,    0, v);
    DISASM_OPIV_VXI_INSN(vmsle,     1, v);
    DISASM_OPIV__XI_INSN(vmsgtu,    0);
    DISASM_OPIV__XI_INSN(vmsgt,     1);

    //0b10_0000
    DISASM_OPIV_VXI_INSN(vsaddu,    0, v);
    DISASM_OPIV_VXI_INSN(vsadd,     1, v);
    DISASM_OPIV_VX__INSN(vssubu,    0);
    DISASM_OPIV_VX__INSN(vssub,     1);
    DISASM_OPIV_VXI_INSN(vsll,      1, v);
    DISASM_INSN("vmv1r.v", vmv1r_v, 0, {&vd, &vs2});
    DISASM_INSN("vmv2r.v", vmv2r_v, 0, {&vd, &vs2});
    DISASM_INSN("vmv4r.v", vmv4r_v, 0, {&vd, &vs2});
    DISASM_INSN("vmv8r.v", vmv8r_v, 0, {&vd, &vs2});
    DISASM_OPIV_VX__INSN(vsmul,     1);
    DISASM_OPIV_VXI_INSN(vsrl,      0, v);
    DISASM_OPIV_VXI_INSN(vsra,      0, v);
    DISASM_OPIV_VXI_INSN(vssrl,     0, v);
    DISASM_OPIV_VXI_INSN(vssra,     0, v);
    DISASM_OPIV_VXI_INSN(vnsrl,     0, w);
    DISASM_OPIV_VXI_INSN(vnsra,     0, w);
    DISASM_OPIV_VXI_INSN(vnclipu,   0, w);
    DISASM_OPIV_VXI_INSN(vnclip,    0, w);

    //0b11_0000
    DISASM_OPIV_S___INSN(vwredsumu, 0);
    DISASM_OPIV_S___INSN(vwredsum,  1);

    //OPMVV/OPMVX
    //0b00_0000
    DISASM_OPIV_VX__INSN(vaaddu,    0);
    DISASM_OPIV_VX__INSN(vaadd,     0);
    DISASM_OPIV_VX__INSN(vasubu,    0);
    DISASM_OPIV_VX__INSN(vasub,     0);

    DISASM_OPIV_S___INSN(vredsum,   1);
    DISASM_OPIV_S___INSN(vredand,   1);
    DISASM_OPIV_S___INSN(vredor,    1);
    DISASM_OPIV_S___INSN(vredxor,   1);
    DISASM_OPIV_S___INSN(vredminu,  0);
    DISASM_OPIV_S___INSN(vredmin,   1);
    DISASM_OPIV_S___INSN(vredmaxu,  0);
    DISASM_OPIV_S___INSN(vredmax,   1);
    DISASM_OPIV__X__INSN(vslide1up,  1);
    DISASM_OPIV__X__INSN(vslide1down,1);

    //0b01_0000
    //VWXUNARY0
    DISASM_INSN("vmv.x.s", vmv_x_s, 0, {&xrd, &vs2});
    DISASM_INSN("vcpop.m", vcpop_m, 0, {&xrd, &vs2, opt, &vm});
    DISASM_INSN("vfirst.m", vfirst_m, 0, {&xrd, &vs2, opt, &vm});

    //VRXUNARY0
    DISASM_INSN("vmv.s.x", vmv_s_x, 0, {&vd, &xrs1});

    //VXUNARY0
    DEFINE_VECTOR_V(vzext_vf2);
    DEFINE_VECTOR_V(vsext_vf2);
    DEFINE_VECTOR_V(vzext_vf4);
    DEFINE_VECTOR_V(vsext_vf4);
    DEFINE_VECTOR_V(vzext_vf8);
    DEFINE_VECTOR_V(vsext_vf8);

    //VMUNARY0
    DEFINE_VECTOR_V(vmsbf_m);
    DEFINE_VECTOR_V(vmsof_m);
    DEFINE_VECTOR_V(vmsif_m);
    DEFINE_VECTOR_V(viota_m);
    DISASM_INSN("vid.v", vid_v, 0, {&vd, opt, &vm});

    DISASM_INSN("vid.v", vid_v, 0, {&vd, opt, &vm});

    DISASM_INSN("vcompress.vm", vcompress_vm, 0, {&vd, &vs2, &vs1});

    DISASM_OPIV_M___INSN(vmandn,    1);
    DISASM_OPIV_M___INSN(vmand,     1);
    DISASM_OPIV_M___INSN(vmor,      1);
    DISASM_OPIV_M___INSN(vmxor,     1);
    DISASM_OPIV_M___INSN(vmorn,     1);
    DISASM_OPIV_M___INSN(vmnand,    1);
    DISASM_OPIV_M___INSN(vmnor,     1);
    DISASM_OPIV_M___INSN(vmxnor,    1);

    //0b10_0000
    DISASM_OPIV_VX__INSN(vdivu,     0);
    DISASM_OPIV_VX__INSN(vdiv,      1);
    DISASM_OPIV_VX__INSN(vremu,     0);
    DISASM_OPIV_VX__INSN(vrem,      1);
    DISASM_OPIV_VX__INSN(vmulhu,    0);
    DISASM_OPIV_VX__INSN(vmul,      1);
    DISASM_OPIV_VX__INSN(vmulhsu,   0);
    DISASM_OPIV_VX__INSN(vmulh,     1);
    DISASM_OPIV_VX__INSN(vmadd,     1);
    DISASM_OPIV_VX__INSN(vnmsub,    1);
    DISASM_OPIV_VX__INSN(vmacc,     1);
    DISASM_OPIV_VX__INSN(vnmsac,    1);

    //0b11_0000
    DISASM_OPIV_VX__INSN(vwaddu,    0);
    DISASM_OPIV_VX__INSN(vwadd,     1);
    DISASM_OPIV_VX__INSN(vwsubu,    0);
    DISASM_OPIV_VX__INSN(vwsub,     1);
    DISASM_OPIV_W___INSN(vwaddu,    0);
    DISASM_OPIV_W___INSN(vwadd,     1);
    DISASM_OPIV_W___INSN(vwsubu,    0);
    DISASM_OPIV_W___INSN(vwsub,     1);
    DISASM_OPIV_VX__INSN(vwmulu,    0);
    DISASM_OPIV_VX__INSN(vwmulsu,   0);
    DISASM_OPIV_VX__INSN(vwmul,     1);
    DISASM_OPIV_VX__INSN(vwmaccu,   0);
    DISASM_OPIV_VX__INSN(vwmacc,    1);
    DISASM_OPIV__X__INSN(vwmaccus,  1);
    DISASM_OPIV_VX__INSN(vwmaccsu,  0);

    #undef DISASM_OPIV_VXI_INSN
    #undef DISASM_OPIV_VX__INSN
    #undef DISASM_OPIV__XI_INSN
    #undef DISASM_OPIV_V___INSN
    #undef DISASM_OPIV_S___INSN
    #undef DISASM_OPIV_W___INSN
    #undef DISASM_OPIV_M___INSN
    #undef DISASM_OPIV__X__INSN
    #undef DISASM_OPIV_VXIM_INSN
    #undef DISASM_OPIV_VX_M_INSN

    #define DISASM_OPIV_VF_INSN(name) \
      DEFINE_VECTOR_VV(name##_vv); \
      DEFINE_VECTOR_VF(name##_vf)

    #define DISASM_OPIV_WF_INSN(name) \
      DEFINE_VECTOR_VV(name##_wv); \
      DEFINE_VECTOR_VF(name##_wf)

    #define DISASM_OPIV_S__INSN(name) \
      DEFINE_VECTOR_VV(name##_vs)

    #define DISASM_OPIV__F_INSN(name) \
      DEFINE_VECTOR_VF(name##_vf)

    #define DISASM_VFUNARY0_INSN(name, suf) \
      DEFINE_VECTOR_V(name##cvt_rtz_xu_f_##suf); \
      DEFINE_VECTOR_V(name##cvt_rtz_x_f_##suf); \
      DEFINE_VECTOR_V(name##cvt_xu_f_##suf); \
      DEFINE_VECTOR_V(name##cvt_x_f_##suf); \
      DEFINE_VECTOR_V(name##cvt_f_xu_##suf); \
      DEFINE_VECTOR_V(name##cvt_f_x_##suf)

    //OPFVV/OPFVF
    //0b00_0000
    DISASM_OPIV_VF_INSN(vfadd);
    DISASM_OPIV_S__INSN(vfredusum);
    DISASM_OPIV_VF_INSN(vfsub);
    DISASM_OPIV_S__INSN(vfredosum);
    DISASM_OPIV_VF_INSN(vfmin);
    DISASM_OPIV_S__INSN(vfredmin);
    DISASM_OPIV_VF_INSN(vfmax);
    DISASM_OPIV_S__INSN(vfredmax);
    DISASM_OPIV_VF_INSN(vfsgnj);
    DISASM_OPIV_VF_INSN(vfsgnjn);
    DISASM_OPIV_VF_INSN(vfsgnjx);
    DISASM_INSN("vfmv.f.s", vfmv_f_s, 0, {&frd, &vs2});
    DISASM_INSN("vfmv.s.f", vfmv_s_f, mask_vfmv_s_f, {&vd, &frs1});
    DISASM_OPIV__F_INSN(vfslide1up);
    DISASM_OPIV__F_INSN(vfslide1down);

    //0b01_0000
    DISASM_INSN("vfmerge.vfm", vfmerge_vfm, 0, {&vd, &vs2, &frs1, &v0});
    DISASM_INSN("vfmv.v.f", vfmv_v_f, 0, {&vd, &frs1});
    DISASM_OPIV_VF_INSN(vmfeq);
    DISASM_OPIV_VF_INSN(vmfle);
    DISASM_OPIV_VF_INSN(vmflt);
    DISASM_OPIV_VF_INSN(vmfne);
    DISASM_OPIV__F_INSN(vmfgt);
    DISASM_OPIV__F_INSN(vmfge);

    //0b10_0000
    DISASM_OPIV_VF_INSN(vfdiv);
    DISASM_OPIV__F_INSN(vfrdiv);

    //vfunary0
    DISASM_VFUNARY0_INSN(vf,  v);
    DISASM_VFUNARY0_INSN(vfw, v);
    DEFINE_VECTOR_V(vfwcvt_f_f_v);

    DISASM_VFUNARY0_INSN(vfn, w);
    DEFINE_VECTOR_V(vfncvt_f_f_w);
    DEFINE_VECTOR_V(vfncvt_rod_f_f_w);

    //vfunary1
    DEFINE_VECTOR_V(vfsqrt_v);
    DEFINE_VECTOR_V(vfrsqrt7_v);
    DEFINE_VECTOR_V(vfrec7_v);
    DEFINE_VECTOR_V(vfclass_v);

    DISASM_OPIV_VF_INSN(vfmul);
    DISASM_OPIV__F_INSN(vfrsub);
    DISASM_OPIV_VF_INSN(vfmadd);
    DISASM_OPIV_VF_INSN(vfnmadd);
    DISASM_OPIV_VF_INSN(vfmsub);
    DISASM_OPIV_VF_INSN(vfnmsub);
    DISASM_OPIV_VF_INSN(vfmacc);
    DISASM_OPIV_VF_INSN(vfnmacc);
    DISASM_OPIV_VF_INSN(vfmsac);
    DISASM_OPIV_VF_INSN(vfnmsac);

    //0b11_0000
    DISASM_OPIV_VF_INSN(vfwadd);
    DISASM_OPIV_S__INSN(vfwredusum);
    DISASM_OPIV_VF_INSN(vfwsub);
    DISASM_OPIV_S__INSN(vfwredosum);
    DISASM_OPIV_WF_INSN(vfwadd);
    DISASM_OPIV_WF_INSN(vfwsub);
    DISASM_OPIV_VF_INSN(vfwmul);
    DISASM_OPIV_VF_INSN(vfwmacc);
    DISASM_OPIV_VF_INSN(vfwnmacc);
    DISASM_OPIV_VF_INSN(vfwmsac);
    DISASM_OPIV_VF_INSN(vfwnmsac);

    #undef DISASM_OPIV_VF_INSN
    #undef DISASM_OPIV__F_INSN
    #undef DISASM_OPIV_S__INSN
    #undef DISASM_OPIV_W__INSN
    #undef DISASM_VFUNARY0_INSN
  }

  if (isa->extension_enabled(EXT_ZVFBFMIN)) {
    DEFINE_VECTOR_V(vfncvtbf16_f_f_w);
    DEFINE_VECTOR_V(vfwcvtbf16_f_f_v);
  }

  if (isa->extension_enabled(EXT_ZVFBFWMA)) {
    DEFINE_VECTOR_VV(vfwmaccbf16_vv);
    DEFINE_VECTOR_VF(vfwmaccbf16_vf);
  }

#define DEFINE_PI3TYPE(code) add_pitype3_insn(this, #code, match_##code, mask_##code);
#define DEFINE_PI4TYPE(code) add_pitype4_insn(this, #code, match_##code, mask_##code);
#define DEFINE_PI5TYPE(code) add_pitype5_insn(this, #code, match_##code, mask_##code);
#define DEFINE_PI6TYPE(code) add_pitype6_insn(this, #code, match_##code, mask_##code);

#define DISASM_8_AND_16_RINSN(code) \
  DEFINE_RTYPE(code##8); \
  DEFINE_RTYPE(code##16);

#define DISASM_8_AND_16_RINSN_ROUND(code) \
  DEFINE_RTYPE(code##8_u); \
  DEFINE_RTYPE(code##16_u);

#define DISASM_8_AND_16_PIINSN(code) \
  DEFINE_PI3TYPE(code##8); \
  DEFINE_PI4TYPE(code##16);

#define DISASM_8_AND_16_PIINSN_ROUND(code) \
  DEFINE_PI3TYPE(code##8_u); \
  DEFINE_PI4TYPE(code##16_u);

#define DISASM_RINSN_AND_ROUND(code) \
  DEFINE_RTYPE(code); \
  DEFINE_RTYPE(code##_u); \
  
  if (isa->extension_enabled(EXT_ZMMUL)) {
    DEFINE_RTYPE(mul);
    DEFINE_RTYPE(mulh);
    DEFINE_RTYPE(mulhu);
    DEFINE_RTYPE(mulhsu);
    DEFINE_RTYPE(mulw);
  }

  if (isa->extension_enabled(EXT_ZBPBO)) {
    DEFINE_RTYPE(min);
    DEFINE_RTYPE(max);
    DEFINE_R3TYPE(cmix);
    DEFINE_RTYPE(pack);
    DEFINE_RTYPE(packu);
    add_insn(new disasm_insn_t("rev", match_grevi | ((isa->get_max_xlen() - 1) << imm_shift), mask_grevi | mask_imm, {&xrd, &xrs1}));
    add_insn(new disasm_insn_t("rev8.h", match_grevi | (0x8 << imm_shift), mask_grevi | mask_imm, {&xrd, &xrs1})); // swap16
    if (isa->get_max_xlen() == 32) {
      DEFINE_R1TYPE(clz);
      DEFINE_R3TYPE(fsr);
      DEFINE_R3TYPE(fsri);
    } else {
      DEFINE_R3TYPE(fsrw);
    }
  }

  if (isa->extension_enabled(EXT_ZPSFOPERAND)) {
    DEFINE_RTYPE(smal)
    DEFINE_RTYPE(radd64);
    DEFINE_RTYPE(uradd64);
    DEFINE_RTYPE(kadd64);
    DEFINE_RTYPE(ukadd64);
    DEFINE_RTYPE(rsub64);
    DEFINE_RTYPE(ursub64);
    DEFINE_RTYPE(ksub64);
    DEFINE_RTYPE(uksub64);
    DEFINE_RTYPE(smar64);
    DEFINE_RTYPE(smsr64);
    DEFINE_RTYPE(umar64);
    DEFINE_RTYPE(umsr64);
    DEFINE_RTYPE(kmar64);
    DEFINE_RTYPE(kmsr64);
    DEFINE_RTYPE(ukmar64);
    DEFINE_RTYPE(ukmsr64);
    DEFINE_RTYPE(smalbb);
    DEFINE_RTYPE(smalbt);
    DEFINE_RTYPE(smaltt);
    DEFINE_RTYPE(smalda);
    DEFINE_RTYPE(smalxda);
    DEFINE_RTYPE(smalds);
    DEFINE_RTYPE(smaldrs);
    DEFINE_RTYPE(smalxds);
    DEFINE_RTYPE(smslda);
    DEFINE_RTYPE(smslxda);
    DEFINE_RTYPE(mulr64);
    DEFINE_RTYPE(mulsr64);
    if (isa->get_max_xlen() == 32) {
      DEFINE_RTYPE(add64);
      DEFINE_RTYPE(sub64);
    }
  }

  if (isa->extension_enabled(EXT_ZPN)) {
    DISASM_8_AND_16_RINSN(add);
    DISASM_8_AND_16_RINSN(radd);
    DISASM_8_AND_16_RINSN(uradd);
    DISASM_8_AND_16_RINSN(kadd);
    DISASM_8_AND_16_RINSN(ukadd);
    DISASM_8_AND_16_RINSN(sub);
    DISASM_8_AND_16_RINSN(rsub);
    DISASM_8_AND_16_RINSN(ursub);
    DISASM_8_AND_16_RINSN(ksub);
    DISASM_8_AND_16_RINSN(uksub);
    DEFINE_RTYPE(cras16);
    DEFINE_RTYPE(rcras16);
    DEFINE_RTYPE(urcras16);
    DEFINE_RTYPE(kcras16);
    DEFINE_RTYPE(ukcras16);
    DEFINE_RTYPE(crsa16);
    DEFINE_RTYPE(rcrsa16);
    DEFINE_RTYPE(urcrsa16);
    DEFINE_RTYPE(kcrsa16);
    DEFINE_RTYPE(ukcrsa16);
    DEFINE_RTYPE(stas16);
    DEFINE_RTYPE(rstas16);
    DEFINE_RTYPE(urstas16);
    DEFINE_RTYPE(kstas16);
    DEFINE_RTYPE(ukstas16);
    DEFINE_RTYPE(stsa16);
    DEFINE_RTYPE(rstsa16);
    DEFINE_RTYPE(urstsa16);
    DEFINE_RTYPE(kstsa16);
    DEFINE_RTYPE(ukstsa16);
    DISASM_8_AND_16_RINSN(sra);
    DISASM_8_AND_16_RINSN(srl);
    DISASM_8_AND_16_RINSN(sll);
    DISASM_8_AND_16_RINSN(ksll);
    DISASM_8_AND_16_RINSN(kslra);
    DISASM_8_AND_16_PIINSN(srai);
    DISASM_8_AND_16_PIINSN(srli);
    DISASM_8_AND_16_PIINSN(slli);
    DISASM_8_AND_16_PIINSN(kslli);
    DISASM_8_AND_16_RINSN_ROUND(sra);
    DISASM_8_AND_16_RINSN_ROUND(srl);
    DISASM_8_AND_16_RINSN_ROUND(kslra);
    DISASM_8_AND_16_PIINSN_ROUND(srai);
    DISASM_8_AND_16_PIINSN_ROUND(srli);

    DISASM_8_AND_16_RINSN(cmpeq);
    DISASM_8_AND_16_RINSN(scmplt);
    DISASM_8_AND_16_RINSN(scmple);
    DISASM_8_AND_16_RINSN(ucmplt);
    DISASM_8_AND_16_RINSN(ucmple);

    DISASM_8_AND_16_RINSN(smul);
    DISASM_8_AND_16_RINSN(smulx);
    DISASM_8_AND_16_RINSN(umul);
    DISASM_8_AND_16_RINSN(umulx);
    DISASM_8_AND_16_RINSN(khm);
    DISASM_8_AND_16_RINSN(khmx);

    DISASM_8_AND_16_RINSN(smin);
    DISASM_8_AND_16_RINSN(umin);
    DISASM_8_AND_16_RINSN(smax);
    DISASM_8_AND_16_RINSN(umax);
    DISASM_8_AND_16_PIINSN(sclip);
    DISASM_8_AND_16_PIINSN(uclip);
    DEFINE_R1TYPE(kabs16);
    DEFINE_R1TYPE(clrs16);
    DEFINE_R1TYPE(clz16);
    DEFINE_R1TYPE(kabs8);
    DEFINE_R1TYPE(clrs8);
    DEFINE_R1TYPE(clz8);

    DEFINE_R1TYPE(sunpkd810);
    DEFINE_R1TYPE(sunpkd820);
    DEFINE_R1TYPE(sunpkd830);
    DEFINE_R1TYPE(sunpkd831);
    DEFINE_R1TYPE(sunpkd832);
    DEFINE_R1TYPE(zunpkd810);
    DEFINE_R1TYPE(zunpkd820);
    DEFINE_R1TYPE(zunpkd830);
    DEFINE_R1TYPE(zunpkd831);
    DEFINE_R1TYPE(zunpkd832);

    DEFINE_RTYPE(pkbb16);
    DEFINE_RTYPE(pkbt16);
    DEFINE_RTYPE(pktb16);
    DEFINE_RTYPE(pktt16);
    DISASM_RINSN_AND_ROUND(smmul);
    DISASM_RINSN_AND_ROUND(kmmac);
    DISASM_RINSN_AND_ROUND(kmmsb);
    DISASM_RINSN_AND_ROUND(kwmmul);
    DISASM_RINSN_AND_ROUND(smmwb);
    DISASM_RINSN_AND_ROUND(smmwt);
    DISASM_RINSN_AND_ROUND(kmmawb);
    DISASM_RINSN_AND_ROUND(kmmawt);
    DISASM_RINSN_AND_ROUND(kmmwb2);
    DISASM_RINSN_AND_ROUND(kmmwt2);
    DISASM_RINSN_AND_ROUND(kmmawb2);
    DISASM_RINSN_AND_ROUND(kmmawt2);
    DEFINE_RTYPE(smbb16)
    DEFINE_RTYPE(smbt16)
    DEFINE_RTYPE(smtt16)
    DEFINE_RTYPE(kmda)
    DEFINE_RTYPE(kmxda)
    DEFINE_RTYPE(smds)
    DEFINE_RTYPE(smdrs)
    DEFINE_RTYPE(smxds)
    DEFINE_RTYPE(kmabb)
    DEFINE_RTYPE(kmabt)
    DEFINE_RTYPE(kmatt)
    DEFINE_RTYPE(kmada)
    DEFINE_RTYPE(kmaxda)
    DEFINE_RTYPE(kmads)
    DEFINE_RTYPE(kmadrs)
    DEFINE_RTYPE(kmaxds)
    DEFINE_RTYPE(kmsda)
    DEFINE_RTYPE(kmsxda)
    DEFINE_RTYPE(sclip32)
    DEFINE_RTYPE(uclip32)
    DEFINE_R1TYPE(clrs32);
    DEFINE_R1TYPE(clz32);
    DEFINE_RTYPE(pbsad);
    DEFINE_RTYPE(pbsada);
    DEFINE_RTYPE(smaqa);
    DEFINE_RTYPE(umaqa);
    DEFINE_RTYPE(smaqa_su);

    DEFINE_RTYPE(kaddh);
    DEFINE_RTYPE(ksubh);
    DEFINE_RTYPE(khmbb);
    DEFINE_RTYPE(khmbt);
    DEFINE_RTYPE(khmtt);
    DEFINE_RTYPE(ukaddh);
    DEFINE_RTYPE(uksubh);
    DEFINE_RTYPE(kaddw);
    DEFINE_RTYPE(ukaddw);
    DEFINE_RTYPE(ksubw);
    DEFINE_RTYPE(uksubw);
    DEFINE_RTYPE(kdmbb);
    DEFINE_RTYPE(kdmbt);
    DEFINE_RTYPE(kdmtt);
    DEFINE_RTYPE(kslraw);
    DEFINE_RTYPE(kslraw_u);
    DEFINE_RTYPE(ksllw);
    DEFINE_PI5TYPE(kslliw);
    DEFINE_RTYPE(kdmabb);
    DEFINE_RTYPE(kdmabt);
    DEFINE_RTYPE(kdmatt);
    DEFINE_RTYPE(kabsw);
    DEFINE_RTYPE(raddw);
    DEFINE_RTYPE(uraddw);
    DEFINE_RTYPE(rsubw);
    DEFINE_RTYPE(ursubw);
    DEFINE_RTYPE(msubr32);
    DEFINE_RTYPE(ave);
    DEFINE_RTYPE(sra_u);
    DEFINE_PI6TYPE(srai_u);
    DEFINE_PI3TYPE(insb);
    DEFINE_RTYPE(maddr32)

    if (isa->get_max_xlen() == 64) {
      DEFINE_RTYPE(add32);
      DEFINE_RTYPE(radd32);
      DEFINE_RTYPE(uradd32);
      DEFINE_RTYPE(kadd32);
      DEFINE_RTYPE(ukadd32);
      DEFINE_RTYPE(sub32);
      DEFINE_RTYPE(rsub32);
      DEFINE_RTYPE(ursub32);
      DEFINE_RTYPE(ksub32);
      DEFINE_RTYPE(uksub32);
      DEFINE_RTYPE(cras32);
      DEFINE_RTYPE(rcras32);
      DEFINE_RTYPE(urcras32);
      DEFINE_RTYPE(kcras32);
      DEFINE_RTYPE(ukcras32);
      DEFINE_RTYPE(crsa32);
      DEFINE_RTYPE(rcrsa32);
      DEFINE_RTYPE(urcrsa32);
      DEFINE_RTYPE(kcrsa32);
      DEFINE_RTYPE(ukcrsa32);
      DEFINE_RTYPE(stas32);
      DEFINE_RTYPE(rstas32);
      DEFINE_RTYPE(urstas32);
      DEFINE_RTYPE(kstas32);
      DEFINE_RTYPE(ukstas32);
      DEFINE_RTYPE(stsa32);
      DEFINE_RTYPE(rstsa32);
      DEFINE_RTYPE(urstsa32);
      DEFINE_RTYPE(kstsa32);
      DEFINE_RTYPE(ukstsa32);
      DEFINE_RTYPE(sra32);
      DEFINE_PI5TYPE(srai32);
      DEFINE_RTYPE(sra32_u);
      DEFINE_PI5TYPE(srai32_u);
      DEFINE_RTYPE(srl32);
      DEFINE_PI5TYPE(srli32);
      DEFINE_RTYPE(srl32_u);
      DEFINE_PI5TYPE(srli32_u);
      DEFINE_RTYPE(sll32);
      DEFINE_PI5TYPE(slli32);
      DEFINE_RTYPE(ksll32);
      DEFINE_PI5TYPE(kslli32);
      DEFINE_RTYPE(kslra32);
      DEFINE_RTYPE(kslra32_u);
      DEFINE_RTYPE(smin32);
      DEFINE_RTYPE(umin32);
      DEFINE_RTYPE(smax32);
      DEFINE_RTYPE(umax32);
      DEFINE_R1TYPE(kabs32);
      DEFINE_RTYPE(khmbb16);
      DEFINE_RTYPE(khmbt16);
      DEFINE_RTYPE(khmtt16);
      DEFINE_RTYPE(kdmbb16);
      DEFINE_RTYPE(kdmbt16);
      DEFINE_RTYPE(kdmtt16);
      DEFINE_RTYPE(kdmabb16);
      DEFINE_RTYPE(kdmabt16);
      DEFINE_RTYPE(kdmatt16);
      DEFINE_RTYPE(smbt32);
      DEFINE_RTYPE(smtt32);
      DEFINE_RTYPE(kmabb32);
      DEFINE_RTYPE(kmabt32);
      DEFINE_RTYPE(kmatt32);
      DEFINE_RTYPE(kmda32);
      DEFINE_RTYPE(kmxda32);
      DEFINE_RTYPE(kmaxda32);
      DEFINE_RTYPE(kmads32);
      DEFINE_RTYPE(kmadrs32);
      DEFINE_RTYPE(kmaxds32);
      DEFINE_RTYPE(kmsda32);
      DEFINE_RTYPE(kmsxda32);
      DEFINE_RTYPE(smds32);
      DEFINE_RTYPE(smdrs32);
      DEFINE_RTYPE(smxds32);
      DEFINE_PI5TYPE(sraiw_u);
      DEFINE_RTYPE(pkbt32);
      DEFINE_RTYPE(pktb32);
    }
  }

  if (isa->extension_enabled(EXT_ZICBOM)) {
    DISASM_INSN("cbo.clean", cbo_clean, 0, {&base_only_address});
    DISASM_INSN("cbo.flush", cbo_flush, 0, {&base_only_address});
    DISASM_INSN("cbo.inval", cbo_inval, 0, {&base_only_address});
  }

  if (isa->extension_enabled(EXT_ZICBOZ)) {
    DISASM_INSN("cbo.zero", cbo_zero, 0, {&base_only_address});
  }

  if (isa->extension_enabled(EXT_ZICOND)) {
    DEFINE_RTYPE(czero_eqz);
    DEFINE_RTYPE(czero_nez);
  }

  if (isa->extension_enabled(EXT_ZIMOP)) {
    DEFINE_R1TYPE(mop_r_0);
    DEFINE_R1TYPE(mop_r_1);
    DEFINE_R1TYPE(mop_r_2);
    DEFINE_R1TYPE(mop_r_3);
    DEFINE_R1TYPE(mop_r_4);
    DEFINE_R1TYPE(mop_r_5);
    DEFINE_R1TYPE(mop_r_6);
    DEFINE_R1TYPE(mop_r_7);
    DEFINE_R1TYPE(mop_r_8);
    DEFINE_R1TYPE(mop_r_9);
    DEFINE_R1TYPE(mop_r_10);
    DEFINE_R1TYPE(mop_r_11);
    DEFINE_R1TYPE(mop_r_12);
    DEFINE_R1TYPE(mop_r_13);
    DEFINE_R1TYPE(mop_r_14);
    DEFINE_R1TYPE(mop_r_15);
    DEFINE_R1TYPE(mop_r_16);
    DEFINE_R1TYPE(mop_r_17);
    DEFINE_R1TYPE(mop_r_18);
    DEFINE_R1TYPE(mop_r_19);
    DEFINE_R1TYPE(mop_r_20);
    DEFINE_R1TYPE(mop_r_21);
    DEFINE_R1TYPE(mop_r_22);
    DEFINE_R1TYPE(mop_r_23);
    DEFINE_R1TYPE(mop_r_24);
    DEFINE_R1TYPE(mop_r_25);
    DEFINE_R1TYPE(mop_r_26);
    DEFINE_R1TYPE(mop_r_27);
    DEFINE_R1TYPE(mop_r_28);
    DEFINE_R1TYPE(mop_r_29);
    DEFINE_R1TYPE(mop_r_30);
    DEFINE_R1TYPE(mop_r_31);
    DEFINE_RTYPE(mop_rr_0);
    DEFINE_RTYPE(mop_rr_1);
    DEFINE_RTYPE(mop_rr_2);
    DEFINE_RTYPE(mop_rr_3);
    DEFINE_RTYPE(mop_rr_4);
    DEFINE_RTYPE(mop_rr_5);
    DEFINE_RTYPE(mop_rr_6);
    DEFINE_RTYPE(mop_rr_7);
  }

  if (isa->extension_enabled(EXT_ZCMOP)) {
    DISASM_INSN("c.mop.1", c_mop_1, 0, {});
    DISASM_INSN("c.mop.3", c_mop_3, 0, {});
    DISASM_INSN("c.mop.5", c_mop_5, 0, {});
    DISASM_INSN("c.mop.7", c_mop_7, 0, {});
    DISASM_INSN("c.mop.9", c_mop_9, 0, {});
    DISASM_INSN("c.mop.11", c_mop_11, 0, {});
    DISASM_INSN("c.mop.13", c_mop_13, 0, {});
    DISASM_INSN("c.mop.15", c_mop_15, 0, {});
  }

  if (isa->extension_enabled(EXT_ZKND) ||
      isa->extension_enabled(EXT_ZKNE)) {
    DISASM_INSN("aes64ks1i", aes64ks1i, 0, {&xrd, &xrs1, &rcon});
    DEFINE_RTYPE(aes64ks2);
  }

  if (isa->extension_enabled(EXT_ZKND)) {
    if(isa->get_max_xlen() == 64) {
      DEFINE_RTYPE(aes64ds);
      DEFINE_RTYPE(aes64dsm);
      DEFINE_R1TYPE(aes64im);
    } else if (isa->get_max_xlen() == 32) {
      DISASM_INSN("aes32dsi", aes32dsi, 0, {&xrd, &xrs1, &xrs2, &bs});
      DISASM_INSN("aes32dsmi", aes32dsmi, 0, {&xrd, &xrs1, &xrs2, &bs});
    }
  }

  if (isa->extension_enabled(EXT_ZKNE)) {
    if(isa->get_max_xlen() == 64) {
      DEFINE_RTYPE(aes64es);
      DEFINE_RTYPE(aes64esm);
    } else if (isa->get_max_xlen() == 32) {
      DISASM_INSN("aes32esi", aes32esi, 0, {&xrd, &xrs1, &xrs2, &bs});
      DISASM_INSN("aes32esmi", aes32esmi, 0, {&xrd, &xrs1, &xrs2, &bs});
    }
  }

  if (isa->extension_enabled(EXT_ZKNH)) {
    DEFINE_R1TYPE(sha256sig0);
    DEFINE_R1TYPE(sha256sig1);
    DEFINE_R1TYPE(sha256sum0);
    DEFINE_R1TYPE(sha256sum1);
    if(isa->get_max_xlen() == 64) {
      DEFINE_R1TYPE(sha512sig0);
      DEFINE_R1TYPE(sha512sig1);
      DEFINE_R1TYPE(sha512sum0);
      DEFINE_R1TYPE(sha512sum1);
    } else if (isa->get_max_xlen() == 32) {
      DEFINE_RTYPE(sha512sig0h);
      DEFINE_RTYPE(sha512sig0l);
      DEFINE_RTYPE(sha512sig1h);
      DEFINE_RTYPE(sha512sig1l);
      DEFINE_RTYPE(sha512sum0r);
      DEFINE_RTYPE(sha512sum1r);
    }
  }

  if (isa->extension_enabled(EXT_ZKSED)) {
    DISASM_INSN("sm4ed", sm4ed, 0, {&xrd, &xrs1, &xrs2, &bs});
    DISASM_INSN("sm4ks", sm4ks, 0, {&xrd, &xrs1, &xrs2, &bs});
  }

  if (isa->extension_enabled(EXT_ZKSH)) {
    DEFINE_R1TYPE(sm3p0);
    DEFINE_R1TYPE(sm3p1);
  }

  if (isa->extension_enabled(EXT_ZVBB)) {
#define DEFINE_VECTOR_VIU_ZIMM6(code) \
  add_vector_viu_z6_insn(this, #code, match_##code, mask_##code)
#define DISASM_VECTOR_VV_VX(name) \
  DEFINE_VECTOR_VV(name##_vv); \
  DEFINE_VECTOR_VX(name##_vx)
#define DISASM_VECTOR_VV_VX_VIU(name) \
  DEFINE_VECTOR_VV(name##_vv); \
  DEFINE_VECTOR_VX(name##_vx); \
  DEFINE_VECTOR_VIU(name##_vi)
#define DISASM_VECTOR_VV_VX_VIU_ZIMM6(name) \
  DEFINE_VECTOR_VV(name##_vv); \
  DEFINE_VECTOR_VX(name##_vx); \
  DEFINE_VECTOR_VIU_ZIMM6(name##_vi)

    DISASM_VECTOR_VV_VX(vandn);
    DEFINE_VECTOR_V(vbrev_v);
    DEFINE_VECTOR_V(vbrev8_v);
    DEFINE_VECTOR_V(vrev8_v);
    DEFINE_VECTOR_V(vclz_v);
    DEFINE_VECTOR_V(vctz_v);
    DEFINE_VECTOR_V(vcpop_v);
    DISASM_VECTOR_VV_VX(vrol);
    DISASM_VECTOR_VV_VX_VIU_ZIMM6(vror);
    DISASM_VECTOR_VV_VX_VIU(vwsll);

#undef DEFINE_VECTOR_VIU_ZIMM6
#undef DISASM_VECTOR_VV_VX
#undef DISASM_VECTOR_VV_VX_VIU
#undef DISASM_VECTOR_VV_VX_VIU_ZIMM6
    }

  if (isa->extension_enabled(EXT_ZVBC)) {
#define DISASM_VECTOR_VV_VX(name) \
    DEFINE_VECTOR_VV(name##_vv); \
    DEFINE_VECTOR_VX(name##_vx)

    DISASM_VECTOR_VV_VX(vclmul);
    DISASM_VECTOR_VV_VX(vclmulh);

#undef DISASM_VECTOR_VV_VX
  }

  if (isa->extension_enabled(EXT_ZVKG)) {
    // Despite its suffix, the vgmul.vv instruction
    // is really ".v", with the form "vgmul.vv vd, vs2".
    DEFINE_VECTOR_V(vgmul_vv);
    DEFINE_VECTOR_VV(vghsh_vv);
  }

  if (isa->extension_enabled(EXT_ZVKNED)) {
    // Despite their suffixes, the vaes*.{vv,vs} instructions
    // are really ".v", with the form "<op>.{vv,vs} vd, vs2".
#define DISASM_VECTOR_VV_VS(name) \
    DEFINE_VECTOR_V(name##_vv); \
    DEFINE_VECTOR_V(name##_vs)

    DISASM_VECTOR_VV_VS(vaesdm);
    DISASM_VECTOR_VV_VS(vaesdf);
    DISASM_VECTOR_VV_VS(vaesem);
    DISASM_VECTOR_VV_VS(vaesef);

    DEFINE_VECTOR_V(vaesz_vs);
    DEFINE_VECTOR_VIU(vaeskf1_vi);
    DEFINE_VECTOR_VIU(vaeskf2_vi);
#undef DISASM_VECTOR_VV_VS
  }

  if (isa->extension_enabled(EXT_ZVKNHA) ||
      isa->extension_enabled(EXT_ZVKNHB)) {
    DEFINE_VECTOR_VV(vsha2ms_vv);
    DEFINE_VECTOR_VV(vsha2ch_vv);
    DEFINE_VECTOR_VV(vsha2cl_vv);
  }

  if (isa->extension_enabled(EXT_ZVKSED)) {
    DEFINE_VECTOR_VIU(vsm4k_vi);
    // Despite their suffixes, the vsm4r.{vv,vs} instructions
    // are really ".v", with the form "vsm4r.{vv,vs} vd, vs2".
    DEFINE_VECTOR_V(vsm4r_vv);
    DEFINE_VECTOR_V(vsm4r_vs);
  }

  if (isa->extension_enabled(EXT_ZVKSH)) {
    DEFINE_VECTOR_VIU(vsm3c_vi);
    DEFINE_VECTOR_VV(vsm3me_vv);
  }

  if (isa->extension_enabled(EXT_ZALASR)) {
    DEFINE_XLOAD_BASE(lb_aq);
    DEFINE_XLOAD_BASE(lh_aq);
    DEFINE_XLOAD_BASE(lw_aq);
    DEFINE_XLOAD_BASE(ld_aq);
    DEFINE_XSTORE_BASE(sb_rl);
    DEFINE_XSTORE_BASE(sh_rl);
    DEFINE_XSTORE_BASE(sw_rl);
    DEFINE_XSTORE_BASE(sd_rl);
  }
}

disassembler_t::disassembler_t(const isa_parser_t *isa)
{
  // highest priority: instructions explicitly enabled
  add_instructions(isa);

  // next-highest priority: other instructions in same base ISA
  std::string fallback_isa_string = std::string("rv") + std::to_string(isa->get_max_xlen()) +
    "gqchv_zfh_zba_zbb_zbc_zbs_zcb_zicbom_zicboz_zicond_zkn_zkr_zks_svinval_zcmop_zimop";
  isa_parser_t fallback_isa(fallback_isa_string.c_str(), DEFAULT_PRIV);
  add_instructions(&fallback_isa);

  // finally: instructions with known opcodes but unknown arguments
  add_unknown_insns(this);

  // Now, reverse the lists, because we search them back-to-front (so that
  // custom instructions later added with add_insn have highest priority).
  for (size_t i = 0; i < HASH_SIZE+1; i++)
    std::reverse(chain[i].begin(), chain[i].end());
}

const disasm_insn_t* disassembler_t::probe_once(insn_t insn, size_t idx) const
{
  for (auto it = chain[idx].rbegin(); it != chain[idx].rend(); ++it)
    if (*(*it) == insn)
      return *it;

  return NULL;
}

const disasm_insn_t* disassembler_t::lookup(insn_t insn) const
{
  if (auto p = probe_once(insn, hash(insn.bits(), MASK1)))
    return p;

  if (auto p = probe_once(insn, hash(insn.bits(), MASK2)))
    return p;

  return probe_once(insn, HASH_SIZE);
}

void NOINLINE disassembler_t::add_insn(disasm_insn_t* insn)
{
  size_t idx =
    (insn->get_mask() & MASK1) == MASK1 ? hash(insn->get_match(), MASK1) :
    (insn->get_mask() & MASK2) == MASK2 ? hash(insn->get_match(), MASK2) :
    HASH_SIZE;

  chain[idx].push_back(insn);
}

disassembler_t::~disassembler_t()
{
  for (size_t i = 0; i < HASH_SIZE+1; i++)
    for (size_t j = 0; j < chain[i].size(); j++)
      delete chain[i][j];
}

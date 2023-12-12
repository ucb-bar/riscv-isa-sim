// vmsif.m rd, vs2, vm
require(P.VU.vsew >= e8 && P.VU.vsew <= e64);
require_vector(true);
require(P.VU.vstart->read(p) == 0);
require_vm;
require(insn.rd() != insn.rs2());

reg_t vl = P.VU.vl->read(p);
reg_t rd_num = insn.rd();
reg_t rs2_num = insn.rs2();

bool has_one = false;
for (reg_t i = P.VU.vstart->read(p); i < vl; ++i) {
  const int midx = i / 64;
  const int mpos = i % 64;
  const uint64_t mmask = UINT64_C(1) << mpos; \

  bool vs2_lsb = ((P.VU.elt<uint64_t>(rs2_num, midx ) >> mpos) & 0x1) == 1;
  bool do_mask = (P.VU.elt<uint64_t>(0, midx) >> mpos) & 0x1;

  if (insn.v_vm() == 1 || (insn.v_vm() == 0 && do_mask)) {
    auto &vd = P.VU.elt<uint64_t>(rd_num, midx, true);
    uint64_t res = 0;
    if (!has_one && !vs2_lsb) {
      res = 1;
    } else if (!has_one && vs2_lsb) {
      has_one = true;
      res = 1;
    }
    vd = (vd & ~mmask) | ((res << mpos) & mmask);
  }
}

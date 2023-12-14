// vmpopc rd, vs2, vm
require(P.VU.vsew >= e8 && P.VU.vsew <= e64);
require_vector(true);
reg_t vl = P.VU.vl->read(p);
reg_t rs2_num = insn.rs2();
require(P.VU.vstart->read(p) == 0);
reg_t popcount = 0;
for (reg_t i=P.VU.vstart->read(p); i<vl; ++i) {
  const int midx = i / 32;
  const int mpos = i % 32;

  bool vs2_lsb = ((P.VU.elt<uint32_t>(rs2_num, midx ) >> mpos) & 0x1) == 1;
  if (insn.v_vm() == 1) {
    popcount += vs2_lsb;
  } else {
    bool do_mask = (P.VU.elt<uint32_t>(0, midx) >> mpos) & 0x1;
    popcount += (vs2_lsb && do_mask);
  }
}
P.VU.vstart->write(0, p);
WRITE_RD(popcount);

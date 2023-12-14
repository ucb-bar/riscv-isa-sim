// vmv_s_x: vd[0] = rs1
require_vector(true);
require(insn.v_vm() == 1);
require(P.VU.vsew >= e8 && P.VU.vsew <= e64);
reg_t vl = P.VU.vl->read(p);

if (vl > 0 && P.VU.vstart->read(p) < vl) {
  reg_t rd_num = insn.rd();
  reg_t sew = P.VU.vsew;

  switch (sew) {
  case e8:
    P.VU.elt<uint8_t>(rd_num, 0, true) = RS1;
    break;
  case e16:
    P.VU.elt<uint16_t>(rd_num, 0, true) = RS1;
    break;
  case e32:
    P.VU.elt<uint32_t>(rd_num, 0, true) = RS1;
    break;
  default:
    P.VU.elt<uint64_t>(rd_num, 0, true) = RS1;
    break;
  }

  vl = 0;
}

P.VU.vstart->write(0, p);

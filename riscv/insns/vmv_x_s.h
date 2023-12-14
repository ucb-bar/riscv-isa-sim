// vmv_x_s: rd = vs2[0]
require_vector(true);
require(insn.v_vm() == 1);
reg_t sew = P.VU.vsew;
reg_t rs2_num = insn.rs2();
reg_t res;

switch (sew) {
case e8:
  res = P.VU.elt<int8_t>(rs2_num, 0);
  break;
case e16:
  res = P.VU.elt<int16_t>(rs2_num, 0);
  break;
case e32:
  res = P.VU.elt<int32_t>(rs2_num, 0);
  break;
case e64:
  res = P.VU.elt<uint64_t>(rs2_num, 0);
  break;
default:
  abort();
}

WRITE_RD(sext_xlen(res));

P.VU.vstart->write(0, p);

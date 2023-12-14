// vfmv_f_s: rd = vs2[0] (rs1=0)
require_vector(true);
require_fp;
require((P.VU.vsew == e16 && p->extension_enabled(EXT_ZVFH)) ||
        (P.VU.vsew == e32 && p->extension_enabled('F')) ||
        (P.VU.vsew == e64 && p->extension_enabled('D')));
require(STATE.frm->read(p) < 0x5);

reg_t rs2_num = insn.rs2();
uint64_t vs2_0 = 0;
const reg_t sew = P.VU.vsew;
switch (sew) {
  case e16:
    vs2_0 = P.VU.elt<uint16_t>(rs2_num, 0);
    break;
  case e32:
    vs2_0 = P.VU.elt<uint32_t>(rs2_num, 0);
    break;
  case e64:
    vs2_0 = P.VU.elt<uint64_t>(rs2_num, 0);
    break;
  default:
    require(0);
    break;
}

// nan_extened
if (FLEN > sew) {
  vs2_0 = vs2_0 | (UINT64_MAX << sew);
}

if (FLEN == 64) {
  WRITE_FRD(f64(vs2_0));
} else {
  WRITE_FRD(f32(vs2_0));
}

P.VU.vstart->write(0, p);

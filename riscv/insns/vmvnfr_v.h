// vmv<nf>r.v vd, vs2
require_vector(true);
const reg_t vd = insn.rd();
const reg_t vs2 = insn.rs2();
const reg_t len = insn.rs1() + 1;
require_align(vd, len);
require_align(vs2, len);
const reg_t size = len * P.VU.vlenb;
const reg_t start = P.VU.vstart->read(p) * (P.VU.vsew >> 3);

//register needs one-by-one copy to keep commitlog correct
if (vd != vs2 && start < size) {
  reg_t i = start / P.VU.vlenb;
  reg_t off = start % P.VU.vlenb;
  if (off) {
    memcpy(&P.VU.elt<uint8_t>(vd + i, off, true),
           &P.VU.elt<uint8_t>(vs2 + i, off), P.VU.vlenb - off);
    i++;
  }

  for (; i < len; ++i) {
    memcpy(&P.VU.elt<uint8_t>(vd + i, 0, true),
           &P.VU.elt<uint8_t>(vs2 + i, 0), P.VU.vlenb);
  }
}

P.VU.vstart->write(0, p);

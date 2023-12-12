//vslideup.vx vd, vs2, rs1
VI_CHECK_SLIDE(true);

const reg_t offset = RS1;
VI_LOOP_BASE
if (P.VU.vstart->read(p) < offset && i < offset)
  continue;

switch (sew) {
case e8: {
  VI_XI_SLIDEUP_PARAMS(e8, offset);
  vd = vs2;
}
break;
case e16: {
  VI_XI_SLIDEUP_PARAMS(e16, offset);
  vd = vs2;
}
break;
case e32: {
  VI_XI_SLIDEUP_PARAMS(e32, offset);
  vd = vs2;
}
break;
default: {
  VI_XI_SLIDEUP_PARAMS(e64, offset);
  vd = vs2;
}
break;
}
VI_LOOP_END

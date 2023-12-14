require_extension(EXT_ZCMT);
STATE.jvt->verify_permissions(insn, false, p);
reg_t jvt = STATE.jvt->read(p);
uint8_t mode = get_field(jvt, JVT_MODE);
reg_t base = jvt & JVT_BASE;
reg_t index = insn.rvc_index();
reg_t target;
switch (mode) {
case 0:  // jump table mode
  if (xlen == 32)
    target = MMU.fetch_jump_table<int32_t>(base + (index << 2));
  else  // xlen = 64
    target = MMU.fetch_jump_table<int64_t>(base + (index << 3));

  if (index >= 32)  // cm.jalt
    WRITE_REG(1, npc);

  set_pc(target & ~reg_t(1));
  break;
default:
  require(0);
  break;
}

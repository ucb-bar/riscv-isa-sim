require_extension('H');
require_novirt();
require_privilege(get_field(STATE.hstatus->read(p), HSTATUS_HU) ? PRV_U : PRV_S);
WRITE_RD(sext_xlen(MMU.guest_load_x<uint32_t>(RS1)));

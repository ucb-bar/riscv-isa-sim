require_extension('H');
require_novirt();
require_privilege(get_field(STATE.hstatus->read(p), HSTATUS_HU) ? PRV_U : PRV_S);
WRITE_RD(MMU.guest_load<int32_t>(RS1));

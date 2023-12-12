require_extension('H');
require_novirt();
require_privilege(get_field(STATE.hstatus->read(p), HSTATUS_HU) ? PRV_U : PRV_S);
MMU.guest_store<uint8_t>(RS1, RS2);

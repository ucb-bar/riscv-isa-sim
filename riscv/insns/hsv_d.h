require_extension('H');
require_rv64;
require_novirt();
require_privilege(get_field(STATE.hstatus->read(p), HSTATUS_HU) ? PRV_U : PRV_S);
MMU.guest_store<uint64_t>(RS1, RS2);

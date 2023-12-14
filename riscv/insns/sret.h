require_extension('S');
reg_t prev_hstatus = STATE.hstatus->read(p);
if (STATE.v) {
  if (STATE.prv == PRV_U || get_field(prev_hstatus, HSTATUS_VTSR))
    require_novirt();
} else {
  require_privilege(get_field(STATE.mstatus->read(p), MSTATUS_TSR) ? PRV_M : PRV_S);
}
reg_t next_pc = p->get_state()->sepc->read(p);
set_pc_and_serialize(next_pc);
reg_t s = STATE.sstatus->read(p);
reg_t prev_prv = get_field(s, MSTATUS_SPP);
s = set_field(s, MSTATUS_SIE, get_field(s, MSTATUS_SPIE));
s = set_field(s, MSTATUS_SPIE, 1);
s = set_field(s, MSTATUS_SPP, PRV_U);
STATE.sstatus->write(s, p);
bool prev_virt = STATE.v;
if (!STATE.v) {
  if (p->extension_enabled('H')) {
    prev_virt = get_field(prev_hstatus, HSTATUS_SPV);
    reg_t new_hstatus = set_field(prev_hstatus, HSTATUS_SPV, 0);
    STATE.hstatus->write(new_hstatus, p);
  }

  STATE.mstatus->write(set_field(STATE.mstatus->read(p), MSTATUS_MPRV, 0), p);
}
p->set_privilege(prev_prv, prev_virt);

require_privilege(PRV_M);
set_pc_and_serialize(p->get_state()->mepc->read(p));
reg_t s = STATE.mstatus->read(p);
reg_t prev_prv = get_field(s, MSTATUS_MPP);
reg_t prev_virt = get_field(s, MSTATUS_MPV);
if (prev_prv != PRV_M)
  s = set_field(s, MSTATUS_MPRV, 0);
s = set_field(s, MSTATUS_MIE, get_field(s, MSTATUS_MPIE));
s = set_field(s, MSTATUS_MPIE, 1);
s = set_field(s, MSTATUS_MPP, p->extension_enabled('U') ? PRV_U : PRV_M);
s = set_field(s, MSTATUS_MPV, 0);
p->put_csr(CSR_MSTATUS, s);
p->set_privilege(prev_prv, prev_virt);

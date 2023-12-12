require(STATE.debug_mode);
set_pc_and_serialize(STATE.dpc->read(p));
p->set_privilege(STATE.dcsr->prv, STATE.dcsr->v);
if (STATE.prv < PRV_M)
  STATE.mstatus->write(STATE.mstatus->read(p) & ~MSTATUS_MPRV, p);

/* We're not in Debug Mode anymore. */
STATE.debug_mode = false;

if (STATE.dcsr->step)
  STATE.single_step = STATE.STEP_STEPPING;

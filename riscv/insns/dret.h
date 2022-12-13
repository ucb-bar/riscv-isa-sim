//require(STATE.debug_mode);
set_pc_and_serialize(STATE.dpc->read());
p->set_privilege(STATE.dcsr->prv);
if (STATE.prv < PRV_M)
  STATE.mstatus->write(STATE.mstatus->read() & ~MSTATUS_MPRV);

/* We're not in Debug Mode anymore. */
STATE.debug_mode = false;

if (STATE.dcsr->step)
  STATE.single_step = STATE.STEP_STEPPING;

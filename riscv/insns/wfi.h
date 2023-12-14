if (STATE.v && STATE.prv == PRV_U) {
  require_novirt();
} else if (get_field(STATE.mstatus->read(p), MSTATUS_TW)) {
  require_privilege(PRV_M);
} else if (STATE.v) { // VS-mode
  if (get_field(STATE.hstatus->read(p), HSTATUS_VTW))
    require_novirt();
} else if (p->extension_enabled('S')) {
  // When S-mode is implemented, then executing WFI in
  // U-mode causes an illegal instruction exception.
  require_privilege(PRV_S);
}
wfi();

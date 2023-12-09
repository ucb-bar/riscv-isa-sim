require_rv32;
require_extension('D');
require_extension(EXT_ZFA);
require_fp;
ui64_f64 ui;
ui.f = FRS1_D;
WRITE_RD(sext32(ui.ui >> 32));

int csr = validate_csr(insn.csr(), true);
reg_t old = p->get_csr(csr, insn, true);
p->put_csr(csr, RS1);
WRITE_RD(sext_xlen(old));
serialize();

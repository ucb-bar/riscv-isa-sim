#include "dtm.h"
#include "debug_defines.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <pthread.h>
#include <stdexcept>

#define RV_X(x, s, n) \
  (((x) >> (s)) & ((1 << (n)) - 1))
#define ENCODE_ITYPE_IMM(x) \
  (RV_X(x, 0, 12) << 20)
#define ENCODE_STYPE_IMM(x) \
  ((RV_X(x, 0, 5) << 7) | (RV_X(x, 5, 7) << 25))
#define ENCODE_SBTYPE_IMM(x) \
  ((RV_X(x, 1, 4) << 8) | (RV_X(x, 5, 6) << 25) | (RV_X(x, 11, 1) << 7) | (RV_X(x, 12, 1) << 31))
#define ENCODE_UTYPE_IMM(x) \
  (RV_X(x, 12, 20) << 12)
#define ENCODE_UJTYPE_IMM(x) \
  ((RV_X(x, 1, 10) << 21) | (RV_X(x, 11, 1) << 20) | (RV_X(x, 12, 8) << 12) | (RV_X(x, 20, 1) << 31))

#define LOAD(xlen, dst, base, imm) \
  (((xlen) == 64 ? 0x00003003 : 0x00002003) \
   | ((dst) << 7) | ((base) << 15) | (uint32_t)ENCODE_ITYPE_IMM(imm))
#define STORE(xlen, src, base, imm) \
  (((xlen) == 64 ? 0x00003023 : 0x00002023) \
   | ((src) << 20) | ((base) << 15) | (uint32_t)ENCODE_STYPE_IMM(imm))
#define JUMP(there, here) (0x6f | (uint32_t)ENCODE_UJTYPE_IMM((there) - (here)))
#define BNE(r1, r2, there, here) (0x1063 | ((r1) << 15) | ((r2) << 20) | (uint32_t)ENCODE_SBTYPE_IMM((there) - (here)))
#define ADDI(dst, src, imm) (0x13 | ((dst) << 7) | ((src) << 15) | (uint32_t)ENCODE_ITYPE_IMM(imm))
#define SRL(dst, src, sh) (0x5033 | ((dst) << 7) | ((src) << 15) | ((sh) << 20))
#define FENCE_I 0x100f
#define EBREAK  0x00100073
#define X0 0
#define S0 8
#define S1 9

#define AC_AR_REGNO(x) ((0x1000 | x) << AC_ACCESS_REGISTER_REGNO_OFFSET)
#define AC_AR_SIZE(x)  (((x == 128)? 4 : (x == 64 ? 3 : 2)) << AC_ACCESS_REGISTER_SIZE_OFFSET)

#define WRITE 1
#define SET 2
#define CLEAR 3
#define CSRRx(type, dst, csr, src) (0x73 | ((type) << 12) | ((dst) << 7) | ((src) << 15) | (uint32_t)((csr) << 20))

#define RUN_AC_OR_DIE(a, b, c, d, e) { \
    uint32_t cmderr = run_abstract_command(a, b, c, d, e);      \
    if (cmderr) {                                               \
      die(cmderr);                                              \
    }                                                           \
  }

uint32_t dtm_t::do_command(dtm_t::req r)
{
  req_buf = r;
  target->switch_to();
  assert(resp_buf.resp == 0);
  return resp_buf.data;
}

uint32_t dtm_t::read(uint32_t addr)
{
  return do_command((req){addr, 1, 0});
}

uint32_t dtm_t::write(uint32_t addr, uint32_t data)
{
  return do_command((req){addr, 2, data});
}

void dtm_t::nop()
{
  do_command((req){0, 0, 0});
}

void dtm_t::select_hart(int hartsel) {
  int dmcontrol = read(DMI_DMCONTROL);
  write (DMI_DMCONTROL, set_field(dmcontrol, DMI_DMCONTROL_HARTSEL, hartsel));
  current_hart = hartsel;
}

int dtm_t::enumerate_harts() {
  int max_hart = (1 << DMI_DMCONTROL_HARTSEL_LENGTH) - 1;
  write(DMI_DMCONTROL, set_field(read(DMI_DMCONTROL), DMI_DMCONTROL_HARTSEL, max_hart));
  read(DMI_DMSTATUS);
  max_hart = get_field(read(DMI_DMCONTROL), DMI_DMCONTROL_HARTSEL);

  int hartsel;
  for (hartsel = 0; hartsel <= max_hart; hartsel++) {
    select_hart(hartsel);
    int dmstatus = read(DMI_DMSTATUS);
    if (get_field(dmstatus, DMI_DMSTATUS_ANYNONEXISTENT))
      break;
  }
  return hartsel;
}

void dtm_t::halt(int hartsel)
{
  if (running) {
    write(DMI_DMCONTROL, DMI_DMCONTROL_DMACTIVE);
    // Read dmstatus to avoid back-to-back writes to dmcontrol.
    read(DMI_DMSTATUS);
  }

  int dmcontrol = DMI_DMCONTROL_HALTREQ | DMI_DMCONTROL_DMACTIVE;
  dmcontrol = set_field(dmcontrol, DMI_DMCONTROL_HARTSEL, hartsel);
  write(DMI_DMCONTROL, dmcontrol);
  int dmstatus;
  do {
    dmstatus = read(DMI_DMSTATUS);
  } while(get_field(dmstatus, DMI_DMSTATUS_ALLHALTED) == 0);
  dmcontrol &= ~DMI_DMCONTROL_HALTREQ;
  write(DMI_DMCONTROL, dmcontrol);
  // Read dmstatus to avoid back-to-back writes to dmcontrol.
  read(DMI_DMSTATUS);
  current_hart = hartsel;
}

void dtm_t::resume(int hartsel)
{
  int dmcontrol = DMI_DMCONTROL_RESUMEREQ | DMI_DMCONTROL_DMACTIVE;
  dmcontrol = set_field(dmcontrol, DMI_DMCONTROL_HARTSEL, hartsel);
  write(DMI_DMCONTROL, dmcontrol);
  int dmstatus;
  do {
    dmstatus = read(DMI_DMSTATUS);
  } while (get_field(dmstatus, DMI_DMSTATUS_ALLRESUMEACK) == 0);
  dmcontrol &= ~DMI_DMCONTROL_RESUMEREQ;
  write(DMI_DMCONTROL, dmcontrol);
  // Read dmstatus to avoid back-to-back writes to dmcontrol.
  read(DMI_DMSTATUS);
  current_hart = hartsel;

  if (running) {
    write(DMI_DMCONTROL, DMI_DMCONTROL_DMACTIVE);
    // Read dmstatus to avoid back-to-back writes to dmcontrol.
    read(DMI_DMSTATUS);
  }
}

uint64_t dtm_t::save_reg(unsigned regno)
{
  uint32_t data[xlen/(8*4)];
  uint32_t command = AC_ACCESS_REGISTER_TRANSFER | AC_AR_SIZE(xlen) | AC_AR_REGNO(regno);
  RUN_AC_OR_DIE(command, 0, 0, data, xlen / (8*4));

  uint64_t result = data[0];
  if (xlen > 32) {
    result |= ((uint64_t)data[1]) << 32;
  }
  return result;
}

void dtm_t::restore_reg(unsigned regno, uint64_t val)
{
  uint32_t data[xlen/(8*4)];
  data[0] = (uint32_t) val;
  if (xlen > 32) {
    data[1] = (uint32_t) (val >> 32);
  }

  uint32_t command = AC_ACCESS_REGISTER_TRANSFER |
    AC_ACCESS_REGISTER_WRITE |
    AC_AR_SIZE(xlen) |
    AC_AR_REGNO(regno);
  
  RUN_AC_OR_DIE(command, 0, 0, data, xlen / (8*4));

}

uint32_t dtm_t::run_abstract_command(uint32_t command,
                                     const uint32_t program[], size_t program_n,
                                     uint32_t data[], size_t data_n)
{ 
  assert(program_n <= ram_words);
  assert(data_n    <= data_words);
  
  for (size_t i = 0; i < program_n; i++) {
    write(DMI_PROGBUF0 + i, program[i]);
  }

  if (get_field(command, AC_ACCESS_REGISTER_WRITE) &&
      get_field(command, AC_ACCESS_REGISTER_TRANSFER)) {
    for (size_t i = 0; i < data_n; i++) {
      write(DMI_DATA0 + i, data[i]);
    }
  }
  
  write(DMI_COMMAND, command);
  
  // Wait for not busy and then check for error.
  uint32_t abstractcs;
  do {
    abstractcs = read(DMI_ABSTRACTCS);
  } while (abstractcs & DMI_ABSTRACTCS_BUSY);

  if ((get_field(command, AC_ACCESS_REGISTER_WRITE) == 0) &&
      get_field(command, AC_ACCESS_REGISTER_TRANSFER)) {
    for (size_t i = 0; i < data_n; i++){
      data[i] = read(DMI_DATA0 + i);
    }
  }
  
  return get_field(abstractcs, DMI_ABSTRACTCS_CMDERR);

}

size_t dtm_t::chunk_align()
{
  return xlen / 8;
}

void dtm_t::read_chunk(uint64_t taddr, size_t len, void* dst)
{
  uint32_t prog[ram_words];
  uint32_t data[data_words];

  uint8_t * curr = (uint8_t*) dst;

  halt(current_hart);

  uint64_t s0 = save_reg(S0);
  uint64_t s1 = save_reg(S1);
  
  prog[0] = LOAD(xlen, S1, S0, 0);
  prog[1] = ADDI(S0, S0, xlen/8);
  prog[2] = EBREAK;

  data[0] = (uint32_t) taddr;
  if (xlen > 32) {
    data[1] = (uint32_t) (taddr >> 32);
  }

  // Write s0 with the address, then execute program buffer.
  // This will get S1 with the data and increment s0.
  uint32_t command = AC_ACCESS_REGISTER_TRANSFER |
    AC_ACCESS_REGISTER_WRITE |
    AC_ACCESS_REGISTER_POSTEXEC |
    AC_AR_SIZE(xlen) | 
    AC_AR_REGNO(S0);

  RUN_AC_OR_DIE(command, prog, 3, data, xlen/(4*8));

  // TODO: could use autoexec here.
  for (size_t i = 0; i < (len * 8 / xlen); i++){
    command = AC_ACCESS_REGISTER_TRANSFER |
      AC_AR_SIZE(xlen) |
      AC_AR_REGNO(S1);
    if ((i + 1) < (len * 8 / xlen)) {
      command |= AC_ACCESS_REGISTER_POSTEXEC;
    }
    
    RUN_AC_OR_DIE(command, 0, 0, data, xlen/(4*8));

    memcpy(curr, data, xlen/8);
    curr += xlen/8;
  }

  restore_reg(S0, s0);
  restore_reg(S1, s1);

  resume(current_hart); 

}

void dtm_t::write_chunk(uint64_t taddr, size_t len, const void* src)
{  
  uint32_t prog[ram_words];
  uint32_t data[data_words];

  const uint8_t * curr = (const uint8_t*) src;

  halt(current_hart);

  uint64_t s0 = save_reg(S0);
  uint64_t s1 = save_reg(S1);
  
  prog[0] = STORE(xlen, S1, S0, 0);
  prog[1] = ADDI(S0, S0, xlen/8);
  prog[2] = EBREAK;
  
  data[0] = (uint32_t) taddr;
  if (xlen > 32) {
    data[1] = (uint32_t) (taddr >> 32);
  }

  // Write the program (not used yet).
  // Write s0 with the address. 
  uint32_t command = AC_ACCESS_REGISTER_TRANSFER |
    AC_ACCESS_REGISTER_WRITE |
    AC_AR_SIZE(xlen) |
    AC_AR_REGNO(S0);
  
  RUN_AC_OR_DIE(command, prog, 3, data, xlen/(4*8));

  // Use Autoexec for more than one word of transfer.
  // Write S1 with data, then execution stores S1 to
  // 0(S0) and increments S0.
  // Each time we write XLEN bits.
  memcpy(data, curr, xlen/8);
  curr += xlen/8;
  
  command = AC_ACCESS_REGISTER_TRANSFER |
    AC_ACCESS_REGISTER_POSTEXEC |
    AC_ACCESS_REGISTER_WRITE | 
    AC_AR_SIZE(xlen) |
    AC_AR_REGNO(S1);

  RUN_AC_OR_DIE(command, 0, 0, data, xlen/(4*8));

  uint32_t abstractcs;
  for (size_t i = 1; i < (len * 8 / xlen); i++){
    if (i == 1) {
      write(DMI_ABSTRACTAUTO, 1 << DMI_ABSTRACTAUTO_AUTOEXECDATA_OFFSET);
    }
    memcpy(data, curr, xlen/8);
    curr += xlen/8;
    if (xlen == 64) {
      write(DMI_DATA0 + 1, data[1]);
    }
    write(DMI_DATA0, data[0]); //Triggers a command w/ autoexec.
    
    do {
      abstractcs = read(DMI_ABSTRACTCS);
    } while (abstractcs & DMI_ABSTRACTCS_BUSY);
    if ( get_field(abstractcs, DMI_ABSTRACTCS_CMDERR)) {
      die(get_field(abstractcs, DMI_ABSTRACTCS_CMDERR));
    }
  }
  if ((len * 8 / xlen) > 1) {
    write(DMI_ABSTRACTAUTO, 0);
  }
  
  restore_reg(S0, s0);
  restore_reg(S1, s1);
  resume(current_hart);
}

void dtm_t::die(uint32_t cmderr)
{
  const char * codes[] = {
    "OK",
    "BUSY",
    "NOT_SUPPORTED",
    "EXCEPTION",
    "HALT/RESUME"
  };
  const char * msg;
  if (cmderr < (sizeof(codes) / sizeof(*codes))){
    msg = codes[cmderr];
  } else {
    msg = "OTHER";
  }
  //throw std::runtime_error("Debug Abstract Command Error #" + std::to_string(cmderr) + "(" +  msg + ")");
  printf("ERROR: %s:%d, Debug Abstract Command Error #%d (%s)", __FILE__, __LINE__, cmderr, msg);
  printf("ERROR: %s:%d, Should die, but allowing simulation to continue and fail.", __FILE__, __LINE__);
  write(DMI_ABSTRACTCS, DMI_ABSTRACTCS_CMDERR);

}

void dtm_t::clear_chunk(uint64_t taddr, size_t len)
{
  uint32_t prog[ram_words];
  uint32_t data[data_words];
  
  halt(current_hart);
  uint64_t s0 = save_reg(S0);
  uint64_t s1 = save_reg(S1);

  uint32_t command;

  // S0 = Addr
  data[0] = (uint32_t) taddr;
  data[1] = (uint32_t) (taddr >> 32);
  command = AC_ACCESS_REGISTER_TRANSFER |
    AC_ACCESS_REGISTER_WRITE |
    AC_AR_SIZE(xlen) |
    AC_AR_REGNO(S0);
  RUN_AC_OR_DIE(command, 0, 0, data, xlen/(4*8));

  // S1 = Addr + len, loop until S0 = S1
  prog[0] = STORE(xlen, X0, S0, 0);
  prog[1] = ADDI(S0, S0, xlen/8);
  prog[2] = BNE(S0, S1, 0*4, 2*4);
  prog[3] = EBREAK;

  data[0] = (uint32_t) (taddr + len);
  data[1] = (uint32_t) ((taddr + len) >> 32);
  command = AC_ACCESS_REGISTER_TRANSFER |
    AC_ACCESS_REGISTER_WRITE |
    AC_AR_SIZE(xlen) |
    AC_AR_REGNO(S1)  |
    AC_ACCESS_REGISTER_POSTEXEC;
  RUN_AC_OR_DIE(command, prog, 4, data, xlen/(4*8));

  restore_reg(S0, s0);
  restore_reg(S1, s1);

  resume(current_hart);
}

uint64_t dtm_t::write_csr(unsigned which, uint64_t data)
{
  return modify_csr(which, data, WRITE);
}

uint64_t dtm_t::set_csr(unsigned which, uint64_t data)
{
  return modify_csr(which, data, SET);
}

uint64_t dtm_t::clear_csr(unsigned which, uint64_t data)
{
  return modify_csr(which, data, CLEAR);
}

uint64_t dtm_t::read_csr(unsigned which)
{
  return set_csr(which, 0);
}

uint64_t dtm_t::modify_csr(unsigned which, uint64_t data, uint32_t type)
{
  halt(current_hart);

  // This code just uses DSCRATCH to save S0
  // and data_base to do the transfer so we don't
  // need to run more commands to save and restore
  // S0.
  uint32_t prog[] = {
    CSRRx(WRITE, S0, CSR_DSCRATCH0, S0),
    LOAD(xlen, S0, X0, data_base),
    CSRRx(type, S0, which, S0),
    STORE(xlen, S0, X0, data_base),
    CSRRx(WRITE, S0, CSR_DSCRATCH0, S0),
    EBREAK
  };

  //TODO: Use transfer = 0. For now both HW and OpenOCD
  // ignore transfer bit, so use "store to X0" NOOP.
  // We sort of need this anyway because run_abstract_command
  // needs the DATA to be written so may as well use the WRITE flag.
  
  uint32_t adata[] = {(uint32_t) data,
                      (uint32_t) (data >> 32)};
  
  uint32_t command = AC_ACCESS_REGISTER_POSTEXEC |
    AC_ACCESS_REGISTER_TRANSFER |
    AC_ACCESS_REGISTER_WRITE |
    AC_AR_SIZE(xlen) |
    AC_AR_REGNO(X0);
  
  RUN_AC_OR_DIE(command, prog, sizeof(prog) / sizeof(*prog), adata, xlen/(4*8));
  
  uint64_t res = read(DMI_DATA0);//adata[0];
  if (xlen == 64)
    res |= read(DMI_DATA0 + 1);//((uint64_t) adata[1]) << 32;
  
  resume(current_hart);
  return res;  
}

size_t dtm_t::chunk_max_size()
{
  // Arbitrary choice. 4k Page size seems reasonable.
  return 4096;
}

uint32_t dtm_t::get_xlen()
{
  // Attempt to read S0 to find out what size it is.
  // You could also attempt to run code, but you need to save registers
  // to do that anyway. If what you really want to do is figure out
  // the size of S0 so you can save it later, then do that.
  uint32_t command = AC_ACCESS_REGISTER_TRANSFER | AC_AR_REGNO(S0);
  uint32_t cmderr;
  
  const uint32_t prog[] = {};
  uint32_t data[] = {};

  cmderr = run_abstract_command(command | AC_AR_SIZE(128), prog, 0, data, 0);
  if (cmderr == 0){
    throw std::runtime_error("FESVR DTM Does not support 128-bit");
    abort();
    return 128;
  }
  write(DMI_ABSTRACTCS, DMI_ABSTRACTCS_CMDERR);

  cmderr = run_abstract_command(command | AC_AR_SIZE(64), prog, 0, data, 0);
  if (cmderr == 0){
    return 64;
  }
  write(DMI_ABSTRACTCS, DMI_ABSTRACTCS_CMDERR);

  cmderr = run_abstract_command(command | AC_AR_SIZE(32), prog, 0, data, 0);
  if (cmderr == 0){
    return 32;
  }
  
  throw std::runtime_error("FESVR DTM can't determine XLEN. Aborting");
}

void dtm_t::fence_i()
{
  halt(current_hart);

  const uint32_t prog[] = {
    FENCE_I,
    EBREAK
  };

  //TODO: Use the transfer = 0.
  uint32_t command = AC_ACCESS_REGISTER_POSTEXEC |
    AC_ACCESS_REGISTER_TRANSFER |
    AC_ACCESS_REGISTER_WRITE |
    AC_AR_SIZE(xlen) |
    AC_AR_REGNO(X0);

  RUN_AC_OR_DIE(command, prog, sizeof(prog)/sizeof(*prog), 0, 0);
  
  resume(current_hart);

}

void host_thread_main(void* arg)
{
  ((dtm_t*)arg)->producer_thread();
}

void dtm_t::reset()
{
  for (int hartsel = 0; hartsel < num_harts; hartsel ++ ){
    select_hart(hartsel);
    // this command also does a halt and resume
    fence_i();
    // after this command, the hart will run from _start.
    write_csr(0x7b1, get_entry_point());
  }
  // In theory any hart can handle the memory accesses,
  // this will enforce that hart 0 handles them.
  select_hart(0);
  read(DMI_DMSTATUS);
} 

void dtm_t::idle()
{
  for (int idle_cycles = 0; idle_cycles < max_idle_cycles; idle_cycles++)
    nop();
}

void dtm_t::producer_thread()
{
  // Learn about the Debug Module and assert things we
  // depend on in this code.

  // Enable the debugger.
  write(DMI_DMCONTROL, DMI_DMCONTROL_DMACTIVE);
  // Poll until the debugger agrees it's enabled.
  while ((read(DMI_DMCONTROL) & DMI_DMCONTROL_DMACTIVE) == 0) ;
    
  // These are checked every time we run an abstract command.
  uint32_t abstractcs = read(DMI_ABSTRACTCS);
  ram_words = get_field(abstractcs, DMI_ABSTRACTCS_PROGSIZE);
  data_words = get_field(abstractcs, DMI_ABSTRACTCS_DATACOUNT);

  // These things are only needed for the 'modify_csr' function.
  // That could be re-written to not use these at some performance
  // overhead.
  uint32_t hartinfo = read(DMI_HARTINFO);
  assert(get_field(hartinfo, DMI_HARTINFO_NSCRATCH) > 0);
  assert(get_field(hartinfo, DMI_HARTINFO_DATAACCESS));

  data_base = get_field(hartinfo, DMI_HARTINFO_DATAADDR);
  
  num_harts = enumerate_harts();
  halt(0);
  // Note: We don't support systems with heterogeneous XLEN.
  // It's possible to do this at the cost of extra cycles.
  xlen = get_xlen();
  resume(0);

  running = true;

  htif_t::run();

  while (true)
    nop();
}

void dtm_t::start_host_thread()
{
  req_wait = false;
  resp_wait = false;

  target = context_t::current();
  host.init(host_thread_main, this);
  host.switch_to();
}

dtm_t::dtm_t(int argc, char** argv)
  : htif_t(argc, argv), running(false)
{
  start_host_thread();
}

dtm_t::~dtm_t()
{
}

void dtm_t::tick(
  bool      req_ready,
  bool      resp_valid,
  resp      resp_bits)
{
  if (!resp_wait) {
    if (!req_wait) {
      req_wait = true;
    } else if (req_ready) {
      req_wait = false;
      resp_wait = true;
    }
  }

  if (resp_valid) {
    assert(resp_wait);
    resp_wait = false;

    resp_buf = resp_bits;
    // update the target with the current context
    target = context_t::current();
    host.switch_to();
  }
}

void dtm_t::return_resp(resp resp_bits){
  resp_buf = resp_bits;
  target = context_t::current();
  host.switch_to();
}

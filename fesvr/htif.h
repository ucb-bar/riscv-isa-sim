// See LICENSE for license details.

#ifndef __HTIF_H
#define __HTIF_H

#include "memif.h"
#include "syscall.h"
#include "device.h"
#include "byteorder.h"
#include <string.h>
#include <map>
#include <vector>
#include <assert.h>
#include <iostream>



static volatile bool signal_exit = false;

static void bad_address(const std::string& situation, reg_t addr)
{
  std::cerr << "Access exception occurred while " << situation << ":\n";
  std::cerr << "Memory address 0x" << std::hex << addr << " is invalid\n";
  exit(-1);
}

class htif_t : public chunked_memif_t
{
 public:
  htif_t();
  htif_t(int argc, char** argv);
  htif_t(const std::vector<std::string>& args);
  virtual ~htif_t();

  virtual void start();
  virtual void stop();

  int run();
  bool done();
  int exit_code();
  void set_expected_xlen(unsigned int m) { expected_xlen = m; }
  virtual memif_t& memif() { return mem; }

  template<typename T> inline T from_target(target_endian<T> n) const
  {
    endianness_t endianness = get_target_endianness();
    assert(endianness == endianness_little || endianness == endianness_big);

    return endianness == endianness_big? n.from_be() : n.from_le();
  }

  template<typename T> inline target_endian<T> to_target(T n) const
  {
    endianness_t endianness = get_target_endianness();
    assert(endianness == endianness_little || endianness == endianness_big);

    return endianness == endianness_big? target_endian<T>::to_be(n) : target_endian<T>::to_le(n);
  }

  addr_t get_tohost_addr() { return tohost_addr; }
  addr_t get_fromhost_addr() { return fromhost_addr; }

 protected:
  virtual void reset() = 0;

  virtual void read_chunk(addr_t taddr, size_t len, void* dst) = 0;
  virtual void write_chunk(addr_t taddr, size_t len, const void* src) = 0;
  virtual void clear_chunk(addr_t taddr, size_t len);

  virtual size_t chunk_align() = 0;
  virtual size_t chunk_max_size() = 0;

  virtual std::map<std::string, uint64_t> load_payload(const std::string& payload, reg_t* entry);
  virtual void load_program();
  virtual void idle() {}

  const std::vector<std::string>& host_args() { return hargs; }
  const std::vector<std::string>& target_args() { return targs; }

  reg_t get_entry_point() { return entry; }

  // indicates that the initial program load can skip writing this address
  // range to memory, because it has already been loaded through a sideband
  virtual bool is_address_preloaded(addr_t, size_t) { return false; }

  // Given an address, return symbol from addr2symbol map
  const char* get_symbol(uint64_t addr);

 protected:
  void parse_arguments(int argc, char ** argv);
  void register_devices();
  void usage(const char * program_name);
  unsigned int expected_xlen = 0;
  memif_t mem;
  reg_t entry;
  bool writezeros;
  std::vector<std::string> hargs;
  std::vector<std::string> targs;
  std::string sig_file;
  unsigned int line_size;
  addr_t sig_addr; // torture
  addr_t sig_len; // torture
  addr_t tohost_addr;
  addr_t fromhost_addr;
  int exitcode;
  bool stopped;

  device_list_t device_list;
  syscall_t syscall_proxy;
  bcd_t bcd;
  std::vector<device_t*> dynamic_devices;
  std::vector<std::string> payloads;

  std::vector<std::string> symbol_elfs;
  std::map<uint64_t, std::string> addr2symbol;

  friend class memif_t;
  friend class syscall_t;
};

/* Alignment guide for emulator.cc options:
  -x, --long-option        Description with max 80 characters --------------->\n\
       +plus-arg-equivalent\n\
 */
#define HTIF_USAGE_OPTIONS \
"HOST OPTIONS\n\
  -h, --help               Display this help and exit\n\
  +h,  +help\n\
       +permissive         The host will ignore any unparsed options up until\n\
                             +permissive-off (Only needed for VCS)\n\
       +permissive-off     Stop ignoring options. This is mandatory if using\n\
                             +permissive (Only needed for VCS)\n\
      --rfb=DISPLAY        Add new remote frame buffer on display DISPLAY\n\
       +rfb=DISPLAY          to be accessible on 5900 + DISPLAY (default = 0)\n\
      --signature=FILE     Write torture test signature to FILE\n\
       +signature=FILE\n\
      --signature-granularity=VAL           Size of each line in signature.\n\
       +signature-granularity=VAL\n\
      --chroot=PATH        Use PATH as location of syscall-servicing binaries\n\
       +chroot=PATH\n\
      --payload=PATH       Load PATH memory as an additional ELF payload\n\
       +payload=PATH\n\
      --symbol-elf=PATH    Populate the symbol table with the ELF file at PATH\n\
       +symbol-elf=PATH\n\
\n\
HOST OPTIONS (currently unsupported)\n\
      --disk=DISK          Add DISK device. Use a ramdisk since this isn't\n\
       +disk=DISK            supported\n\
\n\
TARGET (RISC-V BINARY) OPTIONS\n\
  These are the options passed to the program executing on the emulated RISC-V\n\
  microprocessor.\n"

#define HTIF_LONG_OPTIONS_OPTIND 1024
#define HTIF_LONG_OPTIONS                                               \
{"help",      no_argument,       0, 'h'                          },     \
{"rfb",       optional_argument, 0, HTIF_LONG_OPTIONS_OPTIND     },     \
{"disk",      required_argument, 0, HTIF_LONG_OPTIONS_OPTIND + 1 },     \
{"signature", required_argument, 0, HTIF_LONG_OPTIONS_OPTIND + 2 },     \
{"chroot",    required_argument, 0, HTIF_LONG_OPTIONS_OPTIND + 3 },     \
{"payload",   required_argument, 0, HTIF_LONG_OPTIONS_OPTIND + 4 },     \
{"signature-granularity",    required_argument, 0, HTIF_LONG_OPTIONS_OPTIND + 5 },     \
{"target-argument",          required_argument, 0, HTIF_LONG_OPTIONS_OPTIND + 6 },     \
{"symbol-elf",               required_argument, 0, HTIF_LONG_OPTIONS_OPTIND + 7 },     \
{0, 0, 0, 0}

#endif // __HTIF_H

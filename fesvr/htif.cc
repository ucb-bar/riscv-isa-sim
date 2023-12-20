// See LICENSE for license details.

#include "config.h"
#include "htif.h"
#include "rfb.h"
#include "elfloader.h"
#include "platform.h"
#include "byteorder.h"
#include "trap.h"
#include "../riscv/common.h"
#include <algorithm>
#include <assert.h>
#include <vector>
#include <queue>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <stdio.h>
#include <unistd.h>
#include <signal.h>
#include <getopt.h>

/* Attempt to determine the execution prefix automatically.  autoconf
 * sets PREFIX, and pconfigure sets __PCONFIGURE__PREFIX. */
#if !defined(PREFIX) && defined(__PCONFIGURE__PREFIX)
# define PREFIX __PCONFIGURE__PREFIX
#endif

#ifndef TARGET_ARCH
# define TARGET_ARCH "riscv64-unknown-elf"
#endif

#ifndef TARGET_DIR
# define TARGET_DIR "/" TARGET_ARCH "/bin/"
#endif

static void handle_signal(int sig)
{
  if (sig == SIGABRT || signal_exit) // someone set up us the bomb!
    exit(-1);
  signal_exit = true;
  signal(sig, &handle_signal);
}

htif_t::htif_t()
  : mem(this), entry(DRAM_BASE), sig_addr(0), sig_len(0),
    tohost_addr(0), fromhost_addr(0), exitcode(0), stopped(false),
    syscall_proxy(this)
{
  signal(SIGINT, &handle_signal);
  signal(SIGTERM, &handle_signal);
  signal(SIGABRT, &handle_signal); // we still want to call static destructors
}

htif_t::htif_t(int argc, char** argv) : htif_t()
{
  //Set line size as 16 by default.
  line_size = 16;
  parse_arguments(argc, argv);
  register_devices();
}

htif_t::htif_t(const std::vector<std::string>& args) : htif_t()
{
  int argc = args.size() + 1;
  char * argv[argc];
  argv[0] = (char *) "htif";
  for (unsigned int i = 0; i < args.size(); i++) {
    argv[i+1] = (char *) args[i].c_str();
  }
  //Set line size as 16 by default.
  line_size = 16;
  parse_arguments(argc, argv);
  register_devices();
}

htif_t::~htif_t()
{
  for (auto d : dynamic_devices)
    delete d;
}

void htif_t::start()
{
  if (!targs.empty() && targs[0] != "none") {
    try {
      load_program();
    } catch (const incompat_xlen & err) {
      fprintf(stderr, "Error: cannot execute %d-bit program on RV%d hart\n", err.actual_xlen, err.expected_xlen);
      exit(1);
    }
  }

  reset();
}

std::map<std::string, uint64_t> htif_t::load_payload(const std::string& payload, reg_t* entry)
{
  std::string path;
  if (access(payload.c_str(), F_OK) == 0)
    path = payload;
  else if (payload.find('/') == std::string::npos)
  {
    std::string test_path = PREFIX TARGET_DIR + payload;
    if (access(test_path.c_str(), F_OK) == 0)
      path = test_path;
    else
      throw std::runtime_error(
        "could not open " + payload + "; searched paths:\n" +
        "\t. (current directory)\n" + 
        "\t" + PREFIX TARGET_DIR + " (based on configured --prefix and --with-target)"
      );
  }

  if (path.empty())
    throw std::runtime_error(
        "could not open " + payload +
        " (did you misspell it? If VCS, did you forget +permissive/+permissive-off?)");

  // temporarily construct a memory interface that skips writing bytes
  // that have already been preloaded through a sideband
  class preload_aware_memif_t : public memif_t {
   public:
    preload_aware_memif_t(htif_t* htif) : memif_t(htif), htif(htif) {}

    void write(addr_t taddr, size_t len, const void* src) override
    {
      if (!htif->is_address_preloaded(taddr, len))
        memif_t::write(taddr, len, src);
    }

   private:
    htif_t* htif;
  } preload_aware_memif(this);

  try {
    return load_elf(path.c_str(), &preload_aware_memif, entry, expected_xlen);
  } catch (mem_trap_t& t) {
    bad_address("loading payload " + payload, t.get_tval());
    abort();
  }
}

void htif_t::load_program()
{
  std::map<std::string, uint64_t> symbols = load_payload(targs[0], &entry);

  if (symbols.count("tohost") && symbols.count("fromhost")) {
    tohost_addr = symbols["tohost"];
    fromhost_addr = symbols["fromhost"];
  } else {
    fprintf(stderr, "warning: tohost and fromhost symbols not in ELF; can't communicate with target\n");
  }

  // detect torture tests so we can print the memory signature at the end
  if (symbols.count("begin_signature") && symbols.count("end_signature")) {
    sig_addr = symbols["begin_signature"];
    sig_len = symbols["end_signature"] - sig_addr;
  }

  for (auto payload : payloads) {
    reg_t dummy_entry;
    load_payload(payload, &dummy_entry);
  }

  class nop_memif_t : public memif_t {
   public:
    nop_memif_t(htif_t* htif) : memif_t(htif), htif(htif) {}
    void read(addr_t UNUSED addr, size_t UNUSED len, void UNUSED *bytes) override {}
    void write(addr_t UNUSED taddr, size_t UNUSED len, const void UNUSED *src) override {}
   private:
    htif_t* htif;
  } nop_memif(this);

  reg_t nop_entry;
  for (auto &s : symbol_elfs) {
    std::map<std::string, uint64_t> other_symbols = load_elf(s.c_str(), &nop_memif, &nop_entry,
                                                             expected_xlen);
    symbols.merge(other_symbols);
  }

  for (auto i : symbols) {
    auto it = addr2symbol.find(i.second);
    if ( it == addr2symbol.end())
      addr2symbol[i.second] = i.first;
  }

  return;
}

const char* htif_t::get_symbol(uint64_t addr)
{
  auto it = addr2symbol.find(addr);

  if(it == addr2symbol.end())
      return nullptr;

  return it->second.c_str();
}

void htif_t::stop()
{
  if (!sig_file.empty() && sig_len) // print final torture test signature
  {
    std::vector<uint8_t> buf(sig_len);
    mem.read(sig_addr, sig_len, buf.data());

    std::ofstream sigs(sig_file);
    assert(sigs && "can't open signature file!");
    sigs << std::setfill('0') << std::hex;

    for (addr_t i = 0; i < sig_len; i += line_size)
    {
      for (addr_t j = line_size; j > 0; j--)
          if (i+j <= sig_len)
            sigs << std::setw(2) << (uint16_t)buf[i+j-1];
          else
            sigs << std::setw(2) << (uint16_t)0;
      sigs << '\n';
    }

    sigs.close();
  }

  stopped = true;
}

void htif_t::clear_chunk(addr_t taddr, size_t len)
{
  char zeros[chunk_max_size()];
  memset(zeros, 0, chunk_max_size());

  for (size_t pos = 0; pos < len; pos += chunk_max_size())
    write_chunk(taddr + pos, std::min(len - pos, chunk_max_size()), zeros);
}

int htif_t::run()
{
  start();

  auto enq_func = [](std::queue<reg_t>* q, uint64_t x) { q->push(x); };
  std::queue<reg_t> fromhost_queue;
  std::function<void(reg_t)> fromhost_callback =
    std::bind(enq_func, &fromhost_queue, std::placeholders::_1);

  if (tohost_addr == 0) {
    while (!signal_exit)
      idle();
  }

  while (!signal_exit && exitcode == 0)
  {
    uint64_t tohost;

    try {
      if ((tohost = from_target(mem.read_uint64(tohost_addr))) != 0)
        mem.write_uint64(tohost_addr, target_endian<uint64_t>::zero);
    } catch (mem_trap_t& t) {
      bad_address("accessing tohost", t.get_tval());
    }

    try {
      if (tohost != 0) {
        command_t cmd(mem, tohost, fromhost_callback);
        device_list.handle_command(cmd);
      } else {
        idle();
      }

      device_list.tick();
    } catch (mem_trap_t& t) {
      std::stringstream tohost_hex;
      tohost_hex << std::hex << tohost;
      bad_address("host was accessing memory on behalf of target (tohost = 0x" + tohost_hex.str() + ")", t.get_tval());
    }

    try {
      if (!fromhost_queue.empty() && !mem.read_uint64(fromhost_addr)) {
        mem.write_uint64(fromhost_addr, to_target(fromhost_queue.front()));
        fromhost_queue.pop();
      }
    } catch (mem_trap_t& t) {
      bad_address("accessing fromhost", t.get_tval());
    }
  }

  stop();

  return exit_code();
}

bool htif_t::done()
{
  return stopped;
}

int htif_t::exit_code()
{
  return exitcode >> 1;
}

void htif_t::parse_arguments(int argc, char ** argv)
{
  optind = 0; // reset optind as HTIF may run getopt _after_ others
  while (1) {
    static struct option long_options[] = { HTIF_LONG_OPTIONS };
    int option_index = 0;
    int c = getopt_long(argc, argv, "-h", long_options, &option_index);

    if (c == -1) break;
 retry:
    switch (c) {
      case 'h': usage(argv[0]);
        throw std::invalid_argument("User queried htif_t help text");
      case HTIF_LONG_OPTIONS_OPTIND:
        if (optarg) dynamic_devices.push_back(new rfb_t(atoi(optarg)));
        else        dynamic_devices.push_back(new rfb_t);
        break;
      case HTIF_LONG_OPTIONS_OPTIND + 1:
        // [TODO] Remove once disks are supported again
        throw std::invalid_argument("--disk/+disk unsupported (use a ramdisk)");
        dynamic_devices.push_back(new disk_t(optarg));
        break;
      case HTIF_LONG_OPTIONS_OPTIND + 2:
        sig_file = optarg;
        break;
      case HTIF_LONG_OPTIONS_OPTIND + 3:
        syscall_proxy.set_chroot(optarg);
        break;
      case HTIF_LONG_OPTIONS_OPTIND + 4:
        payloads.push_back(optarg);
        break;
      case HTIF_LONG_OPTIONS_OPTIND + 5:
        line_size = atoi(optarg);
        break;
      case HTIF_LONG_OPTIONS_OPTIND + 6:
        targs.push_back(optarg);
        break;
      case HTIF_LONG_OPTIONS_OPTIND + 7:
        symbol_elfs.push_back(optarg);
        break;
      case '?':
        if (!opterr)
          break;
        throw std::invalid_argument("Unknown argument (did you mean to enable +permissive parsing?)");
      case 1: {
        std::string arg = optarg;
        if (arg == "+h" || arg == "+help") {
          c = 'h';
          optarg = nullptr;
        }
        else if (arg == "+rfb") {
          c = HTIF_LONG_OPTIONS_OPTIND;
          optarg = nullptr;
        }
        else if (arg.find("+rfb=") == 0) {
          c = HTIF_LONG_OPTIONS_OPTIND;
          optarg = optarg + 5;
        }
        else if (arg.find("+disk=") == 0) {
          c = HTIF_LONG_OPTIONS_OPTIND + 1;
          optarg = optarg + 6;
        }
        else if (arg.find("+signature=") == 0) {
          c = HTIF_LONG_OPTIONS_OPTIND + 2;
          optarg = optarg + 11;
        }
        else if (arg.find("+chroot=") == 0) {
          c = HTIF_LONG_OPTIONS_OPTIND + 3;
          optarg = optarg + 8;
        }
        else if (arg.find("+payload=") == 0) {
          c = HTIF_LONG_OPTIONS_OPTIND + 4;
          optarg = optarg + 9;
        }
        else if (arg.find("+signature-granularity=") == 0) {
          c = HTIF_LONG_OPTIONS_OPTIND + 5;
          optarg = optarg + 23;
        }
	else if (arg.find("+target-argument=") == 0) {
	  c = HTIF_LONG_OPTIONS_OPTIND + 6;
	  optarg = optarg + 17;
	}
        else if (arg.find("+symbol-elf=") == 0) {
          c = HTIF_LONG_OPTIONS_OPTIND + 7;
          optarg = optarg + 12;
        }
        else if (arg.find("+permissive-off") == 0) {
          if (opterr)
            throw std::invalid_argument("Found +permissive-off when not parsing permissively");
          opterr = 1;
          break;
        }
        else if (arg.find("+permissive") == 0) {
          if (!opterr)
            throw std::invalid_argument("Found +permissive when already parsing permissively");
          opterr = 0;
          break;
        }
        else {
          if (!opterr)
            break;
          else {
            optind--;
            goto done_processing;
          }
        }
        goto retry;
      }
    }
  }

done_processing:
  while (optind < argc)
    targs.push_back(argv[optind++]);
  if (!targs.size()) {
    usage(argv[0]);
    throw std::invalid_argument("No binary specified (Did you forget it? Did you forget '+permissive-off' if running with +permissive?)");
  }
}

void htif_t::register_devices()
{
  device_list.register_device(&syscall_proxy);
  device_list.register_device(&bcd);
  for (auto d : dynamic_devices)
    device_list.register_device(d);
}

void htif_t::usage(const char * program_name)
{
  printf("Usage: %s [EMULATOR OPTION]... [VERILOG PLUSARG]... [HOST OPTION]... BINARY [TARGET OPTION]...\n ",
         program_name);
  fputs("\
Run a BINARY on the Rocket Chip emulator.\n\
\n\
Mandatory arguments to long options are mandatory for short options too.\n\
\n\
EMULATOR OPTIONS\n\
  Consult emulator.cc if using Verilator or VCS documentation if using VCS\n\
    for available options.\n\
EMUALTOR VERILOG PLUSARGS\n\
  Consult generated-src*/*.plusArgs for available options\n\
", stdout);
  fputs("\n" HTIF_USAGE_OPTIONS, stdout);
}

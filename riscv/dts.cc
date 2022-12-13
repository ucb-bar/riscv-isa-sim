// See LICENSE for license details.

#include "config.h"
#include "dts.h"
#include "libfdt.h"
#include "platform.h"
#include <cassert>
#include <iostream>
#include <sstream>
#include <signal.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/types.h>

std::string make_dts(size_t insns_per_rtc_tick, size_t cpu_hz,
                     reg_t initrd_start, reg_t initrd_end,
                     const char* bootargs,
                     size_t pmpregions,
                     std::vector<processor_t*> procs,
                     std::vector<std::pair<reg_t, mem_t*>> mems)
{
  std::stringstream s;
  s << std::dec <<
         "/dts-v1/;\n"
         "\n"
         "/ {\n"
         "  #address-cells = <2>;\n"
         "  #size-cells = <2>;\n"
         "  compatible = \"ucbbar,spike-bare-dev\";\n"
         "  model = \"ucbbar,spike-bare\";\n"
         "  chosen {\n"
         "    stdout-path = &SERIAL0;\n";
  if (initrd_start < initrd_end) {
    s << "    linux,initrd-start = <" << (size_t)initrd_start << ">;\n"
         "    linux,initrd-end = <" << (size_t)initrd_end << ">;\n";
    if (!bootargs)
      bootargs = "root=/dev/ram console=ttyS0 earlycon";
  } else {
    if (!bootargs)
      bootargs = "console=ttyS0 earlycon";
  }
    s << "    bootargs = \"";
  for (size_t i = 0; i < strlen(bootargs); i++) {
    if (bootargs[i] == '"')
    s << '\\' << bootargs[i];
    else
    s << bootargs[i];
  }
    s << "\";\n";
    s << "  };\n"
         "  cpus {\n"
         "    #address-cells = <1>;\n"
         "    #size-cells = <0>;\n"
         "    timebase-frequency = <" << (cpu_hz/insns_per_rtc_tick) << ">;\n";
  for (size_t i = 0; i < procs.size(); i++) {
    s << "    CPU" << i << ": cpu@" << i << " {\n"
         "      device_type = \"cpu\";\n"
         "      reg = <" << i << ">;\n"
         "      status = \"okay\";\n"
         "      compatible = \"riscv\";\n"
         "      riscv,isa = \"" << procs[i]->get_isa().get_isa_string() << "\";\n"
         "      mmu-type = \"riscv," << (procs[i]->get_isa().get_max_xlen() <= 32 ? "sv32" : "sv57") << "\";\n"
         "      riscv,pmpregions = <" << pmpregions << ">;\n"
         "      riscv,pmpgranularity = <4>;\n"
         "      clock-frequency = <" << cpu_hz << ">;\n"
         "      CPU" << i << "_intc: interrupt-controller {\n"
         "        #address-cells = <2>;\n"
         "        #interrupt-cells = <1>;\n"
         "        interrupt-controller;\n"
         "        compatible = \"riscv,cpu-intc\";\n"
         "      };\n"
         "    };\n";
  }
  s <<   "  };\n";
  for (auto& m : mems) {
    s << std::hex <<
         "  memory@" << m.first << " {\n"
         "    device_type = \"memory\";\n"
         "    reg = <0x" << (m.first >> 32) << " 0x" << (m.first & (uint32_t)-1) <<
                   " 0x" << (m.second->size() >> 16 >> 16) << " 0x" << (m.second->size() & (uint32_t)-1) << ">;\n"
         "  };\n";
  }
  s <<   "  soc {\n"
         "    #address-cells = <2>;\n"
         "    #size-cells = <2>;\n"
         "    compatible = \"ucbbar,spike-bare-soc\", \"simple-bus\";\n"
         "    ranges;\n"
         "    clint@" << CLINT_BASE << " {\n"
         "      compatible = \"riscv,clint0\";\n"
         "      interrupts-extended = <" << std::dec;
  for (size_t i = 0; i < procs.size(); i++)
    s << "&CPU" << i << "_intc 3 &CPU" << i << "_intc 7 ";
  reg_t clintbs = CLINT_BASE;
  reg_t clintsz = CLINT_SIZE;
  s << std::hex << ">;\n"
         "      reg = <0x" << (clintbs >> 32) << " 0x" << (clintbs & (uint32_t)-1) <<
                     " 0x" << (clintsz >> 32) << " 0x" << (clintsz & (uint32_t)-1) << ">;\n"
         "    };\n"
         "    PLIC: plic@" << PLIC_BASE << " {\n"
         "      compatible = \"riscv,plic0\";\n"
         "      #address-cells = <2>;\n"
         "      interrupts-extended = <" << std::dec;
  for (size_t i = 0; i < procs.size(); i++)
    s << "&CPU" << i << "_intc 11 &CPU" << i << "_intc 9 ";
  reg_t plicbs = PLIC_BASE;
  reg_t plicsz = PLIC_SIZE;
  s << std::hex << ">;\n"
         "      reg = <0x" << (plicbs >> 32) << " 0x" << (plicbs & (uint32_t)-1) <<
                     " 0x" << (plicsz >> 32) << " 0x" << (plicsz & (uint32_t)-1) << ">;\n"
         "      riscv,ndev = <0x" << PLIC_NDEV << ">;\n"
         "      riscv,max-priority = <0x" << ((1U << PLIC_PRIO_BITS) - 1) << ">;\n"
         "      #interrupt-cells = <1>;\n"
         "      interrupt-controller;\n"
         "    };\n"
         "    SERIAL0: ns16550@" << NS16550_BASE << " {\n"
         "      compatible = \"ns16550a\";\n"
         "      clock-frequency = <" << std::dec << (cpu_hz/insns_per_rtc_tick) << ">;\n"
         "      interrupt-parent = <&PLIC>;\n"
         "      interrupts = <" << std::dec << NS16550_INTERRUPT_ID;
  reg_t ns16550bs = NS16550_BASE;
  reg_t ns16550sz = NS16550_SIZE;
  s << std::hex << ">;\n"
         "      reg = <0x" << (ns16550bs >> 32) << " 0x" << (ns16550bs & (uint32_t)-1) <<
                     " 0x" << (ns16550sz >> 32) << " 0x" << (ns16550sz & (uint32_t)-1) << ">;\n"
         "      reg-shift = <0x" << NS16550_REG_SHIFT << ">;\n"
         "      reg-io-width = <0x" << NS16550_REG_IO_WIDTH << ">;\n"
         "    };\n"
         "  };\n"
         "  htif {\n"
         "    compatible = \"ucb,htif0\";\n"
         "  };\n"
         "};\n";
  return s.str();
}

std::string dts_compile(const std::string& dts)
{
  // Convert the DTS to DTB
  int dts_pipe[2];
  pid_t dts_pid;

  fflush(NULL); // flush stdout/stderr before forking
  if (pipe(dts_pipe) != 0 || (dts_pid = fork()) < 0) {
    std::cerr << "Failed to fork dts child: " << strerror(errno) << std::endl;
    exit(1);
  }

  // Child process to output dts
  if (dts_pid == 0) {
    close(dts_pipe[0]);
    int step, len = dts.length();
    const char *buf = dts.c_str();
    for (int done = 0; done < len; done += step) {
      step = write(dts_pipe[1], buf+done, len-done);
      if (step == -1) {
        std::cerr << "Failed to write dts: " << strerror(errno) << std::endl;
        exit(1);
      }
    }
    close(dts_pipe[1]);
    exit(0);
  }

  pid_t dtb_pid;
  int dtb_pipe[2];
  if (pipe(dtb_pipe) != 0 || (dtb_pid = fork()) < 0) {
    std::cerr << "Failed to fork dtb child: " << strerror(errno) << std::endl;
    exit(1);
  }

  // Child process to output dtb
  if (dtb_pid == 0) {
    dup2(dts_pipe[0], 0);
    dup2(dtb_pipe[1], 1);
    close(dts_pipe[0]);
    close(dts_pipe[1]);
    close(dtb_pipe[0]);
    close(dtb_pipe[1]);
    execlp(DTC, DTC, "-O", "dtb", (char *)0);
    std::cerr << "Failed to run " DTC ": " << strerror(errno) << std::endl;
    exit(1);
  }

  close(dts_pipe[1]);
  close(dts_pipe[0]);
  close(dtb_pipe[1]);

  // Read-out dtb
  std::stringstream dtb;

  int got;
  char buf[4096];
  while ((got = read(dtb_pipe[0], buf, sizeof(buf))) > 0) {
    dtb.write(buf, got);
  }
  if (got == -1) {
    std::cerr << "Failed to read dtb: " << strerror(errno) << std::endl;
    exit(1);
  }
  close(dtb_pipe[0]);

  // Reap children
  int status;
  waitpid(dts_pid, &status, 0);
  if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
    std::cerr << "Child dts process failed" << std::endl;
    exit(1);
  }
  waitpid(dtb_pid, &status, 0);
  if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
    std::cerr << "Child dtb process failed" << std::endl;
    exit(1);
  }

  return dtb.str();
}

static int fdt_get_node_addr_size(void *fdt, int node, reg_t *addr,
                                  unsigned long *size, const char *field)
{
  int parent, len, i;
  int cell_addr, cell_size;
  const fdt32_t *prop_addr, *prop_size;
  uint64_t temp = 0;

  parent = fdt_parent_offset(fdt, node);
  if (parent < 0)
    return parent;

  cell_addr = fdt_address_cells(fdt, parent);
  if (cell_addr < 1)
    return -ENODEV;

  cell_size = fdt_size_cells(fdt, parent);
  if (cell_size < 0)
    return -ENODEV;

  if (!field)
    return -ENODEV;

  prop_addr = (fdt32_t *)fdt_getprop(fdt, node, field, &len);
  if (!prop_addr)
    return -ENODEV;
  prop_size = prop_addr + cell_addr;

  if (addr) {
    for (i = 0; i < cell_addr; i++)
      temp = (temp << 32) | fdt32_to_cpu(*prop_addr++);
    *addr = temp;
  }
  temp = 0;

  if (size) {
    for (i = 0; i < cell_size; i++)
      temp = (temp << 32) | fdt32_to_cpu(*prop_size++);
    *size = temp;
  }

  return 0;
}

static int check_cpu_node(void *fdt, int cpu_offset)
{
  int len;
  const void *prop;

  if (!fdt || cpu_offset < 0)
    return -EINVAL;

  prop = fdt_getprop(fdt, cpu_offset, "device_type", &len);
  if (!prop || !len)
    return -EINVAL;
  if (strncmp ((char *)prop, "cpu", strlen ("cpu")))
    return -EINVAL;

  return 0;
}

int fdt_get_offset(void *fdt, const char *field)
{
  return fdt_path_offset(fdt, field);
}

int fdt_get_first_subnode(void *fdt, int node)
{
  return fdt_first_subnode(fdt, node);
}

int fdt_get_next_subnode(void *fdt, int node)
{
  return fdt_next_subnode(fdt, node);
}

int fdt_parse_clint(void *fdt, reg_t *clint_addr,
                    const char *compatible)
{
  int nodeoffset, rc;

  nodeoffset = fdt_node_offset_by_compatible(fdt, -1, compatible);
  if (nodeoffset < 0)
    return nodeoffset;

  rc = fdt_get_node_addr_size(fdt, nodeoffset, clint_addr, NULL, "reg");
  if (rc < 0 || !clint_addr)
    return -ENODEV;

  return 0;
}

int fdt_parse_plic(void *fdt, reg_t *plic_addr, uint32_t *ndev,
                   const char *compatible)
{
  int nodeoffset, len, rc;
  const fdt32_t *ndev_p;

  nodeoffset = fdt_node_offset_by_compatible(fdt, -1, compatible);
  if (nodeoffset < 0)
    return nodeoffset;

  rc = fdt_get_node_addr_size(fdt, nodeoffset, plic_addr, NULL, "reg");
  if (rc < 0 || !plic_addr)
    return -ENODEV;

  ndev_p = (fdt32_t *)fdt_getprop(fdt, nodeoffset, "riscv,ndev", &len);
  if (!ndev || !ndev_p)
    return -ENODEV;
  *ndev = fdt32_to_cpu(*ndev_p);

  return 0;
}

int fdt_parse_ns16550(void *fdt, reg_t *ns16550_addr,
                      uint32_t *reg_shift, uint32_t *reg_io_width,
                      const char *compatible)
{
  int nodeoffset, len, rc;
  const fdt32_t *reg_p;

  nodeoffset = fdt_node_offset_by_compatible(fdt, -1, compatible);
  if (nodeoffset < 0)
    return nodeoffset;

  rc = fdt_get_node_addr_size(fdt, nodeoffset, ns16550_addr, NULL, "reg");
  if (rc < 0 || !ns16550_addr)
    return -ENODEV;

  reg_p = (fdt32_t *)fdt_getprop(fdt, nodeoffset, "reg-shift", &len);
  if (reg_shift) {
    if (reg_p) {
      *reg_shift = fdt32_to_cpu(*reg_p);
    } else {
      *reg_shift = NS16550_REG_SHIFT;
    }
  }

  reg_p = (fdt32_t *)fdt_getprop(fdt, nodeoffset, "reg-io-width", &len);
  if (reg_io_width) {
    if (reg_p) {
      *reg_io_width = fdt32_to_cpu(*reg_p);
    } else {
      *reg_io_width = NS16550_REG_IO_WIDTH;
    }
  }

  return 0;
}

int fdt_parse_pmp_num(void *fdt, int cpu_offset, reg_t *pmp_num)
{
  int rc;

  if ((rc = check_cpu_node(fdt, cpu_offset)) < 0)
    return rc;

  rc = fdt_get_node_addr_size(fdt, cpu_offset, pmp_num, NULL,
                              "riscv,pmpregions");
  if (rc < 0 || !pmp_num)
    return -ENODEV;

  return 0;
}

int fdt_parse_pmp_alignment(void *fdt, int cpu_offset, reg_t *pmp_align)
{
  int rc;

  if ((rc = check_cpu_node(fdt, cpu_offset)) < 0)
    return rc;

  rc = fdt_get_node_addr_size(fdt, cpu_offset, pmp_align, NULL,
                              "riscv,pmpgranularity");
  if (rc < 0 || !pmp_align)
    return -ENODEV;

  return 0;
}

int fdt_parse_mmu_type(void *fdt, int cpu_offset, const char **mmu_type)
{
  assert(mmu_type);

  int len, rc;
  const void *prop;

  if ((rc = check_cpu_node(fdt, cpu_offset)) < 0)
    return rc;

  prop = fdt_getprop(fdt, cpu_offset, "mmu-type", &len);
  if (!prop || !len)
    return -EINVAL;

  *mmu_type = (const char *)prop;

  return 0;
}

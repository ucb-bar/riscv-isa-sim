// See LICENSE for license details.

#include "config.h"
#include "elf.h"
#include "memif.h"
#include "byteorder.h"
#include <cstring>
#include <string>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <assert.h>
#include <unistd.h>
#include <stdexcept>
#include <stdlib.h>
#include <stdio.h>
#include <inttypes.h>
#include <vector>
#include <map>
#include <iostream>

std::map<std::string, uint64_t> load_elf(const char* fn, memif_t* memif, reg_t* entry, unsigned required_xlen = 0)
{
  fprintf(stdout, "load_elf\n");
  fflush(stdout);

  int fd = open(fn, O_RDONLY);
  fprintf(stdout, "fd %d\n", fd);
  fflush(stdout);

  struct stat s;
  if (fd == -1)
      throw std::invalid_argument(std::string("Specified ELF can't be opened: ") + strerror(errno));
  if (fstat(fd, &s) < 0)
    abort();
  size_t size = s.st_size;

  fprintf(stdout, "size %" PRIu64 "\n", size);
  fflush(stdout);

  char* buf = (char*)mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0);
  if (buf == MAP_FAILED) {
    fprintf(stdout, "buf == MAP_FAILED\n");
    fflush(stdout);
      throw std::invalid_argument(std::string("Specified ELF can't be mapped: ") + strerror(errno));
  }

  close(fd);

  assert(size >= sizeof(Elf64_Ehdr));
  const Elf64_Ehdr* eh64 = (const Elf64_Ehdr*)buf;
  assert(IS_ELF32(*eh64) || IS_ELF64(*eh64));
  unsigned xlen = IS_ELF32(*eh64) ? 32 : 64;

  fprintf(stdout, "required_xlen: %u xlen: %u\n", required_xlen, xlen);
  fflush(stdout);

  if (required_xlen != 0 && required_xlen != xlen) {
    throw incompat_xlen(required_xlen, xlen);
  }
  assert(IS_ELFLE(*eh64) || IS_ELFBE(*eh64));
  assert(IS_ELF_EXEC(*eh64));
  assert(IS_ELF_RISCV(*eh64) || IS_ELF_EM_NONE(*eh64));
  assert(IS_ELF_VCURRENT(*eh64));

  fprintf(stdout, "elf_assertions done\n");
  fflush(stdout);

  std::vector<uint8_t> zeros;
  std::map<std::string, uint64_t> symbols;

#define LOAD_ELF(ehdr_t, phdr_t, shdr_t, sym_t, bswap)                         \
  do {                                                                         \
    ehdr_t* eh = (ehdr_t*)buf;                                                 \
    phdr_t* ph = (phdr_t*)(buf + bswap(eh->e_phoff));                          \
    *entry = bswap(eh->e_entry);                                               \
    assert(size >= bswap(eh->e_phoff) + bswap(eh->e_phnum) * sizeof(*ph));     \
    for (unsigned i = 0; i < bswap(eh->e_phnum); i++) {                        \
      fprintf(stdout, "MACRO first loop i %d\n", i);                           \
      fflush(stdout);                                                          \
      if (bswap(ph[i].p_type) == PT_LOAD && bswap(ph[i].p_memsz)) {            \
        fprintf(stdout, "ph[i].p_filesz %d\n", bswap(ph[i].p_filesz));         \
        fflush(stdout);                                                        \
        if (bswap(ph[i].p_filesz)) {                                           \
          fprintf(stdout, "p_offset %d p_filesz %d\n",                         \
              bswap(ph[i].p_offset),                                           \
              bswap(ph[i].p_filesz));                                          \
          fflush(stdout);                                                      \
          assert(size >= bswap(ph[i].p_offset) + bswap(ph[i].p_filesz));       \
          memif->write(bswap(ph[i].p_paddr), bswap(ph[i].p_filesz),            \
                       (uint8_t*)buf + bswap(ph[i].p_offset));                 \
          fprintf(stdout, "memif->write done\n");                              \
          fflush(stdout);                                                      \
        }                                                                      \
        if (size_t pad = bswap(ph[i].p_memsz) - bswap(ph[i].p_filesz)) {       \
          zeros.resize(pad);                                                   \
          memif->write(bswap(ph[i].p_paddr) + bswap(ph[i].p_filesz), pad,      \
                       zeros.data());                                          \
        }                                                                      \
      }                                                                        \
    }                                                                          \
    fprintf(stdout, "MACRO first loop done\n");                                \
    fflush(stdout);                                                            \
    shdr_t* sh = (shdr_t*)(buf + bswap(eh->e_shoff));                          \
    assert(size >= bswap(eh->e_shoff) + bswap(eh->e_shnum) * sizeof(*sh));     \
    assert(bswap(eh->e_shstrndx) < bswap(eh->e_shnum));                        \
    assert(size >= bswap(sh[bswap(eh->e_shstrndx)].sh_offset) +                \
                       bswap(sh[bswap(eh->e_shstrndx)].sh_size));              \
    char* shstrtab = buf + bswap(sh[bswap(eh->e_shstrndx)].sh_offset);         \
    unsigned strtabidx = 0, symtabidx = 0;                                     \
    for (unsigned i = 0; i < bswap(eh->e_shnum); i++) {                        \
      unsigned max_len =                                                       \
          bswap(sh[bswap(eh->e_shstrndx)].sh_size) - bswap(sh[i].sh_name);     \
      assert(bswap(sh[i].sh_name) < bswap(sh[bswap(eh->e_shstrndx)].sh_size)); \
      assert(strnlen(shstrtab + bswap(sh[i].sh_name), max_len) < max_len);     \
      if (bswap(sh[i].sh_type) & SHT_NOBITS) continue;                         \
      assert(size >= bswap(sh[i].sh_offset) + bswap(sh[i].sh_size));           \
      if (strcmp(shstrtab + bswap(sh[i].sh_name), ".strtab") == 0)             \
        strtabidx = i;                                                         \
      if (strcmp(shstrtab + bswap(sh[i].sh_name), ".symtab") == 0)             \
        symtabidx = i;                                                         \
    }                                                                          \
    fprintf(stdout, "MACRO second loop done\n");                               \
    fflush(stdout);                                                            \
    if (strtabidx && symtabidx) {                                              \
      char* strtab = buf + bswap(sh[strtabidx].sh_offset);                     \
      sym_t* sym = (sym_t*)(buf + bswap(sh[symtabidx].sh_offset));             \
      for (unsigned i = 0; i < bswap(sh[symtabidx].sh_size) / sizeof(sym_t);   \
           i++) {                                                              \
        unsigned max_len =                                                     \
            bswap(sh[strtabidx].sh_size) - bswap(sym[i].st_name);              \
        assert(bswap(sym[i].st_name) < bswap(sh[strtabidx].sh_size));          \
        assert(strnlen(strtab + bswap(sym[i].st_name), max_len) < max_len);    \
        symbols[strtab + bswap(sym[i].st_name)] = bswap(sym[i].st_value);      \
      }                                                                        \
    }                                                                          \
    fprintf(stdout, "MACRO last if done\n");                                   \
    fflush(stdout);                                                            \
  } while (0)

  fprintf(stdout, "before LOAD_ELF macro call IS_ELFLE(*eh64): %d\n", IS_ELFLE(*eh64));
  fflush(stdout);

  if (IS_ELFLE(*eh64)) {
    if (memif->get_target_endianness() != endianness_little) {
      fprintf(stdout, "specified elf is little endian, but system uses big endian");
      fflush(stdout);
      throw std::invalid_argument("Specified ELF is little endian, but system uses a big-endian memory system. Rerun without --big-endian");
    }
    if (IS_ELF32(*eh64))
      LOAD_ELF(Elf32_Ehdr, Elf32_Phdr, Elf32_Shdr, Elf32_Sym, from_le);
    else
      LOAD_ELF(Elf64_Ehdr, Elf64_Phdr, Elf64_Shdr, Elf64_Sym, from_le);
  } else {
#ifndef RISCV_ENABLE_DUAL_ENDIAN
    std::cout << "RISCV_ENABLE_DUAL_ENDIAN defined & invalid_argument" << std::endl;
    throw std::invalid_argument("Specified ELF is big endian.  Configure with --enable-dual-endian to enable support");
#else
    if (memif->get_target_endianness() != endianness_big) {
      throw std::invalid_argument("Specified ELF is big endian, but system uses a little-endian memory system. Rerun with --big-endian");
    }
    if (IS_ELF32(*eh64))
      LOAD_ELF(Elf32_Ehdr, Elf32_Phdr, Elf32_Shdr, Elf32_Sym, from_be);
    else
      LOAD_ELF(Elf64_Ehdr, Elf64_Phdr, Elf64_Shdr, Elf64_Sym, from_be);
#endif
  }

  fprintf(stdout, "load_elf done\n");
  fflush(stdout);

  munmap(buf, size);

  return symbols;
}

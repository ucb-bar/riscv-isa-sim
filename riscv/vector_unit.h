// See LICENSE for license details.
#ifndef _RISCV_VECTOR_UNIT_H
#define _RISCV_VECTOR_UNIT_H

#include <array>
#include <cstdint>

#include "decode.h"
#include "csrs.h"

class processor_t;

enum VRM{
  RNU = 0,
  RNE,
  RDN,
  ROD,
  INVALID_RM
};

template<uint64_t N>
struct type_usew_t;

template<>
struct type_usew_t<8>
{
  using type=uint8_t;
};

template<>
struct type_usew_t<16>
{
  using type=uint16_t;
};

template<>
struct type_usew_t<32>
{
  using type=uint32_t;
};

template<>
struct type_usew_t<64>
{
  using type=uint64_t;
};

template<uint64_t N>
struct type_sew_t;

template<>
struct type_sew_t<8>
{
  using type=int8_t;
};

template<>
struct type_sew_t<16>
{
  using type=int16_t;
};

template<>
struct type_sew_t<32>
{
  using type=int32_t;
};

template<>
struct type_sew_t<64>
{
  using type=int64_t;
};

// Element Group of 4 32 bits elements (128b total).
using EGU32x4_t = std::array<uint32_t, 4>;

// Element Group of 8 32 bits elements (256b total).
using EGU32x8_t = std::array<uint32_t, 8>;

// Element Group of 4 64 bits elements (256b total).
using EGU64x4_t = std::array<uint64_t, 4>;

// Element Group of 16 8 bits elements (128b total).
using EGU8x16_t = std::array<uint8_t, 16>;

class vectorUnit_t
{
public:
  processor_t* p;
  void *reg_file;
  char reg_referenced[NVPR];
  int setvl_count;
  reg_t vlmax;
  reg_t vlenb;
  csr_t_p vxsat;
  vector_csr_t_p vxrm, vstart, vl, vtype;
  reg_t vma, vta;
  reg_t vsew;
  float vflmul;
  reg_t ELEN, VLEN;
  bool vill;
  bool vstart_alu;

  // vector element for various SEW
  template<class T> T& elt(reg_t vReg, reg_t n, bool is_write = false);
  // vector element group access, where EG is a std::array<T, N>.
  template<typename EG> EG&
  elt_group(reg_t vReg, reg_t n, bool is_write = false);

public:

  void reset();

  vectorUnit_t():
    p(0),
    reg_file(0),
    reg_referenced{0},
    setvl_count(0),
    vlmax(0),
    vlenb(0),
    vxsat(0),
    vxrm(0),
    vstart(0),
    vl(0),
    vtype(0),
    vma(0),
    vta(0),
    vsew(0),
    vflmul(0),
    ELEN(0),
    VLEN(0),
    vill(false),
    vstart_alu(false) {
  }

  ~vectorUnit_t() {
    free(reg_file);
    reg_file = 0;
  }

  reg_t set_vl(int rd, int rs1, reg_t reqVL, reg_t newType);

  reg_t get_vlen() { return VLEN; }
  reg_t get_elen() { return ELEN; }
  reg_t get_slen() { return VLEN; }

  VRM get_vround_mode() {
    return (VRM)(vxrm->read(p));
  }
};
#endif

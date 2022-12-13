// See LICENSE for license details.
#ifndef _RISCV_VECTOR_UNIT_H
#define _RISCV_VECTOR_UNIT_H

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

  // vector element for varies SEW
  template<class T> T& elt(reg_t vReg, reg_t n, bool is_write = false);

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
    return (VRM)(vxrm->read());
  }
};
#endif

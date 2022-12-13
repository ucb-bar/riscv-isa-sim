// See LICENSE for license details.

#ifndef __MEMIF_H
#define __MEMIF_H

#include <stdint.h>
#include <stddef.h>
#include <stdexcept>
#include "byteorder.h"
// for memif_endianness_t
#include "../riscv/cfg.h"

typedef uint64_t reg_t;
typedef int64_t sreg_t;
typedef reg_t addr_t;

class chunked_memif_t
{
public:
  virtual void read_chunk(addr_t taddr, size_t len, void* dst) = 0;
  virtual void write_chunk(addr_t taddr, size_t len, const void* src) = 0;
  virtual void clear_chunk(addr_t taddr, size_t len) = 0;

  virtual size_t chunk_align() = 0;
  virtual size_t chunk_max_size() = 0;

  virtual memif_endianness_t get_target_endianness() const {
    return memif_endianness_little;
  }

  virtual ~chunked_memif_t() = default;
};

class memif_t
{
public:
  memif_t(chunked_memif_t* _cmemif) : cmemif(_cmemif) {}
  virtual ~memif_t(){}

  // read and write byte arrays
  virtual void read(addr_t addr, size_t len, void* bytes);
  virtual void write(addr_t addr, size_t len, const void* bytes);

  // read and write 8-bit words
  virtual target_endian<uint8_t> read_uint8(addr_t addr);
  virtual target_endian<int8_t> read_int8(addr_t addr);
  virtual void write_uint8(addr_t addr, target_endian<uint8_t> val);
  virtual void write_int8(addr_t addr, target_endian<int8_t> val);

  // read and write 16-bit words
  virtual target_endian<uint16_t> read_uint16(addr_t addr);
  virtual target_endian<int16_t> read_int16(addr_t addr);
  virtual void write_uint16(addr_t addr, target_endian<uint16_t> val);
  virtual void write_int16(addr_t addr, target_endian<int16_t> val);

  // read and write 32-bit words
  virtual target_endian<uint32_t> read_uint32(addr_t addr);
  virtual target_endian<int32_t> read_int32(addr_t addr);
  virtual void write_uint32(addr_t addr, target_endian<uint32_t> val);
  virtual void write_int32(addr_t addr, target_endian<int32_t> val);

  // read and write 64-bit words
  virtual target_endian<uint64_t> read_uint64(addr_t addr);
  virtual target_endian<int64_t> read_int64(addr_t addr);
  virtual void write_uint64(addr_t addr, target_endian<uint64_t> val);
  virtual void write_int64(addr_t addr, target_endian<int64_t> val);

  // endianness
  virtual memif_endianness_t get_target_endianness() const {
    return cmemif->get_target_endianness();
  }

protected:
  chunked_memif_t* cmemif;
};

class incompat_xlen : public std::exception {
public:
  const unsigned expected_xlen;
  const unsigned actual_xlen;
  incompat_xlen(unsigned _expected_xlen, unsigned _actual_xlen) : expected_xlen(_expected_xlen), actual_xlen(_actual_xlen) {}
};

#endif // __MEMIF_H

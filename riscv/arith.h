// See LICENSE for license details.

#ifndef _RISCV_ARITH_H
#define _RISCV_ARITH_H

#include <cassert>
#include <cstdint>
#include <climits>
#include <cstddef>
#include <type_traits>

inline uint64_t mulhu(uint64_t a, uint64_t b)
{
  uint64_t t;
  uint32_t y1, y2, y3;
  uint64_t a0 = (uint32_t)a, a1 = a >> 32;
  uint64_t b0 = (uint32_t)b, b1 = b >> 32;

  t = a1*b0 + ((a0*b0) >> 32);
  y1 = t;
  y2 = t >> 32;

  t = a0*b1 + y1;

  t = a1*b1 + y2 + (t >> 32);
  y2 = t;
  y3 = t >> 32;

  return ((uint64_t)y3 << 32) | y2;
}

inline int64_t mulh(int64_t a, int64_t b)
{
  int negate = (a < 0) != (b < 0);
  uint64_t res = mulhu(a < 0 ? -a : a, b < 0 ? -b : b);
  return negate ? ~res + ((uint64_t)a * (uint64_t)b == 0) : res;
}

inline int64_t mulhsu(int64_t a, uint64_t b)
{
  int negate = a < 0;
  uint64_t res = mulhu(a < 0 ? -a : a, b);
  return negate ? ~res + (a * b == 0) : res;
}

//ref:  https://locklessinc.com/articles/sat_arithmetic/
template<typename T, typename UT>
static inline T sat_add(T x, T y, bool &sat)
{
  UT ux = x;
  UT uy = y;
  UT res = ux + uy;
  sat = false;
  int sh = sizeof(T) * 8 - 1;

  /* Calculate overflowed result. (Don't change the sign bit of ux) */
  ux = (ux >> sh) + (((UT)0x1 << sh) - 1);

  /* Force compiler to use cmovns instruction */
  if ((T) ((ux ^ uy) | ~(uy ^ res)) >= 0) {
    res = ux;
    sat = true;
  }

  return res;
}

template<typename T, typename UT>
static inline T sat_add(T x, T y, T z, bool &sat)
{
  bool sat1, sat2;
  T a = y;
  T b = z;
  T res;

  /* Force compiler to use cmovs instruction */
  if (((y ^ z) & (x ^ z)) < 0) {
    a = z;
    b = y;
  }

  res = sat_add<T, UT>(x, a, sat1);
  res = sat_add<T, UT>(res, b, sat2);
  sat = sat1 || sat2;

  return res;
}

template<typename T, typename UT>
static inline T sat_sub(T x, T y, bool &sat)
{
  UT ux = x;
  UT uy = y;
  UT res = ux - uy;
  sat = false;
  int sh = sizeof(T) * 8 - 1;

  /* Calculate overflowed result. (Don't change the sign bit of ux) */
  ux = (ux >> sh) + (((UT)0x1 << sh) - 1);

  /* Force compiler to use cmovns instruction */
  if ((T) ((ux ^ uy) & (ux ^ res)) < 0) {
    res = ux;
    sat = true;
  }

  return res;
}

template<typename T>
T sat_addu(T x, T y, bool &sat)
{
  T res = x + y;
  sat = false;

  sat = res < x;
  res |= -(res < x);

  return res;
}

template<typename T>
T sat_subu(T x, T y, bool &sat)
{
  T res = x - y;
  sat = false;

  sat = !(res <= x);
  res &= -(res <= x);

  return res;
}

static inline uint64_t extract64(uint64_t val, int pos, int len)
{
  assert(pos >= 0 && len > 0 && len <= 64 - pos);
  return (val >> pos) & (~UINT64_C(0) >> (64 - len));
}

static inline uint64_t make_mask64(int pos, int len)
{
    assert(pos >= 0 && len > 0 && pos < 64 && len <= 64);
    return (UINT64_MAX >> (64 - len)) << pos;
}

static inline int popcount(uint64_t val)
{
  val = (val & 0x5555555555555555U) + ((val >>  1) & 0x5555555555555555U);
  val = (val & 0x3333333333333333U) + ((val >>  2) & 0x3333333333333333U);
  val = (val & 0x0f0f0f0f0f0f0f0fU) + ((val >>  4) & 0x0f0f0f0f0f0f0f0fU);
  val = (val & 0x00ff00ff00ff00ffU) + ((val >>  8) & 0x00ff00ff00ff00ffU);
  val = (val & 0x0000ffff0000ffffU) + ((val >> 16) & 0x0000ffff0000ffffU);
  val = (val & 0x00000000ffffffffU) + ((val >> 32) & 0x00000000ffffffffU);
  return val;
}

static inline int ctz(uint64_t val)
{
  if (!val)
    return 0;

  int res = 0;

  if ((val << 32) == 0) res += 32, val >>= 32;
  if ((val << 48) == 0) res += 16, val >>= 16;
  if ((val << 56) == 0) res += 8, val >>= 8;
  if ((val << 60) == 0) res += 4, val >>= 4;
  if ((val << 62) == 0) res += 2, val >>= 2;
  if ((val << 63) == 0) res += 1, val >>= 1;

  return res;
}

static inline int clz(uint64_t val)
{
  if (!val)
    return 0;

  int res = 0;

  if ((val >> 32) == 0) res += 32, val <<= 32;
  if ((val >> 48) == 0) res += 16, val <<= 16;
  if ((val >> 56) == 0) res += 8, val <<= 8;
  if ((val >> 60) == 0) res += 4, val <<= 4;
  if ((val >> 62) == 0) res += 2, val <<= 2;
  if ((val >> 63) == 0) res += 1, val <<= 1;

  return res;
}

// Count number of contiguous 1 bits starting from the LSB.
static inline int cto(uint64_t val)
{
  int res = 0;
  while ((val & 1) == 1)
    val >>= 1, res++;
  return res;
}

static inline int log2(uint64_t val)
{
  if (!val)
    return 0;

  return 63 - clz(val);
}

static inline uint64_t xperm(uint64_t rs1, uint64_t rs2, size_t sz_log2, size_t len)
{
  uint64_t r = 0;
  uint64_t sz = 1LL << sz_log2;
  uint64_t mask = (1LL << sz) - 1;

  assert(sz_log2 <= 6 && len <= 64);

  for (size_t i = 0; i < len; i += sz) {
    uint64_t pos = ((rs2 >> i) & mask) << sz_log2;
    if (pos < len)
      r |= ((rs1 >> pos) & mask) << i;
  }

  return r;
}

// Rotates right an unsigned integer by the given number of bits.
template <typename T>
static inline T rotate_right(T x, std::size_t shiftamt) {
  static_assert(std::is_unsigned<T>::value);
  static constexpr T mask = (8 * sizeof(T)) - 1;
  const std::size_t rshift = shiftamt & mask;
  const std::size_t lshift = (-rshift) & mask;
  return (x << lshift) | (x >> rshift);
}

// Rotates right an unsigned integer by the given number of bits.
template <typename T>
static inline T rotate_left(T x, std::size_t shiftamt) {
  static_assert(std::is_unsigned<T>::value);
  static constexpr T mask = (8 * sizeof(T)) - 1;
  const std::size_t lshift = shiftamt & mask;
  const std::size_t rshift = (-lshift) & mask;
  return (x << lshift) | (x >> rshift);
}

#endif

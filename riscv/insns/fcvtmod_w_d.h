require_extension('D');
require_extension(EXT_ZFA);
require_fp;
uint64_t a = FRS1_D.v;

uint32_t sign = signF64UI(a);
uint32_t exp  = expF64UI(a);
uint64_t frac = fracF64UI(a);

bool inexact = false;
bool invalid = false;

if (exp == 0) {
  inexact = (frac != 0);
  frac = 0;
} else if (exp == 0x7ff) {
  /* inf or NaN */
  invalid = true;
  frac = 0;
} else {
  int true_exp = exp - 1023;
  int shift = true_exp - 52;

  /* Restore implicit bit.  */
  frac |= 1ull << 52;

  /* Shift the fraction into place.  */
  if (shift >= 64) {
    /* The fraction is shifted out entirely.  */
    frac = 0;
  } else  if ((shift >= 0) && (shift < 64)) {
    /* The number is so large we must shift the fraction left.  */
    frac <<= shift;
  } else if ((shift > -64) && (shift < 0)) {
    /* Normal case -- shift right and notice if bits shift out.  */
    inexact = (frac << (64 + shift)) != 0;
    frac >>= -shift;
  } else {
    /* The fraction is shifted out entirely.  */
    frac = 0;
    inexact = true;
  }

  /* Handle overflows */
  if (true_exp > 31 || frac > (sign ? 0x80000000ull : 0x7fffffff)) {
    /* Overflow, for which this operation raises invalid.  */
    invalid = true;
    inexact = false;  /* invalid takes precedence */
  }

  /* Honor the sign.  */
  if (sign) {
    frac = -frac;
  }
}

WRITE_RD(sext32(frac));
STATE.fflags->write(STATE.fflags->read(p) |
		    (inexact ? softfloat_flag_inexact : 0) |
		    (invalid ? softfloat_flag_invalid : 0), p);

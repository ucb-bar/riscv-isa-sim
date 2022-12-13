
#include <fstream>
#include <iostream>

#include "common.h"

//
// Used to model the cryptography extension entropy source.
// See Section 4 of the Scalar Cryptography Extension specificaiton.
class entropy_source {

public:

  // Valid return codes for OPST bits [31:30] when reading seed.
  static const uint32_t OPST_BIST = 0x0 << 30;
  static const uint32_t OPST_WAIT = 0x1 << 30;
  static const uint32_t OPST_ES16 = 0x2 << 30;
  static const uint32_t OPST_DEAD = 0x3 << 30;

  //
  // Other system events
  // ------------------------------------------------------------

  void reset() {
    // Does nothing for now. In the future, can be used to model things
    // like initial BIST states.
  }

  //
  // seed register
  // ------------------------------------------------------------

  void set_seed(reg_t UNUSED val) {
    // Always ignore writes to seed.
    // This CSR is strictly read only. It occupies a RW CSR address
    // to handle the side-effect of the changing seed value on a read.
  }

  //
  // The format of seed is described in Section 4.1 of
  // the scalar cryptography specification.
  reg_t get_seed() {

    uint32_t result  = 0;

    // Currently, always return ES16 (i.e. good randomness) In the future, we
    // can more realistically model things like WAIT states, BIST warm up and
    // maybe scriptable entry into the DEAD state, but until then, this is
    // the bare minimum.
    uint32_t return_status = OPST_ES16;

    if (return_status == OPST_ES16) {

      // Add some sampled entropy into the low 16 bits
      uint16_t entropy = this -> get_two_random_bytes();
      result |= entropy;

    } else if (return_status == OPST_BIST) {

      // Do nothing.

    } else if (return_status == OPST_WAIT) {

      // Do nothing.

    } else if (return_status == OPST_DEAD) {

      // Do nothing. Stay dead.

    } else {

      // Unreachable.

    }

    // Note that if we get here any return_status is anything other than
    // OPST_ES16, then the low 16-bits of the return value must be zero.

    result |= return_status;

    // Result is zero-extended on RV64.
    return (reg_t)result;
  }

  //
  // Utility / support variables and functions.
  // ------------------------------------------------------------

  // The file to read entropy from.
  std::string randomness_source = "/dev/urandom";

  // Read two random bytes from the entropy source file.
  uint16_t get_two_random_bytes() {

    std::ifstream fh(this -> randomness_source, std::ios::binary);

    if (fh.is_open()) {

      uint16_t random_bytes;

      fh.read((char*)(&random_bytes), 2);

      fh.close();

      return random_bytes;

    } else {

      fprintf(stderr, "Could not open randomness source file:\n\t");
      fprintf(stderr, "%s", randomness_source.c_str());
      abort();

    }

  }

};


// See LICENSE for license details.

// This little program finds occurrences of strings like
//   core   0: 0x000000008000c36c (0xfe843783) ld      a5, -24(s0)
// in its inputs, then output the RISC-V instruction with the disassembly
// enclosed hexadecimal number.

#include "config.h"
#include <iostream>
#include <string>
#include <cstdint>
#include <regex>
#include "fesvr/option_parser.h"

#include "disasm.h"
#include "extension.h"

using namespace std;

int main(int UNUSED argc, char** argv)
{
  string s;
  const char* isa_string = DEFAULT_ISA;

  std::function<extension_t*()> extension;
  option_parser_t parser;
  parser.option(0, "extension", 1, [&](const char* s){extension = find_extension(s);});
  parser.option(0, "isa", 1, [&](const char* s){isa_string = s;});
  parser.parse(argv);

  cfg_t cfg;

  isa_parser_t isa(isa_string, DEFAULT_PRIV);
  processor_t p(&isa, &cfg, 0, 0, false, nullptr, cerr);
  if (extension) {
    p.register_extension(extension());
  }

  std::regex reg("^core\\s+\\d+:\\s+0x[0-9a-f]+\\s+\\(0x([0-9a-f]+)\\)", std::regex_constants::icase);
  std::smatch m;
  std::ssub_match sm ;

  while (getline(cin,s)){
    if (regex_search(s, m, reg)){
      // the opcode string
      string op = m[1].str();
      uint32_t bit_num = op.size() * 4;
      uint64_t opcode = strtoull(op.c_str(), nullptr, 16);

      if (bit_num<64){
          opcode = opcode << (64-bit_num) >> (64-bit_num);
      }

      const disasm_insn_t* disasm = p.get_disassembler()->lookup(opcode);
      if (disasm) {
          cout << disasm->get_name() << '\n';
      } else {
          cout << "unknown_op\n";
      }
    }
  }

  return 0;
}

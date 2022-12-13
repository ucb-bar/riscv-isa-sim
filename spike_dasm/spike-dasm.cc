// See LICENSE for license details.

// This little program finds occurrences of strings like
//  DASM(ffabc013)
// in its input, then replaces them with the disassembly
// enclosed hexadecimal number, interpreted as a RISC-V
// instruction.

#include "config.h"
#include "disasm.h"
#include "extension.h"
#include <iostream>
#include <string>
#include <cstdint>
#include <fesvr/option_parser.h>
using namespace std;

int main(int UNUSED argc, char** argv)
{
  string s;
  const char* isa = DEFAULT_ISA;

  std::function<extension_t*()> extension;
  option_parser_t parser;
#ifdef HAVE_DLOPEN
  parser.option(0, "extension", 1, [&](const char* s){extension = find_extension(s);});
#endif
  parser.option(0, "isa", 1, [&](const char* s){isa = s;});
  parser.parse(argv);

  isa_parser_t isa_parser(isa, DEFAULT_PRIV);
  disassembler_t* disassembler = new disassembler_t(&isa_parser);
  if (extension) {
    for (auto disasm_insn : extension()->get_disasms()) {
      disassembler->add_insn(disasm_insn);
    }
  }

  while (getline(cin, s))
  {
    for (size_t pos = 0; (pos = s.find("DASM(", pos)) != string::npos; )
    {
      size_t start = pos;

      pos += strlen("DASM(");

      if (s[pos] == '0' && (s[pos+1] == 'x' || s[pos+1] == 'X'))
        pos += 2;

      if (!isxdigit(s[pos]))
        continue;

      char* endp;
      insn_bits_t bits = strtoull(&s[pos], &endp, 16);
      if (*endp != ')')
        continue;

      string dis = disassembler->disassemble(bits);
      s = s.substr(0, start) + dis + s.substr(endp - &s[0] + 1);
      pos = start + dis.length();
    }

    cout << s << '\n';
  }

  return 0;
}

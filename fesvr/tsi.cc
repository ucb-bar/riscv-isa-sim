#include "tsi.h"
#include <cstdio>
#include <cstdlib>
#include <cinttypes>

#define NHARTS_MAX 16

void tsi_t::host_thread(void *arg)
{
  tsi_t *tsi = static_cast<tsi_t*>(arg);
  tsi->run();

  while (true)
    tsi->target->switch_to();
}

tsi_t::tsi_t(int argc, char** argv) : htif_t(argc, argv)
{
  target = context_t::current();
  host.init(host_thread, this);
}

tsi_t::~tsi_t(void)
{
}

#define MSIP_BASE 0x2000000

// Interrupt core 0 to make it start executing the program in DRAM
void tsi_t::reset()
{
  uint32_t one = 1;

  write_chunk(MSIP_BASE, sizeof(uint32_t), &one);
}

void tsi_t::push_addr(addr_t addr)
{
  for (int i = 0; i < SAI_ADDR_CHUNKS; i++) {
    in_data.push_back(addr & 0xffffffff);
    addr = addr >> 32;
  }
}

void tsi_t::push_len(addr_t len)
{
  for (int i = 0; i < SAI_LEN_CHUNKS; i++) {
    in_data.push_back(len & 0xffffffff);
    len = len >> 32;
  }
}

void tsi_t::read_chunk(addr_t taddr, size_t nbytes, void* dst)
{
  uint32_t *result = static_cast<uint32_t*>(dst);
  size_t len = nbytes / sizeof(uint32_t);

  in_data.push_back(SAI_CMD_READ);
  push_addr(taddr);
  push_len(len - 1);

  for (size_t i = 0; i < len; i++) {
    while (out_data.empty())
      switch_to_target();
    result[i] = out_data.front();
    fprintf(stdout, "tsi_t::read_chunk result[%d] 0x%" PRIx32 "\n", i, result[i]);
    fflush(stdout);
    out_data.pop_front();
  }
}

void tsi_t::write_chunk(addr_t taddr, size_t nbytes, const void* src)
{
  const uint32_t *src_data = static_cast<const uint32_t*>(src);
  size_t len = nbytes / sizeof(uint32_t);

  in_data.push_back(SAI_CMD_WRITE);
  push_addr(taddr);
  push_len(len - 1);

  for (int i = 0; i < len; i++) {
    fprintf(stdout, "tsi_t::write_chunk src_data[%d] 0x%" PRIx32 "\n", i, src_data[i]);
    fflush(stdout);
  }

  in_data.insert(in_data.end(), src_data, src_data + len);
}

void tsi_t::send_word(uint32_t word)
{
  out_data.push_back(word);
}

uint32_t tsi_t::recv_word(void)
{
  uint32_t word = in_data.front();
  in_data.pop_front();
  return word;
}

bool tsi_t::data_available(void)
{
  return !in_data.empty();
}

void tsi_t::switch_to_host(void)
{
  host.switch_to();
}

void tsi_t::switch_to_target(void)
{
  target->switch_to();
}

void tsi_t::tick(bool out_valid, uint32_t out_bits, bool in_ready)
{
  if (out_valid && out_ready())
    out_data.push_back(out_bits);

  if (in_valid() && in_ready)
    in_data.pop_front();
}

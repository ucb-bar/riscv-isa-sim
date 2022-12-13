// See LICENSE for license details.

#ifndef _RISCV_SOCKETIF_H
#define _RISCV_SOCKETIF_H

#include "config.h"

#ifdef HAVE_BOOST_ASIO

#include <boost/algorithm/string.hpp>
#include <boost/regex.hpp>
#include <boost/asio.hpp>

class socketif_t
{
public:
  socketif_t();
  ~socketif_t();

  std::string rin(std::ostream &sout_); // read input command string
  void wout(); // write output to socket

private:
  // the following are needed for command socket interface
  boost::asio::io_service *io_service_ptr;
  boost::asio::ip::tcp::acceptor *acceptor_ptr;
  std::unique_ptr<boost::asio::ip::tcp::socket> socket_ptr;
  boost::asio::streambuf bout;
};

#endif
#endif

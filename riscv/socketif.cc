// See LICENSE for license details.

#include "socketif.h"

#ifdef HAVE_BOOST_ASIO

#include <iostream>

socketif_t::socketif_t()
{
  try { // create socket server
    using boost::asio::ip::tcp;
    io_service_ptr = new boost::asio::io_service;
    acceptor_ptr = new tcp::acceptor(*io_service_ptr, tcp::endpoint(tcp::v4(), 0));
    // acceptor is created passing argument port=0, so O.S. will choose a free port
    std::string name = boost::asio::ip::host_name();
    std::cout << "Listening for debug commands on " << name.substr(0,name.find('.'))
              << " port " << acceptor_ptr->local_endpoint().port() << " ." << std::endl;
    // at the end, add space and some other character for convenience of javascript .split(" ")
  } catch (std::exception& e) {
    std::cerr << e.what() << std::endl;
    exit(-1);
  }
}

socketif_t::~socketif_t()
{
  delete io_service_ptr;
  delete acceptor_ptr;
}

// read input command string
std::string socketif_t::rin(std::ostream &sout_)
{
  std::string s;
  try {
    socket_ptr.reset(new boost::asio::ip::tcp::socket(*io_service_ptr));
    acceptor_ptr->accept(*socket_ptr); // wait for someone to open connection
    boost::asio::streambuf buf;
    boost::asio::read_until(*socket_ptr, buf, "\n"); // wait for command
    s = boost::asio::buffer_cast<const char*>(buf.data());
    boost::erase_all(s, "\r");  // get rid off any cr and lf
    boost::erase_all(s, "\n");
    // The socket client is a web server and it appends the IP of the computer
    // that sent the command from its web browser.

    // For now, erase the IP if it is there.
    boost::regex re(" ((25[0-5]|2[0-4][0-9]|[01]?[0-9]?[0-9])\\.){3}"
                    "(25[0-5]|2[0-4][0-9]|[01]?[0-9]?[0-9])$");
    s = boost::regex_replace(s, re, (std::string)"");

    // TODO: check the IP against the IP used to upload RISC-V source files
  } catch (std::exception& e) {
    std::cerr << e.what() << std::endl;
  }
  // output goes to socket
  sout_.rdbuf(&bout);
  return s;
}

// write sout_ to socket (via bout)
void socketif_t::wout() {
  try {
    boost::system::error_code ignored_error;
    boost::asio::write(*socket_ptr, bout, boost::asio::transfer_all(), ignored_error);
    socket_ptr->close(); // close the socket after each command input/ouput
    //  This is need to in order to make the socket interface
    //  acessible by HTTP GET via a socket client in a web server.
  } catch (std::exception& e) {
    std::cerr << e.what() << std::endl;
  }
}

#endif

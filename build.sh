#!/bin/bash



function build_spike() {
  echo "Generate Protobuf messages"
  echo $PROTOBUF_INSTALL_DIR
  $PROTOBUF_INSTALL_DIR/bin/protoc arch-state.proto --cpp_out=riscv

  if [ ! -d build ]; then
    mkdir -p build
    cd build
    ../configure --prefix=$RISCV --with-boost=no --with-boost-asio=no --with-boost-regex=no
    cd ..
  fi

  cd build
  make -j64
  make install
}

build_spike | tee build.log

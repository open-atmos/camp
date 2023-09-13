#!/usr/bin/env bash

library_path="../../../"
curr_path=$(pwd)
cd $library_path/json-fortran-6.1.0
rm -r build
mkdir build
mkdir install
cd build
cmake -D SKIP_DOC_GEN:BOOL=TRUE -D CMAKE_INSTALL_PREFIX=$(pwd)/../install ..
make install
cd $curr_path

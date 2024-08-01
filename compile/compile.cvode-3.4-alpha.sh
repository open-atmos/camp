#!/usr/bin/env bash
set -e

# get directory of CAMP suite (and force it to be an absolute path)
case "$#" in
    0) camp_suite_dir=../../ ;;
    1) camp_suite_dir=$1     ;;
esac
camp_suite_dir=`cd ${camp_suite_dir} ; pwd`
initial_dir=$(pwd)
case "${BSC_MACHINE}-loadmodules" in
  "mn5-loadmodules")
  . ./load.modules.camp.sh
  ;;
esac
cd ${camp_suite_dir}/cvode-3.4-alpha
rm -rf build
mkdir build
mkdir install || true
mkdir install/examples || true
cd build
cmake -D CMAKE_BUILD_TYPE=debug \
  -D CMAKE_C_FLAGS_DEBUG="-O3" \
  -D CMAKE_C_FLAGS_RELEASE="-O3" \
  -D MPI_ENABLE:BOOL=TRUE \
  -D KLU_ENABLE:BOOL=TRUE \
  -D CUDA_ENABLE:BOOL=FALSE \
  -D CMAKE_C_COMPILER=$(which mpicc) \
  -D EXAMPLES_ENABLE_CUDA=OFF \
  -D KLU_LIBRARY_DIR=${camp_suite_dir}/SuiteSparse/lib \
  -D KLU_INCLUDE_DIR=${camp_suite_dir}/SuiteSparse/include \
  -D CMAKE_INSTALL_PREFIX=$(pwd)/../install \
  -D EXAMPLES_ENABLE_C=OFF \
  ..
make install
cd $initial_dir
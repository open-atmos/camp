#!/usr/bin/env bash
set -e

case "$#" in
    0) camp_suite_dir=../../ ;;
    1) camp_suite_dir=$1     ;;
esac
camp_suite_dir=`cd ${camp_suite_dir} ; pwd`
initial_dir=$(pwd)
case "${BSC_MACHINE}-loadmodules" in
  "mn5-loadmodules")
  module load cmake
  if module list 2>&1 | grep -q "\<intel\>"; then
    module load intel/2023.2.0
    module load impi/2021.10.0
  else
    module load gcc
    module load openmpi/4.1.5-gcc
  fi
  ;;
esac
cd ${camp_suite_dir}/json-fortran-6.1.0
rm -rf build
mkdir build
mkdir install || true
cd build
cmake -D SKIP_DOC_GEN:BOOL=TRUE -D CMAKE_INSTALL_PREFIX=${camp_suite_dir}/json-fortran-6.1.0/install ..
make install
cd $initial_dir

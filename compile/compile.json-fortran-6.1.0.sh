#!/usr/bin/env bash
set -e

if [ $BSC_MACHINE == "mn5" ]; then
    module load cmake
  if module list 2>&1 | grep -q "\<intel\>"; then
    module load intel/2023.2.0
    module load impi/2021.10.0
  else
    module load gcc
    module load openmpi/4.1.5-gcc
  fi
fi
case "$#" in
    0) camp_suite_dir=../../ ;;
    1) camp_suite_dir=$1     ;;
esac
curr_path=$(pwd)
cd $library_path/json-fortran-6.1.0
rm -rf build
mkdir build
mkdir install || true
cd build
cmake -D SKIP_DOC_GEN:BOOL=TRUE -D CMAKE_INSTALL_PREFIX=$(pwd)/../install ..
make install
cd $curr_path

#!/usr/bin/env bash
set -e
case "$#" in
    0) camp_suite_dir="../../" ;;
    1) camp_suite_dir=$1     ;;
esac
camp_suite_dir=`cd ${camp_suite_dir} ; pwd`
initial_dir=$(pwd)

case "${BSC_MACHINE}-loadmodules" in
    "mn5-loadmodules")
  if module list 2>&1 | grep -q "\<gcc\>"; then
    module load gcc
    module load openmpi/4.1.5-gcc
  else
    module load intel/2023.2.0
    module load impi/2021.10.0
  fi
  module load openblas
  module load cmake
  if module list 2>&1 | grep -q "\<cuda\>"; then
    module unload cuda
  fi
  ;;
esac
cd $camp_suite_dir/SuiteSparse
make purge
make BLAS="-L/usr/lib/x86_64-linux-gnu -lopenblas" LAPACK=""
cd $initial_dir
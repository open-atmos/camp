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
  . ./load.modules.camp.sh
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
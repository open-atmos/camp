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
  . ./load.modules.camp.sh
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

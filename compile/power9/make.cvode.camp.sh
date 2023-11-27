#!/usr/bin/env bash
set -e
source make.camp.sh
compile_cvode(){
  curr_path=$(pwd)
  library_path="../../../"
  if [ -z "$SUITE_SPARSE_CAMP_ROOT" ]; then
    SUITE_SPARSE_CAMP_ROOT=$(pwd)/$library_path/SuiteSparse
  fi
  cd $library_path/cvode-3.4-alpha
  cd build
  make install
  cd $curr_path
}

compile_camp_cvode(){
compile_cvode
make_camp
}
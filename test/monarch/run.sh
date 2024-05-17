#!/usr/bin/env bash
set -e
scriptdir="$(dirname "$0")"
cd "$scriptdir"
make_and_check() {
  initial_dir=$(pwd)
  cd ../../build
  unbuffer make | tee output_make.log
  make_exit_status=${PIPESTATUS[0]}
  if [ $make_exit_status -ne 0 ]; then
    exit 1
  fi
  cd $initial_dir
  if grep -q "Scanning dependencies" ../../build/output_make.log; then
    echo "Changes made by 'make' command."
    python checkGPU.py
  fi
}

make_run() {
  initial_dir=$(pwd)
  cd ../../build
  make -j 4
  cd $initial_dir
  python TestMonarch.py
  #python checkGPU.py
}

make_run
#make_and_check

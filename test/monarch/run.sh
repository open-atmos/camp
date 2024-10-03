#!/usr/bin/env bash
set -e
while getopts 'h' flag; do
  case "${flag}" in
    h) echo "Script to use after compiling to run make and the executable.
More info at the README.md included in this directory" &
    exit 1 ;;
  esac
done
scriptdir="$(dirname "$0")"
cd "$scriptdir"
unset I_MPI_PMI_LIBRARY
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

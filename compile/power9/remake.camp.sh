#!/usr/bin/env bash

make_camp(){
  curr_path=$(pwd)
  cd  ../../build
  if ! make -j ${NUMPROC}; then
    exit
  fi
  cd $curr_path
}

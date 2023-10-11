#!/usr/bin/env bash

make_camp(){
  curr_path=$(pwd)
  cd  ../../build
  make || exit 1
  cd $curr_path
}

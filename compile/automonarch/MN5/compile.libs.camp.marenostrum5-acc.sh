#!/usr/bin/env bash
set -e
initial_dir=$(pwd)
cd ../../
./compile.libs.camp.sh
cd $initial_dir
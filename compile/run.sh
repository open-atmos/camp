#!/usr/bin/env bash
set -e

scriptdir="$(dirname "$0")"
cd "$scriptdir"
source load.modules.camp.sh
if [ ! -d ../build ]; then
  ./compile.libs.camp.sh
fi
cd ../test/monarch
./run.sh

#cd ../build
#make -j 4
#cd test_run/chemistry/cb05cl_ae5
#./test_chemistry_cb05cl_ae5.sh

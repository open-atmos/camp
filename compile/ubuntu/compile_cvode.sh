#!/usr/bin/env bash

library_path="../../../"
if [ "$1" == "from_camp_jobs" ]; then
  library_path="../../../../"
fi

if [ -z "$SUITE_SPARSE_CAMP_ROOT" ]; then
	SUITE_SPARSE_CAMP_ROOT=$(pwd)/$library_path/SuiteSparse
fi

#tar -zxvf camp/cvode-3.4-alpha.tar.gz
cd $library_path/cvode-3.4-alpha
#rm -r build
mkdir build
#rm -rf install
mkdir install
mkdir install/examples
cd build
cmake ..
make install

#./cvode-3.4-alpha/build/examples/cvode/serial/cvRoberts_klu

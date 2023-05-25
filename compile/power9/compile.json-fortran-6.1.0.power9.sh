#!/usr/bin/env bash

library_path="../../../"
if [ "$1" == "from_camp_jobs" ]; then
  library_path="../../../../"
fi

cd $library_path/json-fortran-6.1.0
rm -r build
mkdir build
mkdir install
cd build
cmake -D SKIP_DOC_GEN:BOOL=TRUE -D CMAKE_INSTALL_PREFIX=$(pwd)/../install ..
make install

camp_folder=camp
if [ ! -z "$2" ]; then
  camp_folder=camp_jobs/camp$2
fi

if [ "$1" == "from_camp_jobs" ]; then
  cd ../../$camp_folder/build/compile
fi

#!/usr/bin/env bash

library_path="../../../"
if [ "$1" == "from_camp_jobs" ]; then
  library_path="../../../../"
fi

cd $library_path/SuiteSparse
make purge
LOCAL_MACHINE=CGUZMAN
if [ $BSC_MACHINE == "power" ]; then
  make BLAS="-L${EBROOTOPENBLAS}/lib -lopenblas" LAPACK=""
elif [ $BSC_MACHINE == "mn4" ]; then
  make BLAS="-L${INTEL_HOME}/mkl/lib/intel64 -lmkl_intel_lp64 -lmkl_core -lmkl_intel_thread -lpthread -lm" LAPACK=""
elif [ $LOCAL_MACHINE == "CGUZMAN" ]; then
  make BLAS="-L/usr/lib/x86_64-linux-gnu -lopenblas" LAPACK=""
else
  echo "Unknown architecture"
  exit
fi

export SUITE_SPARSE_CAMP_ROOT=$(pwd)/$library_path/

camp_folder=camp
if [ ! -z "$2" ]; then
  camp_folder=camp_jobs/camp$2
fi

if [ "$1" == "from_camp_jobs" ]; then
  cd ../$camp_folder/build/compile
fi

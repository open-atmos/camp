#!/usr/bin/env bash

library_path="../../../"
curr_path=$(pwd)
cd $library_path/SuiteSparse
make purge
LOCAL_MACHINE=CGUZMAN
if [ $BSC_MACHINE == "power" ]; then
  make BLAS="-L${EBROOTOPENBLAS}/lib -lopenblas" LAPACK=""
elif [ $BSC_MACHINE == "mn4" ]; then
  make BLAS="-L${INTEL_HOME}/mkl/lib/intel64 -lmkl_intel_lp64 -lmkl_core -lmkl_intel_thread -lpthread -lm" LAPACK=""
elif [ $LOCAL_MACHINE==CGUZMAN ]; then
  make BLAS="-L/usr/lib/x86_64-linux-gnu -lopenblas" LAPACK=""
else
  echo "Unknown architecture"
  exit
fi
export SUITE_SPARSE_CAMP_ROOT=$(pwd)/$library_path/
cd $curr_path
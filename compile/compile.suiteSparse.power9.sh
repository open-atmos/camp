#!/usr/bin/env bash
set -e
library_path="../.."
curr_path=$(pwd)

if [ "${BSC_MACHINE}" == "mn5" ] ; then
  if module list 2>&1 | grep -q "\<intel\>"; then
    module load cmake
    module load openblas
  else
    module load cmake
    module load gcc
    module load openmpi/4.1.5-gcc
    module load openblas
  fi
  if module list 2>&1 | grep -q "\<cuda\>"; then
    module unload cuda
  fi
fi
cd $library_path/SuiteSparse
make purge
if [ "${BSC_MACHINE}" == "mn5" ]; then
  make BLAS="-L/usr/lib/x86_64-linux-gnu -I$path_Blas_install/include/ -L$path_Blas_install/lib -Wl,-rpath,$path_Blas_install/OpenBLAS/lib -lopenblas" LAPACK=""
else
  make BLAS="-L/usr/lib/x86_64-linux-gnu -lopenblas" LAPACK=""
fi
export SUITE_SPARSE_CAMP_ROOT=$(pwd)/$library_path/
cd $curr_path
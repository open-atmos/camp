#!/usr/bin/env bash
set -e
library_path="../.."
curr_path=$(pwd)

cd $library_path/OpenBLAS
path_Blas_install=$(pwd)/install/
compile_BLAS(){
make
mkdir install || true
make install PREFIX=$path_Blas_install
}
cd $curr_path
#compile_BLAS

cd $library_path/SuiteSparse
make purge
if [ $BSC_MACHINE == "mn5" ]; then
  module load cmake
  module load gcc
  module load openmpi/4.1.5-gcc
  make BLAS="-L/usr/lib/x86_64-linux-gnu -I$path_Blas_install/include/ -L$path_Blas_install/lib -Wl,-rpath,$path_Blas_install/OpenBLAS/lib -lopenblas" LAPACK=""
  #module load intel
  #module load openmpi
  #make BLAS="-L${INTEL_HOME}/mkl/lib/intel64 -lpthread -lm" LAPACK="" #wrong -openmp and missing BLAS
elif [ $BSC_MACHINE == "mn4" ]; then
  make BLAS="-L${INTEL_HOME}/mkl/lib/intel64 -lmkl_intel_lp64 -lmkl_core -lmkl_intel_thread -lpthread -lm" LAPACK=""
else
  make BLAS="-L/usr/lib/x86_64-linux-gnu -lopenblas" LAPACK=""
fi
cd $curr_path
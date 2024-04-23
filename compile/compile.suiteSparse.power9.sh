#!/usr/bin/env bash
set -e
library_path="../.."
curr_path=$(pwd)

if [ $BSC_MACHINE == "mn5" ] ; then
  module load cmake
  module load gcc
  module load openmpi/4.1.5-gcc
  if module list 2>&1 | grep -q "\<cuda\>"; then
    module unload cuda
  fi
fi
cd /gpfs/projects/bsc32/bsc032815/gpupartmc/OpenBLAS
path_Blas_install=/gpfs/projects/bsc32/bsc032815/gpupartmc/OpenBLAS/install
compile_BLAS(){
make
mkdir install
make install PREFIX=$path_Blas_install
}
if [ ! -d /gpfs/projects/bsc32/bsc032815/gpupartmc/OpenBLAS/install ]; then
  echo "Directory OpenBLAS/install does not exists, installing OpenBlas."
  compile_BLAS
fi
cd $curr_path
cd $library_path/SuiteSparse
make purge
if [ $BSC_MACHINE == "mn5" ] || [$BSC_MACHINE == "power"] ; then
  make BLAS="-L/usr/lib/x86_64-linux-gnu -I$path_Blas_install/include/ -L$path_Blas_install/lib -Wl,-rpath,$path_Blas_install/OpenBLAS/lib -lopenblas" LAPACK=""
elif [ $BSC_MACHINE == "mn4" ]; then
  make BLAS="-L${INTEL_HOME}/mkl/lib/intel64 -lmkl_intel_lp64 -lmkl_core -lmkl_intel_thread -lpthread -lm" LAPACK=""
else
  make BLAS="-L/usr/lib/x86_64-linux-gnu -lopenblas" LAPACK=""
fi
export SUITE_SPARSE_CAMP_ROOT=$(pwd)/$library_path/
cd $curr_path
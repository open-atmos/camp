#!/usr/bin/env bash
set -e

library_path="../../"
curr_path=$(pwd)

if [ $BSC_MACHINE == "mn5" ]; then
  module load cmake
  module load gcc
  module load openmpi/4.1.5-gcc
elif [ $BSC_MACHINE == "power" ]; then
  module load GCC/7.3.0-2.30
  module load OpenMPI/3.1.0-GCC-7.3.0-2.30
  module load JasPer/1.900.1-foss-2018b
  module load netCDF/4.6.1-foss-2018b
  module load netCDF-Fortran/4.4.4-foss-2018b
  module load ESMF/6.3.0rp1-foss-2018b
  module load CMake/3.15.3-GCCcore-7.3.0
  module load OpenBLAS/0.3.1-GCC-7.3.0-2.30
  module load CUDA/10.1.105-ES
  module load Python/3.7.0-foss-2018b
  module load matplotlib/3.1.1-foss-2018b-Python-3.7.0
fi

if [ -z "$SUITE_SPARSE_CAMP_ROOT" ]; then
	SUITE_SPARSE_CAMP_ROOT=$(pwd)/$library_path/SuiteSparse
fi

cd $library_path/cvode-3.4-alpha
rm -rf build
mkdir build
mkdir install || true
mkdir install/examples || true
cd build
cmake -D CMAKE_BUILD_TYPE=debug \
-D CMAKE_C_FLAGS_DEBUG="-O3" \
-D MPI_ENABLE:BOOL=TRUE \
-D KLU_ENABLE:BOOL=TRUE \
-D CUDA_ENABLE:BOOL=FALSE \
-D CMAKE_C_COMPILER=$(which mpicc) \
-D EXAMPLES_ENABLE_CUDA=OFF \
-D KLU_LIBRARY_DIR=$SUITE_SPARSE_CAMP_ROOT/lib \
-D KLU_INCLUDE_DIR=$SUITE_SPARSE_CAMP_ROOT/include \
-D CMAKE_INSTALL_PREFIX=$(pwd)/../install \
-D EXAMPLES_ENABLE_C=OFF \
..
make install
cd $curr_path
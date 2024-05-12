#!/usr/bin/env bash
set -e

library_path="../../"
curr_path=$(pwd)

if [ "${BSC_MACHINE}" == "mn5" ]; then
    module load cmake
  if module list 2>&1 | grep -q "\<intel\>"; then
    module load intel/2023.2.0
    module load impi/2021.10.0
  else
    echo
    module load gcc
    module load openmpi/4.1.5-gcc
  fi
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
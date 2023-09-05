#!/usr/bin/env bash

library_path="../../../"

if [ -z "$SUITE_SPARSE_CAMP_ROOT" ]; then
	SUITE_SPARSE_CAMP_ROOT=$(pwd)/$library_path/SuiteSparse
fi

#tar -zxvf camp/cvode-3.4-alpha.tar.gz
cd $library_path/cvode-3.4-alpha
rm -r build
mkdir build
rm -rf install
mkdir install
mkdir install/examples
cd build
cmake -D CMAKE_BUILD_TYPE=debug \
-D CMAKE_C_FLAGS="-O3" \
-D MPI_ENABLE:BOOL=TRUE \
-D KLU_ENABLE:BOOL=TRUE \
-D CUDA_ENABLE:BOOL=FALSE \
-D CAMP_PROFILING=ON \
-D CMAKE_C_COMPILER=$(which mpicc) \
-D EXAMPLES_ENABLE_CUDA=OFF \
-D KLU_LIBRARY_DIR=$SUITE_SPARSE_CAMP_ROOT/lib \
-D KLU_INCLUDE_DIR=$SUITE_SPARSE_CAMP_ROOT/include \
-D CMAKE_INSTALL_PREFIX=$(pwd)/../install \
-D EXAMPLES_ENABLE_C=OFF \
..
#-D EXAMPLES_INSTALL_PATH=$(pwd)/../install/examples .. \
#-D CMAKE_CXX_FLAGS="-O3 -lcudart -lcublas" \
#-D CMAKE_C_FLAGS ="-O3 -lcudart -lcublas" \
#-D CMAKE_CUDA_FLAGS="-Xcompiler="-fpermissive" -lcudart -lcublas" \
#-D EXAMPLES_ENABLE_C=OFF \

#make -j 4 #not working
make install

#./cvode-3.4-alpha/build/examples/cvode/serial/cvRoberts_klu

#!/usr/bin/env bash
set -e

library_path="../../"
curr_path=$(pwd)

source load.modules.camp.sh
export JSON_FORTRAN_HOME=$(pwd)/$library_path/json-fortran-6.1.0/install/jsonfortran-gnu-6.1.0
export SUNDIALS_HOME=$(pwd)/$library_path/cvode-3.4-alpha/install
export SUITE_SPARSE_HOME=$(pwd)/$library_path/SuiteSparse

cd ../
rm -rf build
mkdir build
cd build

if [ "${BSC_MACHINE}" == "mn5" ]; then
  mpifort=$(which mpifort)
fi
cmake -D CMAKE_C_COMPILER=$(which mpicc) \
-D CMAKE_BUILD_TYPE=debug \
-D CMAKE_C_FLAGS_DEBUG="-std=c99 -O3 -g" \
-D CMAKE_Fortran_FLAGS_DEBUG="-g -O3" \
-D CMAKE_C_FLAGS_RELEASE="-std=c99 -O3" \
-D CMAKE_Fortran_FLAGS_RELEASE="" \
-D CMAKE_Fortran_COMPILER=$mpifort \
-D ENABLE_DEBUG=OFF \
-D FAILURE_DETAIL=OFF \
-D ENABLE_MPI=ON \
-D ENABLE_GPU=OFF \
-D ENABLE_CAMP_PROFILE_SOLVING=ON \
-D ENABLE_GSL:BOOL=FALSE \
-D ENABLE_NETCDF=ON \
..

ln -sf ../test/monarch/settings
ln -sf ../test/monarch/out
make -j 4 VERBOSE=1
cd $curr_path

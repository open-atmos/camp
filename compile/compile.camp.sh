#!/usr/bin/env bash
set -e

scriptdir="$(dirname "$0")"
cd "$scriptdir"
# get directory of CAMP suite (and force it to be an absolute path)
case "$#" in
    0) camp_suite_dir=../../ ;;
    1) camp_suite_dir=$1     ;;
esac
camp_suite_dir=`cd ${camp_suite_dir} ; pwd`
initial_dir=$(pwd)

case "${BSC_MACHINE}-loadmodules" in
    "mn5-loadmodules")
  if ! module list 2>&1 | grep -q "\<python\>"; then
    source load.modules.camp.sh
  fi
  ;;
esac
cd ${camp_suite_dir}/camp
if [ "${BSC_MACHINE}" == "mn5" ]; then
  if module list 2>&1 | grep -q "\<intel\>"; then
    if module list 2>&1 | grep -q "\<nvidia-hpc-sdk\>"; then
      #failing maybe because it misses the same on CMakeLists
      mpifort=$(which mpifort)
    else
      mpifort=$(which mpiifort)
    fi
  else
    mpifort=$(which mpifort)
  fi
fi
export JSON_FORTRAN_HOME=${camp_suite_dir}/json-fortran-6.1.0/install/jsonfortran-gnu-6.1.0
export SUNDIALS_HOME=${camp_suite_dir}/cvode-3.4-alpha/install
export SUITE_SPARSE_HOME=${camp_suite_dir}/SuiteSparse
cd ${camp_suite_dir}/camp
rm -rf build
mkdir build
cd build
cmake -D CMAKE_C_COMPILER=$(which mpicc) \
  -D CMAKE_BUILD_TYPE=debug \
  -D CMAKE_C_FLAGS_DEBUG="-O3 -g" \
  -D CMAKE_Fortran_FLAGS_DEBUG="-g -O3" \
  -D CMAKE_C_FLAGS_RELEASE="-O3" \
  -D CMAKE_Fortran_FLAGS_RELEASE="" \
  -D CMAKE_Fortran_COMPILER=$mpifort \
  -D ENABLE_DEBUG=OFF \
  -D FAILURE_DETAIL=OFF \
  -D ENABLE_MPI=ON \
  -D ENABLE_GPU=ON \
  -D ENABLE_PROFILE_SOLVING=ON \
  -D ENABLE_GSL:BOOL=FALSE \
  -D ENABLE_NETCDF=ON \
  ..

ln -sf ${camp_suite_dir}/camp/test/monarch/settings
ln -sf ${camp_suite_dir}/camp/test/monarch/out
make -j 4 VERBOSE=1
cd $initial_dir

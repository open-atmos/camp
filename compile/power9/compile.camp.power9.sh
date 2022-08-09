#!/usr/bin/env bash
#MONARCH P9 compilation
module load bsc/commands
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
#module load GSL/2.4-GCCcore-7.3.0

export NETCDF_FORTRAN_HOME=${EBROOTNETCDFMINFORTRAN}
export NETCDF_HOME=${EBROOTNETCDF}

export NETCDF_FORTRAN_LIB="/gpfs/projects/bsc32/software/rhel/7.5/ppc64le/POWER9/software/netCDF-Fortran/4.4.4-foss-2018b/lib/libnetcdff.so"
export NETCDF_INCLUDE_DIR="/gpfs/projects/bsc32/software/rhel/7.5/ppc64le/POWER9/software/netCDF/4.6.1-foss-2018b/include"

relative_path="../../../"
if [ "$1" == "from_camp_jobs" ]; then
  relative_path="../../../../"
fi

export SUNDIALS_HOME=$(pwd)/$relative_path/cvode-3.4-alpha/install
export SUITE_SPARSE_HOME=$(pwd)/$relative_path/SuiteSparse
export JSON_FORTRAN_HOME=$(pwd)/$relative_path/json-fortran-6.1.0/install/jsonfortran-gnu-6.1.0
#export GSL_HOME=${GSL_DIR}

cd ../../
#rm -rf build
mkdir build
cd build

cmake -D CMAKE_C_COMPILER=$(which mpicc) \
-D CMAKE_BUILD_TYPE=release \
-D CMAKE_C_FLAGS_DEBUG="" \
-D CMAKE_Fortran_FLAGS_DEBUG="-g" \
-D CMAKE_C_FLAGS_RELEASE="-std=c99" \
-D CMAKE_Fortran_FLAGS_RELEASE="" \
-D CMAKE_Fortran_COMPILER=$(which mpifort) \
-D ENABLE_JSON=ON \
-D ENABLE_SUNDIALS=ON \
-D ENABLE_TESTS=OFF \
-D ENABLE_DEBUG=OFF \
-D FAILURE_DETAIL=OFF \
-D ENABLE_CXX=OFF \
-D ENABLE_MPI=ON \
-D ENABLE_GPU=ON \
-D ENABLE_GSL:BOOL=FALSE \
-D ENABLE_RESET_JAC_SOLVING=ON \
-D ENABLE_DEBUG_GPU=ON \
-D ENABLE_ODE_GPU=ON \
-D ENABLE_NETCDF=OFF \
-D ENABLE_BOOTCAMP=OFF \
-D ENABLE_DATA=OFF \
-D use_maxrregcount64=ON \
..

make -j 4 VERBOSE=1
cd ../compile/power9
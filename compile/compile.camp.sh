#!/usr/bin/env bash
set -e

relative_path="../../"
curr_path=$(pwd)

mpifort=$(which mpifort)
export JSON_FORTRAN_HOME=$(pwd)/$relative_path/json-fortran-6.1.0/install/jsonfortran-gnu-6.1.0
if [ $BSC_MACHINE == "mn5" ]; then
  if ! module list 2>&1 | grep -q "\<netcdf/c-4.9.2_fortran-4.6.1_cxx4-4.3.1-gcc-ompi\>"; then #mod not load
    module load cmake
    module load gcc/11.4.0
    module load openmpi/4.1.5-gcc
    module load mkl
    module load hdf5/1.14.1-2-gcc-ompi
    module load python/3.12.1-gcc
    module load cuda
    module load pnetcdf/1.12.3-gcc-ompi
    module load netcdf/c-4.9.2_fortran-4.6.1_cxx4-4.3.1-gcc-ompi
  fi
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
  export NETCDF_FORTRAN_HOME=${EBROOTNETCDFMINFORTRAN}
  export NETCDF_HOME=${EBROOTNETCDF}
  export NETCDF_FORTRAN_LIB="/gpfs/projects/bsc32/software/rhel/7.5/ppc64le/POWER9/software/netCDF-Fortran/4.4.4-foss-2018b/lib/libnetcdff.so"
  export NETCDF_INCLUDE_DIR="/gpfs/projects/bsc32/software/rhel/7.5/ppc64le/POWER9/software/netCDF/4.6.1-foss-2018b/include"
elif [ $BSC_MACHINE == "mn4" ]; then
  export JSON_FORTRAN_HOME=$(pwd)/$relative_path/json-fortran-6.1.0/install/jsonfortran-intel-6.1.0
  mpifort=$(which mpiifort) #intel fortran
  module load cmake
  module load gsl
  module load jasper/1.900.1
  module load netcdf/4.4.1.1
  module load hdf5/1.8.19
  module load libpng/1.5.13
else
  if ! command -v mpicc &> /dev/null; then
      echo "MPI is not installed. Installing..."
      sudo apt update
      sudo apt install -y mpi-default-dev
  fi
fi
export SUNDIALS_HOME=$(pwd)/$relative_path/cvode-3.4-alpha/install
export SUITE_SPARSE_HOME=$(pwd)/$relative_path/SuiteSparse

cd ../
rm -rf build
mkdir build
cd build

cmake -D CMAKE_C_COMPILER=$(which mpicc) \
-D CMAKE_BUILD_TYPE=debug \
-D CMAKE_C_FLAGS_DEBUG="-std=c99 -g -O3" \
-D CMAKE_Fortran_FLAGS_DEBUG="-g -O3" \
-D CMAKE_C_FLAGS_RELEASE="-std=c99 -O3" \
-D CMAKE_Fortran_FLAGS_RELEASE="" \
-D CMAKE_Fortran_COMPILER=$mpifort \
-D DISABLE_TESTS=ON \
-D ENABLE_DEBUG=OFF \
-D FAILURE_DETAIL=OFF \
-D ENABLE_MPI=ON \
-D ENABLE_GPU=ON \
-D ENABLE_GSL:BOOL=FALSE \
-D ENABLE_NETCDF=ON \
..

ln -sf ../test/monarch/settings
ln -sf ../test/monarch/out
make -j 4 VERBOSE=1
cd $curr_path

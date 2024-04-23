#!/usr/bin/env bash
set -e

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

cd ../test/monarch
./run.sh

#cd ../build
#make -j 4
#cd test_run/chemistry/cb05cl_ae5
#./test_chemistry_cb05cl_ae5.sh

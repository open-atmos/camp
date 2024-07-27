#!/usr/bin/env bash

if [ "${BSC_MACHINE}" == "mn5" ]; then
  #module use /apps/GPP/modulefiles/applications /apps/GPP/modulefiles/compilers /apps/GPP/modulefiles/tools /apps/GPP/modulefiles/libraries /apps/GPP/modulefiles/environment /apps/GPP/modulefiles/libs #It use an utility non-existent in default modules: NetCDF-Python
  module load bsc
  if [ "$1" == "gcc" ]; then
    module load gcc
    module load openmpi/4.1.5-gcc
    module load mkl
    if module --raw --redirect show cuda 2>/dev/null | grep -q cuda ; then
      module load hdf5/1.14.1-2-gcc-ompi
      module load pnetcdf/1.12.3-gcc-ompi
      module load netcdf/c-4.9.2_fortran-4.6.1_cxx4-4.3.1-gcc-ompi
      module load cuda
    else
      module load hdf5/1.14.1-2-gcc-openmpi
      module load pnetcdf/1.12.3-gcc-openmpi
      module load netcdf/c-4.9.2_fortran-4.6.1_cxx4-4.3.1_hdf5-1.14.1-2_pnetcdf-1.12.3-gcc-openmpi
    fi
    module load python/3.12.1-gcc
  else
    if module --raw --redirect show cuda 2>/dev/null | grep -q cuda ; then
      module load intel/2023.2.0
      module load impi/2021.10.0
      module load mkl/2023.2.0
      module load hdf5/1.14.1-2
      module load pnetcdf/1.12.3
      module load netcdf
      module load cuda
    else
      module load intel/2023.2.0
      module load impi/2021.10.0
      module load mkl/2023.2.0
      module load hdf5/1.14.1-2
      module load pnetcdf/1.12.3
      module load netcdf/2023-06-14
    fi
    module load python
  fi
  module load cmake
fi
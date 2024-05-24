#!/usr/bin/env bash

if [ "${BSC_MACHINE}" == "mn5" ]; then
  module load bsc
  if module list 2>&1 | grep -q "\<gcc\>"; then
    module load gcc
    module load openmpi/4.1.5-gcc
    module load mkl
    if module --raw --redirect show cuda | grep -q cuda ; then
      module load cuda
      module load hdf5/1.14.1-2-gcc-ompi
      module load pnetcdf/1.12.3-gcc-ompi
      module load netcdf/c-4.9.2_fortran-4.6.1_cxx4-4.3.1-gcc-ompi
    else
      module load hdf5/1.14.1-2-gcc-openmpi
      module load pnetcdf/1.12.3-gcc-openmpi
      module load netcdf/c-4.9.2_fortran-4.6.1_cxx4-4.3.1_hdf5-1.14.1-2_pnetcdf-1.12.3-gcc-openmpi
    fi
    module load python/3.12.1-gcc
  else
    module load intel/2023.2.0
    module load impi/2021.10.0
    module load mkl/2023.2.0
    module load hdf5/1.14.1-2
    module load pnetcdf/1.12.3
    if module --raw --redirect show cuda | grep -q cuda ; then
      module load netcdf
      module load cuda
    else
      module load netcdf/2023-06-14
    fi
    module load python
  fi
  module load cmake
fi
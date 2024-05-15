#!/usr/bin/env bash

if [ "${BSC_MACHINE}" == "mn5" ]; then
  if ! module list 2>&1 | grep -q "\<python\>"; then #mod not load
    module load bsc
    module load cmake
    if module list 2>&1 | grep -q "\<intel\>"; then
      module load intel/2023.2.0
      module load impi/2021.10.0
      module load mkl/2023.2.0
      module load hdf5/1.14.1-2
      module load pnetcdf/1.12.3
    if [[ "${HOSTNAME}" == *"alogin"* ]]; then
     # module load netcdf
      #module load intel/2023.2.0-sycl_cuda
      module load cuda
      module load netcdf
    else
      module load netcdf/2023-06-14
    fi
    module load python
    else
      module load cmake
      module load gcc
      module load openmpi/4.1.5-gcc
      module load mkl
      if [[ "${HOSTNAME}" == *"alogin"* ]]; then
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
    fi
  fi
fi
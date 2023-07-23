#!/usr/bin/env bash

if [ ! -z ${BSC_MACHINE+x} ]; then
if [ $BSC_MACHINE == "power" ]; then
#MONARCH P9 compilation
module load GCC/7.3.0-2.30
module load OpenMPI/3.1.0-GCC-7.3.0-2.30
module load bsc/commands
module load JasPer/1.900.1-foss-2018b
module load netCDF/4.6.1-foss-2018b
module load netCDF-Fortran/4.4.4-foss-2018b
module load ESMF/6.3.0rp1-foss-2018b
module load CMake/3.15.3-GCCcore-7.3.0
module load OpenBLAS/0.3.1-GCC-7.3.0-2.30
module load CUDA/10.1.105-ES
export NETCDF_FORTRAN_HOME=${EBROOTNETCDFMINFORTRAN}
export NETCDF_HOME=${EBROOTNETCDF}
export NETCDF_FORTRAN_LIB="/gpfs/projects/bsc32/software/rhel/7.5/ppc64le/POWER9/software/netCDF-Fortran/4.4.4-foss-2018b/lib/libnetcdff.so"
export NETCDF_INCLUDE_DIR="/gpfs/projects/bsc32/software/rhel/7.5/ppc64le/POWER9/software/netCDF/4.6.1-foss-2018b/include"
elif [ $BSC_MACHINE == "mn4" ]; then
  echo "mn4"
  module load gsl
  module load jasper/1.900.1
  module load netcdf/4.4.1.1
  module load hdf5/1.8.19
  module load libpng/1.5.13
else
  echo "Unknown architecture"
  exit
fi
fi

if [ "$1" == "from_camp_jobs" ]; then
  echo "Running from_camp_jobs folder"
fi

camp_folder=camp
if [ ! -z "$2" ]; then
  echo "Running job" $2
fi

./compile.json-fortran-6.1.0.power9.sh $1 $2
./compile.suiteSparse.power9.sh $1 $2
./compile.cvode-3.4-alpha.power9.sh $1 $2
./compile.camp.sh $1 $2


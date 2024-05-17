#!/bin/sh

# get directory of CAMP suite (and force it to be an absolute path)
case "$#" in
    0) camp_suite_dir=../../ ;;
    1) camp_suite_dir=$1     ;;
esac
camp_suite_dir=`cd ${camp_suite_dir} ; pwd`

# current directory
initial_dir=`pwd`

# load modules for boxmodel for different HPC machines ("-loadmodules" suffix used in install.sh)
case "${BSC_MACHINE}-loadmodules" in
    "mn5-loadmodules")
	module purge
	module load cmake
	module load gcc
	module load openmpi/4.1.5-gcc
	module load mkl
	module load hdf5/1.14.1-2-gcc-openmpi
	module load python/3.12.1-gcc
	module load pnetcdf/1.12.3-gcc-openmpi
	module load netcdf/c-4.9.2_fortran-4.6.1_cxx4-4.3.1_hdf5-1.14.1-2_pnetcdf-1.12.3-gcc-openmpi
	;;
esac

# compile CAMP
cd ${camp_suite_dir}/camp/build
make boxmodel_v2

# come back to initial directory
cd ${initial_dir}

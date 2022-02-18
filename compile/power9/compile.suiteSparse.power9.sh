#module purge
#BSC modules
#module load gcc/6.4.0
#module load openmpi/3.0.0
#module load hdf4/4.2.13
#module load hdf5/1.8.20
#module load pnetcdf/1.9.0
#module load netcdf/4.4.1.1
#module load lapack/3.8.0
cd ../../../SuiteSparse
make BLAS="-L${EBROOTOPENBLAS}/lib -lopenblas" LAPACK=""
export SUITE_SPARSE_CAMP_ROOT=$(pwd)/../../../
cd ../camp/build/compile

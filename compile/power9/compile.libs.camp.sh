
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
module load CUDA/10.1.105-EShttps://rediris.zoom.us/j/87861333712


export NETCDF_FORTRAN_HOME=${EBROOTNETCDFMINFORTRAN}
export NETCDF_HOME=${EBROOTNETCDF}

export NETCDF_FORTRAN_LIB="/gpfs/projects/bsc32/software/rhel/7.5/ppc64le/POWER9/software/netCDF-Fortran/4.4.4-foss-2018b/lib/libnetcdff.so"
export NETCDF_INCLUDE_DIR="/gpfs/projects/bsc32/software/rhel/7.5/ppc64le/POWER9/software/netCDF/4.6.1-foss-2018b/include"


#cd SuiteSparse
#make BLAS="-L${INTEL_HOME}/mkl/lib/intel64 -lmkl_intel_lp64 -lmkl_core -lmkl_intel_thread -lpthread -lm" LAPACK=""
#export SUITE_SPARSE_CAMP_ROOT=$(pwd)
#cd ..
#if [ -z "$SUITE_SPARSE_CAMP_ROOT" ]; then
#        SUITE_SPARSE_CAMP_ROOT=$(pwd)/SuiteSparse
#fi
#tar -zxvf camp/cvode-3.4-alpha.tar.gz

#cd cvode-3.4-alpha
#mkdir build
#mkdir install
#mkdir install/examples
#cd build
#cmake -D CMAKE_BUILD_TYPE=release -D MPI_ENABLE:BOOL=TRUE -D KLU_ENABLE:BOOL=TRUE -D KLU_LIBRARY_DIR=$SUITE_SPARSE_CAMP_ROOT/lib -D KLU_INCLUDE_DIR=$SUITE_SPARSE_CAMP_ROOT/include -D CMAKE_INSTALL_PREFIX=$(pwd)/../install -D EXAMPLES_INSTALL_PATH=$(pwd)/../install/examples ..
#make install
#cd ../..

#cd json-fortran-6.1.0
#mkdir build
#mkdir install
#cd build
#cmake -D SKIP_DOC_GEN:BOOL=TRUE -D CMAKE_INSTALL_PREFIX=$(pwd)/../install ..
#make install
#cd ../..

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
./compile.camp.power9.sh $1 $2

#export SUNDIALS_HOME=$(pwd)/cvode-3.4-alpha/install
#export SUITE_SPARSE_HOME=$(pwd)/SuiteSparse
#export JSON_FORTRAN_HOME=$(pwd)/json-fortran-6.1.0/install/jsonfortran-gnu-6.1.0
#export GSL_HOME=${GSL_DIR}
#cd camp
#rm -fr build
#mkdir build
#cd build
#cmake -D CMAKE_C_COMPILER=$(which mpicc) -D CMAKE_Fortran_COMPILER=$(which mpiifort) -D CMAKE_BUILD_TYPE=release -D CMAKE_C_FLAGS_DEBUG="-std=c99 -g -traceback -fp-stack-check" -D CMAKE_C_FLAGS_RELEASE="-std=c99 -O3 -DNDEBUG" -D CMAKE_Fortran_FLAGS_DEBUG="-g -traceback" -D ENABLE_JSON=ON -D ENABLE_SUNDIALS=ON -D ENABLE_MPI=ON -D ENABLE_GSL=ON -D ENABLE_TESTS=ON ..
#make
#cd ../..


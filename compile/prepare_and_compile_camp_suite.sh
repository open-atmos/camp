#!/usr/bin/env bash

####################################################
#        PREPARE AND COMPILE CAMP SUITE
####################################################
#
# This script manages the preparation of the directories 
# of the CAMP suite components (namely json-fortran,
# SuiteSparse, cvode, and camp itself) and their compilation
# on the BSC's HPC machines (e.g. MN5gpp, MN5acc).
# For each CAMP component, the user specifies the desired
# instructions, namely: "1" for copying the component directory
# from a source directory, "2" for cloning the component directory
# from the corresponding git URL, "3" for compiling the component.
# Note that flags "1" and "2" are exclusive.
# The default directory structure created (in /home/bsc/`whoami`) is:
#   camp_suite
#      |-- camp
#      |-- cvode
#      |-- json-fortran-6.1.0
#      |-- SuiteSparse
#
# Versions:
#       08/05/2024: Creation (Herve Petetin, herve.petetin@bsc.es)
#
####################################################

# indicate the library path where all CAMP suite components 
# (json-fortran, SuiteSparse, cvode, camp) will be located
camp_suite_dir=/home/bsc/`whoami`/camp_suite

# indicate the source directory from where to copy the CAMP component(s)
# (useful only if flag "1" activated)
source_dir=/gpfs/projects/bsc32/bsc032815/gpupartmc

# select if you want to print the compilation outputs, and if not, if you want to clean the log files
verbose=0
clean=1




# current directory
initial_dir=`pwd`

# define colors for bash
RED='\033[0;31m'  #(red)
NC='\033[0m'      #(default color)

# usage instructions and/or other information  
echo -e "$RED|==========================================$NC"
echo -e "$RED|       PREPARE AND COMPILE CAMP SUITE$NC"
echo -e "$RED|==========================================$NC"
if [ "$#" -eq 0 ] ; then
    echo -e "$RED| Usage:  ./prepare_and_compile_camp_suite.sh <lib>:<flag(s)> [...]$NC"
    echo -e "$RED|        with lib the CAMP component library(ies) among json-fortran, SuiteSparse, cvode, camp$NC"
    echo -e "$RED|             flag=1 for copying the directory from another one,$NC"
    echo -e "$RED|             flag=2 for git-cloning from the web,$NC"
    echo -e "$RED|             flag=3 for compiling the library$NC"
    echo -e "$RED| Examples:$NC"
    echo -e "$RED#(copy/compile json-fortran and clone/compile SuiteSparse)$NC"
    echo "./prepare_and_compile_camp_suite.sh json-fortran:13 SuiteSparse:23"
    echo -e "$RED#(clone/compile all components)$NC"
    echo "./prepare_and_compile_camp_suite.sh json-fortran:23 SuiteSparse:23 cvode:23 camp:23"
    echo -e "$RED#(clone/compile json-fortran, SuiteSparse and cvode, and only compile camp)$NC"
    echo "./prepare_and_compile_camp_suite.sh json-fortran:23 SuiteSparse:23 cvode:23 camp:3"
    exit 0
fi
echo -e "$RED| Inputs:$NC"
echo -e "$RED|    components: $@$NC"
echo -e "$RED|    camp_suite_dir: $camp_suite_dir$NC"


# indicate if this script is working on the current HPC machine, exit if not
case "${BSC_MACHINE}" in
    "mn5") echo -e "$RED| HPC machine (${BSC_MACHINE}) handled by this script $NC" ;;
    *) echo -e "$RED| HPC machine (${BSC_MACHINE}) not yet handled by this script. Exiting...$NC"
       exit 0
       ;;
esac

# create directory for the camp suite
echo -e "$RED| Create the camp suite directory$NC"
mkdir -p ${camp_suite_dir}


#########################################################
#  prepare and compile json-fortran
#########################################################
function prepare_and_compile_jsonfortran()
{
    # prepare directories
    if [[ $flags == *1* ]] ; then
	cp -rf ${source_dir}/json-fortran-6.1.0 ${camp_suite_dir}	
    elif [[ $flags == *2* ]] ; then
	cd ${camp_suite_dir}
        git clone https://github.com/jacobwilliams/json-fortran.git json-fortran-6.1.0
        cd ${camp_suite_dir}/json-fortran-6.1.0
	git checkout tags/6.1.0
    fi

    if [[ $flags == *3* ]] ; then
	# load modules
	case "${BSC_MACHINE}" in
	    "mn5")
		module purge
		module load cmake
		if module list 2>&1 | grep -q "\<intel\>"; then
		    module load oneapi
		else
		    module load gcc
		    module load openmpi/4.1.5-gcc
		fi
		;;
	esac
	
	# compile
	cd ${camp_suite_dir}/json-fortran-6.1.0
	rm -rf build
	mkdir build
	mkdir install || true
	cd build
	cmake -D SKIP_DOC_GEN:BOOL=TRUE -D CMAKE_INSTALL_PREFIX=${camp_suite_dir}/json-fortran-6.1.0/install ..
	make install
    fi
}



#########################################################
#  prepare and compile SuiteSparse
#########################################################
function prepare_and_compile_suitesparse()
{
    # prepare directories
    if [[ $flags == *1* ]] ; then
	cp -rf ${source_dir}/SuiteSparse ${camp_suite_dir}
    fi
    if [[ $flags == *2* ]] ; then
	cd ${camp_suite_dir}
	git clone https://github.com/DrTimothyAldenDavis/SuiteSparse.git
	cd SuiteSparse
	git checkout v5.1.0
	suitesparse_patch=./suitesparse_patch.patch
	echo 'diff --git a/SuiteSparse_config/SuiteSparse_config.mk b/SuiteSparse_config/SuiteSparse_config.mk' > ${suitesparse_patch}
	echo 'index bb26ac3a3..a3e63d66f 100644' >> ${suitesparse_patch}
	echo '--- a/SuiteSparse_config/SuiteSparse_config.mk' >> ${suitesparse_patch}
	echo '+++ b/SuiteSparse_config/SuiteSparse_config.mk' >> ${suitesparse_patch}
	echo '@@ -115,7 +115,7 @@ SUITESPARSE_VERSION = 5.1.0' >> ${suitesparse_patch}
	echo '             CC = icc -D_GNU_SOURCE' >> ${suitesparse_patch}
	echo '             CXX = $(CC)' >> ${suitesparse_patch}
	echo '             CFOPENMP = -qopenmp -I$(MKLROOT)/include' >> ${suitesparse_patch}
	echo '-	    LDFLAGS += -openmp' >> ${suitesparse_patch}
	echo '+	    LDFLAGS += -qopenmp' >> ${suitesparse_patch}
	echo '             LDLIBS += -lm -lirc' >> ${suitesparse_patch}
	echo '         endif' >> ${suitesparse_patch}
	echo '         ifneq ($(shell which ifort 2>/dev/null),)' >> ${suitesparse_patch}
	echo '' >> ${suitesparse_patch}
	git apply suitesparse_patch.patch
    fi


    if [[ $flags == *3* ]] ; then
	# load modules
	case "${BSC_MACHINE}" in
	    "mn5")
		module purge
		module load cmake
		module load openblas
		if module list 2>&1 | grep -q "\<intel\>"; then
		    module load oneapi
		else
		    module load gcc
		    module load openmpi/4.1.5-gcc
		fi
		if module list 2>&1 | grep -q "\<cuda\>"; then
		    module unload cuda
		fi
		;;
	esac
	
	# compile
	cd ${camp_suite_dir}/SuiteSparse
	make purge
	if [ "${BSC_MACHINE}" == "mn5" ] || ["${BSC_MACHINE}" == "power"] ; then
	    make BLAS="-L/usr/lib/x86_64-linux-gnu -I$path_Blas_install/include/ -L$path_Blas_install/lib -Wl,-rpath,$path_Blas_install/OpenBLAS/lib -lopenblas" LAPACK=""
	elif [ "${BSC_MACHINE}" == "mn4" ]; then
	    make BLAS="-L${INTEL_HOME}/mkl/lib/intel64 -lmkl_intel_lp64 -lmkl_core -lmkl_intel_thread -lpthread -lm" LAPACK=""
	else
	    make BLAS="-L/usr/lib/x86_64-linux-gnu -lopenblas" LAPACK=""
	fi
    fi
}
    


#########################################################
#  prepare and compile cvode
#########################################################
function prepare_and_compile_cvode()
{
    # prepare directories
    if [[ $flags == *1* ]] ; then
        cp -rf ${source_dir}/cvode ${camp_suite_dir}
    fi
    if [[ $flags == *2* ]] ; then
	cd ${camp_suite_dir}
	git clone https://github.com/mattldawson/cvode.git cvode
    fi


    if [[ $flags == *3* ]] ; then
        # load modules
	case "${BSC_MACHINE}" in
	    "mn5")
		module purge
		module load cmake
		if module list 2>&1 | grep -q "\<intel\>"; then
		    module load oneapi
		else
		    module load gcc
		    module load openmpi/4.1.5-gcc
		fi
		;;
	    "power")
		module purge
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
	esac
	
	# compile
	if [ -z "$SUITE_SPARSE_CAMP_ROOT" ]; then
            SUITE_SPARSE_CAMP_ROOT=${camp_suite_dir}/SuiteSparse
	fi
	cd ${camp_suite_dir}/cvode
	rm -rf build
	mkdir build
	mkdir install || true
	mkdir install/examples || true
	cd build
	cmake -D CMAKE_BUILD_TYPE=debug \
	      -D CMAKE_C_FLAGS_DEBUG="-O3" \
	      -D MPI_ENABLE:BOOL=TRUE \
	      -D KLU_ENABLE:BOOL=TRUE \
	      -D CUDA_ENABLE:BOOL=FALSE \
	      -D CMAKE_C_COMPILER=$(which mpicc) \
	      -D EXAMPLES_ENABLE_CUDA=OFF \
	      -D KLU_LIBRARY_DIR=$SUITE_SPARSE_CAMP_ROOT/lib \
	      -D KLU_INCLUDE_DIR=$SUITE_SPARSE_CAMP_ROOT/include \
	      -D CMAKE_INSTALL_PREFIX=$(pwd)/../install \
	      -D EXAMPLES_ENABLE_C=OFF \
	      ..
	make install
    fi
}
    


#########################################################
#  prepare and compile camp
#########################################################
function prepare_and_compile_camp()
{	
    # prepare directories
    if [[ $flags == *1* ]] ; then
        cp -rf ${source_dir}/camp ${camp_suite_dir}
    fi
    if [[ $flags == *2* ]] ; then
        cd ${camp_suite_dir}
	git clone https://earth.bsc.es/gitlab/ac/camp.git
    fi


    if [[ $flags == *3* ]] ; then
	# load modules
	case "${BSC_MACHINE}" in
	    "mn5")
		module purge
		if ! module list 2>&1 | grep -q "\<python/3.12.1-gcc\>"; then 
		    module purge
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
		mpifort=$(which mpifort)
		;;
	esac
	
	# compile
	export JSON_FORTRAN_HOME=${camp_suite_dir}/json-fortran-6.1.0/install/jsonfortran-gnu-6.1.0
	export SUNDIALS_HOME=${camp_suite_dir}/cvode/install
	export SUITE_SPARSE_HOME=${camp_suite_dir}/SuiteSparse
	cd ${camp_suite_dir}/camp
	rm -rf build
	mkdir build
	cd build
	cmake -D CMAKE_C_COMPILER=$(which mpicc) \
	      -D CMAKE_BUILD_TYPE=debug \
	      -D CMAKE_C_FLAGS_DEBUG="-std=c99 -O3 -g" \
	      -D CMAKE_Fortran_FLAGS_DEBUG="-g -O3" \
	      -D CMAKE_C_FLAGS_RELEASE="-std=c99 -O3" \
	      -D CMAKE_Fortran_FLAGS_RELEASE="" \
	      -D CMAKE_Fortran_COMPILER=$mpifort \
	      -D ENABLE_DEBUG=OFF \
	      -D FAILURE_DETAIL=OFF \
	      -D ENABLE_MPI=ON \
	      -D ENABLE_PROFILE_SOLVING=ON \
	      -D ENABLE_GSL:BOOL=FALSE \
	      -D ENABLE_NETCDF=ON \
	      ..
	
	ln -sf ../test/monarch/settings
	ln -sf ../test/monarch/out
	make -j 4 VERBOSE=1
    fi
}




#########################################################
#  main
#########################################################

# loop on the CAMP suite components to prepare and compile
echo -e "$RED| Start loop on CAMP suite components$NC"
for componentflags in "$@"; do

    # get component and flags
    component=$(echo "$componentflags" | awk -F':' '{print $1}')
    flags=$(echo "$componentflags" | awk -F':' '{print $2}')
    
    echo -e "$RED| Prepare and compile $component with flag(s) $flags$NC"

    # check that flags 1 and 2 are not requested together
    if [[ $flags == *1* ]] && [[ $flags == *2* ]] ; then
	echo -e "$RED ERROR: Wrong flag specification ($flags). Please choose either 1 (copying) or 2 (cloning). Exiting...$NC"
	exit 0
    fi

    # loop on components
    start_time=$(date +%s)
    if [ $verbose -eq 0 ] ; then
	case $component in
	    'json-fortran') prepare_and_compile_jsonfortran > ${camp_suite_dir}/log_jsonfortran 2>&1 ;;
    	    'SuiteSparse')  prepare_and_compile_suitesparse > ${camp_suite_dir}/log_suitesparse 2>&1 ;;
    	    'cvode')        prepare_and_compile_cvode       > ${camp_suite_dir}/log_cvode       2>&1 ;;
    	    'camp')         prepare_and_compile_camp        > ${camp_suite_dir}/log_camp        2>&1 ;;
        esac	
    else
	case $component in
	    'json-fortran') prepare_and_compile_jsonfortran ;;
	    'SuiteSparse')  prepare_and_compile_suitesparse ;;
	    'cvode')        prepare_and_compile_cvode       ;;
	    'camp')         prepare_and_compile_camp        ;;
	esac
    fi
    
    # compute execution duration
    end_time=$(date +%s)
    duration_seconds=$((end_time - start_time))
    duration_minutes=$(echo "scale=2; $duration_seconds/60" | bc)    
    echo "|    (duration: $duration_seconds seconds or $duration_minutes minutes)"
    
done

# clean
if [ $verbose -eq 0 ] && [ $clean -eq 1 ] ; then
    rm -f ${camp_suite_dir}/log_*
fi

echo -e "$RED| Successfully completed.$NC"

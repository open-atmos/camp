#!/usr/bin/env bash

####################################################
#        INSTALL THE CAMP SUITE
####################################################
#
# This script manages the preparation of the directories
# of the CAMP suite components (namely json-fortran,
# SuiteSparse, cvode, camp itself, as well as the
# camp boxmodel within camp) and their compilation
# on the BSC's HPC machines (e.g. MN5gpp, MN5acc).
# For each CAMP component, the user specifies the desired
# instructions, namely: "1" for copying the component directory
# from a source directory, "2" for cloning the component directory
# from the corresponding git URL, "3" for compiling the component.
# Note that flags "1" and "2" are exclusive.
# The default directory structure created (in /home/bsc/`whoami`) is:
#   <camp_suite_dir>
#      |-- camp
#            |-- boxmodel
#      |-- cvode
#      |-- json-fortran-6.1.0
#      |-- SuiteSparse
#
# Versions:
#       08/05/2024: Creation (Herve Petetin, herve.petetin@bsc.es)
#
####################################################

# indicate the library path where all CAMP suite components
# (json-fortran, SuiteSparse, cvode, camp and boxmodel
# within camp) will be located
if [ "$1" == "camp:1" ]; then
  # if you first requested to prepare camp directory, then do it in the current one
  camp_suite_dir=$(pwd)
else
  # if not, then we can assume you are already located in the <camp_suite_dir>/camp/ directory, but check that
  if [ ! -d compile ]; then
    echo -e "$RED| ERROR: It seems CAMP is not yet installed, or at least you are not located there.$NC"
    echo -e "$RED| The required compile.<component>.sh scripts are located in camp/compile.$NC"
    echo -e "$RED| Go in the camp directory if it already exists.$NC"
    echo -e "$RED| If not, start cloning camp before trying again the installation of the other components:$NC"
    echo "./install.sh camp:1"
    exit 0
  else
    camp_suite_dir=$(pwd)/.. #(toward the parent directory of camp)
  fi
fi
# force it to be an absolute path
camp_suite_dir=$(
  cd ${camp_suite_dir}
  pwd
)

# Indicate the source directory from where to copy the CAMP component(s),
# not the CAMP directory itself but its parent directory
# (useful only if flag "1" activated)
source_dir=/gpfs/projects/bsc32/bsc032815/gpupartmc

# select if you want to print the compilation outputs, and if not, if you want to clean the log files
verbose=0
clean=0

############# STARTING FROM HERE, BASIC USERS SHOULD NOT NEED TO MODIFY ANYTHING #############

# current directory
initial_dir=$(pwd)

# define colors for bash
RED='\033[0;31m' #(red)
NC='\033[0m'     #(default color)

# usage instructions and/or other information
echo -e "$RED|==========================================$NC"
echo -e "$RED|       INSTALL THE CAMP SUITE$NC"
echo -e "$RED|==========================================$NC"
if [ "$#" -eq 0 ]; then
  echo -e "$RED| Quick installation from scratch:$NC  ./install.sh start"
  echo -e "$RED| (this will git-clone the different CAMP components, except CAMP itself that is already there, $NC"
  echo -e "$RED|  and compile everything; i.e. jsonfortran:12 suitesparse:12 cvode:12 camp:2 boxmodel:2)$NC"
  echo -e "$RED| $NC"
  echo -e "$RED| For more options:$NC ./install.sh <lib>:<flag(s)> [...]"
  echo -e "$RED|        with <lib> the CAMP component library(ies) among jsonfortran, suitesparse, cvode, camp, boxmodel$NC"
  echo -e "$RED|             <flag=0> for copying the directory from another one,$NC"
  echo -e "$RED|             <flag=1> for git-cloning from the web,$NC"
  echo -e "$RED|             <flag=2> for compiling the library$NC"
  echo -e "$RED| Examples:$NC"
  echo -e "$RED#(copy/compile jsonfortran and clone/compile suitesparse)$NC"
  echo "./install.sh jsonfortran:02 suitesparse:12"
  echo -e "$RED#(git-clone/compile all components, and compile boxmodel)$NC"
  echo "./install.sh jsonfortran:12 suitesparse:12 cvode:12 camp:12 boxmodel:2"
  echo -e "$RED#(git-clone/compile jsonfortran, suitesparse and cvode, and compile camp and boxmodel)$NC"
  echo "./install.sh jsonfortran:12 suitesparse:12 cvode:12 camp:2 boxmodel:2"
  exit 0
fi

# get arguments
if [ "$1" == "start" ]; then
  # get input arguments for quick start
  arguments="jsonfortran:12 suitesparse:12 cvode:12 camp:2 boxmodel:2"
else
  # get input arguments prescribed by the user
  arguments=$@
fi

# print information
echo -e "$RED| Inputs:$NC"
echo -e "$RED|    components: $NC $arguments"
echo -e "$RED|    camp_suite_dir:$NC $camp_suite_dir"

# indicate if this script is working on the current HPC machine, exit if not
case "${BSC_MACHINE}" in
"mn5") echo -e "$RED| HPC machine (${BSC_MACHINE}) handled by this script $NC" ;;
*)
  echo -e "$RED| HPC machine (${BSC_MACHINE}) not yet handled by this script. Exiting...$NC"
  exit 0
  ;;
esac

# create directory for the camp suite
echo -e "$RED| Create the camp suite directory$NC"
mkdir -p ${camp_suite_dir}

#########################################################
#  install json-fortran
#########################################################
function install_jsonfortran() {
  # prepare directories
  if [[ $flags == *0* ]]; then
    #rm -rf ${camp_suite_dir}/json-fortran-6.1.0
    cp -rf ${source_dir}/json-fortran-6.1.0 ${camp_suite_dir}
  elif [[ $flags == *1* ]]; then
    # git-clone
    rm -rf ${camp_suite_dir}/json-fortran-6.1.0
    cd ${camp_suite_dir}
    git clone https://github.com/jacobwilliams/json-fortran.git json-fortran-6.1.0
    cd ${camp_suite_dir}/json-fortran-6.1.0
    git checkout tags/6.1.0
  fi

  if [[ $flags == *2* ]]; then
    # load modules and compile
    ${camp_suite_dir}/camp/compile/compile.json-fortran-6.1.0.sh ${camp_suite_dir}
  fi
  cd $initial_dir
}

#########################################################
#  install SuiteSparse
#########################################################
function install_suitesparse() {
  # prepare directories
  if [[ $flags == *0* ]]; then
    rm -rf ${camp_suite_dir}/SuiteSparse
    cp -rf ${source_dir}/SuiteSparse ${camp_suite_dir}
  fi
  if [[ $flags == *1* ]]; then
    # git-clone
    rm -rf ${camp_suite_dir}/SuiteSparse
    cd ${camp_suite_dir}
    git clone https://github.com/DrTimothyAldenDavis/SuiteSparse.git
    cd SuiteSparse
    git checkout v5.1.0
    suitesparse_patch=./suitesparse_patch.patch
    echo 'diff --git a/SuiteSparse_config/SuiteSparse_config.mk b/SuiteSparse_config/SuiteSparse_config.mk' >${suitesparse_patch}
    echo 'index bb26ac3a3..a3e63d66f 100644' >>${suitesparse_patch}
    echo '--- a/SuiteSparse_config/SuiteSparse_config.mk' >>${suitesparse_patch}
    echo '+++ b/SuiteSparse_config/SuiteSparse_config.mk' >>${suitesparse_patch}
    echo '@@ -115,7 +115,7 @@ SUITESPARSE_VERSION = 5.1.0' >>${suitesparse_patch}
    echo '             CC = icc -D_GNU_SOURCE' >>${suitesparse_patch}
    echo '             CXX = $(CC)' >>${suitesparse_patch}
    echo '             CFOPENMP = -qopenmp -I$(MKLROOT)/include' >>${suitesparse_patch}
    echo '-	    LDFLAGS += -openmp' >>${suitesparse_patch}
    echo '+	    LDFLAGS += -qopenmp' >>${suitesparse_patch}
    echo '             LDLIBS += -lm -lirc' >>${suitesparse_patch}
    echo '         endif' >>${suitesparse_patch}
    echo '         ifneq ($(shell which ifort 2>/dev/null),)' >>${suitesparse_patch}
    echo '' >>${suitesparse_patch}
    git apply suitesparse_patch.patch
  fi

  if [[ $flags == *2* ]]; then
    # load modules and compile
    ${camp_suite_dir}/camp/compile/compile.suiteSparse.sh ${camp_suite_dir}
  fi
  cd $initial_dir
}

#########################################################
#  install cvode
#########################################################
function install_cvode() {
  # prepare directories
  if [[ $flags == *0* ]]; then
    rm -rf ${camp_suite_dir}/cvode-3.4-alpha
    cp -rf ${source_dir}/cvode-3.4-alpha ${camp_suite_dir}
  fi
  if [[ $flags == *1* ]]; then
    # git-clone
    rm -rf ${camp_suite_dir}/cvode-3.4-alpha
    cd ${camp_suite_dir}
    git clone https://github.com/mattldawson/cvode.git cvode-3.4-alpha
  fi
  if [[ $flags == *2* ]]; then
    # load modules and compile
    ${camp_suite_dir}/camp/compile/compile.cvode-3.4-alpha.sh ${camp_suite_dir}
  fi
  cd $initial_dir
}

#########################################################
#  install camp
#########################################################
function install_camp() {
  if [[ $flags == *1* ]]; then
    # git-clone
    git clone https://earth.bsc.es/gitlab/ac/camp.git
    rm -rf ${camp_suite_dir}/camp
    cd ${camp_suite_dir}
  fi

  if [[ $flags == *2* ]]; then
    # load modules and compile
    source ${camp_suite_dir}/camp/compile/load.modules.camp.sh
    ${camp_suite_dir}/camp/compile/compile.camp.sh ${camp_suite_dir}
  fi
  cd $initial_dir
}

#########################################################
#  install boxmodel
#########################################################
function install_boxmodel() {
  if [[ $flags == *2* ]]; then
    # load modules and compile
    ${camp_suite_dir}/camp/compile/compile.boxmodel.sh ${camp_suite_dir}
  fi
  cd $initial_dir
}

#########################################################
#  main
#########################################################

# check if correct flags combination are requted, and
# check if the modules to be load are specified for the current HPC machine
for componentflags in $arguments; do

  # get component and flags
  component=$(echo "$componentflags" | awk -F':' '{print $1}')
  flags=$(echo "$componentflags" | awk -F':' '{print $2}')

  # check that flags 1 and 2 are not requested together
  if [[ $flags == *0* ]] && [[ $flags == *1* ]]; then
    echo -e "$RED| ERROR: Wrong flag specification ($flags). Please choose either 1 (copying) or 2 (cloning). Exiting...$NC"
    exit 0
  fi

  # check if the HPC machine name appears in the load.module.XXX.sh script (if compilation is requested)
  if [[ $flags == *2* ]]; then
    case $component in
    'jsonfortran') ok=$(grep '"'${BSC_MACHINE}'-loadmodules"' compile/compile.json-fortran-6.1.0.sh | wc -l) ;;
    'suitesparse') ok=$(grep '"'${BSC_MACHINE}'-loadmodules"' compile/compile.suiteSparse.sh | wc -l) ;;
    'cvode') ok=$(grep '"'${BSC_MACHINE}'-loadmodules"' compile/compile.cvode-3.4-alpha.sh | wc -l) ;;
    'camp') ok=$(grep '"'${BSC_MACHINE}'-loadmodules"' compile/compile.camp.sh | wc -l) ;;
    'boxmodel') ok=$(grep '"'${BSC_MACHINE}'-loadmodules"' compile/compile.boxmodel.sh | wc -l) ;;
    esac
    if [ $ok -ne 1 ]; then
      echo -e "$RED| ERROR: Module load instructions missing for this HPC machine (${BSC_MACHINE}). Exiting...$NC"
      exit 0
    fi
  fi

done

# loop on the CAMP suite components to prepare and compile
echo -e "$RED| Start loop on CAMP suite components$NC"
for componentflags in $arguments; do

  # get component and flags
  component=$(echo "$componentflags" | awk -F':' '{print $1}')
  flags=$(echo "$componentflags" | awk -F':' '{print $2}')

  echo -e "$RED| Prepare and compile $component with flag(s) $flags$NC"
  echo -e "$RED|   (check compilation log files: ${camp_suite_dir}/log_$component)$Nc"

  # loop on components
  start_time=$(date +%s)
  if [ $verbose -eq 0 ]; then
    case $component in
    'jsonfortran') install_jsonfortran >${camp_suite_dir}/log_$component 2>&1 ;;
    'suitesparse') install_suitesparse >${camp_suite_dir}/log_$component 2>&1 ;;
    'cvode') install_cvode >${camp_suite_dir}/log_$component 2>&1 ;;
    'camp') install_camp >${camp_suite_dir}/log_$component 2>&1 ;;
    'boxmodel') install_boxmodel >${camp_suite_dir}/log_$component 2>&1 ;;
    esac

    # check errors in compilation (not completed yet, need to handle non-important errors)
    #nerrors=`grep Error ${camp_suite_dir}/log_$component | wc -l`
    #if [ "$nerrors" -gt 0 ] ; then
    #    tail -f 100 ${camp_suite_dir}/log_$component
    #    echo -e "$RED| ERROR: during compilation of $component. Check log file:$NC"
    #    echo ${camp_suite_dir}/log_$component
    #    echo -e "$RED| Exiting.$NC"
    #    exit 0
    #fi

  else
    case $component in
    'jsonfortran') install_jsonfortran ;;
    'suitesparse') install_suitesparse ;;
    'cvode') install_cvode ;;
    'camp') install_camp ;;
    'boxmodel') install_boxmodel ;;
    esac
  fi

  # compute execution duration
  end_time=$(date +%s)
  duration_seconds=$((end_time - start_time))
  duration_minutes=$(echo "scale=2; $duration_seconds/60" | bc)
  echo -e "$RED|$NC    (duration: $duration_seconds seconds or $duration_minutes minutes)"

done

# clean
if [ $verbose -eq 0 ] && [ $clean -eq 1 ]; then
  rm -f ${camp_suite_dir}/log_*
fi

echo -e "$RED| Successfully completed.$NC"

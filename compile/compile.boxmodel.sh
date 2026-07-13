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
	module load intel
	source ${camp_suite_dir}/camp/compile/load.modules.camp.sh
	;;
esac

# compile CAMP
cd ${camp_suite_dir}/camp/build
make boxmodel_v2

# come back to initial directory
cd ${initial_dir}

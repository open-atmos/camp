#!/usr/bin/env bash
#SBATCH --qos=acc_debug
##SBATCH --qos=acc_bsces
#SBATCH --job-name=camp_test_monarch
#SBATCH --output=out_sbatch.txt
#SBATCH --error=err_sbatch.txt
#SBATCH --ntasks=20
##SBATCH --nodes=1
##SBATCH --ntasks-per-node=160
##SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH -A bsc32

set -e
while getopts 'h' flag; do
  case "${flag}" in
    h) echo "Script to run a sample test, used for development" &
    exit 1 ;;
  esac
done

scriptdir="$(dirname "$0")"
cd "$scriptdir"
if [ ! -d ../build ]; then
  ./compile.libs.camp.sh
fi
if ! module list 2>&1 | grep -q "\<python\>"; then
  source load.modules.camp.sh
fi

run_boxmodel(){
#first setup:
#cd ../boxmodel && sbatch submit_boxmodel_job config_examples/debug
cd ../build && make -j 8
JOBID=16078386 && cd /gpfs/scratch/bsc32/bsc032815/run/${JOBID}
PATH_TO_CAMP=../../gpupartmc/camp && mpirun -n 1 ${PATH_TO_CAMP}/build/boxmodel_v2 config.json interface_boxmodel.json simu > out.log
}
#run_boxmodel

#gpu test:
cd ../test/monarch && ./run.sh


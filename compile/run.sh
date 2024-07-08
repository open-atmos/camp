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



pwd
if [ ! -d ../build ]; then
  ./compile.libs.camp.sh
fi
if ! module list 2>&1 | grep -q "\<python\>"; then
  source load.modules.camp.sh
fi
cd ../test/monarch
./run.sh
#!/usr/bin/env bash
##SBATCH --qos=debug
#SBATCH --job-name=camp_test_monarch
#SBATCH --output=out_sbatch.txt
#SBATCH --error=err_sbatch.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=160
#SBATCH --gres=gpu:4
#SBATCH --exclusive
set -e

scriptdir="$(dirname "$0")"
cd "$scriptdir"
if [ ! -d ../build ]; then
  ./compile.libs.camp.sh
fi
source load.modules.camp.sh
cd ../test/monarch
./run.sh
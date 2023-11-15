#!/usr/bin/env bash
##SBATCH --qos=debug
#SBATCH --job-name=camp_test_monarch
#SBATCH --output=out_sbatch.txt
#SBATCH --error=err_sbatch.txt
#SBATCH --ntasks-per-core=1
#SBATCH -n 80
##SBATCH --nodes=2
##SBATCH --ntasks-per-node=160
#SBATCH --gres=gpu:4
#SBATCH --exclusive

set -e
make_run(){
  curr_path=$(pwd)
  cd ../../build
  make -j 4
  cd $curr_path
  python TestMonarch1.py
  #python TestMonarch2.py
  #python TestMonarch3.py
  #python TestMonarch4.py
}
time make_run
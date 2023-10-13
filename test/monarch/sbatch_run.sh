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
make_run(){
  curr_path=$(pwd)
  cd ../../build
  make
  cd $curr_path
  python TestMonarch1.py
  python TestMonarch2.py
  python TestMonarch3.py
}
time make_run
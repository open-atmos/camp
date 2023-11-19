#!/usr/bin/env bash
##SBATCH --qos=debug
##SBATCH -t 00:09:00
#SBATCH --job-name=camp_test_monarch
#SBATCH --output=out_sbatch.txt
#SBATCH --error=err_sbatch.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:4
#SBATCH --exclusive

set -e
make_run(){
  curr_path=$(pwd)
  cd ../../build
  make -j 4
  cd $curr_path
  python TestMonarch1.py
  python TestMonarch2.py
  #python TestMonarch3.py
  python TestMonarch4.py
}
time make_run
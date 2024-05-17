#!/usr/bin/env bash
##SBATCH --qos=debug
#SBATCH -t 00:09:00
#SBATCH --job-name=camp_test_monarch
#SBATCH --output=out_sbatch.txt
#SBATCH --error=err_sbatch.txt
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=160
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:4
#SBATCH --exclusive

set -e
make_run(){
  initial_dir=$(pwd)
  cd ../../build
  make -j 4
  cd $initial_dir
  python TestMonarch.py
}
time make_run
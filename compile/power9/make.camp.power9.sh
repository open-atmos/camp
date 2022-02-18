#!/usr/bin/env bash
#SBATCH --qos=debug
#SBATCH --job-name=test_cb05_232
#SBATCH --output=out_test_monarch.txt
###SBATCH --output=%j_outcb05_.txt
#SBATCH --error=err_test_monarch.txt
##SBATCH --ntasks=1
#SBATCH --ntasks=40
#SBATCH --gres=gpu:1

export SUNDIALS_HOME=$(pwd)/../../../cvode-3.4-alpha/install
export SUITE_SPARSE_HOME=$(pwd)/../../../SuiteSparse
export JSON_FORTRAN_HOME=$(pwd)/../../../json-fortran-6.1.0/install/jsonfortran-gnu-6.1.0
#export GSL_HOME=${GSL_DIR}

cd ../../
cd build
make -j 4

cd test_run/monarch

FILE=TestMonarch.py
if test -f "$FILE"; then
  python $FILE
  #./test_monarch_1.sh MPI
else
  echo "Running old commits with file test_monarch_1.py ."
  python test_monarch_1.py
fi

cd ../../../../camp/compile/power9

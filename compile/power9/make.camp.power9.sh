#!/usr/bin/env bash

export SUNDIALS_HOME=$(pwd)/../../../cvode-3.4-alpha/install
export SUITE_SPARSE_HOME=$(pwd)/../../../SuiteSparse
export JSON_FORTRAN_HOME=$(pwd)/../../../json-fortran-6.1.0/install/jsonfortran-gnu-6.1.0
#export GSL_HOME=${GSL_DIR}

is_sbatch="true"
#is_sbatch="false"

mkdir_if_not_exists(){
  if [ ! -d $1 ]; then
      mkdir $1
  fi
}

rm_old_logs(){
find $1 -type f -mtime +90 -exec rm -rf {} \;
}


if [ $is_sbatch == "true" ]; then

  rm_old_logs log/out/
  rm_old_logs log/err/



  #./make.camp.power9.sbatch.sh

else

  cd ../../
  cd build
  make -j 4

  cd test_run/monarch

  FILE=TestMonarch.py
  if test -f "$FILE"; then
    python $FILE
    cd ../../

    #./test_monarch_1.sh MPI
    #./test_run/chemistry/cb05cl_ae5/test_chemistry_cb05cl_ae5.sh
    #./unit_test_aero_rep_single_particle
  else
    echo "Running old commits with file test_monarch_1.py ."
    python test_monarch_1.py
  fi
  cd ../../camp/compile/power9

fi
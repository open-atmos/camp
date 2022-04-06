#!/usr/bin/env bash
#SBATCH --qos=debug
#SBATCH --job-name=test_monarch
#SBATCH --output=log/out/%j.txt
#SBATCH --error=log/err/%j.txt
#SBATCH --ntasks=1
##SBATCH --ntasks=40
##SBATCH --gres=gpu:1
##SBATCH --exclusive

export SUNDIALS_HOME=$(pwd)/../../../cvode-3.4-alpha/install
export SUITE_SPARSE_HOME=$(pwd)/../../../SuiteSparse
export JSON_FORTRAN_HOME=$(pwd)/../../../json-fortran-6.1.0/install/jsonfortran-gnu-6.1.0
#export GSL_HOME=${GSL_DIR}

#Copy contents (src, conf)

#echo "Job"
#echo %j

cd ../../
cd build
#make -j 4

cd test_run/monarch

FILE=TestMonarch.py
if test -f "$FILE"; then
  #python $FILE
  cd ../../

  #./test_monarch_1.sh MPI
  #./test_run/chemistry/cb05cl_ae5/test_chemistry_cb05cl_ae5.sh
  #./unit_test_aero_rep_single_particle
else
  echo "Running old commits with file test_monarch_1.py ."
  python test_monarch_1.py
fi
cd ../../camp/compile/power9

mv_log(){
  src_path=log/err/*
  dst_path=../../test/monarch/exports/log/err
  mv $src_path $dst_path
  src_path=log/out/*
  dst_path=../../test/monarch/exports/log/out
  mv $src_path $dst_path
}

mv_out(){

}
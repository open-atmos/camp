#!/usr/bin/env bash
##SBATCH --qos=debug
#SBATCH --job-name=camp_test_monarch
#SBATCH --output=log/out/%j.txt
#SBATCH --error=log/err/%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=160
#SBATCH --gres=gpu:4
#SBATCH --exclusive

relative_path="../../../"
if [ ! -z "$1" ]; then
  relative_path="../../../../"
fi

export SUNDIALS_HOME=$(pwd)/$relative_path/cvode-3.4-alpha/install
export SUITE_SPARSE_HOME=$(pwd)/$relative_path/SuiteSparse
export JSON_FORTRAN_HOME=$(pwd)/$relative_path/json-fortran-6.1.0/install/jsonfortran-gnu-6.1.0
#export GSL_HOME=${GSL_DIR}

compile_run(){

  id=$1

  echo "Starting job " "$id"

  cd ../../../camp_jobs/camp$id/compile/power9

  ./compile.camp.power9.sh "from_camp_jobs" $id
  cd ../../build/test_run/monarch

  #cd ../../build
  #make -j 4
  #cd test_run/monarch

  FILE=TestMonarch.py
  if test -f "$FILE"; then
    python $FILE  "$id"
    srun --qos=debug --ntasks=1 cp -r -u ../../../test/monarch/exports/* ../../../../../camp/test/monarch/exports/
    #cp -r -u ../../../test/monarch/exports/* ../../../../../camp/test/monarch/exports/
    cd ../../

    #./test_monarch_1.sh MPI
    #./test_run/chemistry/cb05cl_ae5/test_chemistry_cb05cl_ae5.sh
    #./unit_test_aero_rep_single_particle
  else
    echo "Running old commits with file test_monarch_1.py ."
    python test_monarch_1.py
  fi

  cd ../../../
  #srun --qos=debug --ntasks=1 rm -rf camp_jobs/camp$id
  cd camp/compile/power9
}

compile_run $1
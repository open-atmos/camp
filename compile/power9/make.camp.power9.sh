#!/usr/bin/env bash

recompile(){
#./gcc.compile.cvode-3.4-alpha.power9.sh
#./gcc.make.cvode-3.4-alpha.power9.sh
#./compile.cvode-3.4-alpha.power9.sh
#./compile.camp.power9.sh
#./gcc.compile.camp.power9.sh
pwd
}
recompile

export SUNDIALS_HOME=$(pwd)/../../../cvode-3.4-alpha/install
export SUITE_SPARSE_HOME=$(pwd)/../../../SuiteSparse
export JSON_FORTRAN_HOME=$(pwd)/../../../json-fortran-6.1.0/install/jsonfortran-gnu-6.1.0
#export GSL_HOME=${GSL_DIR}

if [ "$1" == "1" ]; then
  is_sbatch="true"
else
  #is_sbatch="true"
  is_sbatch="false"
fi

mkdir_if_not_exists(){
  if [ ! -d $1 ]; then
      mkdir $1
  fi
}

rm_old_logs(){
find $1 -type f -mtime +15 -exec rm -rf {} \;
}
rm_old_dirs_jobs(){
find $1 -type d -ctime +30 -exec rm -rf {} +
}

if [ $is_sbatch == "true" ]; then

  rm_old_logs log/out/
  rm_old_logs log/err/

  id=$(date +%s%N)
  cd ../../..
  mkdir_if_not_exists camp_jobs
  rm_old_dirs_jobs camp_jobs/
  echo "Copying camp folder to" camp_jobs/camp$id
  cp -r camp camp_jobs/camp$id
  cd camp/compile/power9

  echo "Sending job " $job_id
  job_id=$(sbatch --parsable ./sbatch.make.camp.power9.sh "$id")
  #./sbatch.make.camp.power9.sh  "$id"

else

  cd  ../../build
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

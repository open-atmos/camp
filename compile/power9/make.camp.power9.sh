#!/usr/bin/env bash

export SUNDIALS_HOME=$(pwd)/../../../cvode-3.4-alpha/install
export SUITE_SPARSE_HOME=$(pwd)/../../../SuiteSparse
if [ $BSC_MACHINE == "power" ]; then
  export JSON_FORTRAN_HOME=$(pwd)/../../../json-fortran-6.1.0/install/jsonfortran-gnu-6.1.0
elif [ $BSC_MACHINE == "mn4" ]; then
  export JSON_FORTRAN_HOME=$(pwd)/../../../json-fortran-6.1.0/install/jsonfortran-intel-6.1.0
else
  echo "Unknown architecture"
  exit
fi

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
#echo "rm_old_logs finish"
} #echo $(rm_old_logs)
rm_old_dirs_jobs(){
find $1 -type d -ctime +30 -exec rm -rf {} +
}

mkdir_if_not_exists "../../build/test_run"
mkdir_if_not_exists "../../build/test_run/monarch"
mkdir_if_not_exists "../../build/test_run/monarch/out"

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
  echo "Sent job_id" $job_id

else

  cd  ../../build
  if ! make -j ${NUMPROC}; then
    exit
  fi
  cd ../test/monarch

  #echo "make end"

  FILE=TestMonarch.py
  #FILE=./test_run/chemistry/cb05cl_ae5/test_chemistry_cb05cl_ae5.sh
  #FILE=./unit_test_aero_rep_single_particle
  #FILE=./new_make.sh
  if [ "$FILE" == TestMonarch.py ]; then
    #echo "python TestMonarch.py start"
    python $FILE
    cd ../../compile/power9
  elif [ "$FILE" == test_monarch_1.py ]; then
    echo "Running old commits with file test_monarch_1.py ."
    python  $FILE
    cd ../../camp/compile/power9
  else
    #cd ../../build
    cd ../../compile/power9
    time $FILE
  fi

fi
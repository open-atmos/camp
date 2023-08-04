#!/usr/bin/env bash

make_base(){
cd /gpfs/scratch/bsc32/bsc32815/gpupartmc/camp/compile/power9
if ! ./make.camp.power9.sh; then
  exit
fi
cd /gpfs/scratch/bsc32/bsc32815/a591/nmmb-monarch/MODEL/SRC_LIBS/camp/compile/power9
}
#make_base
#./compile.cvode-3.4-alpha.power9.sh

export SUNDIALS_HOME=$(pwd)/../../../cvode-3.4-alpha/install
export SUITE_SPARSE_HOME=$(pwd)/../../../SuiteSparse
export JSON_FORTRAN_HOME=$(pwd)/../../../json-fortran-6.1.0/install/jsonfortran-gnu-6.1.0

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

rm_old_logs(){
find $1 -type f -mtime +15 -exec rm -rf {} \;
}
rm_old_dirs_jobs(){
find $1 -type d -ctime +30 -exec rm -rf {} +
}

mkdir -p "../../build/test_run"
mkdir -p "../../build/test_run/monarch"
mkdir -p "../../build/test_run/monarch/out"

if [ $is_sbatch == "true" ]; then

  rm_old_logs log/out/
  rm_old_logs log/err/

  id=$(date +%s%N)
  cd ../../..
  mkdir -p camp_jobs
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

  FILE=TestMonarch.py
  #FILE=./test_run/chemistry/cb05cl_ae5/test_chemistry_cb05cl_ae5.sh
  #FILE=./unit_test_aero_rep_single_particle
  #FILE=./new_make.sh
  if [ "$FILE" == TestMonarch.py ]; then
    log_path="/gpfs/scratch/bsc32/bsc32815/a591/nmmb-monarch/MODEL/SRC_LIBS/camp/compile/power9/log.txt"
    echo "Generating log file at " $log_path
    #python $FILE > $log_path #if(used_print_double)
    #python $FILE 2>&1 | tee $log_path #if(used_print_double)
    python $FILE
    scripts/compare_netcdf.sh #if(cell_netcdf)
    cd ../../compile/power9
  elif [ "$FILE" == test_monarch_1.py ]; then
    echo "Running old commits with file test_monarch_1.py ."
    python  $FILE
    cd ../../camp/compile/power9
  else
    cd ../../compile/power9
    time $FILE
  fi
  #diff log_cpu.txt log.txt > diff.txt #if(used_print_double)
  #scripts/merge_mpi_out.sh #if (export_double_mpi)
fi
#!/usr/bin/env bash

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

cd  ../../build
if ! make -j ${NUMPROC}; then
  exit
fi

cd ../test/monarch
FILE=TestMonarch.py
#FILE=./test_run/chemistry/cb05cl_ae5/test_chemistry_cb05cl_ae5.sh
#FILE=./unit_test_aero_rep_single_particle
#FILE=./new_make.sh

compare_runs(){
    #log_path="/gpfs/scratch/bsc32/bsc32815/a591/nmmb-monarch/MODEL/SRC_LIBS/camp/compile/power9/log_gpu.txt"
    log_path="../../compile/power9/log_cpu.txt"
    #echo "Generating log file at " $log_path
    python $FILE > $log_path
    #python $FILE 2>&1 | tee $log_path
    cells=1
    sed -i 's/conf.caseBase = "CPU One-cell"/conf.caseBase = "GPU BDF"/g' $FILE
    sed -i 's/conf.cells = \[1\]/conf.cells = \['"$cells"'\]/g' $FILE
    log_path="../../compile/power9/log_gpu.txt"
    #python $FILE 2>&1 | tee $log_path
    python $FILE > $log_path
    sed -i 's/conf.caseBase = "GPU BDF"/conf.caseBase = "CPU One-cell"/g' $FILE
    sed -i 's/conf.cells = \['"$cells"'\]/conf.cells = \[1\]/g' $FILE
    #python translate_netcdf.py
    cd ../../compile/power9
    diff log_cpu.txt log_gpu.txt 2>&1 | tee diff.txt
}

compare_cell(){
    #log_path="/gpfs/scratch/bsc32/bsc32815/a591/nmmb-monarch/MODEL/SRC_LIBS/camp/compile/power9/log_gpu.txt"
    log_path="../../compile/power9/log_cpu.txt"
    #echo "Generating log file at " $log_path
    #python $FILE > $log_path
    python $FILE 2>&1 | tee $log_path
    cd ../../compile/power9
    csplit -z "log_cpu.txt" '/end cell/' '{*}'
    diff xx00 xx01 2>&1 | tee diff.txt
}

if [ "$FILE" == TestMonarch.py ]; then
  compare_runs
  #compare_cell
  #python $FILE
  #log_path="../../compile/power9/log_cpu.txt"
  #python $FILE 2>&1 | tee "../../compile/power9/log_cpu.txt"
elif [ "$FILE" == test_monarch_1.py ]; then
  echo "Running old commits with file test_monarch_1.py ."
  python  $FILE
  cd ../../camp/compile/power9
else
  cd ../../compile/power9
  time $FILE

fi
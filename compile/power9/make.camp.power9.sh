#!/usr/bin/env bash

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
  #log_path="/gpfs/scratch/bsc32/bsc32815/a591/nmmb-monarch/MODEL/SRC_LIBS/camp/compile/power9/log_cpu.txt"
  #echo "Generating log file at " $log_path
  #python $FILE > $log_path
  #python $FILE 2>&1 | tee $log_path
  python $FILE
  cd ../../compile/power9
elif [ "$FILE" == test_monarch_1.py ]; then
  echo "Running old commits with file test_monarch_1.py ."
  python  $FILE
  cd ../../camp/compile/power9
else
  cd ../../compile/power9
  time $FILE
fi
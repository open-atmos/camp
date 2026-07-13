#!/usr/bin/env bash
#todo: enable "USE_PRINT_ARRAYS"
#todo: enable "CAMP_DEBUG_NVECTOR"
set -e

dev_check_cvode_changes() {
  #Serve to know if recompile cvode or not
  initial_dir=$(pwd)
  cd  ../../cvode/build
  output=$(make -j 8 2>&1)
  if echo "$output" | grep -q -i "up to date"; then
    echo "No changes were made by make."
  else
    echo "Make performed changes."
  fi
  cd $initial_dir
}

diff_testMONARCH(){
  cd ../build
  make -j 8 || exit 1
  build_dir=$(pwd)
  FILE=TestMonarch.py
  cd ../test/monarch
  #log_path="/gpfs/scratch/bsc32/bsc32815/a591/nmmb-monarch/MODEL/SRC_LIBS/camp/compile/log_gpu.txt"
  log_path="${build_dir}/log_cpu.txt"
  #echo "Generating log file at " $log_path
  python $FILE > "${build_dir}/log_cpu.txt"
  #python $FILE 2>&1 | tee $log_path
  sed -i 's/conf.caseBase = "CPU"/conf.caseBase = "GPU"/g' $FILE
  log_path="${build_dir}/log_gpu.txt"
  #python $FILE 2>&1 | tee $log_path
  python $FILE > $log_path
  sed -i 's/conf.caseBase = "GPU"/conf.caseBase = "CPU"/g' $FILE
  cd ${build_dir}
  diff log_cpu.txt log_gpu.txt 2>&1 | tee diff.txt
}
#diff_testMONARCH

#./compile.cvode.sh
#./compile.camp.sh
diff_check(){
  FILE=../test/unit_rxn_data/test_rxn_photolysis.F90
  sed -i 's/load_gpu=100/load_gpu=0/g' $FILE
  ./check.sh > "../build/log_cpu.txt" | true
  sed -i 's/load_gpu=0/load_gpu=100/g' $FILE
  ./check.sh > "../build/log_gpu.txt" | true
  diff ../build/log_cpu.txt ../build/log_gpu.txt 2>&1 | tee ../build/diff.txt
}
diff_check
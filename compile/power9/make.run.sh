#!/usr/bin/env bash
source remake.camp.sh

main(){

compile_camp

FILE=TestMonarch.py
#FILE=./camp_v1_paper_binned
#FILE=stats_monarch_netcdf.py
#FILE=./test_run/chemistry/cb05cl_ae5/test_chemistry_cb05cl_ae5.sh
#FILE=./unit_test_aero_rep_single_particle
#FILE=./new_make.sh

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
  cd ../../test/monarch
  #compare_cell
  python $FILE
  #log_path="../../compile/power9/log_cpu.txt"
  #python $FILE 2>&1 | tee "../../compile/power9/log_cpu.txt"
elif [ "$FILE" == ./camp_v1_paper_binned ]; then
  cd ../../build/data_run/CAMP_v1_paper/binned/
  ./test_monarch_binned.sh
else
  cd ../../compile/power9
  $FILE
fi
}
main
#!/usr/bin/env bash
source remake.camp.sh

main(){

make_camp

FILE=TestMonarch.py
#FILE=./camp_v1_paper_binned
#FILE=stats_monarch_netcdf.py
#FILE=./test_run/chemistry/cb05cl_ae5/test_chemistry_cb05cl_ae5.sh
#FILE=./unit_test_aero_rep_single_particle
#FILE=./new_make.sh

if [ "$FILE" == TestMonarch.py ]; then
  cd ../../test/monarch
  python $FILE
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
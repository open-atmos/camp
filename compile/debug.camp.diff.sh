#!/usr/bin/env bash
#todo: enable "USE_PRINT_ARRAYS"
#todo: enable "USE_BCG"
#todo: enable "CAMP_DEBUG_NVECTOR"
make_camp(){
  initial_dir=$(pwd)
  cd  ../build
  make || exit 1
  cd $initial_dir
}
make_camp
FILE=diff_TestMonarch.py
cd ../test/monarch
#log_path="/gpfs/scratch/bsc32/bsc32815/a591/nmmb-monarch/MODEL/SRC_LIBS/camp/compile/log_gpu.txt"
log_path="../compile/log_cpu.txt"
#echo "Generating log file at " $log_path
python $FILE > $log_path
#python $FILE 2>&1 | tee $log_path
#cells=1
sed -i 's/conf.caseBase = "CPU One-cell"/conf.caseBase = "GPU BDF"/g' $FILE
#sed -i 's/conf.cells = \[1\]/conf.cells = \['"$cells"'\]/g' $FILE
log_path="../power9/log_gpu.txt"
#python $FILE 2>&1 | tee $log_path
python $FILE > $log_path
sed -i 's/conf.caseBase = "GPU BDF"/conf.caseBase = "CPU One-cell"/g' $FILE
#sed -i 's/conf.cells = \['"$cells"'\]/conf.cells = \[1\]/g' $FILE
cd ../power9
diff log_cpu.txt log_gpu.txt 2>&1 | tee diff.txt
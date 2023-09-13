#!/usr/bin/env bash
source remake.camp.sh
compile
FILE=TestMonarch_diff.py
cd ../../test/monarch
#log_path="/gpfs/scratch/bsc32/bsc32815/a591/nmmb-monarch/MODEL/SRC_LIBS/camp/compile/power9/log_gpu.txt"
log_path="../../compile/power9/log_cpu.txt"
#echo "Generating log file at " $log_path
python $FILE > $log_path
#python $FILE 2>&1 | tee $log_path
#cells=1
sed -i 's/conf.caseBase = "CPU One-cell"/conf.caseBase = "GPU BDF"/g' $FILE
#sed -i 's/conf.cells = \[1\]/conf.cells = \['"$cells"'\]/g' $FILE
log_path="../../compile/power9/log_gpu.txt"
#python $FILE 2>&1 | tee $log_path
python $FILE > $log_path
sed -i 's/conf.caseBase = "GPU BDF"/conf.caseBase = "CPU One-cell"/g' $FILE
#sed -i 's/conf.cells = \['"$cells"'\]/conf.cells = \[1\]/g' $FILE
cd ../../compile/power9
diff log_cpu.txt log_gpu.txt 2>&1 | tee diff.txt
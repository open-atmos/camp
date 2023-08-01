#!/usr/bin/env bash

make_run(){
  cd  ../../build
  if ! make -j 4; then
    exit
  fi
  cd ../test/monarch
  python TestMonarch.py
}

make_run_merge(){
BASE_PATH=`pwd`
cd $1
echo "BASE_PATH", $1
make_run
#cell_0timestep_0mpirank_0.nc
n_cells=2 #match with TestMonarch.py
n_ranks=10 #match with TestMonarch.py
merged_name="cellstimestep_0mpiranks.txt"
for (( i=0; i<$n_cells; i++ ))
do
  for (( j=0; j<$n_ranks; j++ ))
  do
    #echo "Welcome $i times"
    file_name="cell_"$i"timestep_0mpirank_"$j".nc"
    ncdump $file_name >> $merged_name
  done
done
cd $BASE_PATH
}
main(){
BASE_PATH=`pwd`
CPU_PATH=/gpfs/scratch/bsc32/bsc32815/gpupartmc/camp/test/monarch/
make_run_merge $CPU_PATH
make_run_merge $BASE_PATH
merged_name="cellstimestep_0mpiranks.nc"
diff $CPU_PATH"/"$merged_name $BASE_PATH"/"$merged_name > diff.txt
echo "Generated file at" $BASE_PATH"/"diff.txt
}
main
echo "End"
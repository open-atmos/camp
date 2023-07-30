#!/usr/bin/env bash

mkdir_if_not_exists(){
  if [ ! -d $1 ]; then
      mkdir $1
  fi
}
mkdir_if_not_exists "out/export_double_mpi"
cd out/export_double_mpi
n_files=$(ls -1 | wc -l)
echo "n_files $n_files"
file_base_name="export_data.txt"
for (( i=0; i<$n_files; i++ ))
do
  echo "Welcome $i times"
  file_name="$i$file_base_name"
  cat $file_name >> "../$file_base_name"
  rm $file_name
done
cd ../../

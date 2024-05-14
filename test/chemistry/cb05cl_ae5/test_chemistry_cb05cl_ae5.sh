#!/bin/bash

# exit on error
set -e
# turn on command echoing

# make sure that the current directory is the one where this script is
cd ${0%/*}
# make the output directory if it doesn't exist
mkdir -p out

((counter = 1))
while [ true ]
do
  echo Attempt $counter

if [[ $1 = "MPI" ]]; then
  exec_str="mpirun -v -np 1 ../../../test_chemistry_cb05cl_ae5"
else
  if [ -z ${SLURM_TASK_PID+x} ]; then
    exec_str="../../../test_chemistry_cb05cl_ae5"
  else
    exec_str="mpirun -v -np 1 --bind-to none  ../../../test_chemistry_cb05cl_ae5"
  fi
fi

if ! $exec_str; then
	  echo Failure "$counter"
	  if [ "$counter" -gt 0 ]
	  then
		  echo FAIL
		  exit 1
	  fi
	  echo retrying...
  else
	  echo PASS
	  exit 0
  fi
  ((counter++))
done

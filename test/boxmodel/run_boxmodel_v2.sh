#!/bin/bash

# exit on error
set -e
# turn on command echoing
set -v
# make sure that the current directory is the one where this script is
cd ${0%/*}
pwd
# make the output directory if it doesn't exist
mkdir -p out

(
  cd ../../build
  make boxmodel_v2
)

((counter = 1))
while [ true ]
do
  echo Attempt $counter

# exec_str="srun ../../build/boxmodel_v2 my_config_file.json interface_boxmodel.json out/boxmodel_v2_test"
exec_str="srun ../../build/boxmodel_v2 full_multiphase/config_vbs.json interface_boxmodel.json out/boxmodel_v2_test"


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


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
  cd ../build
  make full_multiphase_boxmodel
)

((counter = 1))
while [ true ]
do
  echo Attempt $counter

exec_str="../build/full_multiphase_boxmodel  full_multiphase/config_vbs.json interface_monarch_binned_external_mixing_vbs.json out/full_multiphase_cb05_external_mixing_vbs"

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


#!/bin/bash

# exit on error
set -e
# turn on command echoing
set -v
# make sure that the current directory is the one where this script is
cd ${0%/*}
# make the output directory if it doesn't exist
mkdir -p out
# copy the compare file to the output directory
cp mod37/monarch_mod37_comp.txt out/monarch_mod37_comp.txt

((counter = 1))
while [ true ]
do
  echo Attempt $counter

if [[ $1 == "MPI" ]]; then
  exec_str="mpirun -v -np 2 ../../mock_monarch mod37/config_monarch_mod37.json mod37/interface_monarch_mod37.json out/monarch_mod37"
else
  if [ $HOSTNAME == "p9login2"  ]; then # My plogin2 bashrc reserves a node by default through salloc - cguzman
    exec_str="mpirun -v  -np 1 --bind-to none  ../../mock_monarch mod37/config_monarch_mod37.json mod37/interface_monarch_mod37.json out/monarch_mod37"
  else
    exec_str=" ../../mock_monarch mod37/config_monarch_mod37.json mod37/interface_monarch_mod37.json out/monarch_mod37"
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


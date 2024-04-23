#!/bin/bash

# exit on error
set -e
# turn on command echoing
set -v
# make sure that the current directory is the one where this script is
cd ${0%/*}
# make the output directory if it doesn't exist
mkdir -p out

if [[ $1 = "MPI" ]]; then
  exec_str="mpirun -v -np 2 ../../test_rxn_aqueous_equilibrium"
else
  if [ -z ${SLURM_TASK_PID+x} ]; then
    exec_str="../../test_rxn_aqueous_equilibrium"
  else
    exec_str="mpirun -v -np 1 --bind-to none  ../../test_rxn_aqueous_equilibrium"
  fi
fi

if ! $exec_str; then
  echo FAIL
  exit 1
else
  echo PASS
  exit 0
fi
done

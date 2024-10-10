#!/bin/bash

module load papi
source ${EXTRAE_HOME}/etc/extrae.sh

export EXTRAE_CONFIG_FILE=./mpi_extrae.xml
export LD_PRELOAD=${EXTRAE_HOME}/lib/libcudampitrace.so #Trying CUDA
#export LD_PRELOAD=${EXTRAE_HOME}/lib/libmpitracef.so # For Fortran apps (Default)
#export LD_PRELOAD=${EXTRAE_HOME}/lib/libmpitrace.so # For C apps



## Run the desired program
$*


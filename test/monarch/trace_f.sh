#!/bin/bash

source env_extrae.sh
source ${EXTRAE_HOME}/etc/extrae.sh

export EXTRAE_CONFIG_FILE=./mpi_extrae.xml
export LD_PRELOAD=${EXTRAE_HOME}/lib/libmpitracef.so # For Fortran apps

## Run the desired program
$*


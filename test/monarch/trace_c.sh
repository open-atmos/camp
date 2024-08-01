#!/bin/bash

source ${EXTRAE_HOME}/etc/extrae.sh

export EXTRAE_CONFIG_FILE=./mpi_extrae.xml
export LD_PRELOAD=${EXTRAE_HOME}/lib/libmpitrace.so # For C apps

## Run the desired program
$*


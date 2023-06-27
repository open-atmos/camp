#!/usr/bin/env bash

#export SUNDIALS_HOME=$(pwd)/../../../cvode-3.4-alpha/install
#export SUITE_SPARSE_HOME=$(pwd)/../../../SuiteSparse
#export LD_LIBRARY_PATH=$(pwd)/../../../SuiteSparse:$(pwd)/../../../cvode-3.4-alpha/install:$LD_LIBRARY_PATH

cd ../../../cvode-3.4-alpha/install/examples/cvode/serial
make
ddt --connect cvDisc_dns
#./cvRoberts_dns


#ddt --connect ../../../
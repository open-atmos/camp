#!/usr/bin/env bash
set -e

#run cb05
cd  ../../build
make -j 4
./test_run/chemistry/cb05cl_ae5/test_chemistry_cb05cl_ae5.sh > ../compile/power9/log.txt
cd ../compile/power9/

#compare with the main branch results
diff log_main.txt log.txt 2>&1 | tee diff.txt
#!/usr/bin/env bash
set -e

test_accuracy_gpu_vs_cpu(){
unset I_MPI_PMI_LIBRARY
exec_str="python checkGPU.py"
if ! $exec_str; then 
    echo FAIL
    exit 1
else
    echo PASS
    exit 0
fi
}
test_accuracy_gpu_vs_cpu

#./cuda_sanitizer.sh # Check tool cuda sanitizer: Warnings, memory allocs, etc #TODO: Add to check, now is manually (because error of cuda context is not taking into account)
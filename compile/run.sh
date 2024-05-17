#!/usr/bin/env bash
##SBATCH --qos=debug
#SBATCH --job-name=camp_test_monarch
#SBATCH --output=out_sbatch.txt
#SBATCH --error=err_sbatch.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=160
#SBATCH --gres=gpu:4
#SBATCH --exclusive
set -e

#wget --tries=3 --timeout=5 -q --spider google.com && echo "Networked" || echo "Non-networked"

#if timeout 30s git ls-remote --tags > /dev/null 2>&1; then
#if timeout 5s git clone  https://github.com/jacobwilliams/json-fortran.git json-fortran-6.1.0; then
if timeout 2s wget -q --spider http://google.com; then
    # Note: it takes 2~4 sec to get to here.
    echo "git server IS available"
else
    # Note: it takes 30 seconds (as specified by `timeout`) to get to here.
    echo "git server is NOT available"
fi
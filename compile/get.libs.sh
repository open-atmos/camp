#!/usr/bin/env bash
set -e

if [ "${BSC_MACHINE}" != "mn5" ]; then
  echo "Not detected a machine for get libraries. Can not place
   the dependencies 'SuiteSparse,json-fortran,cvode' in path '../camp',
   please update this script or do it manually"
  exit 0
fi

library_path="../../"
curr_path=$(pwd)
dst_path=$curr_path/$library_path
src_path=/gpfs/projects/bsc32/bsc032815/gpupartmc
if [ ! -d "${src_path}" ]; then
  echo "ERROR: $src_path does not exist, contact the administrator
   for the path to the libraries, and update
   the variable 'src_path' with the correct path"
   exit 1
fi

cp -rf $src_path/json-fortran-6.1.0 $dst_path
cp -rf $src_path/cvode-3.4-alpha $dst_path
cp -rf $src_path/SuiteSparse $dst_path

#!/usr/bin/env bash

export SUNDIALS_HOME=$(pwd)/../../../cvode-3.4-alpha/install
export SUITE_SPARSE_HOME=$(pwd)/../../../SuiteSparse
if [ $BSC_MACHINE == "power" ]; then
  export JSON_FORTRAN_HOME=$(pwd)/../../../json-fortran-6.1.0/install/jsonfortran-gnu-6.1.0
elif [ $BSC_MACHINE == "mn4" ]; then
  export JSON_FORTRAN_HOME=$(pwd)/../../../json-fortran-6.1.0/install/jsonfortran-intel-6.1.0
else
  echo "Unknown architecture"
  exit
fi

bcmake(){
cd build
cmake ..
make -j 4
}
#bcmake
bmake(){
make clean
if make -j 4; then
  path0=$(pwd)
  path_lib=/gpfs/scratch/bsc32/bsc32815/gpupartmc/json-fortran-6.1.0/install/jsonfortran-gnu-6.1.0/lib
  export LD_LIBRARY_PATH=$path_lib:$LD_LIBRARY_PATH
  IS_DDT_OPEN=false
  if pidof -x $(ps cax | grep ddt) >/dev/null; then
      ddt --connect mpirun -v -np 1 mock_monarch
  else
    mpirun -v -np 1 mock_monarch
  fi

fi
}
bmake

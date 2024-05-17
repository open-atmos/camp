compile(){
   export SUNDIALS_HOME=$(pwd)/../../cvode-3.4-alpha/install
   export SUITE_SPARSE_HOME=$(pwd)/../../SuiteSparse
   export JSON_FORTRAN_HOME=$(pwd)/../../json-fortran-6.1.0/install/jsonfortran-gnu-6.1.0

   if [ "${BSC_MACHINE}" == "mn5" ]; then
     export JSON_FORTRAN_HOME=$(pwd)/../../json-fortran-6.1.0/install/jsonfortran-gnu-6.1.0
   else
     echo "Unknown architecture"
     exit
   fi

   initial_dir=$(pwd)
   cd  ../build
   if ! make -j ${NUMPROC}; then
     exit
   fi
   cd $initial_dir
}
time compile
cd ../build
cuda-memcheck mock_monarch 2>&1 | tee "../compile/a.txt"
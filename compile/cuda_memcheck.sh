compile(){
   export SUNDIALS_HOME=$(pwd)/../../cvode-3.4-alpha/install
   export SUITE_SPARSE_HOME=$(pwd)/../../SuiteSparse
   export JSON_FORTRAN_HOME=$(pwd)/../../json-fortran-6.1.0/install/jsonfortran-gnu-6.1.0

   if [ $BSC_MACHINE == "power" ]; then
     export JSON_FORTRAN_HOME=$(pwd)/../../json-fortran-6.1.0/install/jsonfortran-gnu-6.1.0
   elif [ $BSC_MACHINE == "mn4" ]; then
     export JSON_FORTRAN_HOME=$(pwd)/../../json-fortran-6.1.0/install/jsonfortran-intel-6.1.0
   else
     echo "Unknown architecture"
     exit
   fi

   curr_path=$(pwd)
   cd  ../build
   if ! make -j ${NUMPROC}; then
     exit
   fi
   cd $curr_path
}
time compile
cd ../build
cuda-memcheck mock_monarch 2>&1 | tee "../compile/a.txt"
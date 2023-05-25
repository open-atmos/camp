#BSC modules
#module load gcc/6.4.0
#module load openmpi/3.0.0
#module load hdf4/4.2.13
#module load hdf5/1.8.20
#module load pnetcdf/1.9.0
#module load netcdf/4.4.1.1
#module load lapack/3.8.0

library_path="../../../"
if [ "$1" == "from_camp_jobs" ]; then
  library_path="../../../../"
fi

cd $library_path/SuiteSparse
make purge
LOCAL_MACHINE=CGUZMAN
if [ $BSC_MACHINE == "power" ]; then
  make BLAS="-L${EBROOTOPENBLAS}/lib -lopenblas" LAPACK=""
elif [ $BSC_MACHINE == "mn4" ]; then
  make BLAS="-L${INTEL_HOME}/mkl/lib/intel64 -lmkl_intel_lp64 -lmkl_core -lmkl_intel_thread -lpthread -lm" LAPACK=""
elif [ LOCAL_MACHINE==CGUZMAN ]; then
  echo "WARNING: LOCAL_MACHINE==CGUZMAN"
  if ! dpkg-query -W -f='${Status}' libopenblas-dev 2>/dev/null | grep -q "installed"; then
    echo "Installing OPENBLAS"
    sudo apt install libopenblas-dev
  fi
  make BLAS="-L/lib64 -lopenblas"
else
  echo "Unknown architecture"
  exit
fi

export SUITE_SPARSE_CAMP_ROOT=$(pwd)/$library_path/

camp_folder=camp
if [ ! -z "$2" ]; then
  camp_folder=camp_jobs/camp$2
fi

if [ "$1" == "from_camp_jobs" ]; then
  cd ../$camp_folder/build/compile
fi

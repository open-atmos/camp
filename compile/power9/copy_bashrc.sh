# Source global definitions
if [ -f /etc/bashrc ]; then
	. /etc/bashrc
fi

# If not running interactively (non bash console gui), don't do anything. Used for exit bashrc when using autosubmit, avoiding to load unneeded modules
[ -z "$PS1" ] && return

if [ $BSC_MACHINE == "power" ]; then

  module purge
  #module unuse /apps/modules/modulefiles/applications #todo check if needed
  module use /gpfs/projects/bsc32/software/rhel/7.4/ppc64le/POWER9/modules/all/
  #module purge #todo check if needed
  module load bsc/commands
  module load CUDA/10.1.105-ES
  #Adding pgcc:
  #module load pgi
  #module load cmake
  #pgcc

  module load GCC/7.3.0-2.30
  module load OpenMPI/3.1.0-GCC-7.3.0-2.30
  module load JasPer/1.900.1-foss-2018b
  module load netCDF/4.6.1-foss-2018b
  module load netCDF-Fortran/4.4.4-foss-2018b
  module load ESMF/6.3.0rp1-foss-2018b
  module load ESMF/6.3.0rp1-foss-2018b
  module load CMake/3.15.3-GCCcore-7.3.0
  module load OpenBLAS/0.3.1-GCC-7.3.0-2.30
  module load Python/3.7.0-foss-2018b
  module load matplotlib/3.1.1-foss-2018b-Python-3.7.0
  #module load pgi
  #module load GSL/2.4-GCCcore-7.3.0
  export ALLINEA_FORCE_CUDA_VERSION=9.2 #needed for ddt cuda
  module load ddt

  load_ncview(){
    module load netcdf
    module load ncview
  }
#load_ncview

#QOS=""
QOS="--qos=debug"
#Uncomment for interactive session, which is faster for debugging than sbatch (useful specially for monarch or profiling)
#salloc --x11 $QOS --exclusive --tasks-per-node=160 --nodes=1 --gres=gpu:4
fi



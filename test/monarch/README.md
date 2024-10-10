CAMP GPU : Instructions for Marenostrum cluster
======

#TODO: Copy paste comments from CVODE and add information about differences
#TODO: One section: All options listed here in the README and some sections putting more context
#TODO: Organize files on folder (e.g. profileCPU folder)
#TODO: WARNING: Output is saved in this folder:
#TODO: Define each file and move the output files to the same folder (e.g. Extrae output with the stats output)
    # e.g. cells: 10000, mpi_processes: 2, gpu_load: 50
    # 5000->rank 0 ; 5000->rank 1
    # 2500->CPU rank 0; 2500->GPU rank 0; 2500->CPU rank 1; 2500->GPU rank 1

*If you previously run another branch, run `compile.sh `.*

Run `compile.sh ` and `run.sh ` for developing the GPU test.

We recommend to modify the file `TestMonarch.py` for testing the GPU version. It includes multiple configuration variables, such as number of cells, CPU or GPU version, MPI processes, etc.

The intended behaviour of `TestMonarch.py` is to compare the CPU and GPU versions. It runs both versions, saving the output concentrations and execution times, and using that data to calculate the accuracy error and speedup. 

# Re-use simulation data

You can set:

`conf.is_import = True`

To read previous experiments data. This is useful to develop the python interface, since it does not need to run the CPU or GPU case.

`conf.is_import_base = True`

To read just the base case data. This is useful to develop the GPU version since it avoids the execution of the CPU case.

# Profiling CPU

Run `python profileCPU.py` for an example of a 
CPU execution 1 day simulation.

At follows there is an explanation of the relevant
variables to tune the execution:

`conf.mpiProcessesCaseBase = x`

Where x is the desired number of MPI cores,

`conf.cells = [x]`

Where x represents the number of grid-cells to compute.
A cell is the minimal region obtained after the domain
decomposition of an Earth Science Model, such as
the atmosphere. It can be also referred as the _dx_ component
from a differential equation. It can also be referred as the systems of Ordinary Diferential Equation (ODE) to solve.

Each ODE computes the predicted value of a chemical specie.
Thus, there are as much ODE as chemical species.

`conf.timeSteps = x`

Where x is the number of time-steps to simulate. The
time-step size is set to 2 minutes.

For fast profiling a good number of time-steps could be
from 1 to 60.

For complete profiling, a good number is 720 time-steps,
which includes a full day simulation representing
daylight and night, where the chemical reactions change.

`conf.caseBase = "x"`

Where x may be `CPU One-cell` for CPU execution and
`GPU BDF` for GPU execution.

## Adding a profiling tool

You can add a profiling tool in a similar way than
it is implemented for the GPU profiling.

You can check file "mainMonarch.py" the code related
to the variable "ncu", like:

`if conf.profileCuda == "ncu":`

You should see that the command line that calls the
GPU profiler "ncu" is added to
the rest of the command line to run the program.

# Profiling GPU

Run `python profileGPU.py` for an example of profiling
the GPU with Nvidia profilers.

Set:

`conf.mpiProcessesCaseBase = x`

Where x is the desired MPI cores. For instance, use 1
to profile a single GPU, use all the available
cores on a node to profile all the GPUs.

`conf.cells = [x]`

Where x represents the number of grid-cells to compute.
See [Profiling CPU](#Profiling-CPU) for cells description.

`conf.timeSteps = X`

Where x is the number of time-steps. It is set to 1
because the performance over all time-steps
should be very similar, in addition that the profiling
already repeats multiple times the kernel for
profiling.

`conf.profileCuda = "x"`

Where x is "ncu" for using Nvidia Nsight Compute profiler.

`conf.caseBase = "x"`

Where is "GPU BDF" for GPU execution.
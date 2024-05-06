CAMP GPU : Instructions for [CTE-POWER](https://www.bsc.es/user-support/power.php) cluster
======

Run `run.sh `for developing the GPU test.

We recommend to modify the file `TestMonarch.py` for testing
the GPU version. It includes multiple configuration variables, 
such as number of cells, case, MPI processes, etc.

# Profiling CPU

Run `python profilingCPU.py` for an example of a 
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
from a differential equation. It can also be referred 
as the systems of Ordinary 
Diferential Equation (ODE) to solve.

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

# Profiling GPU

Run `python profilingGPU.py` for an example of profiling
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

Where x is "nsight" for using Nsight, and "nvprof" 
for the "nvprof" profiler, both from Nvidia.

`conf.caseBase = "x"`

Where is "GPU BDF" for GPU execution.
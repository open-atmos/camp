CAMP GPU : Instructions for [CTE-POWER](https://www.bsc.es/user-support/power.php) cluster
======

Run `run.sh `for developing the GPU test.

We recommend to modify the file `TestMonarch.py` for testing
the GPU version. It includes multiple configuration variables, 
such as number of cells, case, MPI processes, etc.

# Profiling

Run `python profiling.py` for an example of a 
CPU execution 1 day simulation.

At follows there is an explanation of the relevant
variables to tune the execution:

`conf.mpiProcessesCaseBase = x`

Where x is your desired number of MPI cores to use,

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
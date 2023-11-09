TestMonarch
======
List of configuration variables:

* conf = Class object with configuration variables.
* chemFile = Chemical mechanism (e.g. CB05, CB05+S0A...)
* diff_cells = "Realistic" for different initial conditions between cells. "Ideal" for the same.
* profileCuda = Profile GPU through nvprof.
* mpi = Use "mpirun" or not. Recommended to use default value "yes"
* mpiProcessesList = Number of MPI processes. The number of cells is divided between them.
* cells = List with total number of cells. The program is repeated for each cell in the list. This number is divided between the number of MPI processes.
* timesteps = Number of time-steps iterations. 
* time_step_dt = Time-step size [min]
* caseBase = Case to be compared with caseOptim (e.g. Speedup is defined as time caseBase/caseOptim)
* caseOptim = Case Optimized (e.g. Speedup is defined as time caseBase/caseOptim)
* plotYKey = Metric to evaluate (e.g. Speedup of the linear solver)

Plotting elements:
* Axe X: By default corresponds to "timesteps" value. It uses "cells" in case of more than one value in "cells". Then,  the mean over "time-steps" is calculated.
* Axe Y: Corresponding to "plotYKey".
* PlotTitle and Legend = Constructed automatically through conf. variables such as caseBase.
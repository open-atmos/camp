TEST_MONARCH_1
======
List of configuration variables:

* conf = Container with the configuration variables
* chem_file = Chemical mechanism (e.g. CB05, CB05+S0A...)
* diff_cells = "Realistic" for different initial conditions between cells. "Ideal" for the same.
* profileCuda = Profile GPU through nvprof.
* mpi = Use "mpirun" or not. Recommended to use default value "yes"
* mpiProcessesList = Number of MPI processes. The number of cells is divided between them.
* cells = Total number of cells.
* timesteps = Number of time-steps iterations.
* TIME_STEP = Time-step size [min]
* caseBase = Case to be compared with caseOptim (e.g. Speedup is defined as time caseBase/caseOptim)
* caseOptim = Case Optimized (e.g. Speedup is defined as time caseBase/caseOptim)
* plot_y_key = Metric to evaluate (e.g. Speedup of the linear solver)
* MAPE_tol = Tolerance for MAPE calculation. Should be equal to CVODE tolerance.

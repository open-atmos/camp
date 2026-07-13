This folder contains reports useful for possible optimizations.

cvode_vectorization: Reports vectorization issues of the CPU CAMP version running CVODE. Developed by victor.correal@bsc.es. For more info, see file "CAMP_CPU_PROFILING.pdf".

From the report and the file, we can summarize that there are some loops that can be vectorized. However, not all the loops reported are used on CAMP. We should only look at the functions used, related to the KLU linear solver and Newton iteration functions. To facilitate identify these functions, one can use a debugger or look at the GPU code. The GPU code should also be check to see if a vectorization improvement can be applied for both CPU and GPU solvers.

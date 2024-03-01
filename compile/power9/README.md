CAMP GPU : Instructions for [CTE-POWER](https://www.bsc.es/user-support/power.php) cluster
======

On the version for the Block-cells paper, ensure CAMP and CVODE are in the same version.

Run "./compile.libs.camp.sh" to compile CAMP from scratch including dependencies (SuiteSparse, JSON-fortran, CVODE and CAMP)

When switching from another version, ensure to recompile CAMP or the dependencies. 
You can use compile.camp.power9.sh and compile.cvode-3.4-alpha.power9.sh.

Run "./make.camp.power9.sh" to execute "TestMonarch.py" test

We recommend to use the file "TestMonarch.py" for testing the GPU branch. It includes multiple configuration variables, such as number of cells, case (One-cell,Multi-cells...), MPI processes, etc. More info about the test can be found at "camp/test/monarch/".

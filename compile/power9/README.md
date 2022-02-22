CAMP GPU : Instructions for [CTE-POWER](https://www.bsc.es/user-support/power.php) cluster
======

Run "./compile.libs.camp.sh" to compile CAMP and its dependencies (JSON,SUITESPARSE and CVODE)

Run "./make.camp.power9.sh" to execute "TestMonarch.py" of CAMP

We recommend to use the file "TestMonarch.py" for testing the GPU branch. It includes multiple configuration variables, such as number of cells, case (One-cell,Multi-cells...), MPI processes, etc. More info about the test can be found at "camp/test/monarch/".

CAMP
======

CAMP: Chemistry Across Multiple Phases

[![Latest version](https://img.shields.io/github/tag/open-atmos/camp.svg?label=version)](https://github.com/open-atmos/camp/blob/main/ChangeLog.md)
[![CI Status](https://github.com/open-atmos/camp/actions/workflows/main.yml/badge.svg)](https://github.com/open-atmos/camp/actions/workflows/main.yml)
[![License](https://img.shields.io/github/license/open-atmos/camp.svg)](https://github.com/open-atmos/camp/blob/main/COPYING)

<http://open-atmos.org/camp/>

The full CAMP documentation, including the BootCAMP tutorial is available [here](https://open-atmos.github.io/camp).

References:

   * [M. Dawson, C. Guzman, J. H. Curtis, M. Acosta, S. Zhu, D. Dabdub,
     A. Conley, M. West, N. Riemer, and O. Jorba (2022),
     Chemistry Across Multiple Phases (CAMP) version 1.0: An
     Integrated multi-phase chemistry model, Geosci. Model Dev. 15](https://doi.org/10.5194/gmd-15-3663-2022)

Copyright (C) 2017&ndash;2021 Barcelona Supercomputing Center and the
University of Illinois at Urbana&ndash;Champaign


Dependencies
============

Required dependencies:

   * Fortran 2008 compiler - <https://gcc.gnu.org/fortran/> or similar
   * CMake version 2.6.4 or higher - <http://www.cmake.org/>
   * json-fortran for JSON input file support -
     <https://github.com/jacobwilliams/json-fortran>
   * SuiteSparse - <http://faculty.cse.tamu.edu/davis/SuiteSparse/SuiteSparse-5.1.0.tar.gz>
   * A modified version of CVODE 3.1.2 - provided here in `cvode-3.4-alpha.tar.gz`

Optional dependencies:

   * GSL for Jacobian evaluation and random number generators -
     <http://www.gnu.org/software/gsl/>


Starting with CAMP
==================

# Installation and compilation
1. Create a camp_suite directory on MN5. This will be the main directory of the CAMP suite where will be located camp itself and its dependencies (e.g. jsonfortran, suitesparse, cvode)
2. Go in camp_suite/ and run `git clone ghttps://earth.bsc.es/gitlab/ac/camp.git` 
3. Go in camp_suite/camp and run `./install.sh start`. This will install and compile jsonfortran, suitesparse, cvode and compile camp and boxmodel (since these two components are already installed through the previous git-clone). You can `./install.sh` alone to print more information about this installation script.

# Running a first CAMP-boxmodel simulation
1. Go in camp_suite/camp/boxmodel and ensure the sbatch-related header of `submit_boxmodel_job` is as follows:
```bash
#!/bin/bash 
#SBATCH --job-name=boxmodel_v2
#SBATCH --account=bsc32
#SBATCH --qos=gp_debug
#SBATCH --time=0:10:00
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --ntasks=1
```
2. Run `sbatch submit_boxmodel_job config_examples/cb05-2prod-monodisperse-simple-organics`. The argument here corresponds to a directory where CAMP can find the two required JSON input files, namely `config.json` and `interface_boxmodel.json`. This will submit the CAMP-boxmodel job to MN5. When the job start, two `.out` and `.err` files will allow you to monitor the job while running. When the job finishes, these files are renamed into `out.log` and `err.log` and moved to the output directory created for your run in `/gpfs/scratch/bsc32/<username>/out/<job_id>` where are saved the CAMP output netcdf files. 

# CAMP Development

Validate your developments running check.sh from the compile/ folder
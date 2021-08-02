
CAMP
======

CAMP: Chemistry Across Multiple Phases

[![Latest version](https://img.shields.io/github/tag/open-atmos/camp.svg?label=version)](https://github.com/open-atmos/camp/blob/main/ChangeLog.md)
[![Docker build status](https://img.shields.io/docker/automated/open-atmos/camp.svg)](https://cloud.docker.com/swarm/compdyn/repository/docker/openatmos/camp/builds)
[![CI Status](https://github.com/open-atmos/camp/actions/workflows/main.yml/badge.svg)](https://github.com/open-atmos/camp/actions/workflows/main.yml)
[![License](https://img.shields.io/github/license/open-atmos/camp.svg)](https://github.com/open-atmos/camp/blob/main/COPYING)

Version -.-.-
Released -

<http://open-atmos.org/camp/>

References:

   * M. Dawson, C. Guzman, J. H. Curtis, M. Acosta, S. Zhu, D. Dabdub,
     A. Conley, M. West, N. Riemer, and O. Jorba (2021),
     Chemistry Across Multiple Phases (CAMP) version 1.0: An
     Integrated multi-phase chemistry model, in preparation

Copyright (C) 2017&ndash;2021 Barcelona Supercomputing Center and the
University of Illinois at Urbana&ndash;Champaign


Running CAMP with Docker
==========================

This is the fastest way to get running.

* **_Step 1:_** Install [Docker Community Edition](https://www.docker.com/community-edition).
    * On Linux and MacOS this is straightforward. [Download from here](https://store.docker.com/search?type=edition&offering=community).
    * On Windows the best version is [Docker Community Edition for Windows](https://store.docker.com/editions/community/docker-ce-desktop-windows), which requires Windows 10 Pro/Edu.

* **_Step 2:_** Run the CAMP test suite with:

```text
docker run -it --rm openatmos/camp bash -c 'cd /build; make test'
```


Dependencies
============

Required dependencies:

   * Fortran 2008 compiler - <https://gcc.gnu.org/fortran/> or similar
   * CMake version 2.6.4 or higher - <http://www.cmake.org/>
   * json-fortran for JSON input file support -
     <https://github.com/jacobwilliams/json-fortran>
   * SuiteSparse - <http://faculty.cse.tamu.edu/davis/SuiteSparse/SuiteSparse-5.1.0.tar.gz>
   * A modified version of CVODE 3.1.2 - <https://github.com/mattldawson/cvode>

Optional dependencies:

   * GSL for Jacobian evaluation and random number generators -
     <http://www.gnu.org/software/gsl/>


Installation
============

1. Install cmake, SuiteSparse, and CVODE (see above).

2. Unpack CAMP:

        tar xzvf camp-1.0.0.tar.gz

3. Change into the main CAMP directory (where this README file is
   located):

        cd camp-1.0.0

4. Make a directory called `build` and change into it:

        mkdir build
        cd build

5. If desired, set environment variables to indicate the install
   locations of supporting libraries. If running `echo $SHELL`
   indicates that you are running `bash`, then you can do something
   like:

        export JSON_FORTRAN_HOME=${HOME}/opt
        export SUITE_SPARSE_HOME=${HOME}/opt
        export SUNDIALS_HOME=${HOME}/opt
        export GSL_HOME=${HOME}/opt

   Of course the exact directories will depend on where the libraries
   are installed. You only need to set variables for libraries
   installed in non-default locations, and only for those libraries
   you want to use.

   If `echo $SHELL` instead is `tcsh` or similar, then the environment
   variables can be set like `setenv NETCDF_HOME /` and similarly.

6. Run cmake with the main CAMP directory as an argument (note the
   double-c):

        ccmake ..

7. Inside ccmake press `c` to configure, edit the values as needed,
   press `c` again, then `g` to generate. Optional libraries can be
   activated by setting the `ENABLE` variable to `ON`. For a parallel
   build, toggle advanced mode with `t` and set the
   `CMAKE_Fortran_COMPILER` to `mpif90`, then reconfigure.

8. Optionally, enable compiler warnings by pressing `t` inside ccmake
   to enable advanced options and then setting `CMAKE_Fortran_FLAGS`
   to:

        -O2 -g -fimplicit-none -W -Wall -Wconversion -Wunderflow -Wimplicit-interface -Wno-compare-reals -Wno-unused -Wno-unused-parameter -Wno-unused-dummy-argument -fbounds-check

8. Compile CAMP and test it as follows. Some tests may fail due to
   bad random initial conditions, so re-run the tests a few times to
   see if failures persist.

        make
        make test



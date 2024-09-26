#
# Copyright (C) 2022 Barcelona Supercomputing Center and University of
# Illinois at Urbana-Champaign
# SPDX-License-Identifier: MIT
#

from mainMonarch import *


def all_timesteps():
  conf = TestMonarch()
  conf.timeSteps = 1
  conf.loads_gpu = [50]
  #todo set maximum of code line length for python, C, C++, Fortran
  conf.cells = [1]
  conf.mpiProcessesCaseBase = 1
  conf.caseBase = "CPU One-cell"
  #conf.caseBase = "GPU BDF"
  conf.mpiProcessesCaseOptimList = [1]
  #conf.casesOptim = ["GPU BDF"] #todo trigger exception if this is commented
  #conf.is_import = True
  #conf.is_import_base = True
  #conf.profileCuda = "ncu" #todo comment Need allocated note on MN5
  #conf.profileCuda = "nsys"
  #conf.profileExtrae = True
  datay = run_main(conf)
  plot_cases(conf, datay)


if __name__ == "__main__":
  all_timesteps()
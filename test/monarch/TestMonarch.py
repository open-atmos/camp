#
# Copyright (C) 2022 Barcelona Supercomputing Center and University of
# Illinois at Urbana-Champaign
# SPDX-License-Identifier: MIT
#

from mainMonarch import *


def all_timesteps():
  conf = TestMonarch()
  conf.timeSteps = 1
  conf.loads_gpu = [1]
  conf.cells = [100]
  conf.mpiProcessesCaseBase = 10
  conf.caseBase = "CPU One-cell"
  #conf.caseBase = "GPU BDF"
  conf.mpiProcessesCaseOptimList = [1]
  #conf.casesOptim = ["GPU BDF"]
  #conf.is_import = True
  #conf.is_import_base = True
  #conf.profileCuda = "ncu"
  #conf.profileCuda = "nsys"
  #conf.profileExtrae = True
  datay = run_main(conf)
  plot_cases(conf, datay)


if __name__ == "__main__":
  all_timesteps()
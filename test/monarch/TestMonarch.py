#
# Copyright (C) 2022 Barcelona Supercomputing Center and University of
# Illinois at Urbana-Champaign
# SPDX-License-Identifier: MIT
#

from mainMonarch import *


def all_timesteps():
  conf = TestMonarch()
  conf.casesOptim = []
  conf.mpiProcessesCaseBase = 2
  conf.mpiProcessesCaseOptimList = [2]
  conf.cells = [100]
  conf.timeSteps = 10
  conf.gpu_percentage = 100
  conf.caseBase = "CPU One-cell"
  #conf.caseBase = "GPU BDF"
  #conf.casesOptim = ["CPU One-cell"]
  conf.casesOptim = ["GPU BDF"]
  #conf.is_import = True
  conf.is_import_base = True
  #conf.profileCuda = "nsight"
  run_main(conf)
  #plot_cases(conf, run_main(conf))


if __name__ == "__main__":
  all_timesteps()

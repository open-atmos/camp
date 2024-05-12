#
# Copyright (C) 2022 Barcelona Supercomputing Center and University of
# Illinois at Urbana-Champaign
# SPDX-License-Identifier: MIT
#

from mainMonarch import *


def all_timesteps():
  conf = TestMonarch()
  conf.casesOptim = []
  conf.mpiProcessesCaseBase = 1
  conf.mpiProcessesCaseOptimList = [1]
  conf.cells = [1]
  conf.timeSteps = 1
  conf.caseBase = "CPU One-cell"
  #conf.caseBase = "GPU BDF"
  #conf.casesOptim = ["CPU One-cell"]
  conf.casesOptim = ["GPU BDF"]
  #conf.is_import = True
  #conf.is_import_base = True
  run_main(conf)
  #plot_cases(conf, run_main(conf))


if __name__ == "__main__":
  all_timesteps()

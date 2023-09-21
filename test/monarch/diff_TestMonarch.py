#
# Copyright (C) 2022 Barcelona Supercomputing Center and University of
# Illinois at Urbana-Champaign
# SPDX-License-Identifier: MIT
#

from run import *


def all_timesteps():
  conf = TestMonarch()

  conf.chemFile = "cb05_paperV2"
  #conf.chemFile = "monarch_cb05"

  conf.diffCellsL = []
  conf.diffCellsL.append("Realistic")
  #conf.diffCellsL.append("Ideal")

  conf.profileCuda = ""

  #conf.is_import = True

  conf.nGPUsCaseBase = 1
  # conf.nGPUsCaseBase = 2

  conf.nGPUsCaseOptimList = [1]
  # conf.nGPUsCaseOptimList = [1,2]

  conf.mpiProcessesCaseBase = 1
  # conf.mpiProcessesCaseBase = 2

  conf.mpiProcessesCaseOptimList.append(1)
  # conf.mpiProcessesCaseOptimList.append(2)

  conf.allocatedNodes = 1

  conf.allocatedTasksPerNode = 160

  conf.cells = [2]

  conf.timeSteps = 3

  conf.timeStepsDt = 2

  conf.caseBase = "CPU One-cell"
  conf.casesOptim = []
  conf.plotYKey = "NRMSE"

  """END OF CONFIGURATION VARIABLES"""

  check_run(conf)


if __name__ == "__main__":
  all_timesteps()

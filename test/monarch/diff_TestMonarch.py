#
# Copyright (C) 2022 Barcelona Supercomputing Center and University of
# Illinois at Urbana-Champaign
# SPDX-License-Identifier: MIT
#

from mainMonarch import *


def all_timesteps():
  conf = TestMonarch()
  conf.chemFile = "cb05_paperV2"
  #conf.chemFile = "monarch_cb05"
  conf.diffCellsL = []
  conf.diffCellsL.append("Realistic")
  #conf.diffCellsL.append("Ideal")
  conf.profileCuda = ""
  conf.nGPUsCaseBase = 1
  conf.nGPUsCaseOptimList = [1]
  conf.mpiProcessesCaseBase = 1
  conf.mpiProcessesCaseOptimList.append(1)
  conf.allocatedNodes = 1
  conf.allocatedTasksPerNode = 160
  conf.cells = [2]
  conf.timeSteps = 3
  conf.caseBase = "CPU One-cell"
  conf.casesOptim = []
  conf.plotYKey = "NRMSE"
  """END OF CONFIGURATION VARIABLES"""
  run_main(conf)


if __name__ == "__main__":
  all_timesteps()

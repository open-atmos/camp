#
# Copyright (C) 2022 Barcelona Supercomputing Center and University of
# Illinois at Urbana-Champaign
# SPDX-License-Identifier: MIT
#

from mainMonarch import *


def checkGPU():
  conf = TestMonarch()
  conf.chemFile = "cb05_paperV2"
  conf.diffCellsL = []
  conf.diffCellsL.append("Realistic")
  conf.nGPUsCaseBase = 1
  conf.nGPUsCaseOptimList = [1]
  conf.mpiProcessesCaseBase = 1
  conf.mpiProcessesCaseOptimList.append(1)
  conf.allocatedNodes = 1
  conf.allocatedTasksPerNode = 160
  conf.cells = [10]
  conf.timeSteps = 3
  conf.timeStepsDt = 2
  conf.caseBase = "CPU One-cell"
  conf.casesOptim = []
  conf.casesOptim.append("GPU BDF")
  conf.plotYKey = "Speedup timecvStep"
  """END OF CONFIGURATION VARIABLES"""
  run_main(conf)

if __name__ == "__main__":
  checkGPU()
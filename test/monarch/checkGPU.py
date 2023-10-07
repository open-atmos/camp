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
  conf.nGPUsCaseOptimList = [2]
  conf.mpiProcessesCaseBase = 20
  conf.mpiProcessesCaseOptimList.append(20)
  conf.allocatedNodes = 1
  conf.allocatedTasksPerNode = 160
  conf.cells = [40]
  conf.timeSteps = 3
  conf.timeStepsDt = 2
  conf.caseBase = "CPU One-cell"
  conf.casesOptim = []
  conf.casesOptim.append("GPU BDF")
  conf.plotYKey = "Speedup"
  """END OF CONFIGURATION VARIABLES"""
  run_main(conf)

if __name__ == "__main__":
  checkGPU()
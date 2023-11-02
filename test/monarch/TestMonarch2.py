#
# Copyright (C) 2022 Barcelona Supercomputing Center and University of
# Illinois at Urbana-Champaign
# SPDX-License-Identifier: MIT
#

from checkGPU import *


def all_timesteps():
  conf = TestMonarch()
  conf.chemFile = "cb05_paperV2"
  # conf.chemFile = "monarch_cb05"
  conf.diffCellsL = []
  conf.diffCellsL.append("Realistic")
  # conf.diffCellsL.append("Ideal")
  conf.profileCuda = ""
  # conf.profileCuda = "nvprof"
  # conf.profileCuda = "nsight"
  #conf.is_import = True
  conf.nGPUsCaseBase = 1
  conf.nGPUsCaseOptimList = [2]
  conf.mpiProcessesCaseBase = 20
  conf.mpiProcessesCaseOptimList = [20]
  conf.allocatedNodes = 1
  conf.allocatedTasksPerNode = 160
  conf.cells = [100000]
  # conf.cells = [100, 500, 1000, 5000, 10000]
  conf.timeSteps = 720
  conf.caseBase = "CPU One-cell"
  # conf.caseBase = "GPU BDF"
  conf.casesOptim = []
  # conf.casesOptim.append("CPU One-cell")
  # conf.casesOptim.append("CPU EBI")
  conf.casesOptim.append("GPU BDF")
  conf.plotYKey = "Speedup timecvStep"
  # conf.plotXKey = "GPUs"

  run_main(conf)


if __name__ == "__main__":
  all_timesteps()

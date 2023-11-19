#
# Copyright (C) 2022 Barcelona Supercomputing Center and University of
# Illinois at Urbana-Champaign
# SPDX-License-Identifier: MIT
#

from mainMonarch import *


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
  conf.is_import = True
  conf.mpiProcessesCaseBase = 20
  conf.mpiProcessesCaseOptimList = [20]
  conf.cells = [500]
  conf.timeSteps = 1
  conf.caseBase = "CPU One-cell"
  # çconf.caseBase = "GPU BDF"
  conf.casesOptim = []
  # conf.casesOptim.append("CPU One-cell")
  conf.casesOptim.append("GPU BDF")
  conf.plotYKey = "Speedup timecvStep"

  run_main(conf)


if __name__ == "__main__":
  all_timesteps()

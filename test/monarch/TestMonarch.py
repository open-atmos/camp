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
  # conf.profileCuda = "nvprof"
  #conf.profileCuda = "nsight"
  # conf.profileCuda = "nsightSummary"

  #conf.is_import = True

  # conf.commit = "MATCH_IMPORTED_CONF"
  conf.commit = ""

  conf.nGPUsCaseBase = 1
  # conf.nGPUsCaseBase = 2

  # conf.nGPUsCaseOptimList = [1]
  conf.nGPUsCaseOptimList = [1]
  # conf.nGPUsCaseOptimList = [1,2]

  conf.mpiProcessesCaseBase = 1
  # conf.mpiProcessesCaseBase = 2

  conf.mpiProcessesCaseOptimList.append(1)
  # conf.mpiProcessesCaseOptimList.append(2)
  # conf.mpiProcessesCaseOptimList = [10,20,40]

  conf.allocatedNodes = 1
  # conf.allocatedNodes = 4
  # conf.allocatedNodes = get_allocatedNodes_sbatch() #todo

  conf.allocatedTasksPerNode = 160
  # conf.allocatedTasksPerNode = 40
  # conf.allocatedTasksPerNode = 320
  # conf.allocatedTasksPerNode = get_ntasksPerNode_sbatch() #todo

  conf.cells = [20]
  # conf.cells = [100, 500, 1000, 5000, 10000]
  # conf.cells = [50000,100000,500000,1000000]

  conf.timeSteps = 5

  conf.timeStepsDt = 2

  # conf.caseBase = "CPU EBI"
  conf.caseBase = "CPU One-cell"
  # conf.caseBase = "CPU Multi-cells"
  # conf.caseBase = "CPU New"
  # conf.caseBase="GPU Multi-cells"
  # conf.caseBase="GPU Block-cellsN"
  # conf.caseBase="GPU Block-cells1"
  #conf.caseBase = "GPU BDF"
  # conf.caseBase = "GPU CPU"
  # conf.caseBase = "GPU maxrregcount-64" #wrong 10,000 cells
  # conf.caseBase = "GPU maxrregcount-24" #Minimum
  # conf.caseBase = "GPU maxrregcount-62"
  # conf.caseBase = "GPU maxrregcount-68"
  # conf.caseBase = "GPU maxrregcount-48"

  conf.casesOptim = []
  # conf.casesOptim.append("CPU One-cell")
  # conf.casesOptim.append("CPU Multi-cells")
  # conf.casesOptim.append("CPU New")
  # conf.casesOptim.append("GPU One-cell")
  # conf.casesOptim.append("GPU Multi-cells")
  # conf.casesOptim.append("GPU Block-cellsNhalf")
  # conf.casesOptim.append("GPU Block-cellsN")
  # conf.casesOptim.append("GPU Block-cells1")
  # conf.casesOptim.append("CPU EBI")
  conf.casesOptim.append("GPU BDF")
  # conf.casesOptim.append("GPU CPU")
  # conf.casesOptim.append("GPU maxrregcount-64") #wrong 10,000 cells
  # conf.casesOptim.append("GPU maxrregcount-68")
  # conf.casesOptim.append("GPU maxrregcount-62")
  # conf.casesOptim.append("GPU maxrregcount-24")
  # conf.casesOptim.append("CPU IMPORT_NETCDF")

  # conf.plotYKey = "Speedup timeCVode"
  # conf.plotYKey = "Speedup normalized counterLS"
  # conf.plotYKey = "Speedup normalized timeLS"
  # conf.plotYKey = "Speedup normalized computational timeLS"
  # conf.plotYKey = "Speedup counterBCG"
  # conf.plotYKey = "Speedup normalized counterBCG"
  # conf.plotYKey = "Speedup total iterations - counterBCG"
  # conf.plotYKey = "Speedup BCG iteration (Comp.timeLS/counterBCG)"
  #conf.plotYKey = "Speedup timecvStep"
  # conf.plotYKey = "Speedup timecvStep normalized by countercvStep"
  # conf.plotYKey = "Speedup countercvStep"
  # conf.plotYKey = "Speedup device timecvStep"
  # conf.plotYKey = "Percentage data transfers CPU-GPU [%]"
  conf.plotYKey = "NRMSE"

  #conf.use_monarch = True #better run stats_monarch_netcdf.py

  #conf.is_export_netcdf = True

  # conf.plotXKey = "MPI processes"
  # conf.plotXKey = "GPUs"

  """END OF CONFIGURATION VARIABLES"""

  check_run(conf)

if __name__ == "__main__":
  all_timesteps()

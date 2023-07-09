#
# Copyright (C) 2022 Barcelona Supercomputing Center and University of
# Illinois at Urbana-Champaign
# SPDX-License-Identifier: MIT
#

from run import *

def all_timesteps():
    conf = TestMonarch()

    # conf.chemFile = "simple"
    conf.chemFile = "monarch_cb05"
    #conf.chemFile = "monarch_binned"

    conf.diffCellsL = []
    conf.diffCellsL.append("Realistic")
    # conf.diffCellsL.append("Ideal")

    conf.profileCuda = ""
    #conf.profileCuda = "nvprof"
    #conf.profileCuda = "nsight"
    #conf.profileCuda = "nsightSummary"

    conf.is_export = get_is_sbatch()
    # conf.is_export = True
    # conf.is_export = False

    # conf.is_import = True
    conf.is_import = False

    # conf.commit = "MATCH_IMPORTED_CONF"
    conf.commit = ""

    conf.nGPUsCaseBase = 1
    # conf.nGPUsCaseBase = 2

    # conf.nGPUsCaseOptimList = [1]
    conf.nGPUsCaseOptimList = [1]
    # conf.nGPUsCaseOptimList = [1,2]

    conf.mpi = "yes"
    # conf.mpi = "no"

    conf.mpiProcessesCaseBase = 1
    #conf.mpiProcessesCaseBase = 2

    conf.mpiProcessesCaseOptimList.append(1)
    #conf.mpiProcessesCaseOptimList.append(2)
    # conf.mpiProcessesCaseOptimList = [10,20,40]

    conf.allocatedNodes = 1
    # conf.allocatedNodes = 4
    # conf.allocatedNodes = get_allocatedNodes_sbatch() #todo

    conf.allocatedTasksPerNode = 160
    # conf.allocatedTasksPerNode = 40
    # conf.allocatedTasksPerNode = 320
    # conf.allocatedTasksPerNode = get_ntasksPerNode_sbatch() #todo

    conf.cells = [10]
    # conf.cells = [100, 500, 1000, 5000, 10000]
    # conf.cells = [50000,100000,500000,1000000]

    conf.timeSteps = 10
    #conf.timeSteps = 720

    conf.timeStepsDt = 2

    # conf.caseBase = "CPU EBI"
    conf.caseBase = "CPU One-cell"
    #conf.caseBase = "CPU Multi-cells"
    #conf.caseBase = "CPU New"
    # conf.caseBase="GPU Multi-cells"
    # conf.caseBase="GPU Block-cellsN"
    # conf.caseBase="GPU Block-cells1"
    #conf.caseBase = "GPU BDF"
    #conf.caseBase = "GPU CPU"
    # conf.caseBase = "GPU maxrregcount-64" #wrong 10,000 cells
    # conf.caseBase = "GPU maxrregcount-24" #Minimum
    # conf.caseBase = "GPU maxrregcount-62"
    # conf.caseBase = "GPU maxrregcount-68"
    # conf.caseBase = "GPU maxrregcount-48"

    conf.casesOptim = []
    #conf.casesOptim.append("CPU One-cell")
    #conf.casesOptim.append("CPU Multi-cells")
    #conf.casesOptim.append("CPU New")
    #conf.casesOptim.append("GPU One-cell")
    # conf.casesOptim.append("GPU Multi-cells")
    # conf.casesOptim.append("GPU Block-cellsNhalf")
    # conf.casesOptim.append("GPU Block-cellsN")
    # conf.casesOptim.append("GPU Block-cells1")
    # conf.casesOptim.append("CPU EBI")
    conf.casesOptim.append("GPU BDF")
    #conf.casesOptim.append("GPU CPU")
    # conf.casesOptim.append("GPU maxrregcount-64") #wrong 10,000 cells
    # conf.casesOptim.append("GPU maxrregcount-68")
    # conf.casesOptim.append("GPU maxrregcount-62")
    # conf.casesOptim.append("GPU maxrregcount-24")
    #conf.casesOptim.append("CPU IMPORT_NETCDF")

    #conf.plotYKey = "Speedup timeCVode"
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
    conf.plotYKey = "MAPE"
    # conf.plotYKey ="SMAPE"
    # conf.plotYKey ="NRMSE"
    # conf.MAPETol = 1.0E-6

    # conf.plotXKey = "MPI processes"
    # conf.plotXKey = "GPUs"

    """END OF CONFIGURATION VARIABLES"""

    # Utility functions
    # remove_to_tmp(conf,"1661337164911019079")
    conf.results_file = "_solver_stats.csv"
    if conf.plotYKey == "NRMSE" or conf.plotYKey == "MAPE" or conf.plotYKey == "SMAPE":
        conf.results_file = '_results_all_cells.csv'
        conf.is_export = False
        conf.is_import = False
    jsonFile = open("settings/monarch_box_binned/cb05_abs_tol.json")
    jsonData = json.load(jsonFile)
    conf.MAPETol = jsonData["camp-data"][0]["value"]  # Default: 1.0E-4
    jsonData.clear()
    if conf.plotYKey == "":
        print("conf.plotYKey is empty")
    if conf.chemFile == "monarch_binned":
        if conf.timeStepsDt != 2:
            print("Warning: Setting timeStepsDt to 2, since it is the usual value for monarch_binned")
        conf.timeStepsDt = 2
    elif conf.chemFile == "monarch_cb05" or conf.chemFile == "cb05_mechanism_yarwood2005":
        conf.timeStepsDt = 3
        if "Realistic" in conf.diffCellsL:
            conf.diffCellsL = ["Ideal"]
    elif conf.chemFile == "cb05_mechanism_yarwood2005":
        print("ERROR: Not tested in testmonarch.py, configuration taken from monarch branch 209 and tested in monarch for the camp paper")
        raise
    if not conf.caseBase:
        print("ERROR: caseBase is empty")
        raise
    if conf.caseBase == "CPU EBI":
        print("Warning: Disable CAMP_PROFILING in CVODE to better profiling")
    if conf.caseBase == "CPU EBI" and conf.chemFile != "monarch_cb05":
        print("Error: Set conf.chemFile = monarch_cb05 to run CPU EBI")
        raise Exception
    for caseOptim in conf.casesOptim:
        if caseOptim == "CPU EBI":
            print("Warning: Disable CAMP_PROFILING in CVODE to better profiling")
        if caseOptim == "CPU EBI" and conf.chemFile != "monarch_cb05":
            print("Error: Set conf.chemFile = monarch_cb05 to run CPU EBI")
            raise Exception
    for i, mpiProcesses in enumerate(conf.mpiProcessesCaseOptimList):
        for j, cellsProcesses in enumerate(conf.cells):
            nCells = int(cellsProcesses / mpiProcesses)
            if nCells == 0:
                print("WARNING: Configured less cells than MPI processes, setting 1 cell per process")
                conf.mpiProcessesCaseOptimList[i] = cellsProcesses

    run_diffCells(conf)

    if get_is_sbatch() is False:
        plot_cases(conf)


if __name__ == "__main__":
  all_timesteps()

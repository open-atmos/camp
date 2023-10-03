import matplotlib as mpl

mpl.use('TkAgg')
# import plot_functions #comment to save ~2s execution time
import math_functions
import os
import numpy as np
import json
import subprocess
import time
from pandas import read_csv as pd_read_csv


class TestMonarch:
  def __init__(self):
    # Case configuration
    self.chemFile = "cb05_paperV2"
    self.diffCells = ""
    self.timeSteps = 1
    self.timeStepsDt = 2
    self.case = []
    self.nCells = 1
    self.caseGpuCpu = ""
    self.caseMulticellsOnecell = ""
    self.mpiProcesses = 1
    self.allocatedNodes = 1
    self.allocatedTasksPerNode = 160
    self.nGPUs = 1
    # Cases configuration
    self.is_start_cases_attributes = True
    self.diffCellsL = ""
    self.mpiProcessesCaseBase = 1
    self.mpiProcessesCaseOptimList = []
    self.nGPUsCaseOptimList = [1]
    self.cells = [100]
    self.caseBase = ""
    self.casesOptim = [""]
    self.plotYKey = ""
    self.plotXKey = ""
    self.is_import = False
    self.profileCuda = ""
    self.is_out = True
    # Auxiliary
    self.is_start_auxiliary_attributes = True
    self.sbatch_job_id = ""
    self.outBase = []
    self.outOptim = []
    self.datacolumns = []
    self.stdColumns = []
    self.exportPath = "exports"
    self.legend = []
    self.results_file = "_solver_stats.csv"
    self.plotTitle = ""
    self.nCellsProcesses = 1
    self.campSolverConfigFile = "settings/config_variables_c_solver.txt"

def read_csv(conf,data_path):
  with open(data_path) as f:
    if conf.case is conf.caseBase:
      conf.outBase = [float(line.rstrip('\n')) for line in f]
    else:
      conf.outOptim = [float(line.rstrip('\n')) for line in f]


def write_camp_config_file(conf):
  try:
    file1 = open(conf.campSolverConfigFile, "w")
    if conf.caseGpuCpu == "CPU":
      file1.write("USE_CPU=ON\n")
    else:
      file1.write("USE_CPU=OFF\n")
    file1.write(str(conf.nGPUs) + "\n")
    file1.write("IS_EXPORT_STATE=ON\n")
    file1.close()
  except Exception as e:
    print("write_camp_config_file fails", e)


def run(conf):
  if conf.caseGpuCpu == "GPU":
    maxCoresPerNode = 40
    if conf.mpiProcesses > maxCoresPerNode and conf.mpiProcesses % maxCoresPerNode != 0:
      print(
        "ERROR: MORE THAN 40 MPI PROCESSES AND NOT MULTIPLE OF 40, WHEN CTE-POWER ONLY HAS 40 CORES PER NODE\n");
      raise
    maxnDevices = 4
    maxCoresPerDevice = maxCoresPerNode / maxnDevices
    maxCores = int(maxCoresPerDevice * conf.nGPUs)
    if conf.mpiProcesses != maxCores and (conf.mpiProcesses != 1 and maxCores == 10):
      print("WARNING: conf.mpiProcesses != maxCores, ",
            conf.mpiProcesses, "!=", maxCores,
            "conf.mpiProcesses changed from ", conf.mpiProcesses, "to ", maxCores)
      conf.mpiProcesses = maxCores
      conf.mpiProcessesCaseOptimList[0] = maxCores
      raise
  exec_str = ""
  try:
    ddt_pid = subprocess.check_output('pidof -x $(ps cax | grep ddt)', shell=True)
    if ddt_pid:
      exec_str += 'ddt --connect '
  except Exception:
    pass
  exec_str += "mpirun -v -np " + str(conf.mpiProcesses) + " --bind-to core "
  if conf.profileCuda == "nvprof" and conf.caseGpuCpu == "GPU":
    pathNvprof = "../../compile/power9/" + conf.caseMulticellsOnecell \
                 + str(conf.nCells) + "Cells" + ".nvprof "
    exec_str += "nvprof --analysis-metrics -f -o " + pathNvprof
    print("Saving profiling file in ", os.path.abspath(os.getcwd()) \
          + "/" + pathNvprof + ".nvprof")
  elif conf.profileCuda == "nsight" and conf.caseGpuCpu == "GPU":
    exec_str += "/apps/NVIDIA-HPC-SDK/21.3/Linux_ppc64le/21.3/profilers/Nsight_Compute/ncu "
    pathNvprof = "../../compile/power9/" + conf.caseMulticellsOnecell \
                 + str(conf.nCells) + "Cells "
    exec_str += "--set full -f -o " + pathNvprof  # last working version
    print("Saving nsight file in ", os.path.abspath(os.getcwd()) \
          + "/" + pathNvprof + ".ncu-rep")
  path_exec = "../../build/mock_monarch"
  exec_str += path_exec
  write_camp_config_file(conf)
  print("exec_str:", exec_str, conf.diffCells, conf.caseGpuCpu,
        conf.caseMulticellsOnecell, "ncellsPerMPIProcess:",
        conf.nCells, "nGPUs:", conf.nGPUs)
  conf_name = "settings/TestMonarch.json"
  with open(conf_name, 'w', encoding='utf-8') as jsonFile:
    json.dump(conf.__dict__, jsonFile, indent=4, sort_keys=False)
  nCellsStr = str(conf.nCells)
  if conf.nCells >= 1000:
    nCellsStr = str(int(conf.nCells/1000))+"k"
  if conf.caseGpuCpu == "GPU":
    caseGpuCpuName = str(conf.nGPUs) + conf.caseGpuCpu
  else:
    caseGpuCpuName = str(conf.mpiProcesses) + "CPUcores"
  if not conf.is_import:
    os.system(exec_str)
  if conf.is_out:
    data_path = "out/state" + caseGpuCpuName + nCellsStr + "cells" \
                + str(conf.timeSteps) + "tsteps.csv"
    try:
      if not conf.is_import:
        os.rename("out/state.csv", data_path)
      start = time.time()
      #df = pd_read_csv(data_path)
      #conf.outBase = df.to_dict('list')
      with open(data_path) as f:
        if conf.case is conf.caseBase:
          conf.outBase = [float(line.rstrip('\n')) for line in f]
        else:
          conf.outOptim = [float(line.rstrip('\n')) for line in f]
      print("read state + to_dict", time.time() - start)
    except FileNotFoundError as e:
      raise FileNotFoundError("Check enable EXPORT_STATE in CAMP code") from e
  data_path = "out/stats" + caseGpuCpuName + nCellsStr + "cells" \
              + str(conf.timeSteps) + "tsteps.csv"
  if not conf.is_import:
    os.rename("out/stats.csv", data_path)
  nRows_csv = conf.timeSteps * conf.nCells * conf.mpiProcesses
  df = pd_read_csv(data_path, nrows=nRows_csv)
  data = df.to_dict('list')
  y_key_words = conf.plotYKey.split()
  y_key = y_key_words[-1]
  data = data[y_key]
  return data


def run_cases(conf):
  # Base case
  conf.mpiProcesses = conf.mpiProcessesCaseBase
  if conf.nCellsProcesses % conf.mpiProcesses != 0:
    print("ERROR: On base case conf.nCellsProcesses % conf.mpiProcesses != 0, nCellsProcesses, mpiProcesses",
          conf.nCellsProcesses,
          conf.mpiProcesses)
    raise
  conf.nCells = int(conf.nCellsProcesses / conf.mpiProcesses)
  conf.nGPUs = conf.nGPUsCaseBase

  cases_words = conf.caseBase.split()
  conf.caseGpuCpu = cases_words[0]
  conf.caseMulticellsOnecell = cases_words[1]

  conf.case = conf.caseBase
  baseData = run(conf)

  # OptimCases
  datacases = []
  stdCases = []
  for nGPUs in conf.nGPUsCaseOptimList:
    conf.nGPUs = nGPUs
    for mpiProcessesCaseOptim in conf.mpiProcessesCaseOptimList:
      conf.mpiProcesses = mpiProcessesCaseOptim
      if conf.nCellsProcesses % conf.mpiProcesses != 0:
        print("WARNING: On optim case conf.nCellsProcesses % conf.mpiProcesses != 0,nCellsProcesses, mpiProcesses",
              conf.nCellsProcesses, conf.mpiProcesses)
      conf.nCells = int(conf.nCellsProcesses / conf.mpiProcesses)
      for caseOptim in conf.casesOptim:
        cases_words = caseOptim.split()
        conf.caseGpuCpu = cases_words[0]
        conf.caseMulticellsOnecell = cases_words[1]
        conf.case = caseOptim
        optimData = run(conf)
        if conf.is_out:
          math_functions.check_NRMSE(conf.outBase, conf.outOptim, conf.nCellsProcesses)
        datay = [0.] * len(optimData)
        for i in range(len(optimData)):
          datay[i] = baseData[i] / optimData[i]
        if len(conf.cells) > 1 or conf.plotXKey == "GPUs":
          datacases.append(np.mean(datay))
          stdCases.append(np.std(datay))
        else:
          datacases.append([elem for elem in datay])
  return datacases, stdCases


def run_cells(conf):
  datacells = []
  stdCells = []
  for i in range(len(conf.cells)):
    conf.nCellsProcesses = conf.cells[i]
    datacases, stdCases = run_cases(conf)
    if len(conf.cells) > 1 or conf.plotXKey == "GPUs":
      datacells.append(datacases)
      stdCells.append(stdCases)
    else:
      datacells = datacases
      stdCells = stdCases
  if len(conf.cells) > 1:
    datacellsTranspose = np.transpose(datacells)
    datacells = datacellsTranspose.tolist()
    stdCellsTranspose = np.transpose(stdCells)
    stdCells = stdCellsTranspose.tolist()
  return datacells, stdCells


def run_diffCells(conf):
  conf.datacolumns = []
  conf.stdColumns = []
  for i, diff_cells in enumerate(conf.diffCellsL):
    conf.diffCells = diff_cells
    datacells, stdcells = run_cells(conf)
    conf.datacolumns += datacells
    conf.stdColumns += stdcells


def plot_cases(conf):
  cases_words = conf.caseBase.split()
  conf.caseGpuCpu = cases_words[0]
  conf.caseMulticellsOnecell = cases_words[1]
  conf.legend = []
  cases_words = conf.casesOptim[0].split()
  conf.caseGpuCpu = cases_words[0]
  conf.caseMulticellsOnecell = cases_words[1]
  last_arch_optim = conf.caseGpuCpu
  last_case_optim = conf.caseMulticellsOnecell
  is_same_arch_optim = True
  is_same_case_optim = True
  for caseOptim in conf.casesOptim:
    cases_words = caseOptim.split()
    conf.caseGpuCpu = cases_words[0]
    conf.caseMulticellsOnecell = cases_words[1]
    if last_arch_optim != conf.caseGpuCpu:
      is_same_arch_optim = False
    last_arch_optim = conf.caseGpuCpu
    if last_case_optim != conf.caseMulticellsOnecell:
      is_same_case_optim = False
    last_case_optim = conf.caseMulticellsOnecell
  is_same_diff_cells = False
  for diff_cells in conf.diffCellsL:
    conf.diffCells = diff_cells
    for nGPUs in conf.nGPUsCaseOptimList:
      for mpiProcessesCaseOptim in conf.mpiProcessesCaseOptimList:
        for caseOptim in conf.casesOptim:
          cases_words = caseOptim.split()
          conf.caseGpuCpu = cases_words[0]
          conf.caseMulticellsOnecell = cases_words[1]
          case_multicells_onecell_name = ""
          if conf.caseMulticellsOnecell.find("BDF") != -1 or conf.caseMulticellsOnecell.find(
              "maxrregcount") != -1:
            is_same_diff_cells = True
          legend_name = ""
          if len(conf.diffCellsL) > 1:
            legend_name += conf.diffCells + " "
          if len(conf.nGPUsCaseOptimList) > 1 and conf.caseGpuCpu == "GPU" \
              and len(conf.cells) > 1:
            legend_name += str(nGPUs) + " GPU "
          elif not is_same_arch_optim:
            legend_name += conf.caseGpuCpu + " "
          if not is_same_case_optim:
            legend_name += case_multicells_onecell_name
          if not legend_name == "":
            conf.legend.append(legend_name)
  conf.plotTitle = ""
  if not is_same_diff_cells and len(conf.diffCellsL) == 1:
    conf.plotTitle += conf.diffCells + " test: "
  if is_same_arch_optim:
    if conf.plotXKey == "GPUs":
      conf.plotTitle += ""
    else:
      if conf.caseGpuCpu == "GPU" and len(conf.nGPUsCaseOptimList) == 1 and conf.nGPUsCaseOptimList[0] > 1:
        conf.plotTitle += str(conf.nGPUsCaseOptimList[0]) + " GPUs "
      else:
        conf.plotTitle += conf.caseGpuCpu + " "
  if conf.plotXKey == "GPUs":
    conf.plotTitle += "GPU "
  if len(conf.legend) == 1 or not conf.legend or len(conf.diffCellsL) > 1:
    if len(conf.mpiProcessesCaseOptimList) > 1:
      legend_name += str(mpiProcessesCaseOptim) + " MPI "
    if len(conf.diffCellsL) > 1:
      conf.plotTitle += "Implementations "
  else:
    conf.plotTitle += "Implementations "
  namey = conf.plotYKey
  if len(conf.cells) > 1:
    namey += " [Mean and \u03C3]"
    conf.plotTitle += ""
    datax = conf.cells
    plot_x_key = "Cells"
  elif conf.plotXKey == "GPUs":
    if len(conf.cells) > 1:
      conf.plotTitle += ", Cells: " + str(conf.cells[0])
      datax = conf.nGPUsCaseOptimList
      plot_x_key = conf.plotXKey
    else:
      datax = conf.nGPUsCaseOptimList
      plot_x_key = "GPUs"
  else:
    conf.plotTitle += ", Cells: " + str(conf.cells[0])
    datax = list(range(1, conf.timeSteps + 1, 1))
    plot_x_key = "Timesteps"
  namex = plot_x_key
  datay = conf.datacolumns
  if conf.allocatedNodes != 1:
    print("Nodes:", conf.allocatedNodes)
  if namex == "Timesteps":
    print("Mean:", format(np.mean(datay), '.2e'), "Std", format(np.std(datay), '.2e'))
  else:
    print("Std", conf.stdColumns)
  print(namex, ":", datax[-1])
  if conf.legend:
    print("plotTitle: ", conf.plotTitle, " legend:", conf.legend)
  else:
    print("plotTitle: ", conf.plotTitle)
  print(namey, ":", datay)


def run_main(conf):
  if conf.is_out:
    if len(conf.mpiProcessesCaseOptimList) > 1 or conf.mpiProcessesCaseBase != conf.mpiProcessesCaseOptimList[0]:
      print(
        "Disabled out error check because number of processes should be the same for accuracy calculation, only speedup can use different number")
      conf.is_out = False
    if "Realistic" in conf.diffCellsL:
      conf.diffCellsL = ["Ideal"]
  if "Realistic" in conf.diffCells and \
      conf.mpiProcessesCaseBase not in \
      conf.mpiProcessesCaseOptimList:
    print("ERROR: Wrong conf, MPI and cells are exported in different order, set same MPIs for both cases")
    raise
  if not conf.caseBase:
    print("ERROR: caseBase is empty")
    raise
  for i, mpiProcesses in enumerate(conf.mpiProcessesCaseOptimList):
    for j, cellsProcesses in enumerate(conf.cells):
      nCells = int(cellsProcesses / mpiProcesses)
      if nCells == 0:
        print("WARNING: Configured less cells than MPI processes, setting 1 cell per process")
        conf.mpiProcessesCaseOptimList[i] = cellsProcesses

  run_diffCells(conf)
  plot_cases(conf)

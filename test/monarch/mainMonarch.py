import matplotlib as mpl

mpl.use('TkAgg')
#import plot_functions  # comment to save ~2s execution time
import math_functions
import os
import numpy as np
import json
import subprocess
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
    # Cases configuration
    self.diffCellsL = ""
    self.mpiProcessesCaseBase = 1
    self.mpiProcessesCaseOptimList = []
    self.cells = [100]
    self.caseBase = ""
    self.casesOptim = [""]
    self.plotYKey = ""
    self.is_import = False
    self.profileCuda = ""
    self.is_out = True
    # Auxiliary
    self.sbatch_job_id = ""
    self.exportPath = "exports"
    self.results_file = "_solver_stats.csv"
    self.nCellsProcesses = 1
    self.campConf = "settings/config_variables_c_solver.txt"


# from line_profiler_pycharm import profile
# @profile
def run(conf):
  exec_str = ""
  try:
    ddt_pid = subprocess.check_output(
      'pidof -x $(ps cax | grep ddt)', shell=True)
    if ddt_pid:
      exec_str += 'ddt --connect '
  except Exception:
    pass
  maxCoresPerNode = 40
  if conf.mpiProcesses <= maxCoresPerNode:
    exec_str += "mpirun -v -np " + str(
      conf.mpiProcesses) + " --bind-to core " #for plogin (fails squeue)
  else:
    exec_str += "srun --cpu-bind=core -n " + str(
      conf.mpiProcesses) + " " #foqueue (slow plogin)
  if (conf.profileCuda == "nvprof" and conf.caseGpuCpu ==
      "GPU"):
    pathNvprof = ("../../compile/power9/" +
                  conf.caseMulticellsOnecell
                  + str(conf.nCells) + "Cells" + ".nvprof ")
    exec_str += ("nvprof --analysis-metrics -f -o " +
                 pathNvprof)
    print("Saving profiling file in ",
          os.path.abspath(os.getcwd())
          + "/" + pathNvprof + ".nvprof")
  elif (conf.profileCuda == "nsight" and conf.caseGpuCpu
        == "GPU"):
    exec_str += ("/apps/NVIDIA-HPC-SDK/21.3/Linux_ppc64le"
                 "/21.3/profilers/Nsight_Compute/ncu ")
    pathNvprof = ("../../compile/power9/" +
                  conf.caseMulticellsOnecell
                  + str(conf.nCells) + "Cells ")
    exec_str += "--set full -f -o " + pathNvprof  # last
    print("Saving nsight file in ",
          os.path.abspath(os.getcwd())
          + "/" + pathNvprof + ".ncu-rep")
  path_exec = "../../build/mock_monarch"
  exec_str += path_exec
  try:
    file1 = open(conf.campConf, "w")
    if conf.caseGpuCpu == "CPU":
      file1.write("USE_CPU=ON\n")
    else:
      file1.write("USE_CPU=OFF\n")
    file1.close()
  except Exception as e:
    print("write_camp_config_file fails", e)
  print("exec_str:", exec_str, conf.diffCells,
        conf.caseGpuCpu,
        conf.caseMulticellsOnecell, "ncellsPerMPIProcess:",
        conf.nCells)
  conf_name = "settings/TestMonarch.json"
  with open(conf_name, 'w', encoding='utf-8') as jsonFile:
    json.dump(conf.__dict__, jsonFile, indent=4,
              sort_keys=False)
  nCellsStr = str(conf.nCells)
  if conf.nCells >= 1000:
    nCellsStr = str(int(conf.nCells / 1000)) + "k"
  caseGpuCpuName=conf.caseGpuCpu+str(conf.mpiProcesses) + "cores"
  out = 0
  is_import = False
  data_path = ("out/stats" + caseGpuCpuName + nCellsStr +
               "cells" + str(conf.timeSteps) + "tsteps.csv")
  print("data_path", data_path)
  data_path2 = ("out/state" + caseGpuCpuName + nCellsStr +
                "cells" + str(conf.timeSteps) + "tsteps.csv")
  if conf.is_import and os.path.exists(data_path):
    nRows_csv = (conf.timeSteps * conf.nCells *conf.mpiProcesses)
    df = pd_read_csv(data_path, nrows=nRows_csv)
    data = df.to_dict('list')
    y_key_words = conf.plotYKey.split()
    y_key = y_key_words[-1]
    data = data[y_key]
    print("data stats",data)
    if data:
      is_import = True
    if conf.is_out:
      if os.path.exists(data_path2):
        is_import = True
        df = pd_read_csv(data_path2, header=None,
                         names=["Column1"])
        out = df["Column1"].tolist()
        if out:
          is_import = True
      else:
        is_import = False
  if not is_import:
    os.system(exec_str)
    os.rename("out/stats.csv", data_path)
    if conf.is_out:
      os.rename("out/state.csv", data_path2)
    nRows_csv = (conf.timeSteps * conf.nCells *conf.mpiProcesses)
    df = pd_read_csv(data_path, nrows=nRows_csv)
    data = df.to_dict('list')
    y_key_words = conf.plotYKey.split()
    y_key = y_key_words[-1]
    data = data[y_key]
    print("data stats",data)
    if conf.is_out:
      if os.path.exists(data_path2):
        df = pd_read_csv(data_path2, header=None,
                         names=["Column1"])
        out = df["Column1"].tolist()
  return data[0], out


# @profile
def run_cases(conf):
  # Base case
  conf.mpiProcesses = conf.mpiProcessesCaseBase
  if conf.nCellsProcesses % conf.mpiProcesses != 0:
    print(
      "ERROR: On base case conf.nCellsProcesses % "
      "conf.mpiProcesses != 0, nCellsProcesses, "
      "mpiProcesses",
      conf.nCellsProcesses,
      conf.mpiProcesses)
    raise
  conf.nCells = int(
    conf.nCellsProcesses / conf.mpiProcesses)
  cases_words = conf.caseBase.split()
  conf.caseGpuCpu = cases_words[0]
  conf.caseMulticellsOnecell = cases_words[1]
  conf.case = conf.caseBase
  timeBase, valuesBase = run(conf)
  # OptimCases
  datacases = []
  for mpiProcessesCaseOptim in (
      conf.mpiProcessesCaseOptimList):
    conf.mpiProcesses = mpiProcessesCaseOptim
    if conf.nCellsProcesses % conf.mpiProcesses != 0:
      print(
        "WARNING: On optim case conf.nCellsProcesses % "
        "conf.mpiProcesses != 0,nCellsProcesses, "
        "mpiProcesses",
        conf.nCellsProcesses, conf.mpiProcesses)
    conf.nCells = int(
      conf.nCellsProcesses / conf.mpiProcesses)
    for caseOptim in conf.casesOptim:
      cases_words = caseOptim.split()
      conf.caseGpuCpu = cases_words[0]
      conf.caseMulticellsOnecell = cases_words[1]
      conf.case = caseOptim
      timeOptim, valuesOptim = run(conf)
      if conf.is_out:
        math_functions.check_NRMSE(valuesBase,
                                   valuesOptim,
                                   conf.nCellsProcesses)
      datay = timeBase / timeOptim
      datacases.append(datay)

  return datacases


def run_cells(conf):
  datacells = []
  for i in range(len(conf.cells)):
    conf.nCellsProcesses = conf.cells[i]
    datacases = run_cases(conf)
    datacells.append(datacases)
  if len(conf.cells) > 1:
    datacellsTranspose = np.transpose(datacells)
    datacells = datacellsTranspose.tolist()
  return datacells


def run_diffCells(conf):
  datacolumns = []
  for i, diff_cells in enumerate(conf.diffCellsL):
    conf.diffCells = diff_cells
    datacells = run_cells(conf)
    datacolumns += datacells
  return datacolumns


def plot_cases(conf, datay):
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
  legend = []
  for diff_cells in conf.diffCellsL:
    conf.diffCells = diff_cells
    for mpiProcessesCaseOptim in (
        conf.mpiProcessesCaseOptimList):
      for caseOptim in conf.casesOptim:
        cases_words = caseOptim.split()
        conf.caseGpuCpu = cases_words[0]
        conf.caseMulticellsOnecell = cases_words[1]
        case_multicells_onecell_name = ""
        if (conf.caseMulticellsOnecell.find(
            "BDF") != -1 or
            conf.caseMulticellsOnecell.find(
              "maxrregcount") != -1):
          is_same_diff_cells = True
        legend_name = ""
        if len(conf.diffCellsL) > 1:
          legend_name += conf.diffCells + " "
        elif not is_same_arch_optim:
          legend_name += conf.caseGpuCpu + " "
        if not is_same_case_optim:
          legend_name += case_multicells_onecell_name
        if not legend_name == "":
          legend.append(legend_name)
  plotTitle = ""
  nGPUsOptim = [int(i / 10) for i in conf.mpiProcessesCaseOptimList]
  if not is_same_diff_cells and len(conf.diffCellsL) == 1:
    plotTitle += conf.diffCells + " test: "
  plotXKey = ""
  if len(nGPUsOptim) > 1:
    plotXKey = "GPUs"
  if is_same_arch_optim:
    if plotXKey == "GPUs":
      plotTitle += ""
    else:
      if conf.caseGpuCpu == "GPU" and len(
          nGPUsOptim) == 1 and \
          conf.mpiProcessesCaseOptimList[0] > 1:
        plotTitle += str(
          int(nGPUsOptim[0])) + " GPUs "
      else:
        plotTitle += conf.caseGpuCpu + " "
  if plotXKey == "GPUs":
    plotTitle += "GPU "
  if len(legend) == 1 or not legend or len(
      conf.diffCellsL) > 1:
    if len(conf.mpiProcessesCaseOptimList) > 1:
      legend_name += str(mpiProcessesCaseOptim) + " MPI "
    if len(conf.diffCellsL) > 1:
      plotTitle += "Implementations "
  else:
    plotTitle += "Implementations "
  cases_words = conf.caseBase.split()
  conf.caseGpuCpu = cases_words[0]
  plotTitle += "vs " + str(
    conf.mpiProcessesCaseBase) + " Cores CPU"
  namey = "Speedup"
  if len(conf.cells) > 1:
    plotTitle += ""
    datax = conf.cells
    plot_x_key = "Cells"
  elif plotXKey == "GPUs":
    datax = nGPUsOptim
    if len(conf.cells) > 1:
      plotTitle += ", Cells: " + str(conf.cells[0])
      plot_x_key = plotXKey
    else:
      plot_x_key = "GPUs"
  else:
    plotTitle += ", Cells: " + str(conf.cells[0])
    datax = list(range(1, conf.timeSteps + 1, 1))
    plot_x_key = "Timesteps"
  namex = plot_x_key
  print(namex, ":", datax[0],"to",datax[-1])
  if legend:
    print("plotTitle: ", plotTitle, " legend:", legend)
  else:
    print("plotTitle: ", plotTitle)
  print(namey, ":", datay)
  #plot_functions.plotsns(namex, namey, datax, datay, plotTitle, legend)


def run_main(conf):
  if conf.is_out:
    if (len(
        conf.mpiProcessesCaseOptimList) > 1 or
        conf.mpiProcessesCaseBase !=
        conf.mpiProcessesCaseOptimList[0]):
      print(
        "Disabled out error check because number of "
        "processes should be the same for calculate "
        "accuracy, only speedup can use different number")
      conf.is_out = False
  for i, mpiProcesses in enumerate(
      conf.mpiProcessesCaseOptimList):
    for j, cellsProcesses in enumerate(conf.cells):
      nCells = int(cellsProcesses / mpiProcesses)
      if nCells == 0:
        print(
          "WARNING: Configured less cells than MPI "
          "processes, setting 1 cell per process")
        conf.mpiProcessesCaseOptimList[i] = cellsProcesses

  datay = run_diffCells(conf)
  plot_cases(conf, datay)

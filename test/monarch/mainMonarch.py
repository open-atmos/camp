import matplotlib as mpl

mpl.use('TkAgg')
# import plot_functions #comment to save ~2s execution time
import math_functions
import sys
import os
import numpy as np
import datetime
import json
from pathlib import Path
import zipfile
import subprocess
import time


class TestMonarch:
  def __init__(self):
    # Case configuration
    self._chemFile = "cb05_paperV2"
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
    self.out = []
    self.datacolumns = []
    self.stdColumns = []
    self.exportPath = "exports"
    self.legend = []
    self.results_file = "_solver_stats.csv"
    self.plotTitle = ""
    self.nCellsProcesses = 1
    self.campSolverConfigFile = "settings/config_variables_c_solver.txt"

  @property
  def chemFile(self):
    return self._chemFile

  @chemFile.setter
  def chemFile(self, new_chemFile):
    self._chemFile = new_chemFile


def getCaseName(conf):
  case_multicells_onecell_name = ""
  if conf.caseMulticellsOnecell == "Block-cellsN":
    case_multicells_onecell_name += "Block-cells (N)"
  elif conf.caseMulticellsOnecell == "Block-cells1":
    case_multicells_onecell_name += "Block-cells (1)"
  elif conf.caseMulticellsOnecell == "Block-cellsNhalf":
    case_multicells_onecell_name += "Block-cells (N/2)"
  elif conf.caseMulticellsOnecell.find("One") != -1:
    case_multicells_onecell_name += "Base version"
  else:
    case_multicells_onecell_name += conf.caseMulticellsOnecell
  return case_multicells_onecell_name


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


def export(conf, data_path):
  data_path_abs = os.path.abspath(os.getcwd()) + "/" + data_path
  exportPath = conf.exportPath
  if len(sys.argv) > 1:
    conf.sbatch_job_id = sys.argv[1]
  print(os.path.abspath(os.getcwd()) + "/" + exportPath)
  if not os.path.exists(exportPath):
    os.makedirs(exportPath)
  conf_dir = exportPath + "/conf"
  if not os.path.exists(conf_dir):
    os.makedirs(conf_dir)
  now = datetime.datetime.now()
  basename = now.strftime("%d-%m-%Y-%H.%M.%S") + "-" + conf.sbatch_job_id
  conf_path = conf_dir + "/" + basename + ".json"
  with open(conf_path, 'w', encoding='utf-8') as jsonFile:
    json.dump(conf.__dict__, jsonFile, indent=4, sort_keys=False)
  conf_name = basename + ".json"
  path_to_zip_file = conf_dir + "/" + basename + ".zip"
  zipfile.ZipFile(path_to_zip_file, mode='w').write(conf_path, arcname=conf_name)
  os.remove(conf_path)
  print("Configuration saved to", os.path.abspath(os.getcwd()) + conf_name)
  path_to_zip_file = exportPath + "/data"
  if not os.path.exists(path_to_zip_file):
    os.makedirs(path_to_zip_file)
  path_to_zip_file = exportPath + "/data/" + basename + ".zip"
  new_data_name = basename + ".csv"
  new_data_path = exportPath + "/data/" + new_data_name
  os.rename(data_path_abs, new_data_path)
  zipfile.ZipFile(path_to_zip_file, mode='w').write(new_data_path, arcname=new_data_name)
  os.rename(new_data_path, data_path_abs)
  print("Data saved to", os.path.abspath(os.getcwd()) + "/" + path_to_zip_file)
  if os.path.getsize(exportPath) > 1000000000:
    print("WARNING: More than 1GB saved in ", os.path.abspath(os.getcwd()) + "/" + exportPath)
    # raise


def run(conf):
  if conf.caseGpuCpu == "GPU":
    maxCoresPerNode = 40  # CTE-POWER specs
    if conf.mpiProcesses > maxCoresPerNode and conf.mpiProcesses % maxCoresPerNode != 0:
      print(
        "ERROR: MORE THAN 40 MPI PROCESSES AND NOT MULTIPLE OF 40, WHEN CTE-POWER ONLY HAS 40 CORES PER NODE\n");
      raise
    maxnDevices = 4  # CTE-POWER specs
    maxCoresPerDevice = maxCoresPerNode / maxnDevices
    maxCores = int(maxCoresPerDevice * conf.nGPUs)
    if conf.mpiProcesses != maxCores and (conf.mpiProcesses != 1 and maxCores == 10):
      print("WARNING: conf.mpiProcesses != maxCores, ",
            conf.mpiProcesses, "!=", maxCores,
            "conf.mpiProcesses changed from ", conf.mpiProcesses, "to ", maxCores)
      conf.mpiProcesses = maxCores
      conf.mpiProcessesCaseOptimList[0] = maxCores
      raise

    # if conf.mpiProcesses != maxCores:
    # print("ERROR: conf.mpiProcesses != maxCores, ",
    #      conf.mpiProcesses, "!=", maxCores)
    # raise
  exec_str = ""
  try:
    ddt_pid = subprocess.check_output('pidof -x $(ps cax | grep ddt)', shell=True)
    # ddt_pid = False
    if ddt_pid:
      exec_str += 'ddt --connect '
  except Exception:
    pass
  exec_str += "mpirun -v -np " + str(conf.mpiProcesses) + " --bind-to core "  # fails on monarch cte-power

  if conf.profileCuda == "nvprof" and conf.caseGpuCpu == "GPU":
    pathNvprof = "../../compile/power9/" + conf.caseMulticellsOnecell \
                 + str(conf.nCells) + "Cells" + ".nvprof "
    exec_str += "nvprof --analysis-metrics -f -o " + pathNvprof  # all metrics
    # exec_str += "nvprof --print-gpu-trace " #registers per thread
    # --print-gpu-summary
    print("Saving profiling file in ", os.path.abspath(os.getcwd()) \
          + "/" + pathNvprof + ".nvprof")
  elif conf.profileCuda == "nsight" and conf.caseGpuCpu == "GPU":
    print("TODO TRY TO USE /apps/NVIDIA-HPC-SDK/21.3/Linux_ppc64le/21.3/profilers/Nsight_Compute/ncu")
    exec_str += "/apps/NVIDIA-HPC-SDK/20.9/Linux_ppc64le/2020/profilers/Nsight_Compute/ncu "
    pathNvprof = "../../compile/power9/" + conf.caseMulticellsOnecell \
                 + str(conf.nCells) + "Cells "
    exec_str += "--set full -f -o " + pathNvprof  # last working version
    print("Saving nsight file in ", os.path.abspath(os.getcwd()) \
          + "/" + pathNvprof + ".ncu-rep")
  elif conf.profileCuda == "nsightSummary" and conf.caseGpuCpu == "GPU":
    exec_str += "/apps/NVIDIA-HPC-SDK/20.9/Linux_ppc64le/2020/profilers/Nsight_Compute/ncu "
    pathNvprof = "../../compile/power9/" + conf.caseMulticellsOnecell \
                 + str(conf.nCells) + "Cells "
    exec_str += ""
    print("CUDASaving nsight file in ", os.path.abspath(os.getcwd()) \
          + "/" + pathNvprof)

  path_exec = "../../build/mock_monarch"
  exec_str += path_exec

  # CAMP solver option GPU-CPU
  write_camp_config_file(conf)

  print("exec_str:", exec_str, conf.diffCells, conf.caseGpuCpu,
        conf.caseMulticellsOnecell, "ncellsPerMPIProcess:",
        conf.nCells, "nGPUs:", conf.nGPUs)

  conf_name = "settings/TestMonarch.json"
  with open(conf_name, 'w', encoding='utf-8') as jsonFile:
    json.dump(conf.__dict__, jsonFile, indent=4, sort_keys=False)

  data_path = ""
  if not conf.is_import:
    os.system(exec_str)
    if conf.case is conf.caseBase:
      os.rename("out/state.csv", "out/state0.csv")
      os.rename("out/stats.csv", "out/stats0.csv")
    else:
      os.rename("out/state.csv", "out/state1.csv")
      os.rename("out/stats.csv", "out/stats1.csv")
  #OUT
  if False:
    if conf.case is conf.caseBase:
      data_path = "out/state0.csv"
    else:
      data_path = "out/state1.csv"
    try:
      with open(data_path) as f:
        conf.out = [float(line.rstrip('\n')) for line in f]
    except FileNotFoundError as e:
      raise FileNotFoundError("Check enable EXPORT_STATE in CAMP code") from e
  #SPEEDUP
  if conf.case is conf.caseBase:
    data_path = "out/stats0.csv"
  else:
    data_path = "out/stats1.csv"
  nrows_csv = conf.timeSteps * conf.nCells * conf.mpiProcesses
  data = math_functions.read_solver_stats(data_path, nrows_csv)

  print("conf.results_file", data_path)
  return data


def run_case(conf):
  data = run(conf)
  if "timeLS" in conf.plotYKey and "computational" in conf.plotYKey \
      and "GPU" in conf.case:
    for i in range(len(data["timeLS"])):
      data["timeLS"][i] = data["timeLS"][i] - data["timeBiconjGradMemcpy"][i]
  y_key_words = conf.plotYKey.split()
  y_key = y_key_words[-1]
  if "normalized" in conf.plotYKey:
    nSystemsOfCells = 1
    if "One-cell" in conf.case:
      nSystemsOfCells = conf.nCells
    if y_key == "timeLS":
      for i in range(len(data[y_key])):
        data[y_key][i] = data[y_key][i] / (data["counterLS"][i] / nSystemsOfCells)
    elif y_key == "timecvStep":
      for i in range(len(data[y_key])):
        data[y_key][i] = data[y_key][i] / (data["countercvStep"][i] * nSystemsOfCells)
    else:
      for i in range(len(data[y_key])):
        data[y_key][i] = data[y_key][i] / nSystemsOfCells
  print("run_case", conf.case, y_key, ":", data[y_key])

  return data


def run_cases(conf):
  # Base case
  conf.mpiProcesses = conf.mpiProcessesCaseBase
  if conf.nCellsProcesses % conf.mpiProcesses != 0:
    print("WARNING: On base case conf.nCellsProcesses % conf.mpiProcesses != 0, nCellsProcesses, mpiProcesses",
          conf.nCellsProcesses,
          conf.mpiProcesses)
    raise
  conf.nCells = int(conf.nCellsProcesses / conf.mpiProcesses)
  conf.nGPUs = conf.nGPUsCaseBase

  cases_words = conf.caseBase.split()
  conf.caseGpuCpu = cases_words[0]
  conf.caseMulticellsOnecell = cases_words[1]

  conf.case = conf.caseBase
  dataCaseBase = run_case(conf)
  data = {"caseBase": dataCaseBase}

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
        # raise
      conf.nCells = int(conf.nCellsProcesses / conf.mpiProcesses)
      for caseOptim in conf.casesOptim:
        if conf.plotXKey == "MPI processes":
          if (caseOptim == conf.caseBase and mpiProcessesCaseOptim == conf.mpiProcessesCaseBase) \
              or (caseOptim != conf.caseBase and mpiProcessesCaseOptim != conf.mpiProcessesCaseBase):
            continue
        cases_words = caseOptim.split()
        conf.caseGpuCpu = cases_words[0]
        conf.caseMulticellsOnecell = cases_words[1]

        conf.case = caseOptim
        data["caseOptim"] = run_case(conf)

        if conf.plotYKey == "NRMSE":
          nCellsProcesses = [conf.nCellsProcesses]
          datay = math_functions.calculate_NRMSE(
            data, conf.timeSteps, nCellsProcesses)
        elif "Speedup" in conf.plotYKey:
          y_key_words = conf.plotYKey.split()
          y_key = y_key_words[-1]
          datay = math_functions.calculate_speedup(data, y_key)
        else:
          raise Exception("Not found plot function for conf.plotYKey")

        if len(conf.cells) > 1 or conf.plotXKey == "MPI processes" \
            or conf.plotXKey == "GPUs":
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


# Anything regarding different initial conditions is applied to both cases (Base and Optims/s)
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
  case_multicells_onecell_name = ""

  case_gpu_cpu_name = ""
  if conf.caseGpuCpu == "CPU":
    case_gpu_cpu_name = "CPU"
  elif conf.caseGpuCpu == "GPU":
    if conf.mpiProcessesCaseBase > 1:
      case_gpu_cpu_name += str(conf.mpiProcessesCaseBase) + " MPI "
    case_gpu_cpu_name += str(conf.nGPUsCaseBase) + " GPU"
  else:
    case_gpu_cpu_name = conf.caseGpuCpu

  baseCaseName = ""
  if conf.plotYKey != "Percentage data transfers CPU-GPU [%]":  # Speedup
    baseCaseName = "vs " + case_gpu_cpu_name + " " + case_multicells_onecell_name

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
          if conf.plotXKey == "MPI processes":
            if (caseOptim == conf.caseBase and mpiProcessesCaseOptim == conf.mpiProcessesCaseBase) \
                or (caseOptim != conf.caseBase and mpiProcessesCaseOptim != conf.mpiProcessesCaseBase):
              continue
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
          if len(conf.mpiProcessesCaseOptimList) > 1 and conf.plotXKey == "MPI processes":
            legend_name += str(mpiProcessesCaseOptim) + " MPI "
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
    if conf.plotXKey == "MPI processes":
      conf.plotTitle += "CPU "
    elif conf.plotXKey == "GPUs":
      conf.plotTitle += ""
    else:
      if conf.caseGpuCpu == "GPU" and len(conf.nGPUsCaseOptimList) == 1 and conf.nGPUsCaseOptimList[0] > 1:
        conf.plotTitle += str(conf.nGPUsCaseOptimList[0]) + " GPUs "
      else:
        conf.plotTitle += conf.caseGpuCpu + " "
  if conf.plotXKey == "GPUs":
    conf.plotTitle += "GPU "
  if conf.plotXKey == "MPI processes":
    conf.plotTitle += "Speedup against 1 MPI CPU-based version"
  if len(conf.legend) == 1 or not conf.legend or len(conf.diffCellsL) > 1:
    if len(conf.mpiProcessesCaseOptimList) > 1:
      legend_name += str(mpiProcessesCaseOptim) + " MPI "
    # conf.plotTitle += case_multicells_onecell_name + " "
    if len(conf.diffCellsL) > 1:
      conf.plotTitle += "Implementations "
  else:
    conf.plotTitle += "Implementations "
  if not conf.plotXKey == "MPI processes":
    conf.plotTitle += baseCaseName

  namey = conf.plotYKey
  if conf.plotYKey == "Speedup normalized computational timeLS":
    namey = "Speedup linear solver kernel"
  if conf.plotYKey == "Speedup counterLS":
    namey = "Speedup iterations CAMP solving"
  if conf.plotYKey == "Speedup normalized timeLS":
    namey = "Speedup linear solver"
  if conf.plotYKey == "Speedup timecvStep":
    namey = "Speedup"
  if conf.plotYKey == "Speedup countercvStep":
    namey = "Speedup iterations BDF loop"
  if conf.plotYKey == "Speedup timeCVode":
    namey = "Speedup CAMP solving"
  if conf.plotYKey == "NRMSE":
    namey = "NRMSE [%]"
  if conf.plotYKey == "Speedup counterBCG":
    namey = "Speedup solving iterations BCG"

  if len(conf.cells) > 1:
    namey += " [Mean and \u03C3]"
    conf.plotTitle += ""
    datax = conf.cells
    plot_x_key = "Cells"
  elif conf.plotXKey == "MPI processes":
    conf.plotTitle += ", Cells: " + str(conf.cells[0])
    datax = conf.mpiProcessesCaseOptimList
    plot_x_key = conf.plotXKey
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
  print(namex, ":", datax)
  if conf.legend:
    print("plotTitle: ", conf.plotTitle, " legend:", conf.legend)
  else:
    print("plotTitle: ", conf.plotTitle)
  print(namey, ":", datay)
  if conf.plotYKey == "NRMSE":
    print("SUCCESS")


def run_main(conf):
  if conf.is_out:
    if len(conf.mpiProcessesCaseOptimList) > 1 or conf.mpiProcessesCaseBase != conf.mpiProcessesCaseOptimList[0]:
      print("Disabled out error check because number of processes should be the same for NRMSE, only speedup can use different number")
      conf.is_out=False
  if conf.plotYKey == "":
    print("conf.plotYKey is empty")
  if conf.chemFile == "cb05_paperV2":
    if conf.timeStepsDt != 2:
      print("Warning: Setting timeStepsDt to 2, since it is the usual value for", conf.chemFile)
    conf.timeStepsDt = 2
  elif conf.chemFile == "monarch_cb05":
    conf.timeStepsDt = 3
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
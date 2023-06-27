import matplotlib as mpl

mpl.use('TkAgg')
#import plot_functions #comment to save ~2s execution time
import math_functions
import sys, getopt
import os
import numpy as np
import datetime
import json
from pathlib import Path
import zipfile
from os import walk
import subprocess
import time


class TestMonarch:
  def __init__(self):
    # Case configuration
    self._chemFile = "monarch_binned"
    self.diffCells = ""
    self.mpi = "yes"
    self.timeSteps = 1
    self.timeStepsDt = 2
    self.MAPETol = 1.0E-4
    self.commit = ""
    self.case = []
    self.nCells = 1
    self.caseGpuCpu = ""
    self.caseMulticellsOnecell = ""
    self.mpiProcesses = 1
    self.allocatedNodes = 1
    self.allocatedTasksPerNode = 160
    self.nGPUs = 1
    self.nCellsGPUPerc = 4  # between 0 and 10 (represents 0%-100%)
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
    self.is_export = False
    self.is_import = False
    self.profileCuda = ""
    # Auxiliary
    self.is_start_auxiliary_attributes = True
    self.sbatch_job_id = ""
    self.datacolumns = []
    self.stdColumns = []
    self.exportPath = "exports"
    self.legend = []
    self.results_file = "_solver_stats.csv"
    self.plotTitle = ""
    self.nCellsProcesses = 1
    self.itsolverConfigFile = "settings/itsolver_options.txt"
    self.campSolverConfigFile = "settings/config_variables_c_solver.txt"

  @property
  def chemFile(self):
    return self._chemFile

  @chemFile.setter
  def chemFile(self, new_chemFile):
    self._chemFile = new_chemFile


def get_is_sbatch():
  try:
    if sys.argv[1]:
      return True
    else:
      return False
  except Exception:
    return False


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


def write_itsolver_config_file(conf):
  file1 = open(conf.itsolverConfigFile, "w")
  cells_method_str = "CELLS_METHOD=" + conf.caseMulticellsOnecell
  file1.write(cells_method_str)
  # print("Saved", cells_method_str)
  file1.close()


def set_import_netcdf(conf, bool_import_netcdf):
  if conf.caseMulticellsOnecell == "IMPORT_NETCDF":
    conf.diffCells = "Ideal"
    savePath = os.getcwd()
    # print("savePath",savePath)
    os.chdir("../../build")
    cmake_str = "cmake -D ENABLE_IMPORT_NETCDF=" + str(bool_import_netcdf) + " .."
    os.system(cmake_str)
    os.system("make -j 4")
    os.chdir(savePath)


def write_camp_config_file(conf):
  try:
    file1 = open(conf.campSolverConfigFile, "w")
    if conf.caseGpuCpu == "CPU":
      file1.write("USE_CPU=ON\n")
    else:
      file1.write("USE_CPU=OFF\n")
    if conf.caseMulticellsOnecell == "BDF" or conf.caseMulticellsOnecell.find("maxrregcount") != -1:
      if conf.chemFile == "monarch_binned" or conf.chemFile == "cb05_mechanism_yarwood2005":
        print(
          "Error:", conf.chemFile, "can not run GPU BDF, disable GPU BDF or use a valid chemFile like monarch_cb05")
        raise
      else:
        file1.write("USE_GPU_CVODE=ON\n")
    else:
      file1.write("USE_GPU_CVODE=OFF\n")
    file1.write(str(conf.nGPUs) + "\n")
    file1.write(str(conf.nCellsGPUPerc) + "\n")
    file1.close()
  except Exception as e:
    print("write_camp_config_file fails", e)


def get_commit_hash():
  try:
    commit = subprocess.check_output(
      ['git', 'rev-parse', '--short', 'HEAD']).decode().strip()
  # Not copying .git folder into docker container
  except subprocess.CalledProcessError:
    commit = ""
  # print(' > Git Hash: {}'.format(commit))
  return str(commit)


def remove_to_tmp(conf, sbatch_job_id):
  exportPath = conf.exportPath

  now = datetime.datetime.now()
  tmp_dir = exportPath + "/tmp/" + now.strftime("%d-%m-%Y")  # + extra_str_name
  if not os.path.exists(tmp_dir):
    os.makedirs(tmp_dir)
    os.makedirs(tmp_dir + "/conf")
    os.makedirs(tmp_dir + "/data")

  conf_path = exportPath + "/conf"
  filenames = next(walk(conf_path), (None, None, []))[2]

  if not filenames:
    print("WARNING: Import folder is empty. Path:", os.path.abspath(os.getcwd()) + "/" + conf_path)
    raise

  data_path = exportPath + "/data/"
  for filename in filenames:
    dir_to_extract = conf_path + "/"
    basename = os.path.splitext(filename)[0]
    path_to_zip_file = dir_to_extract + basename + ".zip"
    with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
      zip_ref.extractall(dir_to_extract)
    conf_name = conf_path + "/" + basename + ".json"
    jsonFile = open(conf_name)
    conf_imported = json.load(jsonFile)
    os.remove(conf_name)
    conf_dict = vars(conf)
    # print("conf_dict",conf_dict)
    is_same_conf_case = True
    if conf_imported["sbatch_job_id"] == sbatch_job_id or conf_imported["sbatch_job_id"] == "-" + sbatch_job_id:
      tmp_dir_conf = tmp_dir + "/conf/" + basename + ".zip"
      # os.remove(path_to_zip_file)
      os.rename(path_to_zip_file, tmp_dir_conf)
      # print("Moved conf from",os.path.abspath(os.getcwd()) + "/" + path_to_zip_file, "to", tmp_dir)
      path_to_zip_file = data_path + basename + ".zip"
      # os.remove(path_to_zip_file)
      tmp_dir_data = tmp_dir + "/data/" + basename + ".zip"
      os.rename(path_to_zip_file, tmp_dir_data)
      print("Moved data from", os.path.abspath(os.getcwd()) + "/" + path_to_zip_file, "to",
            os.path.abspath(os.getcwd()) + "/" + tmp_dir)
  raise
  return True


def import_data(conf, tmp_path):
  is_import = False
  exportPath = conf.exportPath
  new_path = tmp_path
  if not os.path.exists(exportPath):
    return False, new_path
  conf_path = exportPath + "/conf"
  if not os.path.exists(conf_path):
    return False, new_path
  filenames = next(walk(conf_path), (None, None, []))[2]
  if not filenames:
    print("WARNING: Import folder is empty. Path:", os.path.abspath(os.getcwd()) + "/" + conf_path)
    return False, new_path
  data_path = exportPath + "/data/"
  # print("filenames:",filenames)
  # print("conf_path",os.path.abspath(os.getcwd())+"/"+conf_path)
  conf_defaultClass = TestMonarch()
  conf_default = vars(conf_defaultClass)
  for filename in filenames:
    dir_to_extract = conf_path + "/"
    basename = os.path.splitext(filename)[0]
    path_to_zip_file = dir_to_extract + basename + ".zip"
    # print("import_data path_to_zip_file",path_to_zip_file)
    with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
      zip_ref.extractall(dir_to_extract)
    conf_name = conf_path + "/" + basename + ".json"
    jsonFile = open(conf_name)
    conf_imported = json.load(jsonFile)
    os.remove(conf_name)
    conf_dict = vars(conf)
    # print("conf_dict",conf_dict)
    is_same_conf_case = True
    for confKey in conf_dict:
      # print("confKey",confKey)
      if confKey == "is_start_cases_attributes":
        # print("BREAK")
        break
      if conf_imported["timeSteps"] >= conf_dict["timeSteps"]:
        conf_imported["timeSteps"] = conf_dict["timeSteps"]
      if conf_dict["commit"] == "MATCH_IMPORTED_CONF":
        conf.commit = get_commit_hash()
      else:
        conf_imported["commit"] = conf_dict["commit"]
      # print("confKey",confKey)
      if confKey not in conf_imported:
        conf_imported[confKey] = conf_default[confKey]
      # if "allocatedTasksPerNode" not in conf_imported:
      # conf_imported["allocatedTasksPerNode"] = 160
      # if "allocatedNodes" not in conf_imported:
      # conf_imported["allocatedNodes"] = 1
      # if "nGPUs" not in conf_imported:
      # conf_imported["nGPUs"] = 1
      if conf_imported[confKey] != conf_dict[confKey]:
        # print(conf_dict[confKey])
        is_same_conf_case = False
    # print("basename",basename)
    # if basename == "16-04-2022-02.30.57-1649810070774628350":
    # print("conf_imported",conf_imported,"conf_dict",conf_dict)
    if is_same_conf_case:
      is_import = True
      dir_to_extract = data_path
      path_to_zip_file = data_path + basename + ".zip"
      try:
        with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
          zip_ref.extractall(dir_to_extract)
      except BaseException as err:
        print("path_to_zip_file", path_to_zip_file)
        print(err)
        raise
      new_path = os.path.abspath(os.getcwd()) + "/" + dir_to_extract + basename + ".csv"
      print("Imported data from", new_path)
      break
  return is_import, new_path


def export(conf, data_path):
  data_path_abs = os.path.abspath(os.getcwd()) + "/" + data_path
  exportPath = conf.exportPath
  conf.commit = get_commit_hash()
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
  print("Configuration saved to", os.path.abspath(os.getcwd()) + path_to_zip_file)
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
    #print("ERROR: conf.mpiProcesses != maxCores, ",
    #      conf.mpiProcesses, "!=", maxCores)
    #raise
  exec_str = ""
  try:
    ddt_pid = subprocess.check_output('pidof -x $(ps cax | grep ddt)', shell=True)
    if ddt_pid:
      exec_str += 'ddt --connect '
      #exec_str +=""
  except Exception:
    pass
  if conf.mpi == "yes":
    if os.getenv("BSC_MACHINE") == "power":
      exec_str += "mpirun -v -np " + str(conf.mpiProcesses) + " --bind-to core "  # fails on monarch cte-power
      # exec_str += "srun --cpu-bind=core -n " + str(conf.mpiProcesses) +" " #fine only with salloc or sbatch (no login nodes) and no ddt
      # exec_str += "srun --cpu-bind=core -n " + str(conf.mpiProcesses) +" --gres=gpu:1 " #fail because it needs 40 tasks allocated, despite I only use 1
  # salloc --x11 --qos=debug --tasks-per-node=160 --nodes=1 --gres=gpu:4
  elif os.getenv("BSC_MACHINE") == "mn4":
    exec_str += "mpirun -np " + str(conf.mpiProcesses) + " --bind-to core "
  else:
    print("Error python run - Unknown BSC_MACHINE")
    raise
  # exec_str+="srun -n "+str(conf.mpiProcesses)+" "
  # exec_str += "gprof " #to see cpu most time consuming functions
  if conf.profileCuda == "nvprof" and conf.caseGpuCpu == "GPU":
    pathNvprof = "../../compile/power9/"  # "../../../nvprof/"
    Path(pathNvprof).mkdir(parents=True, exist_ok=True)
    pathNvprof = pathNvprof + conf.caseMulticellsOnecell \
                 + str(conf.nCells) + "Cells" + ".nvprof "
    exec_str += "nvprof --analysis-metrics -f -o " + pathNvprof  # all metrics
    # exec_str += "nvprof --print-gpu-trace " #registers per thread
    # --print-gpu-summary
    print("Saving profiling file in ", os.path.abspath(os.getcwd()) \
          + "/" + pathNvprof)
  elif conf.profileCuda == "nsight" and conf.caseGpuCpu == "GPU":
    exec_str += "/apps/NVIDIA-HPC-SDK/20.9/Linux_ppc64le/2020/profilers/Nsight_Compute/ncu "
    pathNvprof = "../../compile/power9/"  # "../../../nvprof/nsight"
    Path(pathNvprof).mkdir(parents=True, exist_ok=True)
    pathNvprof = pathNvprof + conf.caseMulticellsOnecell \
                 + str(conf.nCells) + "Cells "
    exec_str += "--set full -f -o " + pathNvprof  # last working version
    # exec_str += " "  # summary
    # exec_str += "--mode=launch " #fail #gui attach
    print("CUDASaving nsight file in ", os.path.abspath(os.getcwd()) \
          + "/" + pathNvprof)
  elif conf.profileCuda == "nsightSummary" and conf.caseGpuCpu == "GPU":
    exec_str += "/apps/NVIDIA-HPC-SDK/20.9/Linux_ppc64le/2020/profilers/Nsight_Compute/ncu "
    pathNvprof = "../../compile/power9/"  # "../../../nvprof/nsight"
    Path(pathNvprof).mkdir(parents=True, exist_ok=True)
    pathNvprof = pathNvprof + conf.caseMulticellsOnecell \
                 + str(conf.nCells) + "Cells "
    exec_str += ""  # summary
    # exec_str += "--mode=launch " #fail #gui attach
    print("CUDASaving nsight file in ", os.path.abspath(os.getcwd()) \
          + "/" + pathNvprof)

  path_exec = "../../build/mock_monarch"
  exec_str += path_exec

  set_import_netcdf(conf, True)
  # CAMP solver option GPU-CPU
  write_camp_config_file(conf)

  # Onecell-Multicells ModelDataCPU
  write_itsolver_config_file(conf)

  print("exec_str:", exec_str, conf.diffCells, conf.caseGpuCpu,
        conf.caseMulticellsOnecell, "ncellsPerMPIProcess:",
        conf.nCells, "nGPUs:", conf.nGPUs)

  conf_name = "settings/TestMonarch.json"
  with open(conf_name, 'w', encoding='utf-8') as jsonFile:
    json.dump(conf.__dict__, jsonFile, indent=4, sort_keys=False)

  data_name = conf.chemFile + '_' + conf.caseMulticellsOnecell + conf.results_file
  tmp_path = 'out/' + data_name

  if conf.is_import and conf.plotYKey != "MAPE":
    is_import, data_path = import_data(conf, tmp_path)
  else:
    is_import, data_path = False, tmp_path

  if not is_import:
    os.system(exec_str)
    if conf.is_export:
      export(conf, data_path)
    set_import_netcdf(conf, False)

  data = math_functions.read_solver_stats(data_path, conf.timeSteps)
  if is_import:
    os.remove(data_path)

  return data


def run_case(conf):
  data = run(conf)
  if "timeLS" in conf.plotYKey and "computational" in conf.plotYKey \
      and "GPU" in conf.case:
    for i in range(len(data["timeLS"])):
      data["timeLS"][i] = data["timeLS"][i] - data["timeBiconjGradMemcpy"][i]
  # if conf.plotYKey != "MAPE":
  # print("data",data)
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
    else:  # counterBCG and other counters
      for i in range(len(data[y_key])):
        data[y_key][i] = data[y_key][i] / nSystemsOfCells

  if "(Comp.timeLS/counterBCG)" in conf.plotYKey and "GPU" in conf.case:
    for i in range(len(data["timeLS"])):
      data["timeLS"][i] = data["timeLS"][i] - data["timeBiconjGradMemcpy"][i]
    for i in range(len(data["timeLS"])):
      data["timeLS"][i] = data["timeLS"][i] \
                          / data["counterBCG"][i]

    for j in range(len(data["timeLS"])):
      data["timeLS"][j] = data["timeLS"][j] \
                          / data["counterBCG"][j]

  if conf.plotYKey != "MAPE":
    print("run_case", conf.case, y_key, ":", data[y_key])
  # print("data",data)

  return data


def run_cases(conf):
  # Run base case
  conf.mpiProcesses = conf.mpiProcessesCaseBase
  if conf.nCellsProcesses % conf.mpiProcesses != 0:
    print("WARNING: On base case conf.nCellsProcesses % conf.mpiProcesses != 0, nCellsProcesses, mpiProcesses", conf.nCellsProcesses,
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

  # Run OptimCases
  datacases = []
  stdCases = []
  for nGPUs in conf.nGPUsCaseOptimList:
    conf.nGPUs = nGPUs
    for mpiProcessesCaseOptim in conf.mpiProcessesCaseOptimList:
      conf.mpiProcesses = mpiProcessesCaseOptim
      if conf.nCellsProcesses % conf.mpiProcesses != 0:
        print("WARNING: On optim case conf.nCellsProcesses % conf.mpiProcesses != 0,nCellsProcesses, mpiProcesses",
              conf.nCellsProcesses, conf.mpiProcesses)
        #raise
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

        # calculate measures between caseBase and caseOptim
        if conf.plotYKey == "NRMSE":
          datay = math_functions.calculate_NMRSE(data, conf.timeSteps)
        elif conf.plotYKey == "MAPE":
          datay = math_functions.calculate_MAPE(data, conf.timeSteps, conf.MAPETol)
        elif conf.plotYKey == "SMAPE":
          datay = math_functions.calculate_SMAPE(data, conf.timeSteps)
        elif "Speedup" in conf.plotYKey:
          y_key_words = conf.plotYKey.split()
          y_key = y_key_words[-1]
          datay = math_functions.calculate_speedup(data, y_key)
        elif conf.plotYKey == "Percentage data transfers CPU-GPU [%]":
          y_key = "timeBiconjGradMemcpy"
          print("elif conf.plotYKey==Time data transfers")
          datay = math_functions.calculate_BCGPercTimeDataTransfers(data, y_key)
        else:
          raise Exception("Not found plot function for conf.plotYKey")

        if len(conf.cells) > 1 or conf.plotXKey == "MPI processes" \
            or conf.plotXKey == "GPUs":
          datacases.append(round(np.mean(datay), 2))
          stdCases.append(round(np.std(datay), 2))
        else:
          # datacases.append([round(elem, 2) for elem in datay])
          datacases.append([round(elem, 2) for elem in datay])

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

  # print("datacells",datacells)

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
  # Set plot info
  cases_words = conf.caseBase.split()
  conf.caseGpuCpu = cases_words[0]
  conf.caseMulticellsOnecell = cases_words[1]
  # case_multicells_onecell_name = getCaseName(conf)
  case_multicells_onecell_name = ""

  case_gpu_cpu_name = ""
  if conf.caseGpuCpu == "CPU":
    # case_gpu_cpu_name = str(conf.mpiProcessesCaseBase) + " MPI CPU Cores"
    # case_gpu_cpu_name = "1 CPU ("+str(conf.mpiProcessesCaseBase)+" MPI cores)"
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
    # print(last_case_optim,conf.caseMulticellsOnecell)
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
          # case_multicells_onecell_name = getCaseName(conf)
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
        # conf.plotTitle += str(nGPUs) + " " + conf.caseGpuCpu + " "
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
  if conf.plotYKey == "MAPE":
    namey = "MAPE [%]"
  if conf.plotYKey == "Speedup counterBCG":
    namey = "Speedup solving iterations BCG"

  if len(conf.cells) > 1:
    namey += " [Mean and \u03C3]"
    conf.plotTitle += ""
    # conf.plotTitle += ", " + str(conf.timeSteps) + " timesteps"
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
    print("Mean:", round(np.mean(datay), 2), "Std", round(np.std(datay), 2))
  else:
    print("Std", conf.stdColumns)
  print(namex, ":", datax)
  if conf.legend:
    print("plotTitle: ", conf.plotTitle, " legend:", conf.legend)
  else:
    print("plotTitle: ", conf.plotTitle)
  print(namey, ":", datay)

  #plot_functions.plotsns(namex, namey, datax, datay, conf.stdColumns, conf.plotTitle, conf.legend)

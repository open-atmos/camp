
#
# Copyright (C) 2022 Barcelona Supercomputing Center and University of
# Illinois at Urbana-Champaign
# SPDX-License-Identifier: MIT
#

import matplotlib as mpl

mpl.use('TkAgg')
import plot_functions #comment to save ~2s execution time
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
        self.itsolverConfigFile = "itsolver_options.txt"
        self.campSolverConfigFile = "config_variables_c_solver.txt"

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
    # if conf.caseMulticellsOnecell != "BDF" and conf.caseGpuCpu == "GPU":
    # case_multicells_onecell_name = "LS "
    if conf.caseMulticellsOnecell == "Block-cellsN":
        case_multicells_onecell_name += "Block-cells (N)"
    elif conf.caseMulticellsOnecell == "Block-cells1":
        case_multicells_onecell_name += "Block-cells (1)"
    elif conf.caseMulticellsOnecell == "Block-cellsNhalf":
        case_multicells_onecell_name += "Block-cells (N/2)"
    elif conf.caseMulticellsOnecell.find("maxrregcount") != -1:
        #case_multicells_onecell_name += ""
        #print("WARNING: Changed name maxrregcount to", case_multicells_onecell_name)
        case_multicells_onecell_name += conf.caseMulticellsOnecell
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


def write_camp_config_file(conf):
    try:
        file1 = open(conf.campSolverConfigFile, "w")

        if conf.caseGpuCpu == "CPU":
            file1.write("USE_CPU=ON\n")
        else:
            file1.write("USE_CPU=OFF\n")

        if conf.caseMulticellsOnecell == "BDF" or conf.caseMulticellsOnecell.find("maxrregcount") != -1:
            # print("FOUND MAXRREGCOUNT")
            if conf.chemFile == "monarch_binned":
                print(
                    "Error: monarch_binned can not run GPU BDF, disable GPU BDF or use a valid chemFile like monarch_cb05")
                raise
            else:
                file1.write("USE_GPU_CVODE=ON\n")
        elif conf.caseMulticellsOnecell == "CPU":
            file1.write("USE_GPU_CVODE=2\n")
        else:
            file1.write("USE_GPU_CVODE=OFF\n")

        file1.write(str(conf.nGPUs))

        file1.close()

        # new_path = os.path.abspath(os.getcwd()) + "/" + conf.campSolverConfigFile
        # print("write_camp_config_file in", new_path)
        # conf.debug_path = new_path
        # file_exists = os.path.exists(conf.debug_path)
        # print(file_exists)

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
    # print("filenames:",filenames)
    # print("conf_path",os.path.abspath(os.getcwd())+"/"+conf_path)
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
    exec_str = ""
    if conf.mpi == "yes":
        exec_str += "mpirun -v -np " + str(conf.mpiProcesses) + " --bind-to core "
        # exec_str+="srun -n "+str(conf.mpiProcesses)+" "

    if conf.profileCuda == "nvprof" and conf.caseGpuCpu == "GPU":
        pathNvprof = "../../compile/power9/"#"../../../nvprof/"
        Path(pathNvprof).mkdir(parents=True, exist_ok=True)
        pathNvprof = pathNvprof+ conf.caseMulticellsOnecell \
                     + str(conf.nCells) + "Cells" +  ".nvprof "
        #exec_str += "nvprof --analysis-metrics -f -o " + pathNvprof #all metrics
        exec_str += "nvprof --print-gpu-trace " #registers per thread
        # --print-gpu-summary
        print("Saving profiling file in ", os.path.abspath(os.getcwd()) \
              + "/" + pathNvprof)
    elif conf.profileCuda == "nsight" and conf.caseGpuCpu == "GPU":
        pathNvprof = "../../compile/power9/"#"../../../nvprof/nsight"
        Path(pathNvprof).mkdir(parents=True, exist_ok=True)
        pathNvprof = pathNvprof + conf.caseMulticellsOnecell \
                     + str(conf.nCells) + "Cells "
        #exec_str += "/apps/NVIDIA-HPC-SDK/20.9/Linux_ppc64le/2020/profilers/Nsight_Compute/ncu --set full -f -o" + pathNvprof #last working version
        exec_str += "/apps/NVIDIA-HPC-SDK/20.9/Linux_ppc64le/2020/profilers/Nsight_Compute/ncu "#last working version
        #wrong exec_str += "/apps/NVIDIA-HPC-SDK/21.3/Linux_ppc64le/21.3/profilers/Nsight_Compute/ncu --set full -f -o " + pathNvprof
        #wrongexec_str += "/apps/NVIDIA-HPC-SDK/21.3/Linux_ppc64le/21.3/profilers/Nsight_Compute/ncu "
        #wrong exec_str += "/apps/NVIDIA-HPC-SDK/21.9/Linux_ppc64le/21.9/profilers/Nsight_Compute/ncu "
        #wrong exec_str += "/apps/NVIDIA-HPC-SDK/21.9/Linux_ppc64le/2021/profilers/Nsight_Compute/ncu "
        #wrong exec_str += "/apps/NVIDIA-HPC-SDK/22.2/Linux_ppc64le/22.2/profilers/Nsight_Compute/ncu "
        #wrong exec_str += "/apps/NVIDIA-HPC-SDK/22.3/Linux_ppc64le/22.3/profilers/Nsight_Compute/ncu "

        print("Saving nsight file in ", os.path.abspath(os.getcwd()) \
              + "/" + pathNvprof)

    path_exec = "../../build/mock_monarch"
    exec_str += path_exec

    # CAMP solver option GPU-CPU
    write_camp_config_file(conf)

    # Onecell-Multicells itsolver
    write_itsolver_config_file(conf)

    print("exec_str:", exec_str, conf.diffCells, conf.caseGpuCpu, conf.caseMulticellsOnecell, conf.mpiProcesses,
          conf.nGPUs)

    conf_name = "TestMonarch.json"
    with open(conf_name, 'w', encoding='utf-8') as jsonFile:
        json.dump(conf.__dict__, jsonFile, indent=4, sort_keys=False)

    data_name = conf.chemFile + '_' + conf.caseMulticellsOnecell + conf.results_file
    tmp_path = '../../build/test_run/monarch/out/' + data_name

    if conf.is_import and conf.plotYKey != "MAPE":
        is_import, data_path = import_data(conf, tmp_path)
    else:
        is_import, data_path = False, tmp_path
    # print("IMPORT DATA", is_import)

    # print(data_path)
    if not is_import:
        # Main
        # if conf.caseGpuCpu.find("maxrregcount") != -1:
        #   conf.caseGpuCpu = "maxrregcount"
        os.system(exec_str)
        if conf.is_export:
            export(conf, data_path)

    # new_path = os.path.abspath(os.getcwd()) + "/" + data_path
    # print("data_path", new_path)

    data = math_functions.read_solver_stats(data_path, conf.timeSteps)
    #y_key_words = conf.plotYKey.split()
    #y_key = y_key_words[-1]
    #print(data[y_key])

    # print("The size of the dictionary is {} bytes".format(sys.getsizeof(data)))
    # print("The size of the dictionary is {} bytes".format(sys.getsizeof(data["timeLS"])))

    if is_import:
        os.remove(data_path)

    return data


def run_case(conf):
    data = run(conf)
    if "timeLS" in conf.plotYKey and "computational" in conf.plotYKey \
        and "GPU" in conf.case:
        for i in range(len(data["timeLS"])):
            data["timeLS"][i] = data["timeLS"][i] - data["timeBiconjGradMemcpy"][i]
    #if conf.plotYKey != "MAPE":
        #print("data",data)
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
    #print("data",data)

    return data


def run_cases(conf):
    # Run base case
    conf.mpiProcesses = conf.mpiProcessesCaseBase
    if conf.nCellsProcesses % conf.mpiProcesses !=0:
        print("ERROR: DIVISION OF CELLS/PROCESSES NOT INTEGER")
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
            if conf.nCellsProcesses % conf.mpiProcesses !=0:
                print("ERROR: DIVISION OF CELLS/PROCESSES NOT INTEGER")
                raise
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
                    # print("WARNING: Check y_key is correct:",y_key)
                    datay = math_functions.calculate_speedup(data, y_key)
                elif conf.plotYKey == "Percentage data transfers CPU-GPU [%]":
                    y_key = "timeBiconjGradMemcpy"
                    print("elif conf.plotYKey==Time data transfers")
                    datay = math_functions.calculate_BCGPercTimeDataTransfers(data, y_key)
                else:
                    raise Exception("Not found plot function for conf.plotYKey")

                if len(conf.cells) > 1 or conf.plotXKey == "MPI processes":
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

        if len(conf.cells) == 1:
            datacells = datacases
            stdCells = stdCases
        else:
            datacells.append(datacases)
            stdCells.append(stdCases)

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
    case_multicells_onecell_name = getCaseName(conf)
    # if conf.caseMulticellsOnecell.find("One-cell") != -1:
    #    case_multicells_onecell_name = "Base version"

    case_gpu_cpu_name = ""
    if conf.caseGpuCpu == "CPU":
        case_gpu_cpu_name = str(conf.mpiProcessesCaseBase) + " MPI" + " CPU"
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
                    case_multicells_onecell_name = getCaseName(conf)
                    if conf.caseMulticellsOnecell.find("BDF") != -1 or conf.caseMulticellsOnecell.find(
                        "maxrregcount") != -1:
                        is_same_diff_cells = True
                    legend_name = ""
                    if len(conf.diffCellsL) > 1:
                        legend_name += conf.diffCells + " "
                    if len(conf.mpiProcessesCaseOptimList) > 1 and conf.plotXKey == "MPI processes":
                        legend_name += str(mpiProcessesCaseOptim) + " MPI "
                    if len(conf.nGPUsCaseOptimList) > 1 and conf.caseGpuCpu == "GPU":
                        legend_name += str(nGPUs) + "GPU "
                    elif not is_same_arch_optim:
                        legend_name += conf.caseGpuCpu + " "
                    if not is_same_case_optim:
                        legend_name += case_multicells_onecell_name
                    if not legend_name == "":
                        conf.legend.append(legend_name)

    conf.plotTitle = ""
    if not is_same_diff_cells and len(conf.diffCellsL) == 1:
        conf.plotTitle += conf.diffCells + " test: "
    if len(conf.mpiProcessesCaseOptimList) == 1 and conf.caseGpuCpu == "CPU":
        # if len(conf.mpiProcessesCaseOptimList) == 1:
        conf.plotTitle += str(mpiProcessesCaseOptim) + " MPI "
    if len(conf.nGPUsCaseOptimList) == 1 and conf.plotXKey == "GPUs":
        # conf.plotTitle += str(nGPUs) + " GPUs "
        conf.plotTitle += " GPUs "
    # print("is_same_arch_optim",is_same_arch_optim)
    if is_same_arch_optim:
        if conf.plotXKey == "MPI processes":
            conf.plotTitle += "CPU "
        elif conf.plotXKey == "GPUs":
            conf.plotTitle += ""
        else:
            if conf.caseGpuCpu == "GPU" and len(conf.nGPUsCaseOptimList) == 1 and conf.nGPUsCaseOptimList[0] > 1 :
                conf.plotTitle += str(conf.nGPUsCaseOptimList[0]) + " GPUs "
            else:
                conf.plotTitle += conf.caseGpuCpu + " "
    if len(conf.legend) == 1 or not conf.legend or len(conf.diffCellsL) > 1:
        if len(conf.mpiProcessesCaseOptimList) > 1:
            legend_name += str(mpiProcessesCaseOptim) + " MPI "
        conf.plotTitle += case_multicells_onecell_name + " "
        if len(conf.diffCellsL) > 1:
            conf.plotTitle += "Implementations "
    else:
        if conf.plotXKey == "GPUs":
            conf.plotTitle += "GPU "
        if conf.plotXKey == "MPI processes":
            conf.plotTitle += "Speedup against 1 MPI CPU-based version"
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
        namey = "Reduction in solver iterations"

    if len(conf.cells) > 1:
        namey += " [Mean and \u03C3]"
        # namey += " [Average]"
        #conf.plotTitle += ", " + str(conf.timeSteps)+" timesteps"
        conf.plotTitle += ""
        datax = conf.cells
        plot_x_key = "Cells"
    elif conf.plotXKey == "MPI processes":
        conf.plotTitle += ", Cells: " + str(conf.cells[0])
        datax = conf.mpiProcessesCaseOptimList
        plot_x_key = conf.plotXKey
    elif conf.plotXKey == "GPUs":
        conf.plotTitle += ", Cells: " + str(conf.cells[0])
        datax = conf.nGPUsCaseOptimList
        plot_x_key = conf.plotXKey
    else:
        conf.plotTitle += ", Cells: " + str(conf.cells[0])
        datax = list(range(1, conf.timeSteps + 1, 1))
        plot_x_key = "Timesteps"

    namex = plot_x_key
    datay = conf.datacolumns

    print("Nodes:", conf.allocatedNodes)
    if namex == "Timesteps":
        print("Mean:", round(np.mean(datay), 2))
        print("Std", round(np.std(datay), 2))
    else:
        print("Std", conf.stdColumns)
    print(namex, ":", datax)
    print("plotTitle: ", conf.plotTitle, " legend:", conf.legend)
    print(namey, ":", datay)


    plot_functions.plotsns(namex, namey, datax, datay, conf.stdColumns, conf.plotTitle, conf.legend)


def all_timesteps():
    conf = TestMonarch()

    # conf.chemFile = "simple"
    #conf.chemFile = "monarch_cb05"
    conf.chemFile = "monarch_binned"

    conf.diffCellsL = []
    conf.diffCellsL.append("Realistic")
    conf.diffCellsL.append("Ideal")

    conf.profileCuda = ""
    #conf.profileCuda = "nvprof"
    #conf.profileCuda = "nsight"

    conf.is_export = get_is_sbatch()
    #conf.is_export = True
    #conf.is_export = False

    conf.is_import = True
    # conf.is_import = False

    # conf.commit = "MATCH_IMPORTED_CONF"
    conf.commit = ""

    conf.nGPUsCaseBase = 1
     #conf.nGPUsCaseBase = 2

    conf.nGPUsCaseOptimList = [1]
    #conf.nGPUsCaseOptimList = [2]
    # conf.nGPUsCaseOptimList = [1,2,3,4]

    conf.mpi = "yes"
    # conf.mpi = "no"

    conf.mpiProcessesCaseBase = 1
    #conf.mpiProcessesCaseBase = 40

    #conf.mpiProcessesCaseOptimList.append(1)
    conf.mpiProcessesCaseOptimList.append(1)
    # conf.mpiProcessesCaseOptimList = [10,20,40]
    # conf.mpiProcessesCaseOptimList = [1,4,8,16,32,40]

    conf.allocatedNodes = 1
    # conf.allocatedNodes = 4
    # conf.allocatedNodes = get_allocatedNodes_sbatch() #todo

    conf.allocatedTasksPerNode = 160
    # conf.allocatedTasksPerNode = 40
    # conf.allocatedTasksPerNode = 320
    # conf.allocatedTasksPerNode = get_ntasksPerNode_sbatch() #todo

    #conf.cells = [1000]
    conf.cells = [100, 500, 1000, 5000, 10000]
    # conf.cells = [50000,100000,500000,1000000]

    #conf.timeSteps = 1
    conf.timeSteps = 720

    conf.timeStepsDt = 2

    #conf.caseBase = "CPU EBI"
    #conf.caseBase = "CPU One-cell"
    #conf.caseBase = "CPU Multi-cells"
    # conf.caseBase="GPU Multi-cells"
    conf.caseBase="GPU Block-cellsN"
    #conf.caseBase="GPU Block-cells1"
    #conf.caseBase = "GPU BDF"
    #conf.caseBase = "GPU CPU"
    #conf.caseBase = "GPU maxrregcount-64" #wrong 10,000 cells
    # conf.caseBase = "GPU maxrregcount-24" #Minimum
    # conf.caseBase = "GPU maxrregcount-62"
    # conf.caseBase = "GPU maxrregcount-68"
    # conf.caseBase = "GPU maxrregcount-48"

    conf.casesOptim = []
    #conf.casesOptim.append("CPU One-cell")
    #conf.casesOptim.append("CPU Multi-cells")
    # conf.casesOptim.append("GPU One-cell")
    # conf.casesOptim.append("GPU Multi-cells")
    # conf.casesOptim.append("GPU Block-cellsNhalf")
    #conf.casesOptim.append("GPU Block-cellsN")
    conf.casesOptim.append("GPU Block-cells1")
    #conf.casesOptim.append("CPU EBI")
    #conf.casesOptim.append("GPU BDF")
    #conf.casesOptim.append("GPU CPU")
    #conf.casesOptim.append("GPU maxrregcount-64") #wrong 10,000 cells
    # conf.casesOptim.append("GPU maxrregcount-68")
    # conf.casesOptim.append("GPU maxrregcount-62")
    # conf.casesOptim.append("GPU maxrregcount-24")

    #conf.plotYKey = "Speedup timeCVode"
    # conf.plotYKey = "Speedup counterLS"
    #conf.plotYKey = "Speedup normalized timeLS"
    # conf.plotYKey = "Speedup normalized computational timeLS"
    conf.plotYKey = "Speedup counterBCG"
    #conf.plotYKey = "Speedup normalized counterBCG" #not used
    #conf.plotYKey = "Speedup total iterations - counterBCG" #not used
    # conf.plotYKey = "Speedup BCG iteration (Comp.timeLS/counterBCG)" #not used
    #conf.plotYKey = "Speedup timecvStep"
    # conf.plotYKey = "Speedup timecvStep normalized by countercvStep"
    #conf.plotYKey = "Speedup countercvStep"
    #conf.plotYKey = "Speedup device timecvStep" //pending fix bug https://earth.bsc.es/gitlab/ac/camp/-/issues/124
    # conf.plotYKey = "Percentage data transfers CPU-GPU [%]"
    #conf.plotYKey = "MAPE"
    # conf.plotYKey ="SMAPE"
    # conf.plotYKey ="NRMSE"
    # conf.MAPETol = 1.0E-6

    # conf.plotXKey = "MPI processes"
    # conf.plotXKey = "GPUs"

    """END OF CONFIGURATION VARIABLES"""

    # Utility functions
    #remove_to_tmp(conf,"1661337164911019079")

    conf.results_file = "_solver_stats.csv"
    if conf.plotYKey == "NRMSE" or conf.plotYKey == "MAPE" or conf.plotYKey == "SMAPE":
        conf.results_file = '_results_all_cells.csv'
        conf.is_export = False
        conf.is_import = False
    if (conf.nGPUsCaseBase >2 and conf.mpiProcessesCaseBase < 30):
        print("ERROR: nGPUsCaseBase is more than 2 but MPI processes is not enough, use 40 MPI processes or reduce GPUs to 1 or 2")
        raise
    jsonFile = open("monarch_box_binned/cb05_abs_tol.json")
    jsonData = json.load(jsonFile)
    conf.MAPETol = jsonData["camp-data"][0]["value"]  # Default: 1.0E-4
    jsonData.clear()

    if not os.path.exists('out'):
        os.makedirs('out')

    if conf.plotYKey == "":
        print("conf.plotYKey is empty")

    if conf.chemFile == "monarch_binned":
        if conf.timeStepsDt != 2:
            print("Warning: Setting timeStepsDt to 2, since it is the usual value for monarch_binned")
        conf.timeStepsDt = 2
    elif conf.chemFile == "monarch_cb05":
        if conf.timeStepsDt != 3:
            print("Warning: Setting timeStepsDt to 3, since it is the usual value for monarch_cb05")
        conf.timeStepsDt = 3
        if "Realistic" in conf.diffCellsL:
            print("Warning: Setting Ideal, chemFile == monarch_cb05 only has ideal case implemented")
            conf.diffCellsL = ["Ideal"]

    if not conf.caseBase:
        print("ERROR: caseBase is empty")
        raise

    if conf.caseBase  == "CPU EBI":
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
    #print("main start")
    all_timesteps()

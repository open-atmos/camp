#
# -----------------------------------------------------------------
# Programmer(s): Matthew Dawson, Christian G. Ruiz and
# Mario Acosta
# -----------------------------------------------------------------
# Copyright (C) 2022 Barcelona Supercomputing Center and University of
# Illinois at Urbana-Champaign
# SPDX-License-Identifier: MIT
#

import matplotlib as mpl

mpl.use('TkAgg')
import matplotlib.pyplot as plt
import csv
import sys, getopt
import os
import numpy as np
from pylab import imread, subplot, imshow, show
import plot_functions
import datetime
import pandas as pd
import seaborn as sns
import time
import json
from pathlib import Path
import shutil
import zipfile

class TestMonarch:
    def __init__(self):
        # Configuration
        self._chemFile = "monarch_binned"
        self.diffCellsL = ""
        self.profileCuda = False
        self.mpi = "yes"
        self.mpiProcessesList = [1]
        self.cells = [100]
        self.timeSteps = 1
        self.timeStepsDt = 2
        self.caseBase = ""
        self.casesOptim = [""]
        self.plotYKey = ""
        self.MAPETol = 1.0E-4
        self.readData = False
        # Auxiliar
        self.diffCells = ""
        self.datacolumns = []
        self.exportPath= "test/monarch/exports"
        self.legend = []
        self.case = []
        self.results_file = "_solver_stats.csv"
        self.plotTitle = ""
        self.savePlot = True
        self.mpiProcesses = 1
        self.nCells = 1
        self.nCellsCase = 1
        self.caseGpuCpu = ""
        self.caseMulticellsOnecell = ""
        self.itsolverConfigFile = "itsolver_options.txt"
        self.campSolverConfigFile = "config_variables_c_solver.txt"

    @property
    def chemFile(self):
        return self._chemFile

    @chemFile.setter
    def chemFile(self, new_chemFile):
        self._chemFile = new_chemFile

    def __del__(self):
        if os.path.exists(self.itsolverConfigFile):
            os.remove(self.itsolverConfigFile)
        if os.path.exists(self.campSolverConfigFile):
            os.remove(self.campSolverConfigFile)

def write_itsolver_config_file(conf):
    file1 = open(conf.itsolverConfigFile, "w")

    cells_method_str = "CELLS_METHOD=" + conf.caseMulticellsOnecell
    file1.write(cells_method_str)
    # print("Saved", cells_method_str)

    file1.close()

def write_camp_config_file(conf):
    file1 = open(conf.campSolverConfigFile, "w")

    if conf.caseGpuCpu == "CPU":
        file1.write("USE_CPU=ON\n")
    else:
        file1.write("USE_CPU=OFF\n")

    if conf.caseMulticellsOnecell == "CVODE":
        if conf.chemFile == "monarch_binned":
            print(
                "Error: monarch_binned can not run GPU CVODE, disable GPU CVODE or use a valid chemFile like monarch_cb05")
            raise
        else:
            file1.write("USE_GPU_CVODE=ON\n")
    else:
        file1.write("USE_GPU_CVODE=OFF\n")

    file1.close()

def read_conf():

    jsonFile = open(last_file_created)
    jsonData = json.load(jsonFile)
    print(jsonData)

def export(conf, tmp_path):

    init_path = os.path.abspath(os.getcwd())
    tmp_path_abs = os.path.abspath(os.getcwd()) + "/" + tmp_path
    exportPath = conf.exportPath

    os.chdir("../../..")
    if not os.path.exists(exportPath):
        os.makedirs(exportPath)

    conf_path = exportPath + "/conf"
    path_to_zip_file = conf_path + ".zip"
    if os.path.exists(path_to_zip_file):
        directory_to_extract_to = conf_path
        with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
            zip_ref.extractall(directory_to_extract_to)
    else:
        os.makedirs(conf_path)

    now = datetime.datetime.now()
    dt_string = now.strftime("%d-%m-%Y-%H.%M.%S")
    conf_name = conf_path + "/" + dt_string
    with open(conf_name, 'w', encoding='utf-8') as jsonFile:
        json.dump(conf.__dict__, jsonFile, indent=4, sort_keys=True)

    path_to_zip_file = conf_path
    shutil.make_archive(path_to_zip_file, 'zip', conf_path)
    print("Configuration saved to", os.path.abspath(os.getcwd()) + path_to_zip_file)
    #print("Configuration saved to", os.path.abspath(os.getcwd())+"/"+conf_path)

    path_to_zip_file = exportPath +"/data"
    if not os.path.exists(path_to_zip_file):
        os.makedirs(path_to_zip_file)

    path_to_zip_file = exportPath +"/data/" + dt_string + ".zip"
    zipfile.ZipFile(path_to_zip_file, mode='w').write(tmp_path_abs)
    #zipfile.ZipFile('hello.zip', mode='w').write("hello.csv")

    #shutil.make_archive(path_to_zip_file, 'zip', tmp_path_abs)
    #data_path = exportPath + "/data/" + dt_string
    #shutil.copy2(tmp_path_abs,data_path)
    print("Data saved to", os.path.abspath(os.getcwd())+"/"+path_to_zip_file)

    if os.path.getsize(exportPath) > 1000000000:
        print("ERROR: More than 1GB saved in ", os.path.abspath(os.getcwd())+"/"+exportPath)
        raise

    os.chdir(init_path)

def run(conf):
    exec_str = ""
    if conf.mpi == "yes":
        exec_str += "mpirun -v -np " + str(conf.mpiProcesses) + " --bind-to none "
        # exec_str+="srun -n "+str(conf.mpiProcesses)+" "

    if conf.profileCuda and conf.caseGpuCpu == "GPU":
        pathNvprof = "nvprof/"
        Path(pathNvprof).mkdir(parents=True, exist_ok=True)
        exec_str += "nvprof --analysis-metrics -f -o " + pathNvprof + \
                    conf.chemFile + \
                    conf.caseMulticellsOnecell + str(conf.nCells) + ".nvprof "
        # --print-gpu-summary
        print("Nvprof file saved in ", os.path.abspath(os.getcwd()) \
              + "/" + pathNvprof)

    exec_str += "../../mock_monarch"

    # CAMP solver option GPU-CPU
    write_camp_config_file(conf)

    # Onecell-Multicells itsolver
    write_itsolver_config_file(conf)

    print(exec_str, conf.caseGpuCpu, conf.caseMulticellsOnecell)

    #if save_folder
    #if not os.path.exists(save_folder):
    #if os.path.exists(save_folder):
    #os.makedirs(save_folder)

    conf_name = "TestMonarch.json"
    with open(conf_name, 'w', encoding='utf-8') as jsonFile:
        json.dump(conf.__dict__, jsonFile, indent=4, sort_keys=True)

     # Main
    os.system(exec_str)

    data_name = conf.chemFile + '_' + conf.caseMulticellsOnecell + conf.results_file
    tmp_path = 'out/' + data_name

    export(conf,tmp_path)

    data = {}
    plot_functions.read_solver_stats(tmp_path, data)

    return data


def run_case(conf):
    data = run(conf)

    #if conf.readData:

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
                data[y_key][i] = data[y_key][i] * nSystemsOfCells \
                                 / data["counterLS"][i]
        elif y_key == "timecvStep":
            for i in range(len(data[y_key])):
                data[y_key][i] = data[y_key][i] * nSystemsOfCells \
                                 / data["countercvStep"][i]
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

    print("run_case", conf.case, y_key, ":", data[y_key])

    return data


def run_cases(conf):
    # Run base case
    conf.mpiProcesses = conf.mpiProcessesList[0]
    conf.nCells = int(conf.nCellsCase / conf.mpiProcesses)

    cases_words = conf.caseBase.split()
    conf.caseGpuCpu = cases_words[0]
    conf.caseMulticellsOnecell = cases_words[1]

    conf.case = conf.caseBase
    dataCaseBase = run_case(conf)
    data = {conf.caseBase: dataCaseBase}

    # Run OptimCases
    conf.mpiProcesses = conf.mpiProcessesList[-1]
    conf.nCells = int(conf.nCellsCase / conf.mpiProcesses)
    if conf.nCells == 0:
        print("WARNING: Configured less cells than MPI processes, setting 1 cell per process")
        conf.nCells = 1

    datacases = []
    for caseOptim in conf.casesOptim:
        cases_words = caseOptim.split()
        conf.caseGpuCpu = cases_words[0]
        conf.caseMulticellsOnecell = cases_words[1]

        conf.case = caseOptim
        data[caseOptim] = run_case(conf)

        # calculate measures between caseBase and caseOptim
        if conf.plotYKey == "NRMSE":
            datay = plot_functions.calculate_NMRSE(data, conf.timeSteps)
        elif conf.plotYKey == "MAPE":
            datay = plot_functions.calculate_MAPE(data, conf.timeSteps, conf.MAPETol)
        elif conf.plotYKey == "SMAPE":
            datay = plot_functions.calculate_SMAPE(data, conf.timeSteps)
        elif "Speedup" in conf.plotYKey:
            y_key_words = conf.plotYKey.split()
            y_key = y_key_words[-1]
            # print("WARNING: Check y_key is correct:",y_key)
            datay = plot_functions.calculate_speedup(data, y_key)
        elif conf.plotYKey == "Percentage data transfers CPU-GPU [%]":
            y_key = "timeBiconjGradMemcpy"
            print("elif conf.plotYKey==Time data transfers")
            datay = plot_functions.calculate_BCGPercTimeDataTransfers(data, y_key)
        else:
            raise Exception("Not found plot function for conf.plotYKey")

        if len(conf.cells) > 1:  # Mean timeSteps
            datacases.append(round(np.mean(datay), 2))
        else:
            datacases.append([round(elem, 2) for elem in datay])

        data.pop(caseOptim)

    return datacases


def run_cells(conf):
    datacells = []
    for i in range(len(conf.cells)):
        conf.nCellsCase = conf.cells[i]
        datacases = run_cases(conf)

        if len(conf.cells) > 1:  # Mean timeSteps
            datacells.append(datacases)
        else:
            datacells = datacases

    if len(conf.cells) > 1:  # Mean timeSteps
        datacellsTranspose = np.transpose(datacells)
        datacells = datacellsTranspose.tolist()

    return datacells


def run_diffCells(conf):
    for diff_cells in conf.diffCellsL:
        conf.diffCells = diff_cells
        data = run_cells(conf)
        conf.datacolumns += data


def getCaseName(conf):
    if conf.caseMulticellsOnecell != "CVODE" and conf.caseGpuCpu == "GPU":
        # case_multicells_onecell_name = "LS "
        case_multicells_onecell_name = ""
    else:
        case_multicells_onecell_name = ""

    if conf.caseMulticellsOnecell == "Block-cellsN":
        case_multicells_onecell_name += "Block-cells (N)"
    elif conf.caseMulticellsOnecell == "Block-cells1":
        case_multicells_onecell_name += "Block-cells (1)"
    elif conf.caseMulticellsOnecell == "Block-cellsNhalf":
        case_multicells_onecell_name += "Block-cells (N/2)"
    else:
        case_multicells_onecell_name += conf.caseMulticellsOnecell

    return case_multicells_onecell_name

def saveConf(conf, namex, namey, datax, datay, plot_title, legend):

    init_path = os.path.abspath(os.getcwd())

    os.chdir("../../.." )
    save_folder = "test/monarch/savedConf/"
    if os.path.getsize(save_folder) > 1000000000:
        print("WARNING: More than 1GB in", os.path.abspath(os.getcwd())+"/"+save_folder)
    save_folder = save_folder + namey + " " + plot_title + " Timesteps " + str(conf.timeSteps)
    save_folder = save_folder.replace(":", "")
    #now = datetime.datetime.now()
    #dt_string = now.strftime("%d-%m-%Y %H.%M.%S")
    dt_string = "saveConf"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    save_path = save_folder + "/" + dt_string + ".json"
    with open(save_path, 'w', encoding='utf-8') as jsonFile:
        json.dump(conf.__dict__, jsonFile, indent=4, sort_keys=True)

    print("Configuration saved to", os.path.abspath(os.getcwd())+"/"+save_path)

    #os.chdir(init_path)

def plot_cases(conf):
    # Set plot info
    cases_words = conf.caseBase.split()
    conf.caseGpuCpu = cases_words[0]
    conf.caseMulticellsOnecell = cases_words[1]
    case_multicells_onecell_name = getCaseName(conf)

    case_gpu_cpu_name = conf.caseGpuCpu
    if len(conf.mpiProcessesList) == 2 and conf.caseGpuCpu == "CPU":
        case_gpu_cpu_name = str(conf.mpiProcessesList[0]) + " MPI"

    baseCaseName = ""
    if conf.plotYKey != "Percentage data transfers CPU-GPU [%]":  # Speedup
        baseCaseName = " vs " + case_gpu_cpu_name + " " + case_multicells_onecell_name

    conf.legend = []
    isSameArquiOptim = True
    cases_words = conf.casesOptim[0].split()
    lastArquiOptim = cases_words[0]
    for diff_cells in conf.diffCellsL:
        conf.diffCells = diff_cells
        columnDiffCells = ""
        if len(conf.diffCellsL) > 1:
            columnDiffCells += conf.diffCells + " "

        for caseOptim in conf.casesOptim:
            cases_words = caseOptim.split()
            conf.caseGpuCpu = cases_words[0]
            conf.caseMulticellsOnecell = cases_words[1]
            case_multicells_onecell_name = getCaseName(conf)

            case_gpu_cpu_name = conf.caseGpuCpu
            if len(conf.mpiProcessesList) == 2 and conf.caseGpuCpu == "CPU":
                case_gpu_cpu_name = str(conf.mpiProcessesList[-1]) + " MPI"

            if lastArquiOptim != conf.caseGpuCpu:
                isSameArquiOptim = False
            lastArquiOptim = conf.caseGpuCpu

            column = columnDiffCells
            if len(conf.casesOptim) > 1:
                if isSameArquiOptim:
                    column += case_multicells_onecell_name
                else:
                    column += case_gpu_cpu_name + " " + case_multicells_onecell_name
            conf.legend.append(column)

    optimCaseName = case_gpu_cpu_name + " " + case_multicells_onecell_name
    conf.plotTitle = ""
    if len(conf.casesOptim) > 1:
        if isSameArquiOptim:
            conf.plotTitle += case_gpu_cpu_name + " "
        conf.plotTitle += "Implementations"
    else:
        conf.plotTitle = optimCaseName
    conf.plotTitle += baseCaseName

    if len(conf.diffCellsL) == 1:
        conf.plotTitle += ", " + conf.diffCells + " test"

    namey = conf.plotYKey
    if conf.plotYKey == "Speedup normalized computational timeLS":
        namey = "Speedup without data transfers CPU-GPU"
    if conf.plotYKey == "Speedup counterLS":
        namey = "Speedup iterations CAMP solving"
    if conf.plotYKey == "Speedup normalized timeLS":
        namey = "Speedup linear solver"
    if conf.plotYKey == "Speedup normalized timecvStep":
        namey = "Speedup BDF loop"
    if conf.plotYKey == "Speedup normalized countercvStep":
        namey = "Speedup iterations BDF loop"
    if conf.plotYKey == "Speedup timeCVode":
        namey = "Speedup CAMP solving"
    if conf.plotYKey == "MAPE":
        namey = "MAPE [%]"
    if conf.plotYKey == "Speedup total iterations - counterBCG":
        namey = "Speedup solving iterations BCG"

    if (len(conf.datacolumns) > 1):
        datay = conf.datacolumns
    else:
        datay = conf.datacolumns[0]

    if (len(conf.cells) > 1):
        print_timesteps_title = True
        # print_timesteps_title = False
        if print_timesteps_title:
            conf.plotTitle += ", Timesteps: " + str(conf.timeSteps)
        datax = conf.cells
        plot_x_key = "Cells"
    else:
        conf.plotTitle += ", Cells: " + str(conf.cells[0])
        datax = list(range(1, conf.timeSteps + 1, 1))
        plot_x_key = "Timesteps"

    namex = plot_x_key

    print("plotTitle: ", conf.plotTitle, ", legend:", conf.legend)
    if (namex == "Timesteps"):
        print("Mean:", round(np.mean(datay), 2))
    print(namex, ":", datax)
    print(namey, ":", datay)

    #save_data = True
    #save_data=False
    #if save_data:
    #    saveConf(conf, namex, namey, datax, datay, conf.plotTitle, conf.legend)
    #else:

    #plot_functions.plot(namex, namey, datax, datay, conf.plotTitle, conf.legend, conf.savePlot)


def all_timesteps():
    conf = TestMonarch()

    # conf.chemFile = "simple"
    # conf.chemFile = "monarch_cb05"
    conf.chemFile = "monarch_binned"

    conf.diffCellsL = []
    conf.diffCellsL.append("Realistic")
    # conf.diffCellsL.append("Ideal")

    conf.profileCuda = False
    # conf.profileCuda = True

    conf.mpi = "yes"
    # conf.mpi = "no"

    conf.mpiProcessesList = [1]
    #conf.mpiProcessesList = [40, 1]

    conf.cells = [10]
    # conf.cells = [400,2000,4000]
    # conf.cells = [1,5,10,50,100]
    # conf.cells = [100,500,1000,5000,10000]

    conf.timeSteps = 1
    conf.timeStepsDt = 3

    #conf.caseBase = "CPU One-cell"
    conf.caseBase = "CPU Multi-cells"
    # conf.caseBase="GPU Multi-cells"
    # conf.caseBase="GPU Block-cellsN"
    # conf.caseBase="GPU Block-cells1"

    conf.casesOptim = []
    # conf.casesOptim.append("GPU CVODE")
    # conf.casesOptim.append("GPU Block-cellsNhalf")
    conf.casesOptim.append("GPU Block-cells1")
    # conf.casesOptim.append("GPU Block-cellsN")
    #conf.casesOptim.append("GPU Multi-cells")
    # conf.casesOptim.append("GPU One-cell")
    # conf.casesOptim.append("CPU Multi-cells")

    # conf.plotYKey = "Speedup timeCVode"
    # conf.plotYKey = "Speedup counterLS"
    conf.plotYKey = "Speedup normalized timeLS"
    # conf.plotYKey = "Speedup normalized computational timeLS"
    # conf.plotYKey = "Speedup counterBCG"
    #conf.plotYKey = "Speedup normalized counterBCG"
    # conf.plotYKey = "Speedup total iterations - counterBCG"
    # conf.plotYKey = "Speedup BCG iteration (Comp.timeLS/counterBCG)"
    # conf.plotYKey = "Speedup timecvStep"
    # conf.plotYKey = "Speedup normalized timecvStep"
    # conf.plotYKey = "Speedup normalized countercvStep"
    # conf.plotYKey = "Speedup countercvStep"
    # conf.plotYKey = "Speedup device timecvStep"
    # conf.plotYKey = "Percentage data transfers CPU-GPU [%]"
    # conf.plotYKey="MAPE"
    # conf.plotYKey="SMAPE"
    # conf.plotYKey="NRMSE"
    # conf.MAPETol=1.0E-6

    """END OF CONFIGURATION VARIABLES"""

    conf.results_file = "_solver_stats.csv"
    if conf.plotYKey == "NRMSE" or conf.plotYKey == "MAPE" or conf.plotYKey == "SMAPE":
        conf.results_file = '_results_all_cells.csv'

    jsonFile = open("monarch_box_binned/cb05_abs_tol.json")
    jsonData = json.load(jsonFile)
    conf.MAPETol = jsonData["camp-data"][0]["value"]  # Default: 1.0E-4
    jsonData.clear()

    if not os.path.exists('out'):
        os.makedirs('out')

    if conf.chemFile == "monarch_binned":
        if conf.timeStepsDt != 2:
            print (
                "Warning: Setting timeStepsDt to 2, since it is the usual value for the measures with monarch_binned")
        conf.timeStepsDt = 2
    elif conf.chemFile == "monarch_cb05":
        if conf.timeStepsDt != 3:
            print ("Warning: Setting timeStepsDt to 3, since it is the usual value for the measures with monarch_cb05")
        conf.timeStepsDt = 3
        # if "Realistic" in conf.diffCellsL:
        #  print ("Warning: chemFile == monarch_cb05 only works with Ideal test, setting test to Ideal")
        # conf.diffCellsL = ["Ideal"]

    if not conf.caseBase:
        print("ERROR: caseBase is empty")
        raise

    conf.isSameArquiOptim = True
    cases_words = conf.casesOptim[0].split()
    lastArquiOptim = cases_words[0]

    for caseOptim in conf.casesOptim:
        if caseOptim == conf.caseBase:
            # logger = logging.getLogger(__name__)
            # logger.error(error)
            print ("Error: caseOptim == caseBase")
            raise

    if len(conf.mpiProcessesList) > 2 or len(conf.mpiProcessesList) < 1:
        raise Exception("Length of mpiProcessesList should be between 1 and 2")

    start_time = time.perf_counter()
    conf.datacolumns = []

    run_diffCells(conf)

    end_time = time.perf_counter()
    time_s = end_time - start_time
    if time_s > 1:
        conf.savePlot = True
    else:
        conf.savePlot = False

    plot_cases(conf)

    del conf

def test_copy_conf():
    conf = TestMonarch()

    init_path = os.path.abspath(os.getcwd())
    os.chdir("../../..")
    save_folder = "test/monarch/savedConf/"

    now = datetime.datetime.now()
    dt_string = now.strftime("%d-%m-%Y %H.%M.%S")
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    save_path = save_folder + dt_string + ".json"

    with open(save_path, 'w', encoding='utf-8') as jsonFile:
        json.dump(conf.__dict__, jsonFile, indent=4, sort_keys=True)

    print("Configuration saved to", os.path.abspath(os.getcwd())+"/"+save_path)
    os.chdir(init_path)
    print(os.getcwd())


#test_copy_conf()
#read_conf()
all_timesteps()






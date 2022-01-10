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


class TestMonarch:
    def __init__(self):
        # Configuration
        self._chemFile = "monarch_binned"  # Read.json
        self.diffCellsL = "Realistic"
        self.profileCuda = False
        self.mpi = "yes"
        self.mpiProcessesList = [1]
        self.cells = [100]  # [ 5,10]
        self.timeSteps = 1  # 5
        self.timeStepsDt = 2
        self.caseBase = "CPU Multi-cells"
        self.casesOptim = ["GPU Block-cells1"]
        self.plotYKey = "Speedup normalized timeLS"
        self.MAPETol = 1.0E-4
        # Auxiliar
        self.diffCells = ""
        self.datacolumns = []
        self.legend = []
        self.column = ""
        self.columnDiffCells = ""
        self.casesL = []
        self.cases = []
        self.results_file = "_solver_stats.csv"
        self.diffArquiOptim = False
        self.plotTitle = ""
        self.savePlot = True
        self.mpiProcesses = 1
        self.nCells = 1
        self.casesGpuCpu = []
        self.caseGpuCpu = ""
        self.casesMulticellsOnecells = []
        self.caseMulticellsOnecell = ""
        self.nCellsCase = 1
        self.itsolverConfigFile = "itsolver_options"
        self.campSolverConfigFile = "config_variables_c_solver.txt"


    @property
    def chemFile(self):
        return self._chemFile

    @chemFile.setter
    def chemFile(self, new_chemFile):
        if new_chemFile not in self._chemFile:
            raise
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

    if conf.chemFile == "monarch_binned":
        file1.write("USE_F_CPU=ON\n")
    else:
        file1.write("USE_F_CPU=OFF\n")

    file1.close()


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

    with open('TestMonarch.json', 'w', encoding='utf-8') as jsonFile:
        json.dump(conf.__dict__, jsonFile, indent=4, sort_keys=True)

    # Main
    print(exec_str)
    os.system(exec_str)

    data = {}
    file = 'out/' + conf.chemFile + '_' + conf.caseMulticellsOnecell + conf.results_file
    plot_functions.read_solver_stats(file, data)

    return data


def run_cell(conf):
    y_key_words = conf.plotYKey.split()
    y_key = y_key_words[-1]
    data = {}

    for i in range(len(conf.cases)):

        if len(conf.mpiProcessesList) == len(conf.cases):
            conf.mpiProcesses = conf.mpiProcessesList[i]
            conf.nCells = int(conf.nCellsCase / conf.mpiProcesses)
            if conf.nCells == 0:
                conf.nCells = 1
        else:
            conf.mpiProcesses = conf.mpiProcessesList[0]
            conf.nCells = conf.nCellsCase

        conf.caseMulticellsOnecell = conf.casesMulticellsOnecell[i]
        conf.caseGpuCpu = conf.casesGpuCpu[i]
        data[conf.cases[i]] = run(conf)

        if ("timeLS" in conf.plotYKey and "computational" in conf.plotYKey):
            data = plot_functions.calculate_computational_timeLS( \
                data, "timeBiconjGradMemcpy", conf.cases[i])

        if ("normalized" in conf.plotYKey):
            if (y_key == "counterBCG" or y_key == "timeLS"):
                data = plot_functions.normalize_by_counterLS_and_cells( \
                    data, y_key, conf.nCells, conf.cases[i])
            elif (y_key == "timecvStep"):
                data = plot_functions.normalize_by_countercvStep_and_cells( \
                    data, "timecvStep", conf.nCells, conf.cases[i])
            else:
                raise Exception("Unkown normalized case", y_key)

    if (len(conf.cases) != 2):
        raise Exception("Cases to compare != 2, check cases")

    if ("(Comp.timeLS/counterBCG)" in conf.plotYKey):
        data = plot_functions.calculate_computational_timeLS( \
            data, "timeBiconjGradMemcpy")
        y_key = "timeLS"
        for case in conf.cases:
            for i in range(len(data[case][y_key])):
                data[case][y_key][i] = data[case][y_key][i] \
                                       / data[case]["counterBCG"][i]

    if (conf.plotYKey == "NRMSE"):
        datay = plot_functions.calculate_NMRSE(data, conf.timeSteps)
    elif (conf.plotYKey == "MAPE"):
        datay = plot_functions.calculate_MAPE(data, conf.timeSteps, conf.MAPETol)
    elif (conf.plotYKey == "SMAPE"):
        datay = plot_functions.calculate_SMAPE(data, conf.timeSteps)
    elif ("Speedup" in conf.plotYKey):
        # y_key = conf.plotYKey.replace('Speedup ', '')
        # y_key_words = conf.plotYKey.split()
        # y_key = y_key_words[-1]
        # print(y_key)
        datay = plot_functions.calculate_speedup(data, y_key)
    elif conf.plotYKey == "Percentage data transfers CPU-GPU [%]":
        y_key = "timeBiconjGradMemcpy"
        print("elif conf.plotYKey==Time data transfers")
        datay = plot_functions.calculate_BCGPercTimeDataTransfers(data, y_key)
    else:
        raise Exception("Not found plot function for conf.plotYKey")

    return datay


def run_case(conf):
    datacase = []
    for i in range(len(conf.cells)):

        conf.nCellsCase = conf.cells[i]
        datay_cell = run_cell(conf)

        # print(datay_cell)
        if (len(conf.cells) > 1):
            # Mean timeSteps
            datacase.append(np.mean(datay_cell))
        else:
            datacase = datay_cell

    return datacase


def run_diff_cells(conf):

    #column=columnHeader
    #conf.column = conf.columnDiffCells
    for j in range(len(conf.casesL)):
        conf.cases = conf.casesL[j]
        conf.casesGpuCpu = [""] * len(conf.cases)
        conf.casesMulticellsOnecell = [""] * len(conf.cases)
        cases_multicells_onecell_name = [""] * len(conf.cases)
        cases_gpu_cpu_name = [""] * len(conf.cases)
        for i in range(len(conf.cases)):
            cases_words = conf.cases[i].split()
            conf.casesGpuCpu[i] = cases_words[0]
            conf.casesMulticellsOnecell[i] = cases_words[1]

            if conf.casesMulticellsOnecell[i] == "Block-cellsN":
                cases_multicells_onecell_name[i] = "Block-cells (N)"
            elif conf.casesMulticellsOnecell[i] == "Block-cells1":
                cases_multicells_onecell_name[i] = "Block-cells (1)"
            else:
                cases_multicells_onecell_name[i] = conf.casesMulticellsOnecell[i]

            if (len(conf.mpiProcessesList) == 2):
                if conf.casesGpuCpu[i] == "CPU":
                    cases_gpu_cpu_name[i] = str(conf.mpiProcessesList[i]) + " MPI"
                    # print("conf.casesGpuCpu[i]==CPU",cases_gpu_cpu_name[i])
                # elif conf.casesGpuCpu[i]=="GPU":
                #  cases_gpu_cpu_name[i]=str(gpus) + " GPU" #always 1 GPU, so comment this on the test section
                else:
                    cases_gpu_cpu_name[i] = conf.casesGpuCpu[i]
            else:
                cases_gpu_cpu_name[i] = conf.casesGpuCpu[i]

        conf.column = conf.columnDiffCells
        if len(conf.casesL) > 1:
            if conf.diffArquiOptim:
                conf.column += cases_gpu_cpu_name[1] + " " + cases_multicells_onecell_name[1]
            else:
                conf.column += cases_multicells_onecell_name[1]

        conf.legend.append(conf.column)

        datacase = run_case(conf)
        conf.datacolumns.append(datacase)

    conf.plotTitle = ""
    first_word = cases_gpu_cpu_name[1] + " " + cases_multicells_onecell_name[1]

    if conf.plotYKey == "Percentage data transfers CPU-GPU [%]":
        second_word = ""
    else:  # Speedup
        second_word = " vs " + cases_gpu_cpu_name[0] + " " + cases_multicells_onecell_name[0]

    if (len(conf.casesL) > 1):
        if (not conf.diffArquiOptim):
            conf.plotTitle += cases_gpu_cpu_name[1] + " "
        conf.plotTitle += "Implementations" + second_word
    else:
        conf.plotTitle = first_word + second_word


def plot_cases(conf):
    namey = conf.plotYKey
    if conf.plotYKey == "Speedup normalized computational timeLS":
        namey = "Speedup without data transfers CPU-GPU"
    if conf.plotYKey == "Speedup counterLS":
        namey = "Speedup iterations CVODE solving"
    if conf.plotYKey == "Speedup normalized timeLS":
        namey = "Speedup linear solver"
    if conf.plotYKey == "Speedup timeCVode":
        namey = "Speedup CVODE solving"
    if conf.plotYKey == "MAPE":
        namey = "MAPE [%]"

    if (len(conf.datacolumns) > 1):
        datay = conf.datacolumns
    else:
        datay = conf.datacolumns[0]

    if (len(conf.cells) > 1):
        # print_timesteps_title=True
        print_timesteps_title = False
        if print_timesteps_title:
            # conf.plotTitle+=", Mean over "+str(timeSteps)+ " timeSteps"
            conf.plotTitle += ", Timesteps: " + str(conf.timeSteps)
        datax = conf.cells
        plot_x_key = "Cells"
    else:
        conf.plotTitle += ", Cells: " + str(conf.cells[0])
        datax = list(range(1, conf.timeSteps + 1, 1))
        plot_x_key = "Timesteps"

    namex = plot_x_key

    print("plotTitle: ", conf.plotTitle, ", legend:", conf.legend)
    print(namex,":",datax)
    print(namey, ":", datay)

    #plot_functions.plot(namex, namey, datax, datay, conf.plotTitle, conf.legend, conf.savePlot)


def all_timesteps():
    conf = TestMonarch()

    # conf.chemFile="simple"
    # conf.chemFile="monarch_cb05"
    # conf.chemFile = "monarch_binned"
    conf.chemFile = "monarch_binned"

    conf.diffCellsL = []
    conf.diffCellsL.append("Realistic")
    #conf.diffCellsL.append("Ideal")

    conf.profileCuda = False
    # conf.profileCuda = True

    conf.mpi = "yes"
    # conf.mpi = "no"

    conf.mpiProcessesList = [1]
    #conf.mpiProcessesList =  [40,1]

    conf.cells = [100]
    #conf.cells = [100,1000]
    #conf.cells = [1,5,10,50,100]
    #conf.cells = [100,500,1000,5000,10000]

    conf.timeSteps = 1
    conf.timeStepsDt = 2

    #conf.caseBase = "CPU One-cell"
    conf.caseBase = "CPU Multi-cells"
    # conf.caseBase="GPU Multi-cells"
    # conf.caseBase="GPU Block-cellsN"
    # conf.caseBase="GPU Block-cells1"

    conf.casesOptim = []
    conf.casesOptim.append("GPU Block-cells1")
    #conf.casesOptim.append("GPU Block-cellsN")
    #conf.casesOptim.append("GPU Multi-cells")
    #conf.casesOptim.append("GPU One-cell")
    #conf.casesOptim.append("CPU Multi-cells")

    #conf.plotYKey = "Speedup timeCVode"
    #conf.plotYKey = "Speedup counterLS"
    conf.plotYKey = "Speedup normalized timeLS"
    # conf.plotYKey = "Speedup normalized computational timeLS"
    # conf.plotYKey = "Speedup counterBCG"
    # conf.plotYKey = "Speedup total iterations - counterBCG"
    # conf.plotYKey = "Speedup normalized counterBCG"
    # conf.plotYKey = "Speedup BCG iteration (Comp.timeLS/counterBCG)"
    # conf.plotYKey = "Speedup timecvStep"
    # conf.plotYKey = "Speedup normalized timecvStep"#not needed, is always normalized
    # conf.plotYKey = "Speedup device timecvStep"
    #conf.plotYKey = "Percentage data transfers CPU-GPU [%]"

    # conf.plotYKey="MAPE"
    # conf.plotYKey="SMAPE"
    # conf.plotYKey="NRMSE"
      # MAPE=0
    # conf.MAPETol=1.0E-6 #MAPE~=0.5

    """END OF CONFIGURATION VARIABLES"""

    conf.results_file = "_solver_stats.csv"
    if conf.plotYKey == "NRMSE" or conf.plotYKey == "MAPE" or conf.plotYKey == "SMAPE":
        conf.results_file = '_results_all_cells.csv'

    jsonFile = open("monarch_box_binned/cb05_abs_tol.json")
    jsonData = json.load(jsonFile)
    conf.MAPETol = jsonData["camp-data"][0]["value"] # Default: 1.0E-4
    jsonData.clear()

    if not os.path.exists('out'):
        os.makedirs('out')

    if "total" in conf.plotYKey and "counterBCG" in conf.plotYKey:
        print("WARNING: Remember to enable solveBcgCuda_sum_it")
    if conf.chemFile == "monarch_cb05":
        conf.timeStepsDt = 3

    conf.savePlot = False
    start_time = time.perf_counter()

    conf.casesL = []
    conf.cases = []
    conf.diffArquiOptim = False
    cases_words = conf.casesOptim[0].split()
    lastArquiOptim = cases_words[0]
    for caseOptim in conf.casesOptim:
        if caseOptim == conf.caseBase:
            # logger = logging.getLogger(__name__)
            # logger.error(error)
            print ("Error: caseOptim == caseBase")
            raise
        conf.cases = [conf.caseBase] + [caseOptim]
        conf.casesL.append(conf.cases)

        cases_words = caseOptim.split()
        arqui = cases_words[0]
        if (lastArquiOptim != arqui):
            conf.diffArquiOptim = True
        lastArquiOptim = arqui

    conf.datacolumns = []
    conf.legend = []
    conf.plotTitle = ""
    for diff_cells in conf.diffCellsL:
        conf.diffCells = diff_cells
        if (conf.chemFile == "monarch_cb05"):
            conf.diffCells = "Ideal"

        conf.columnDiffCells = ""
        if len(conf.diffCellsL) > 1:
            conf.columnDiffCells += conf.diffCells + " "

        run_diff_cells(conf)

    end_time = time.perf_counter()
    time_s = end_time - start_time
    if time_s > 60:
        conf.savePlot = True

    if (len(conf.diffCellsL) == 1):
        conf.plotTitle += ", " + conf.diffCells + " test"

    plot_cases(conf)
    del conf


all_timesteps()

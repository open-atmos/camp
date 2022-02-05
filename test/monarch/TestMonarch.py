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
        self.case = []
        self.results_file = "_solver_stats.csv"
        self.isSameArquiOptim = False
        self.plotTitle = ""
        self.savePlot = True
        self.mpiProcesses = 1
        self.nCells = 1
        self.nCellsCase = 1
        self.casesGpuCpu = []
        self.caseGpuCpu = ""
        self.casesMulticellsOnecells = []
        self.caseMulticellsOnecell = ""
        self.itsolverConfigFile = "itsolver_options.txt"
        self.campSolverConfigFile = "config_variables_c_solver.txt"
        self.dataCaseBase = {}

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
    print(exec_str,conf.caseGpuCpu,conf.caseMulticellsOnecell)
    os.system(exec_str)

    data = {}
    file = 'out/' + conf.chemFile + '_' + conf.caseMulticellsOnecell + conf.results_file
    plot_functions.read_solver_stats(file, data)

    return data


def run_case2(conf):

    data = run(conf)

    if "timeLS" in conf.plotYKey and "computational" in conf.plotYKey\
        and "GPU" in conf.case:
        for i in range(len(data["timeLS"])):
            data["timeLS"][i] = data["timeLS"][i] - data["timeBiconjGradMemcpy"][i]

    y_key_words = conf.plotYKey.split()
    y_key = y_key_words[-1]
    if "normalized" in conf.plotYKey:
        if y_key == "counterBCG" or y_key == "timeLS":
            nSystemsOfCells = 1
            if ("One-cell" in conf.case):
                nSystemsOfCells = conf.nCells
            data = plot_functions.normalize_by_counterLS_and_cells2( \
                data, y_key,nSystemsOfCells)

        elif y_key == "timecvStep":
            if "One-cell" in conf.case:
                cells_multiply = conf.nCells
                #print("One-cell")
            elif "Multi-cells" in conf.case:
                cells_multiply = 1
                #print("Multi-cells")
            else:
                raise Exception("normalize_by_countercvStep_and_cells case without One-cell or Multi-cells key name")

            for i in range(len(data[y_key])):
                data[y_key][i] = data[y_key][i] \
                              / data["countercvStep"][i] * cells_multiply
        else:
            raise Exception("Unkown normalized case", y_key)

    if "(Comp.timeLS/counterBCG)" in conf.plotYKey and "GPU" in conf.case:
        y_key = "timeLS"
        for i in range(len(data["timeLS"])):
            data["timeLS"][i] = data["timeLS"][i] - data["timeBiconjGradMemcpy"][i]
        for i in range(len(data["timeLS"])):
            data["timeLS"][i] = data["timeLS"][i] \
                                            / data["counterBCG"][i]

        for j in range(len(data[conf.cases[j]]["timeLS"])):
            data[conf.cases[j]]["timeLS"][j] = data[conf.cases[j]]["timeLS"][j] \
                                            / data[conf.cases[j]]["counterBCG"][j]

    return data


def run_cases2(conf):

    #Run base case
    conf.mpiProcesses = conf.mpiProcessesList[0]
    conf.nCells = int(conf.nCellsCase / conf.mpiProcesses)

    cases_words = conf.caseBase.split()
    conf.caseGpuCpu = cases_words[0]
    conf.caseMulticellsOnecell = cases_words[1]

    conf.case = conf.caseBase
    conf.dataCaseBase = run_case2(conf)
    data = {conf.caseBase: conf.dataCaseBase}

    #Run OptimCases
    #todo recheck if this should be moved ot other place
    if len(conf.mpiProcessesList) == 1:
        i = 0
    elif len(conf.mpiProcessesList) == 2:
        i = 1
    else:
        raise Exception("Length of mpiProcessesList > 2 when it should be 2 or 1")
    conf.mpiProcesses = conf.mpiProcessesList[i]
    conf.nCells = int(conf.nCellsCase / conf.mpiProcesses)
    if conf.nCells == 0:
        conf.nCells = 1

    datacases = []
    for caseOptim in conf.casesOptim:
        cases_words = caseOptim.split()
        conf.caseGpuCpu = cases_words[0]
        conf.caseMulticellsOnecell = cases_words[1]

        conf.case = caseOptim
        data[caseOptim] = run_case2(conf)

        #calculate measures between caseBase and caseOptim
        if conf.plotYKey == "NRMSE":
            datay = plot_functions.calculate_NMRSE(data, conf.timeSteps)
        elif conf.plotYKey == "MAPE":
            datay = plot_functions.calculate_MAPE(data, conf.timeSteps, conf.MAPETol)
        elif conf.plotYKey == "SMAPE":
            datay = plot_functions.calculate_SMAPE(data, conf.timeSteps)
        elif "Speedup" in conf.plotYKey:
            y_key_words = conf.plotYKey.split()
            y_key = y_key_words[-1]
            #print("WARNING: Check y_key is correct:",y_key)
            datay = plot_functions.calculate_speedup(data, y_key)
        elif conf.plotYKey == "Percentage data transfers CPU-GPU [%]":
            y_key = "timeBiconjGradMemcpy"
            print("elif conf.plotYKey==Time data transfers")
            datay = plot_functions.calculate_BCGPercTimeDataTransfers(data, y_key)
        else:
            raise Exception("Not found plot function for conf.plotYKey")

        if len(conf.cells) > 1:  # Mean timeSteps
            #datacases.append(np.mean(datay))
            datacases.append(np.mean(datay))
        else:  # todo
            datacases.append(datay)
            #datacases = datay

        data.pop(caseOptim)

    print("run_cases2 datacases", datacases)

    return datacases

def run_cells2(conf):
    datacells = []
    for i in range(len(conf.cells)):
        conf.nCellsCase = conf.cells[i]
        datacases = run_cases2(conf)

        if len(conf.cells) > 1:  # Mean timeSteps
            datacells.append(datacases)
            #conf.datacolumns.append(datacases)
        else:  # todo
            datacells = datacases
            #conf.datacolumns = datacases
            #conf.datacolumns.append(datacases)

    print("run_cells2 datacells", datacells)

    if len(conf.cells) > 1:  # Mean timeSteps
        datacellsTranspose = np.transpose(datacells)
        datacells = datacellsTranspose.tolist()


    print("run_cells2 datacells", datacells)
    print("run_cells2 conf.datacolumns", conf.datacolumns)

    return datacells

def run_diffCells(conf):

    #todo dont reset datacells, we need to store next iter
    for diff_cells in conf.diffCellsL:
        conf.diffCells = diff_cells
        data = run_cells2(conf)
        #if len(conf.cells) > 1:  # Mean timeSteps
        #conf.datacolumns.append(data)
        conf.datacolumns += data


    #conf.datacolumns = data

    #if len(conf.cells) > 1:  # Mean timeSteps
     #   conf.datacolumns = np.transpose(conf.datacolumns)
        #  conf.datacolumns.append(np.transpose(datacells))
    #else:
        #conf.datacolumns = datacases
        #conf.datacolumns.append(data)

    print("run_diffCells conf.datacolumns", conf.datacolumns)

def plot_cases2(conf):

    #Set plot info

    cases_words = conf.caseBase.split()
    conf.caseGpuCpu = cases_words[0]
    conf.caseMulticellsOnecell = cases_words[1]

    case_multicells_onecell_name = conf.caseMulticellsOnecell
    if conf.caseMulticellsOnecell== "Block-cellsN":
        case_multicells_onecell_name = "Block-cells (N)"
    elif conf.caseMulticellsOnecell == "Block-cells1":
        case_multicells_onecell_name = "Block-cells (1)"
    elif conf.caseMulticellsOnecell == "Block-cellsNhalf":
        case_multicells_onecell_name = "Block-cells (N/2)"

    case_gpu_cpu_name = conf.caseGpuCpu
    if len(conf.mpiProcessesList) == 2 and conf.caseGpuCpu == "CPU":
        case_gpu_cpu_name = str(conf.mpiProcessesList[0]) + " MPI"
        #todo remove casesGpuCpu and others cases not used

    baseCaseName = ""
    if conf.plotYKey != "Percentage data transfers CPU-GPU [%]":  # Speedup
        baseCaseName = " vs " + case_gpu_cpu_name+ " " + case_multicells_onecell_name

    for diff_cells in conf.diffCellsL:
        conf.diffCells = diff_cells
        conf.columnDiffCells = ""
        if len(conf.diffCellsL) > 1:
            conf.columnDiffCells += conf.diffCells + " "

        for caseOptim in conf.casesOptim:
            cases_words = caseOptim.split()
            conf.caseGpuCpu = cases_words[0]
            conf.caseMulticellsOnecell = cases_words[1]

            if conf.caseMulticellsOnecell== "Block-cellsN":
                case_multicells_onecell_name = "Block-cells (N)"
            elif conf.caseMulticellsOnecell == "Block-cells1":
                case_multicells_onecell_name = "Block-cells (1)"
            elif conf.caseMulticellsOnecell == "Block-cellsNhalf":
                case_multicells_onecell_name = "Block-cells (N/2)"
            else:
                case_multicells_onecell_name = conf.caseMulticellsOnecell

            case_gpu_cpu_name = conf.caseGpuCpu
            if len(conf.mpiProcessesList) == 2 and conf.caseGpuCpu == "CPU":
                case_gpu_cpu_name = str(conf.mpiProcessesList[-1]) + " MPI"

            conf.column = conf.columnDiffCells
            #if len(conf.casesL) > 1:
            if len(conf.casesOptim) > 1: #todo fix column with Ideal Realistic cases
                if conf.isSameArquiOptim:
                    conf.column += case_multicells_onecell_name
                else:
                    conf.column += case_gpu_cpu_name + " " + case_multicells_onecell_name
            conf.legend.append(conf.column)

    optimCaseName = case_gpu_cpu_name + " " + case_multicells_onecell_name
    conf.plotTitle = ""
    if len(conf.casesL) > 1:
        if conf.isSameArquiOptim:
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

    plot_functions.plot(namex, namey, datax, datay, conf.plotTitle, conf.legend, conf.savePlot)

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

        if (len(conf.cells) > 1):
            # Mean timeSteps
            datacase.append(np.mean(datay_cell))
        else:
            datacase = datay_cell

    return datacase


def run_diff_cells(conf):

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

            #print(conf.casesMulticellsOnecell[i])

            if conf.casesMulticellsOnecell[i] == "Block-cellsN":
                cases_multicells_onecell_name[i] = "Block-cells (N)"
            elif conf.casesMulticellsOnecell[i] == "Block-cells1":
                cases_multicells_onecell_name[i] = "Block-cells (1)"
            elif conf.casesMulticellsOnecell[i] == "Block-cellsNhalf":
                cases_multicells_onecell_name[i] = "Block-cells (N/2)"
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
            if conf.isSameArquiOptim:
                conf.column += cases_multicells_onecell_name[1]
            else:
                conf.column += cases_gpu_cpu_name[1] + " " + cases_multicells_onecell_name[1]

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
        if conf.isSameArquiOptim:
            conf.plotTitle += cases_gpu_cpu_name[1] + " "
        conf.plotTitle += "Implementations" + second_word
    else:
        conf.plotTitle = first_word + second_word


def plot_cases(conf):
    namey = conf.plotYKey
    if conf.plotYKey == "Speedup normalized computational timeLS":
        namey = "Speedup without data transfers CPU-GPU"
    if conf.plotYKey == "Speedup counterLS":
        namey = "Speedup iterations CAMP solving"
    if conf.plotYKey == "Speedup normalized timeLS":
        namey = "Speedup linear solver"
    if conf.plotYKey == "Speedup timeCVode":
        namey = "Speedup CAMP solving"
    if conf.plotYKey == "MAPE":
        namey = "MAPE [%]"
    if conf.plotYKey == "Speedup total iterations - counterBCG":
        namey = "Speedup solving iterations BCG"

    if len(conf.datacolumns) > 1:
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
    conf.chemFile = "monarch_binned"

    conf.diffCellsL = []
    conf.diffCellsL.append("Realistic")
    conf.diffCellsL.append("Ideal")

    conf.profileCuda = False
    # conf.profileCuda = True

    conf.mpi = "yes"
    # conf.mpi = "no"

    conf.mpiProcessesList = [1]
    #conf.mpiProcessesList = [40,1]

    conf.cells = [100]
    #conf.cells = [100,1000]
    #conf.cells = [1,5,10,50,100]
    #conf.cells = [100,500,1000,5000,10000]

    conf.timeSteps = 5
    conf.timeStepsDt = 2

    conf.caseBase = "CPU One-cell"
    #conf.caseBase = "CPU Multi-cells"
    # conf.caseBase="GPU Multi-cells"
    #conf.caseBase="GPU Block-cellsN"
    #conf.caseBase="GPU Block-cells1"

    conf.casesOptim = []
    #conf.casesOptim.append("GPU Block-cellsNhalf")
    conf.casesOptim.append("GPU Block-cells1")
    #conf.casesOptim.append("GPU Block-cellsN")
    #conf.casesOptim.append("GPU Multi-cells")
    #conf.casesOptim.append("GPU One-cell")
    conf.casesOptim.append("CPU Multi-cells")

    conf.plotYKey = "Speedup timeCVode"
    #conf.plotYKey = "Speedup counterLS"
    #conf.plotYKey = "Speedup normalized timeLS"
    # conf.plotYKey = "Speedup normalized computational timeLS"
    # conf.plotYKey = "Speedup counterBCG"
    #conf.plotYKey = "Speedup total iterations - counterBCG"
    # conf.plotYKey = "Speedup normalized counterBCG"
    # conf.plotYKey = "Speedup BCG iteration (Comp.timeLS/counterBCG)"
    # conf.plotYKey = "Speedup timecvStep"
    # conf.plotYKey = "Speedup normalized timecvStep"#not needed, is always normalized
    # conf.plotYKey = "Speedup device timecvStep"
    #conf.plotYKey = "Percentage data transfers CPU-GPU [%]"
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
    conf.MAPETol = jsonData["camp-data"][0]["value"] # Default: 1.0E-4
    jsonData.clear()

    if not os.path.exists('out'):
        os.makedirs('out')

    if conf.chemFile == "monarch_cb05":
        conf.timeStepsDt = 3
        for i, diff_cells in enumerate(conf.diffCellsL):
            conf.diffCellsL[i] = "Ideal"

    conf.casesL = []
    conf.cases = []
    conf.isSameArquiOptim = True
    cases_words = conf.casesOptim[0].split()
    lastArquiOptim = cases_words[0]

    for caseOptim in conf.casesOptim:
        if caseOptim == conf.caseBase:
            # logger = logging.getLogger(__name__)
            # logger.error(error)
            print ("Error: caseOptim == caseBase")
            raise
        #todo check if this is needed after finishing run_cases2
        conf.cases = [conf.caseBase] + [caseOptim]
        conf.casesL.append(conf.cases)

        cases_words = caseOptim.split()
        arqui = cases_words[0]
        if lastArquiOptim != arqui:
            conf.isSameArquiOptim = False
        lastArquiOptim = arqui

    conf.datacolumns = []
    conf.legend = []
    conf.plotTitle = "" #todo move this
    start_time = time.perf_counter()

    issue4 = True
    #issue4 = False

    if issue4:
     run_diffCells(conf)
    else:
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
    else:
        conf.savePlot = False

    if len(conf.diffCellsL) == 1:
        conf.plotTitle += ", " + conf.diffCells + " test"

    if issue4:
        plot_cases2(conf)
    else:
        plot_cases(conf)

    del conf


all_timesteps()

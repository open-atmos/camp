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


# todo: https://pythonexamples.org/convert-python-class-object-to-json/
class TestMonarch:
    def __init__(self):
        #Configuration
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
        #Auxiliar
        self.diffCells=""
        self.datacolumns = []
        self.legend = []
        self.column = ""
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


    @property
    def chemFile(self):
        return self._chemFile

    @chemFile.setter
    def chemFile(self, new_chemFile):
        if new_chemFile not in self._chemFile:
            raise
        self._chemFile = new_chemFile


# mark = Geeks()

# mark.age = 19

# print(mark.age)

def write_itsolver_config_file(conf):
    file1 = open("itsolver_options.txt", "w")

    cells_method_str = "CELLS_METHOD=" + conf.caseMulticellsOnecell
    file1.write(cells_method_str)
    # print("Saved", cells_method_str)

    file1.close()

def write_camp_config_file(conf):
    file1 = open("config_variables_c_solver.txt", "w")


    if (conf.caseGpuCpu == "CPU"):
        file1.write("USE_CPU=ON\n")
    else:
        file1.write("USE_CPU=OFF\n")
    # print("Saved", conf.caseGpuCpu)

    file1.close()

def run(conf):
    exec_str = ""
    if conf.mpi == "yes":
        exec_str += "mpirun -v -np " + str(conf.mpiProcesses) + " --bind-to none "
        # exec_str+="srun -n "+str(conf.mpiProcesses)+" "

    if (conf.profileCuda and conf.caseGpuCpu == "GPU"):
        pathNvprof = "nvprof/"
        Path(pathNvprof).mkdir(parents=True, exist_ok=True)
        exec_str += "nvprof --analysis-metrics -f -o " + pathNvprof + \
                    conf.chemFile + \
                    conf.caseMulticellsOnecell + str(conf.nCells) + ".nvprof "
        # --print-gpu-summary
        print("Nvprof file saved in ", os.path.abspath(os.getcwd()) \
              + "/" + pathNvprof)

    exec_str += "../../mock_monarch config_" + conf.chemFile + ".json " + "interface_" + conf.chemFile \
                + ".json " + conf.chemFile

    ADD_EMISIONS = "OFF"
    if conf.chemFile == "monarch_binned":
        ADD_EMISIONS = "ON"

    exec_str += " " + ADD_EMISIONS

    # CAMP solver option GPU-CPU
    write_camp_config_file(conf)

    # Onecell-Multicells itsolver
    write_itsolver_config_file(conf)
    if conf.caseGpuCpu == "GPU" and conf.caseMulticellsOnecell != "One-cell":
        # print("conf.caseGpuCpu==GPU and case!=One-cell")
        conf.caseMulticellsOnecell = "Multi-cells"

    # Onecell-Multicells

    print(exec_str + " " + str(conf.nCells) + " " + conf.caseMulticellsOnecell +
          " " + str(conf.timeSteps) + " " + conf.diffCells)
    os.system(
        exec_str + " " + str(conf.nCells) + " " + conf.caseMulticellsOnecell +
        " " + str(conf.timeSteps) + " " + conf.diffCells)

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
            # print("len(conf.mpiProcessesList)==len(conf.cases)",len(conf.cases))
            conf.mpiProcesses = conf.mpiProcessesList[i]
            conf.nCells = int(conf.nCellsCase / conf.mpiProcesses)
            if conf.nCells == 0:
                conf.nCells = 1
        else:
            conf.mpiProcesses = conf.mpiProcessesList[0]
            conf.nCells = conf.nCellsCase

        conf.caseMulticellsOnecell=conf.casesMulticellsOnecell[i]
        conf.caseGpuCpu=conf.casesGpuCpu[i]
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
        # print(data)
        for case in conf.cases:
            for i in range(len(data[case][y_key])):
                data[case][y_key][i] = data[case][y_key][i] \
                                       / data[case]["counterBCG"][i]

    # print(data)

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
        datay = plot_functions.calculate_speedup2(data, y_key)
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

        # print("conf.casesGpuCpu",conf.casesGpuCpu)
        if (len(conf.casesL) > 1):
            conf.column = conf.column + cases_multicells_onecell_name[1]
            if (conf.diffArquiOptim):
                conf.column = conf.column + cases_gpu_cpu_name[1] + " " + cases_multicells_onecell_name[1]

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

    print(namey, ":", datay)

    #plot_functions.plot(namex,namey,datax,datay,conf.plotTitle,conf.legend,conf.savePlot)


def all_timesteps():

    conf = TestMonarch()

    conf2 = {}

    # conf.chemFile="simple"
    # conf.chemFile="monarch_cb05"
    #conf.chemFile = "monarch_binned"
    conf.chemFile = "monarch_binned"

    conf.diffCellsL = []
    conf.diffCellsL.append("Realistic")
    # conf.diffCellsL.append("Ideal")

    conf.profileCuda = False
    # conf.profileCuda = True

    mpi = "yes"
    conf.mpi = "yes"
    # conf.mpi = "no"

    mpiProcessesList = [1]
    conf.mpiProcessesList = [1]
    # conf.mpiProcessesList =  [40,1]

    conf.cells = [100]
    cells = [100]
    # conf.cells = [5,10]
    # conf.cells = [100,500,1000]
    # conf.cells = [1,5,10,50,100]
    # conf.cells = [100,500,1000,5000,10000]

    conf.timeSteps = 1
    timeSteps = 1
    conf.timeStepsDt = 2  # TODO pending send timeStepsDt to mock_monarch

    # conf.caseBase="CPU One-cell"
    conf.caseBase = "CPU Multi-cells"
    # conf.caseBase="GPU Multi-cells"
    # conf.caseBase="GPU Block-cellsN"
    # conf.caseBase="GPU Block-cells1"

    conf.casesOptim = []
    conf.casesOptim.append("GPU Block-cells1")
    # conf.casesOptim.append("GPU Block-cellsN")
    # conf.casesOptim.append("GPU Multi-cells")
    # conf.casesOptim.append("GPU One-cell")
    # conf.casesOptim.append("CPU Multi-cells")

    # conf.cases = ["Historic"]
    # conf.cases = ["CPU One-cell"]
    # conf.cases = ["CPU Multi-cells"]
    # conf.cases = ["GPU One-cell"]

    # conf.plotYKey = "Speedup timeCVode"
    # conf.plotYKey = "Speedup counterLS"
    conf.plotYKey = "Speedup normalized timeLS"
    # conf.plotYKey = "Speedup normalized computational timeLS"
    # conf.plotYKey = "Speedup counterBCG"
    # conf.plotYKey = "Speedup total iterations - counterBCG"
    # conf.plotYKey = "Speedup normalized counterBCG"
    # conf.plotYKey = "Speedup BCG iteration (Comp.timeLS/counterBCG)"
    # conf.plotYKey = "Speedup timecvStep"
    # conf.plotYKey = "Speedup normalized timecvStep"#not needed, is always normalized
    # conf.plotYKey = "Speedup device timecvStep"
    # conf.plotYKey = "Percentage data transfers CPU-GPU [%]"

    # conf.plotYKey = "Percentages solveCVODEGPU" #Pending function
    # conf.plotYKey="MAPE"
    # conf.plotYKey="SMAPE"
    # conf.plotYKey="NRMSE"
    conf.MAPETol = 1.0E-4  # MAPE=0
    # conf.MAPETol=1.0E-6 #MAPE~=0.5

    """END OF CONFIGURATION VARIABLES"""

    conf.results_file = "_solver_stats.csv"
    if conf.plotYKey == "NRMSE" or conf.plotYKey == "MAPE" or conf.plotYKey == "SMAPE":
        conf.results_file = '_results_all_cells.csv'

    if not os.path.exists('out'):
        os.makedirs('out')

    print("WARNING: DEVELOPING CSR")

    if "total" in conf.plotYKey:
        print("WARNING: Remember to enable solveBcgCuda_sum_it")
    elif "counterBCG" in conf.plotYKey:
        print("WARNING: Remember to disable solveBcgCuda_sum_it")

    if conf.chemFile == "monarch_binned":
        print("WARNING: ENSURE DERIV_CPU_ON_GPU IS ON")

    conf.savePlot = False
    start_time = time.perf_counter()

    conf.casesL = []
    conf.cases = []
    conf.diffArquiOptim = False
    cases_words = conf.casesOptim[0].split()
    lastArquiOptim = cases_words[0]
    for caseOptim in conf.casesOptim:
        conf.cases = [conf.caseBase] + [caseOptim]
        conf.casesL.append(conf.cases)

        cases_words = caseOptim.split()
        arqui = cases_words[0]
        if (lastArquiOptim != arqui):
            conf.diffArquiOptim = True
        lastArquiOptim = arqui

    if (conf.cases[0] == "Historic"):
        if (len(conf.cells) < 2):
            # TODO check if still pending
            print("WARNING: PENDING TEST HISTORIC WITH TIMESTEPS AS AXIS X")

        conf.casesL.append(["CPU One-cell", "GPU Block-cells1"])
        conf.casesL.append(["CPU One-cell", "GPU Block-cellsN"])
        conf.casesL.append(["CPU One-cell", "GPU Multi-cells"])
        conf.casesL.append(["CPU One-cell", "CPU Multi-cells"])
        conf.casesL.append(["CPU One-cell", "GPU One-cell"])
    elif (len(conf.casesL) == 0):
        print("len(conf.casesL)==0")
        conf.casesL.append(conf.cases)

    conf.datacolumns = []
    conf.legend = []
    conf.plotTitle = ""
    for diff_cells in conf.diffCellsL:
        conf.diffCells = diff_cells
        if (conf.chemFile == "monarch_cb05"):
            conf.diffCells = "Ideal"
            print("WARNING: ENSURE DERIV_CPU_ON_GPU IS OFF")

        conf.column = ""
        if (len(conf.diffCellsL) > 1):
            conf.column += conf.diffCells + " "

        run_diff_cells(conf)

    end_time = time.perf_counter()
    time_s = end_time - start_time
    # print("time_s:",time_s)
    if time_s > 60:
        conf.savePlot = True

    print("plotTitle", conf.plotTitle)

    if (len(conf.diffCellsL) == 1):
        conf.plotTitle += ", " + conf.diffCells + " test"

    plot_cases(conf)


"""
"""


def plotsns():
    namex = "Cells"
    namey = "Speedup"
    plot_title = "Test plotsns"

    ncol = 4
    # ncol=2
    if (ncol == 4):

        datay2 = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
        datax = [123, 346, 789]
        conf.legend = ["GPU Block-cells(1)",
                  "GPU Block-cells(2)",
                  "GPU Block-cells(3)",
                  "GPU Block-cells(4)"]
    else:
        datay2 = [[1, 2, 3], [4, 5, 6]]
        datax = [123, 346, 789]
        conf.legend = ["GPU Block-cells(1)",
                  "GPU Block-cells(2)"]

    # datay=map(list,)

    # datay=datay2
    datay = list(map(list, zip(*datay2)))  # short circuits at shortest nested list if table is jagged
    # numpy_array = np.array(datay2)
    # transpose = numpy_array.T
    # datay = transpose.tolist()

    print(datay)
    print(datax)

    # print(sns.__version__)
    sns.set_style("whitegrid")

    # sns.set(font_scale=2)
    # sns.set_context("paper", rc={"font.size":8,"axes.titlesize":8,"axes.labelsize":5})
    sns.set_context("paper", font_scale=1.25)

    # data = pd.DataFrame(datay, datax)
    data = pd.DataFrame(datay, datax, columns=legend)

    fig = plt.figure()
    ax = plt.subplot(111)

    ax.set_xlabel(namex)
    ax.set_ylabel(namey)
    # ax.set_title(plot_title)

    legend = True
    if (legend == True):

        print("WARNING: Increase plot window manually to take better screenshot")

        sns.lineplot(data=data, palette="tab10", linewidth=2.5)

        # ax.set_position([box.x0, box.y0 + box.height * 0.1,
        #               box.width, box.height * 0.9])

        # Legend under the plot
        # box = ax.get_position()
        # ax.set_position([box.x0, box.y0 + box.height * 0.1,
        #             box.width, box.height * 0.75])
        # ax.legend(bbox_to_anchor=(0.5, -0.05), loc='upper center',
        #          labels=legend,ncol=4, mode="expand", borderaxespad=0.)
        # fig.subplots_adjust(bottom=0.35)
        # borderaxespad=1. to move down more the legend

        # Legend up the plot (problem: hide title)
        ax.set_title(plot_title, y=1.06)

        ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1),
                  ncol=len(legend), labels=legend, frameon=True, shadow=False, borderaxespad=0.)

        # ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1),
        #          ncol=len(legend), labels=legend,frameon=False, shadow=False, borderaxespad=0.)#fine

        # ax.subplots_adjust(top=0.25) #not work
        # fig.subplots_adjust(top=0.25)

        # legend out of the plot at the right (problem: plot feels very small)
        # sns.lineplot(data=data, palette="tab10", linewidth=2.5)
        # box=ax.get_position()
        # ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])
        # ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0,labels=legend)

    else:
        ax.set_title(plot_title)
        sns.lineplot(data=data, palette="tab10", linewidth=2.5, legend=False)
    plt.show()


# rs = np.random.RandomState(365)
# values = rs.randn(365, 4).cumsum(axis=0)
# dates = pd.date_range("1 1 2016", periods=365, freq="D")
# data = pd.DataFrame(values, dates, legend=["A", "B", "C", "D"])
# data = data.rolling(7).mean()

# plotsns()
all_timesteps()

# import plot_functions  # comment to save ~2s execution time
import math_functions
import os
import numpy as np
import json
import subprocess
from pandas import read_csv as pd_read_csv


# TODO; DEFINE OPTIONS
# TODO: Add option to save path for output files
# TODO: Move profile files to a new folder
class TestMonarch:
    def __init__(self):
        # Case configuration
        self.chemFile = "cb05_paperV2"
        # self.chemFile = "monarch_cb05"
        self.diffCells = ""
        self.timeSteps = 1
        self.timeStepsDt = 2
        self.case = []
        self.nCells = 1
        self.caseGpuCpu = ""
        self.caseMulticellsOnecell = ""
        self.mpiProcesses = 1
        # Cases configuration
        self.diffCellsL = ["Realistic"]
        # self.diffCellsL.append("Ideal")
        self.mpiProcessesCaseBase = 1
        self.mpiProcessesCaseOptimList = [1]
        self.cells = [100]
        self.profileCuda = ""
        # self.profileCuda = "ncu"
        # self.profileCuda = "nsys"
        self.profileExtrae = None
        self.profileValgrind = None
        self.caseBase = "CPU One-cell"
        self.plotYKey = "Speedup timeCVode"
        self.casesOptim = []
        self.is_import = False
        self.is_import_base = False
        self.is_out = True
        self.loads_gpu = [0]  # Percentage of computational load (cells) to GPU
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
            "pidof -x $(ps cax | grep forge)", shell=True
        )
        if ddt_pid:
            exec_str += "ddt --connect "
    except Exception:
        pass
    if conf.profileCuda == "nsys" and conf.caseGpuCpu == "GPU":
        if os.environ["SLURM_JOB_NUM_NODES"] != "1":
            raise Exception(
                "nsys option is for slurm salloc session."
                " It could work on multiple nodes but it needs to be implemented"
            )
        result = subprocess.run(
            ["module list"],
            shell=True,
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
        )
        output = result.stdout + result.stderr
        if "gcc" not in output:
            raise Exception(
                "Missing gcc module for nsys option. To run nsys you need to use GCC for compile and run dependencies"
            )
        exec_str += "/apps/ACC/NVIDIA-HPC-SDK/23.9/Linux_x86_64/23.9/profilers/Nsight_Systems/bin/nsys "
        pathNsight = "../../compile/profile "
        exec_str += (
            "profile -f true --trace=mpi,cuda,nvtx --mpi-impl=openmpi -o "
            + pathNsight
        )
        print(
            "Saving nsight file in ",
            os.path.abspath(os.getcwd()) + "/" + pathNsight,
        )
    if int(os.environ.get("SLURM_JOB_NUM_NODES", 0)) > 1:
        exec_str += "srun --cpu-bind=core -n " + str(conf.mpiProcesses) + " "
    else:
        exec_str += "mpirun -np " + str(conf.mpiProcesses) + " --bind-to core "
    if conf.profileCuda == "ncu" and conf.caseGpuCpu == "GPU":
        # gui: /apps/ACC/NVIDIA-HPC-SDK/24.3/Linux_x86_64/2024/profilers/Nsight_Compute/ncu-ui
        exec_str += "/apps/ACC/NVIDIA-HPC-SDK/23.9/Linux_x86_64/23.9/profilers/Nsight_Compute/ncu "
        pathNsight = "../../compile/profile"
        exec_str += (
            "--target-processes application-only --set full -f -o "
            + pathNsight
            + " "
        )
        print(
            "Saving nsight file in ",
            os.path.abspath(os.getcwd()) + "/" + pathNsight + ".ncu-rep",
        )
    if conf.profileExtrae is not None:
        exec_str += "./trace_f.sh "
    if conf.profileValgrind is not None:
        exec_str += "valgrind --tool=cachegrind "
    path_exec = "../../build/mock_monarch"
    exec_str += path_exec
    try:
        file1 = open(conf.campConf, "w")
        if conf.caseGpuCpu == "GPU":
            file1.write(str(conf.load_gpu) + "\n")
        else:
            file1.write("0\n")
        file1.close()
    except Exception as e:
        print("write_camp_config_file fails", e)
    print("exec_str:", exec_str)
    print(
        conf.diffCells,
        conf.caseGpuCpu,
        conf.caseMulticellsOnecell,
        "ncellsPerMPIProcess:",
        conf.nCells,
        "Cells to GPU:",
        str(conf.load_gpu) + "%",
    )
    conf_name = "settings/TestMonarch.json"
    with open(conf_name, "w", encoding="utf-8") as jsonFile:
        json.dump(conf.__dict__, jsonFile, indent=4, sort_keys=False)
    nCellsStr = str(conf.nCells)
    if conf.nCells >= 1000:
        nCellsStr = str(int(conf.nCells / 1000)) + "k"
    caseGpuCpuName = conf.caseGpuCpu + str(conf.mpiProcesses) + "cores"
    out = 0
    is_import = False
    data_path = "out/stats"
    if conf.caseGpuCpu == "GPU":
        data_path += str(conf.load_gpu)
    data_path += (
        caseGpuCpuName
        + nCellsStr
        + "cells"
        + str(conf.timeSteps)
        + "tsteps.csv"
    )
    print("data_path", data_path)
    data_path2 = "out/state"
    if conf.caseGpuCpu == "GPU":
        data_path2 += str(conf.load_gpu)
    data_path2 += (
        caseGpuCpuName
        + nCellsStr
        + "cells"
        + str(conf.timeSteps)
        + "tsteps.csv"
    )
    if conf.is_import and os.path.exists(data_path):
        nRows_csv = conf.timeSteps * conf.nCells * conf.mpiProcesses
        df = pd_read_csv(data_path, nrows=nRows_csv)
        data = df.to_dict("list")
        y_key_words = conf.plotYKey.split()
        y_key = y_key_words[-1]
        data = data[y_key]
        print(y_key + ":", data)
        if data:
            is_import = True
        if conf.is_out:
            if os.path.exists(data_path2):
                is_import = True
            else:
                is_import = False
    if not is_import:
        os.system(exec_str)
        os.rename("out/stats.csv", data_path)
        if conf.is_out:
            os.rename("out/state.csv", data_path2)
        nRows_csv = conf.timeSteps * conf.nCells * conf.mpiProcesses
        df = pd_read_csv(data_path, nrows=nRows_csv)
        data = df.to_dict("list")
        y_key_words = conf.plotYKey.split()
        y_key = y_key_words[-1]
        data = data[y_key]
        print(y_key + ":", data)
    if conf.is_out:
        if os.path.exists(data_path2):
            df = pd_read_csv(data_path2, header=None, names=["Column1"])
            out = df["Column1"].tolist()
    return data[0], out


# @profile
def run_cases(conf):
    # Base case
    save_is_import = conf.is_import
    if conf.is_import_base:
        conf.is_import = True
    conf.mpiProcesses = conf.mpiProcessesCaseBase
    if conf.nCellsProcesses % conf.mpiProcesses != 0:
        print(
            "WARNING: On base case conf.nCellsProcesses % "
            "conf.mpiProcesses != 0, nCellsProcesses, "
            "mpiProcesses",
            conf.nCellsProcesses,
            conf.mpiProcesses,
            "setting cells to",
            int(conf.nCellsProcesses / conf.mpiProcesses) * conf.mpiProcesses,
        )
    conf.nCells = int(conf.nCellsProcesses / conf.mpiProcesses)
    cases_words = conf.caseBase.split()
    conf.caseGpuCpu = cases_words[0]
    conf.caseMulticellsOnecell = cases_words[1]
    conf.case = conf.caseBase
    timeBase, valuesBase = run(conf)
    # OptimCases
    conf.is_import = save_is_import
    datacases = []
    for mpiProcessesCaseOptim in conf.mpiProcessesCaseOptimList:
        conf.mpiProcesses = mpiProcessesCaseOptim
        if conf.nCellsProcesses % conf.mpiProcesses != 0:
            print(
                "WARNING: On optim case conf.nCellsProcesses % "
                "conf.mpiProcesses != 0,nCellsProcesses, "
                "mpiProcesses",
                conf.nCellsProcesses,
                conf.mpiProcesses,
                "setting cells to",
                int(conf.nCellsProcesses / conf.mpiProcesses)
                * conf.mpiProcesses,
            )
        conf.nCells = int(conf.nCellsProcesses / conf.mpiProcesses)
        for caseOptim in conf.casesOptim:
            cases_words = caseOptim.split()
            conf.caseGpuCpu = cases_words[0]
            conf.caseMulticellsOnecell = cases_words[1]
            conf.case = caseOptim
            timeOptim, valuesOptim = run(conf)
            if conf.is_out:
                math_functions.check_NRMSE(
                    valuesBase, valuesOptim, conf.nCellsProcesses
                )
            datay = timeBase / timeOptim
            print("Speedup", datay)
            datacases.append(datay)
    return datacases


def run_cells(conf):
    data = []
    for i, item in enumerate(conf.cells):
        conf.nCellsProcesses = item
        data += run_cases(conf)
    return data


def run_loads_gpu(conf):
    data = []
    for i, item in enumerate(conf.loads_gpu):
        conf.load_gpu = item
        data += run_cells(conf)
    return data


def run_diffCells(conf):
    data = []
    for i, item in enumerate(conf.diffCellsL):
        conf.diffCells = item
        data += run_loads_gpu(conf)
    return data


def plot_cases(conf, datay):
    try:
        cases_words = conf.casesOptim[0].split()
    except Exception:
        raise Exception(
            "Missing 'conf.casesOptim'. Ensure you have a case enabled such as 'CPU' or 'GPU'"
        )
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
        for mpiProcessesCaseOptim in conf.mpiProcessesCaseOptimList:
            for caseOptim in conf.casesOptim:
                cases_words = caseOptim.split()
                conf.caseGpuCpu = cases_words[0]
                conf.caseMulticellsOnecell = cases_words[1]
                case_multicells_onecell_name = ""
                if (
                    conf.caseMulticellsOnecell.find("BDF") != -1
                    or conf.caseMulticellsOnecell.find("maxrregcount") != -1
                ):
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
    nGPUsOptim = [i if i < 5 else 4 for i in conf.mpiProcessesCaseOptimList]
    if not is_same_diff_cells and len(conf.diffCellsL) == 1:
        plotTitle += conf.diffCells + " test: "
    if is_same_arch_optim:
        if len(nGPUsOptim) > 1:
            plotTitle += ""
        else:
            if (
                conf.caseGpuCpu == "GPU"
                and len(nGPUsOptim) == 1
                and conf.mpiProcessesCaseOptimList[0] > 1
            ):
                plotTitle += str(int(nGPUsOptim[0])) + " GPUs "
                plotTitle += "and " + str(mpiProcessesCaseOptim) + " Cores "
            else:
                plotTitle += conf.caseGpuCpu + " "
    if len(nGPUsOptim) > 1:
        plotTitle += "GPU "
    if len(legend) == 1 or not legend or len(conf.diffCellsL) > 1:
        if len(conf.mpiProcessesCaseOptimList) > 1:
            legend_name += str(mpiProcessesCaseOptim) + " MPI "
        if len(conf.diffCellsL) > 1:
            plotTitle += "Implementations "
    else:
        plotTitle += "Implementations "
    cases_words = conf.caseBase.split()
    conf.caseGpuCpu = cases_words[0]
    plotTitle += "vs " + str(conf.mpiProcessesCaseBase) + " Cores CPU"
    namey = "Speedup"
    if len(conf.cells) > 1:
        namex = "Cells"
        datax = conf.cells
        plotTitle += ""
    elif len(nGPUsOptim) > 1:
        namex = "GPUs"
        datax = nGPUsOptim
        if len(conf.cells) > 1:
            plotTitle += ", Cells: " + str(conf.cells[0])
    elif len(conf.loads_gpu) > 1:
        namex = "Percentage of cells to GPU"
        datax = conf.loads_gpu
        plotTitle += "," + str(conf.cells[0]) + " Cells"
    else:
        plotTitle += (
            ", Cells: "
            + str(conf.cells[0])
            + " Load GPU: "
            + str(conf.loads_gpu[0])
        )
        datax = list(range(1, conf.timeSteps + 1, 1))
        namex = "Time-steps"
    print(namex + ":", datax[0], "to", datax[-1])
    if legend:
        print("plotTitle: ", plotTitle, " legend:", legend)
    else:
        print("plotTitle: ", plotTitle)
    print(namex, ":", datax)
    print(namey, ":", datay)
    # plot_functions.plotsns(namex, namey, datax, datay, plotTitle, legend)


def run_main(conf):
    if conf.profileCuda and os.environ.get("SLURM_JOB_NUM_NODES", 0) != "1":
        raise Exception(
            "CUDA profiling option is for slurm salloc session on Marenostrum 5."
        )
    if conf.is_out and conf.casesOptim:
        if (
            len(conf.mpiProcessesCaseOptimList) > 1
            or conf.mpiProcessesCaseBase != conf.mpiProcessesCaseOptimList[0]
        ):
            print(
                "WARNING: Disabled out error check because number of "
                "processes should be the same for calculate "
                "accuracy, only speedup can use different number"
            )
            conf.is_out = False
    for i, mpiProcesses in enumerate(conf.mpiProcessesCaseOptimList):
        for j, cellsProcesses in enumerate(conf.cells):
            nCells = int(cellsProcesses / mpiProcesses)
            if nCells == 0:
                print(
                    "WARNING: Configured less cells than MPI "
                    "processes, setting 1 cell per process"
                )
                conf.mpiProcessesCaseOptimList[i] = cellsProcesses

    datay = run_diffCells(conf)
    return datay

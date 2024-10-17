#
# Copyright (C) 2022 Barcelona Supercomputing Center and University of
# Illinois at Urbana-Champaign
# SPDX-License-Identifier: MIT
#

from mainMonarch import *


def run_testMonarch():
    conf = TestMonarch()
    conf.timeSteps = 10  # Minimum value of 1
    conf.loads_gpu = [100]  # e.g. 100: GPU-Only 1-99: CPU+GPU
    conf.load_balance = 0  # 0: Fixed, 1: Automatic in runtime
    conf.cells = [10]  # Minimum value of 1
    conf.mpiProcessesCaseBase = 1  # Minimum value of 1
    conf.caseBase = "CPU"  # CPU or GPU
    conf.mpiProcessesCaseOptimList = [1]  # Minimum value of 1
    conf.casesOptim = ["GPU"]  # CPU or GPU
    # conf.is_import = True # Import results for case Base and Optim
    # conf.is_import_base = True # Import results for case Base
    #conf.profileCuda = "ncu"  # ncu or nsys
    # conf.profileCuda = "nsys"# ncu or nsys
    # conf.profileExtrae = True # Enable Extrae profiling
    datay = run_main(conf)  # Run
    plot_cases(conf, datay)  # Print results


if __name__ == "__main__":
    """
    Runs the CPU and GPU versions of the atmospheric chemistry solver.

    The GPU version is validated through calculating the accuracy error 
    with respect to the CPU version. The GPU version can also run the CPU
    solver simultaneously to the GPU solver to accelerate execution. This
    depends on the computational load assigned to both resources.

    This script configures and executes the solver with the specified settings.
    Details on the implementation can be found in C. Guzman's PhD Thesis:
    'Porting of an Atmospheric Chemistry Solver to Parallel CPU-GPU Execution.'

    Parameters:
    - conf.timeSteps (int): Number of time steps to run.
    - conf.loads_gpu (list of int): Percentage of computational load assigned to the GPU.
      For example, 0 means CPU-only, 100 means GPU-only, and values between 1-99 mean
      a combination of CPU and GPU.
    - conf.load_balance (int): Flag to indicate if the load should be balanced at runtime.
      0 for fixed load, 1 for automatic load balancing.
    - conf.cells (list of int): Represents points in the map from the domain decomposition
      of an Earth Science Model, such as the atmosphere. Mathematically, it is a system
      of Ordinary Differential Equations (ODE) to solve. 
      Cells are divided between the MPI processes. 
      Example: 10 cells and 2 MPI processes results to 5 cells per MPI process.
    - conf.mpiProcessesCaseBase (int): Number of MPI processes for the base case.
    - conf.caseBase (str): Base case to run, either "CPU" or "GPU".
    - conf.mpiProcessesCaseOptimList (list of int): List of MPI processes for optimized cases.
    - conf.casesOptim (list of str): List of optimized cases to run, e.g., ["GPU"].
    - conf.is_import (bool): Flag to indicate if the configuration should be imported.
      Useful for debugging the python interface
    - conf.is_import_base (bool): Flag to indicate if the base configuration should be imported.
      Useful for developing a GPU or CPU version.
    - conf.profileCuda (str): Tool to profile CUDA, e.g., "ncu" or "nsys".   
    - conf.profileExtrae (bool): Flag to indicate if Extrae profiling should be enabled.
    """
    run_testMonarch()

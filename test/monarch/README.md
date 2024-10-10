CAMP GPU : Instructions for Marenostrum cluster
======

#TODO: One section: All options listed here in the README and some sections putting more context
#TODO: 
    # e.g. cells: 10000, mpi_processes: 2, gpu_load: 50
    # 5000->rank 0 ; 5000->rank 1
    # 2500->CPU rank 0; 2500->GPU rank 0; 2500->CPU rank 1; 2500->GPU rank 1

This file includes instructions to run the GPU version of the code. All the related files to the GPU code are in this directory.

*If you previously run another branch, run `compile.sh `.*

Run `compile.sh ` and `run.sh ` for developing the GPU test.

We recommend to modify the file `TestMonarch.py` for developing the GPU version. The test is prepared to run the CPU and GPU version and get the speedup and acurracy error between both versions. It includes multiple configuration variables to facilitate development and testing. 

# Profiling CPU

Run `python exampleProfileCPU.py` for an example of a CPU execution 1 day simulation.

# Profiling GPU

Run `python exampleProfileGPU.py` for an example of profiling
the GPU with Nvidia profilers.
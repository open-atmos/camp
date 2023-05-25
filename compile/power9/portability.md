**GPU Portability between different architectures**

Default configuration is for CTE-POWER architecture from Barcelona Supercomputing Center.
Running GPUs in a different architecture may produce an error in some functionalities.
There is a list of detected issues with different architecture:
 - Multi-GPUs should follow same architecture, which corresponds to each GPU connected to two CPUs (in our case, 20 cores per CPU, 2 CPU per node and 4 GPUs per node). Variables endDevice and startDevices should be modified correspondingly to follow a differente architecture.

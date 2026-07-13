**HOW TO RUN BOXMODEL:**

This is an example of how to run the box model with the cb05 configuration.

You can edit the configuration files at *config_examples/cb05-ideal*

cd boxmodel
sbatch submit_boxmodel_job config_examples/cb05-ideal

Run files at: 
cd /gpfs/scratch/bsc32/$BSC_USER/run

Output files:
cd /gpfs/scratch/bsc32/$BSC_USER/out

To manually run again:
cd /gpfs/scratch/bsc32/$BSC_USER/run
mpirun -n 1 $PATH_TO_CAMP_BUILD_BOXMODEL_EXEC config.json interface.json

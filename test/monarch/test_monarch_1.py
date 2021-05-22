import matplotlib as mpl
mpl.use('TkAgg')
import plot_cases

plot_cases.all_timesteps()

#deprecated
#plot_cases.mpi_scalability()
#plot_cases.speedup_cells("Mean")
#plot_cases.speedup_cells("Standard Deviation")
#plot_cases.error_timesteps()
#plot_cases.speedup_timesteps()
#plot_cases.speedup_timesteps_counterBCG()
#plot_cases.debug_no_plot()

import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import csv
import sys, getopt
import os
import numpy as np
from pylab import imread,subplot,imshow,show
import plot_functions

#exec_str="../../mock_monarch config_simple.json interface_simple.json out/simple"
#exec_str="../../mock_monarch config_simple.json interface_simple.json simple"

#config_file="simple"
#config_file="monarch_cb05"
config_file="monarch_binned"

#mpi="yes"
mpi="no"

mpi_threads = 1

exec_str=""
if mpi=="yes":
  exec_str+="mpirun -v -np "+str(mpi_threads)+" --bind-to none "

exec_str+="../../mock_monarch config_"+config_file+".json "+"interface_"+config_file \
  +".json "+config_file

ADD_EMISIONS="OFF"
if config_file=="monarch_binned":
  ADD_EMISIONS="ON"

exec_str+=" "+ADD_EMISIONS

#Read file

#cells = [100,1000]
cells = [2]
#cells = [int(cell/mpi_threads) for cell in cells] #in case divide load between threads
cells = [str(cell) for cell in cells]
#cases_multicells_onecell = ["one-cell","multi-cells"]
#cases_multicells_onecell = ["one-cell"]
cases_multicells_onecell = ["multi-cells"]

#SELECT MANUALLY (future:if arch=cpu then select cpu if not gpu)
cases_gpu_cpu = ["cpu"]
#cases_gpu_cpu = ["gpu"]

plot_x_key="timestep"

#todo fix timeCVode to MPI counters to avoid 0 division
#plot_y_key="timeCVode"
#plot_y_key="timeLS"
plot_y_key="counterLS"

data = {}
#data_list = []

# make the output directory if it doesn't exist
if not os.path.exists('out'):
  os.makedirs('out')

for case in cases_multicells_onecell:

  data_tmp = {}

  file = 'out/'+config_file+'_'+case+'_solver_stats.csv'

  for cell in cells:

    print (exec_str + " " + cell + " " + case)
    os.system(exec_str + " " + cell + " " + case)

    plot_functions.read_solver_stats(file, data_tmp)

  #plot_functions.plot_solver_stats(data_tmp)
  data[case]=data_tmp
  #data_list.append(data_tmp)

#print(data)

if (len(cases_multicells_onecell) == 2):

  base_case_data=data[cases_multicells_onecell[0]][plot_y_key]
  new_case_data=data[cases_multicells_onecell[1]][plot_y_key]

  data,plot_y_key=plot_functions.calculate_speedup( \
    base_case_data,new_case_data,data_tmp,plot_y_key)


plot_title = config_file + ", cells: " + str(cells[0])

plot_functions.plot_solver_stats(data, plot_x_key, plot_y_key, plot_title)

"""
  #not working for cases>1
  with open(cases_gpu_cpu[0]+"_"+case+".csv", 'w') as file:
    writer = csv.writer(file, delimiter=' ')

    keys=[]
    for key in data.keys():
      keys.append(key)
    writer.writerow(keys)
    for value in data.values():
      writer.writerow(value)
"""

#plot_functions.plot_species("out/"+config_file+"_urban_plume_0001.txt")



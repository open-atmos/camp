import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import csv
import sys, getopt
import os
import numpy as np
from pylab import imread,subplot,imshow,show
import plot_functions

def write_config_file(case_gpu_cpu):
  file1 = open("config_variables_c_solver.txt","w")

  #print(case_gpu_cpu)

  if(case_gpu_cpu=="CPU"):
    file1.write("USE_CPU=ON\n")
  else:
    file1.write("USE_CPU=OFF\n")

  file1.close()

def mpi_scalability():

  #config_file="simple"
  #config_file="monarch_cb05"
  config_file="monarch_binned"

  mpi="yes"
  #mpi="no"

  mpiProcessesList = [1,16,40]

  #Read file

  #cells = [100,1000]
  cells = [1000]
  divide_cells_load=True

  #cases_multicells_onecell = ["one-cell","multi-cells"]
  #cases_multicells_onecell = ["one-cell"]
  cases_multicells_onecell = ["multi-cells"]

  plot_x_key = "mpiProcesses"
  timestep_to_plot = 0

  plot_y_key = "timeCVode"
  #plot_y_key = "timeLS"
  #plot_y_key = "counterLS"

  data = {}

  # make the output directory if it doesn't exist
  if not os.path.exists('out'):
    os.makedirs('out')

  plot_title = config_file + ", cells: " + str(cells[0]) \
                +"/"+plot_x_key
               #+ ", divide_cells_load:" + str(divide_cells_load)

  #print(plot_title)

  for case in cases_multicells_onecell:

    data_tmp = {}

    file = 'out/'+config_file+'_'+case+'_solver_stats.csv'

    cells_init=cells

    for mpiProcesses in mpiProcessesList:

      mpiProcesses_str=str(mpiProcesses)

      exec_str=""
      if mpi=="yes":
        exec_str+="mpirun -v -np "+mpiProcesses_str+" --bind-to none "

      exec_str+="../../mock_monarch config_"+config_file+".json "+"interface_"+config_file \
                +".json "+config_file

      ADD_EMISIONS="OFF"
      if config_file=="monarch_binned":
        ADD_EMISIONS="ON"

      exec_str+=" "+ADD_EMISIONS

      #todo improve file by sending to program and create many folders as \
      # cases to store results and avoid execution all time

      if divide_cells_load==True:
        cells = [int(cell/mpiProcesses) for cell in cells_init] #in case divide load between threads

      for cell in cells:

        cell_str=str(cell)
        print(exec_str + " " + cell_str + " " + case)
        os.system(exec_str + " " + cell_str + " " + case)

        plot_functions.read_solver_stats(file, data_tmp)

      data=data_tmp

    #print(data)

    data = plot_functions.get_values_same_timestep(timestep_to_plot,mpiProcessesList, \
                                                   data,plot_x_key,plot_y_key)

    print(data)

    #todo plot values
    plot_functions.plot_solver_stats_mpi(data, plot_x_key, plot_y_key, plot_title)

def speedup_cells(metric):

  #config_file="simple"
  #config_file="monarch_cb05"
  config_file="monarch_binned"

  mpi="yes"
  #mpi="no"

  divide_cells_load=False

  mpiProcessesList = [1]

  #cells = [100,1000]
  cells = [1,10]

  cases_multicells_onecell = ["one-cell","multi-cells"]
  #cases_multicells_onecell = ["one-cell"]
  #cases_multicells_onecell = ["multi-cells"]

  #plot_x_key = "timestep"
  plot_x_key = "Cells"

  #plot_y_key = "timeCVode"
  plot_y_key = "timeLS"
  #plot_y_key = "counterLS"

  data = {}

  # make the output directory if it doesn't exist
  if not os.path.exists('out'):
    os.makedirs('out')

  plot_title = config_file + ", Timesteps: 0-720"
  #plot_title = config_file + ", Timesteps: 720-1400"

  cells_init = cells
  plot_y_key_init = plot_y_key

  for mpiProcesses in mpiProcessesList:

    exec_str=""
    if mpi=="yes":
      exec_str+="mpirun -v -np "+str(mpiProcesses)+" --bind-to none "

    exec_str+="../../mock_monarch config_"+config_file+".json "+"interface_"+config_file \
              +".json "+config_file

    ADD_EMISIONS="OFF"
    if config_file=="monarch_binned":
      ADD_EMISIONS="ON"

    exec_str+=" "+ADD_EMISIONS

    if divide_cells_load==True:
      cells = [int(cell/mpiProcesses) for cell in cells_init] #in case divide load between threads

    for cell in cells:

      cell_str=str(cell)

      data[cell] = {}

      for case in cases_multicells_onecell:

        data[cell][case] = {}

        file = 'out/'+config_file+'_'+case+'_solver_stats.csv'

        print(exec_str + " " + cell_str + " " + case)
        os.system(exec_str + " " + cell_str + " " + case)

        plot_functions.read_solver_stats(file, data[cell][case])

      #print(data)

      if (len(cases_multicells_onecell) == 2):

        if(plot_y_key_init=="timeLS"):
          data[cell], plot_y_key=plot_functions.normalized_timeLS( \
            cases_multicells_onecell,data[cell],cell)

        data[cell],plot_y_key2=plot_functions.calculate_speedup( \
          cases_multicells_onecell,data[cell],"timestep", \
          plot_y_key)

      #data,plot_y_key3 = plot_functions.calculate_std_cell( \
      #  cell,data,plot_x_key, \
      #  plot_y_key2)

      if(metric=="Mean"):
        data,plot_y_key3 = plot_functions.calculate_mean_cell( \
          cell,data,plot_x_key, \
          plot_y_key2)
      elif(metric=="Standard Deviation"):
        data,plot_y_key3 = plot_functions.calculate_std_cell( \
            cell,data,plot_x_key, \
            plot_y_key2)

    #print(data)

    plot_functions.plot_speedup_cells(cells,data[plot_y_key3],plot_x_key, \
                                    plot_y_key3, plot_title)

def rmse_timesteps():

  #config_file="simple"
  #config_file="monarch_cb05"
  config_file="monarch_binned"

  mpi="yes"
  #mpi="no"

  mpiProcessesList = [1]

  #cells = [100,1000]
  cells = [10]
  cell = cells[0]

  timesteps = 60#720=12h
  TIME_STEP = 2

  cases_multicells_onecell = ["one-cell","multi-cells"]
  #cases_multicells_onecell = ["one-cell"]
  #cases_multicells_onecell = ["multi-cells"]

  cases_gpu_cpu = ["CPU"]
  #cases_gpu_cpu = ["GPU"]
  case_gpu_cpu = cases_gpu_cpu[0]

  data = {}

  # make the output directory if it doesn't exist
  if not os.path.exists('out'):
    os.makedirs('out')

  write_config_file(case_gpu_cpu)

  for mpiProcesses in mpiProcessesList:

    exec_str=""
    if mpi=="yes":
      exec_str+="mpirun -v -np "+str(mpiProcesses)+" --bind-to none "

    exec_str+="../../mock_monarch config_"+config_file+".json "+"interface_"+config_file \
              +".json "+config_file

    ADD_EMISIONS="OFF"
    if config_file=="monarch_binned":
      ADD_EMISIONS="ON"

    exec_str+=" "+ADD_EMISIONS

    cell_str=str(cell)

    for case in cases_multicells_onecell:

      print(exec_str +" " + cell_str + " " + case + " " + str(timesteps))
      os.system(exec_str +" " + cell_str + " " + case + " " + str(timesteps))

      data[case] = {}

      #file = 'out/'+config_file+'_'+case+'_solver_stats.csv'
      file = 'out/'+config_file+'_'+case+'_results_all_cells.csv'
      plot_functions.read_solver_stats(file, data[case])

    #print(data)

    if (len(cases_multicells_onecell) == 2):

      #rel_tol 1.d-5, abs_tol 1d-4
      #rel_tol=1.0E-2
      #abs_tol=1.0E-4
      #plot_functions.check_tolerances(data,timesteps,rel_tol,abs_tol)
      NRMSEs=plot_functions.calculate_NMRSE(data,timesteps)

      #data[cell],plot_y_key2=plot_functions.calculate_speedup( \
      #  cases_multicells_onecell,data[cell],"timestep", \
      #  plot_y_key)

    #print(data)

      namex = "Timesteps"
      namey="NRMSE"
      #datax=list(range(TIME_STEP,timesteps*TIME_STEP,TIME_STEP))
      datax=list(range(TIME_STEP,TIME_STEP*(timesteps+1),TIME_STEP))
      datay=NRMSEs
      plot_title="Ideal "+config_file+" "+case_gpu_cpu+", Cells: "+cell_str
      plot_title="Practical "+config_file+" "+case_gpu_cpu+", Cells: "+cell_str
      #plot_title = config_file + ", Timesteps: 720-1400"

      plot_functions.plot(namex,namey,datax,datay,plot_title)


def speedup_timesteps():

  #config_file="simple"
  #config_file="monarch_cb05"
  config_file="monarch_binned"

  mpi="yes"
  #mpi="no"

  divide_cells_load=False

  mpiProcessesList = [1]

  #cells = [100,1000]
  cells = [10]

  cases_multicells_onecell = ["one-cell","multi-cells"]
  #cases_multicells_onecell = ["one-cell"]
  #cases_multicells_onecell = ["multi-cells"]

  #SELECT MANUALLY (future:if arch=cpu then select cpu if not gpu)
  cases_gpu_cpu = ["CPU"]
  #cases_gpu_cpu = ["GPU"]
  #cases_gpu_cpu = ["CPU","GPU"]

  plot_x_key = "timestep"

  #plot_y_key = "timeCVode"
  #plot_y_key = "timeLS"
  plot_y_key = "counterLS"

  remove_iters=0#10 #360

  data = {}

  # make the output directory if it doesn't exist
  if not os.path.exists('out'):
    os.makedirs('out')

  #plot_title = config_file + ", cells: " + str(cells[0])
  #plot_title = config_file + ", cells: " + str(cells[0]) + " Diff cells: temp, press and emissions"
  plot_title = config_file + ", cells: " + str(cells[0]) + " Diff cells: temp and press"
  #plot_title = config_file + ", cells: " + str(cells[0]) + ", Timesteps: 0-72"
  #plot_title = config_file + ", cells: " + str(cells[0]) + ", Timesteps: 720-792"

  cells_init = cells

  for mpiProcesses in mpiProcessesList:

    exec_str=""
    if mpi=="yes":
      exec_str+="mpirun -v -np "+str(mpiProcesses)+" --bind-to none "
      #exec_str+="srun -n "+str(mpiProcesses)+" "

    exec_str+="../../mock_monarch config_"+config_file+".json "+"interface_"+config_file \
              +".json "+config_file

    ADD_EMISIONS="OFF"
    if config_file=="monarch_binned":
      ADD_EMISIONS="ON"

    exec_str+=" "+ADD_EMISIONS

    if divide_cells_load==True:
      cells = [int(cell/mpiProcesses) for cell in cells_init] #in case divide load between threads

    for case in cases_multicells_onecell:

      data_tmp = {}

      file = 'out/'+config_file+'_'+case+'_solver_stats.csv'

      for cell in cells:

        cell_str=str(cell)
        print(exec_str + " " + cell_str + " " + case)
        os.system(exec_str + " " + cell_str + " " + case)

        plot_functions.read_solver_stats(file, data_tmp)

      data[case]=data_tmp

    #print(data)

    if (len(cases_multicells_onecell) == 2):

      if(plot_y_key=="timeLS2"):
        data, plot_y_key=plot_functions.normalized_timeLS( \
          cases_multicells_onecell,data, cells[0])

      data,plot_y_key2=plot_functions.calculate_speedup( \
        cases_multicells_onecell,data,plot_x_key, \
        plot_y_key)

      #print(data[plot_x_key])

      for i in range(remove_iters):
        data[plot_x_key].pop(0)
        data[plot_y_key2].pop(0)
        #print (data[plot_x_key].pop(0))
        #print (data[plot_y_key2].pop(0))

      #print(data[plot_x_key])

    else:
      data = data_tmp
      plot_y_key2=plot_y_key

    #print(data)

    plot_functions.plot_solver_stats(data,plot_x_key, plot_y_key2, plot_title)

def speedup_timesteps_counterBCG():

  #config_file="simple"
  #config_file="monarch_cb05"
  config_file="monarch_binned"

  mpi="yes"
  #mpi="no"

  divide_cells_load=False

  mpiProcessesList = [1]

  #cells = [100,1000]
  cells = [5]

  cases_multicells_onecell = ["one-cell","multi-cells"]
  #cases_multicells_onecell = ["one-cell"]
  #cases_multicells_onecell = ["multi-cells"]

  #cases_gpu_cpu = ["CPU"]
  cases_gpu_cpu = ["GPU"]
  #cases_gpu_cpu = ["CPU","GPU"]

  plot_x_key = "timestep"

  #plot_y_key = "timeCVode"
  #plot_y_key = "timeLS"
  #plot_y_key = "counterLS"
  plot_y_key = "counterBCG"

  remove_iters=0#10 #360

  data = {}

  # make the output directory if it doesn't exist
  if not os.path.exists('out'):
    os.makedirs('out')

  #plot_title = config_file + ", cells: " + str(cells[0])
  #plot_title = config_file + ", cells: " + str(cells[0]) + " Diff cells: temp, press and emissions"
  #plot_title = config_file + ", cells: " + str(cells[0]) + " Diff cells: temp and press"
  plot_title = config_file + ", cells: " + str(cells[0]) + " Ideal case"
  #plot_title = config_file + ", cells: " + str(cells[0]) + ", Timesteps: 0-72"
  #plot_title = config_file + ", cells: " + str(cells[0]) + ", Timesteps: 720-792"

  cells_init = cells

  for case_gpu_cpu in cases_gpu_cpu:

    write_config_file(case_gpu_cpu)

    mpiProcesses = mpiProcessesList[0]

    exec_str=""
    if mpi=="yes":
      exec_str+="mpirun -v -np "+str(mpiProcesses)+" --bind-to none "
      #exec_str+="srun -n "+str(mpiProcesses)+" "

    exec_str+="../../mock_monarch config_"+config_file+".json "+"interface_"+config_file \
              +".json "+config_file

    ADD_EMISIONS="OFF"
    if config_file=="monarch_binned":
      ADD_EMISIONS="ON"

    exec_str+=" "+ADD_EMISIONS

    if divide_cells_load==True:
      cells = [int(cell/mpiProcesses) for cell in cells_init] #in case divide load between threads

    for case in cases_multicells_onecell:

      data_tmp = {}

      file = 'out/'+config_file+'_'+case+'_solver_stats.csv'

      for cell in cells:

        cell_str=str(cell)
        print(exec_str + " " + cell_str + " " + case)
        os.system(exec_str + " " + cell_str + " " + case)

        plot_functions.read_solver_stats(file, data_tmp)

        if(plot_y_key=="counterBCG2"):
          for i in range(len(data_tmp[plot_y_key])):
            #print(data_tmp[plot_y_key][i],data_tmp["counterLS"][i])
            data_tmp[plot_y_key][i]=data_tmp[plot_y_key][i]/\
                                    data_tmp["counterLS"][i]

      data[case]=data_tmp

    #print(data)

    if (len(cases_multicells_onecell) == 2):

      data,plot_y_key2=plot_functions.calculate_speedup( \
      cases_multicells_onecell,data,plot_x_key, \
      plot_y_key)

      #print(data[plot_x_key])

      for i in range(remove_iters):
        data[plot_x_key].pop(0)
        data[plot_y_key2].pop(0)
        #print (data[plot_x_key].pop(0))
        #print (data[plot_y_key2].pop(0))

      #print(data[plot_x_key])

    else:
      data = data_tmp
      plot_y_key2=plot_y_key

    #print(data)

    plot_functions.plot_solver_stats(data,plot_x_key, plot_y_key2, plot_title)


def debug_no_plot():

  #config_file="simple"
  #config_file="monarch_cb05"
  config_file="monarch_binned"

  mpi="yes"
  #mpi="no"

  mpiProcessesList = [1]

  #Read file

  #cells = [100,1000]
  cells = [10]

  timesteps = 2

  #cases_multicells_onecell = ["one-cell","multi-cells"]
  cases_multicells_onecell = ["one-cell"]
  #cases_multicells_onecell = ["multi-cells"]

  #SELECT MANUALLY (future:if arch=cpu then select cpu if not gpu)
  cases_gpu_cpu = ["CPU"]
  #cases_gpu_cpu = ["GPU"]
  case_gpu_cpu = cases_gpu_cpu[0]

  # make the output directory if it doesn't exist
  if not os.path.exists('out'):
    os.makedirs('out')

  plot_title = config_file + ", cells: " + str(cells[0])
  #plot_title = config_file + ", cells: " + str(cells[0]) + ", Timesteps: 0-72"
  #plot_title = config_file + ", cells: " + str(cells[0]) + ", Timesteps: 720-792"

  cells_init = cells

  write_config_file(case_gpu_cpu)

  for mpiProcesses in mpiProcessesList:

    exec_str=""
    if mpi=="yes":
      exec_str+="mpirun -v -np "+str(mpiProcesses)+" --bind-to none "
      #exec_str+="srun -n "+str(mpiProcesses)+" "

    exec_str+="../../mock_monarch config_"+config_file+".json "+"interface_"+config_file \
              +".json "+config_file

    ADD_EMISIONS="OFF"
    if config_file=="monarch_binned":
      ADD_EMISIONS="ON"

    exec_str+=" "+ADD_EMISIONS

    for case in cases_multicells_onecell:

      for cell in cells:

        cell_str=str(cell)
        print(exec_str +" " + cell_str + " " + case + " " + str(timesteps))
        os.system(exec_str +" " + cell_str + " " + case + " " + str(timesteps))




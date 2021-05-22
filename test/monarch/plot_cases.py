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

def run(config_file,diff_cells,mpi,mpiProcessesList,n_cells,timesteps,
        case_gpu_cpu,case,results_file):

  data={}

  #for case_gpu_cpu in cases_gpu_cpu:

  #MPI
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

  #if divide_cells_load==True:
  #  cells = [int(cell/mpiProcesses) for cell in cells_init] #in case divide load between threads

  #GPU-CPU
  write_config_file(case_gpu_cpu)

  #for case in cases_multicells_onecell:

  #Onecell-Multicells

  #plot_title+=case_gpu_cpu+" "+case

  file = 'out/'+config_file+'_'+case+results_file

  print(exec_str +" " + str(n_cells) + " " + case + " " + str(timesteps)+" "+diff_cells)
  os.system(exec_str +" " + str(n_cells) + " " + case + " " + str(timesteps)+" "+diff_cells)

  plot_functions.read_solver_stats(file, data)

  return data

def run_cell(config_file,diff_cells,mpi,mpiProcessesList,n_cells,timesteps,
             cases,cases_gpu_cpu,cases_multicells_onecell,results_file,plot_y_key):

  data={}
  for i in range(len(cases)):
    data[cases[i]] = run(config_file,diff_cells,mpi,mpiProcessesList,
                         n_cells,timesteps,cases_gpu_cpu[i],cases_multicells_onecell[i],results_file)

  if(len(cases)!=2):
    raise Exception("Only one case to compare, check cases")

  if("normalized timeLS" in plot_y_key):
    case_default=cases_multicells_onecell[0]
    data=plot_functions.normalize_timeLS( \
      data,"normalized timeLS",n_cells,case_default)

  #print(data)

  datay=[]
  if(plot_y_key=="NRMSE"):
    datay=plot_functions.calculate_NMRSE(data,timesteps)
  elif(plot_y_key=="MAPE"):
    datay=plot_functions.calculate_MAPE(data,timesteps)
  elif(plot_y_key=="SMAPE"):
    datay=plot_functions.calculate_SMAPE(data,timesteps)
  elif("Speedup" in plot_y_key):
    #y_key = plot_y_key.replace('Speedup ', '')
    y_key_words = plot_y_key.split()
    y_key = y_key_words[-1]
    #print(y_key)
    datay=plot_functions.calculate_speedup2(data,y_key)
  elif(plot_y_key=="BCG % Time data transfers CPU-GPU"):
    datay=plot_functions.calculate_BCGPercTimeDataTransfers(data)
  else:
    raise Exception("Not found plot function for plot_y_key")

  return datay

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

def all_timesteps():

  #config_file="simple"
  #config_file="monarch_cb05"
  config_file="monarch_binned"

  diff_cells="Practical"
  #diff_cells="Ideal"

  mpi="yes"
  #mpi="no"

  divide_cells_load=False

  mpiProcessesList = [1]

  cells = [10]
  #cells = [1,10,100,1000,5000,10000]

  timesteps = 5#720=12h
  TIME_STEP = 2 #pending send TIME_STEP to mock_monarch

  #cases = ["CPU one-cell","CPU multi-cells"]
  cases = ["CPU multi-cells","GPU multi-cells"]

  #plot_y_key = "counterBCG"
  #plot_y_key = "Average BCG internal iterations per call"
  #plot_y_key = "Average BCG time per call" #This metric makes no sense, one-cell would always be faster because is computing way less cells
  #plot_y_key = "Speedup normalized timeLS"

  #plot_y_key = "Speedup timeCVode"
  #plot_y_key = "Speedup counterLS"
  plot_y_key = "Speedup normalized timeLS"

  plot_y_key = "BCG % Time data transfers CPU-GPU"
  #plot_y_key="NRMSE"
  #plot_y_key="MAPE"
  #plot_y_key="SMAPE"

  remove_iters=0#10 #360

  if not os.path.exists('out'):
    os.makedirs('out')


  results_file="_solver_stats.csv"
  if(plot_y_key=="NRMSE" or plot_y_key=="MAPE" or plot_y_key=="SMAPE"):
    results_file='_results_all_cells.csv'

  cases_gpu_cpu=[""]*len(cases)
  cases_multicells_onecell=[""]*len(cases)
  for i in range(len(cases)):
    cases_words=cases[i].split()
    cases_gpu_cpu[i]=cases_words[0]
    cases_multicells_onecell[i]=cases_words[1]

  datay=[]
  for i in range(len(cells)):
    datay_cell = run_cell(config_file,diff_cells,mpi,mpiProcessesList,cells[i],timesteps,
                 cases,cases_gpu_cpu,cases_multicells_onecell,results_file,plot_y_key)

    #print(datay_cell)
    if(len(cells)>1):
      #Default metric
      datay.append(np.mean(datay_cell))
    else:
      datay=datay_cell

  print(datay)

  plot_title=""
  first_word=""
  second_word=""
  third_word=""
  if(cases_gpu_cpu[0]!=cases_gpu_cpu[1]):
    first_word+=cases_gpu_cpu[1] + " "
    second_word+=cases_gpu_cpu[0] + " "
  else:
    third_word+=cases_gpu_cpu[0] + " "
  if(cases_multicells_onecell[0]!=cases_multicells_onecell[1]):
    first_word+=cases_multicells_onecell[1] + " "
    second_word+=cases_multicells_onecell[0] + " "
  else:
    third_word+=cases_multicells_onecell[0] + " "

  plot_title+=first_word + "vs " + second_word + third_word
  plot_title+=diff_cells+" test "+"Group cells"
  if(len(cells)>1):
    plot_title+=", Timesteps:[0-"+str(TIME_STEP*timesteps)+"]"#str(timesteps)
    datax=cells
    plot_x_key="Cells"
    #Default metric
    plot_y_key="Mean "+plot_y_key
  else:
    plot_title+=", Cells: "+str(cells[0])
    datax=list(range(TIME_STEP,TIME_STEP*(timesteps+1),TIME_STEP))
    plot_x_key = "Timesteps"

  namey=plot_y_key
  namex=plot_x_key

  plot_functions.plot(namex,namey,datax,datay,plot_title)


def error_timesteps():

  #config_file="simple"
  #config_file="monarch_cb05"
  config_file="monarch_binned"

  diff_cells="Practical"
  #diff_cells="Ideal"

  #plot_y_key="NRMSE"
  plot_y_key="MAPE"
  #plot_y_key="SMAPE"

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

  #cases_gpu_cpu = ["CPU"]
  #cases_gpu_cpu = ["GPU"]
  cases_gpu_cpu = ["CPU","GPU"]
  case_gpu_cpu = cases_gpu_cpu[0]
  write_config_file(case_gpu_cpu)

  data = {}

  # make the output directory if it doesn't exist
  if not os.path.exists('out'):
    os.makedirs('out')

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

      print(exec_str +" " + cell_str + " " + case + " " + str(timesteps)+" "+diff_cells)
      os.system(exec_str +" " + cell_str + " " + case + " " + str(timesteps)+" "+diff_cells)

      data[case] = {}

      #file = 'out/'+config_file+'_'+case+'_solver_stats.csv'
      file = 'out/'+config_file+'_'+case+'_results_all_cells.csv'
      plot_functions.read_solver_stats(file, data[case])

      if(len(cases_gpu_cpu)==2 and len(cases_multicells_onecell)==2):
        case_gpu_cpu = cases_gpu_cpu[1]
        write_config_file(case_gpu_cpu)

    #print(data)

    if (len(cases_multicells_onecell) == 2):

      errs=[]

      if(plot_y_key=="NRMSE"):
        errs=plot_functions.calculate_NMRSE(data,timesteps)
      if(plot_y_key=="MAPE"):
        errs=plot_functions.calculate_MAPE(data,timesteps)
      if(plot_y_key=="SMAPE"):
        errs=plot_functions.calculate_SMAPE(data,timesteps)

      #data[cell],plot_y_key2=plot_functions.calculate_speedup( \
      #  cases_multicells_onecell,data[cell],"timestep", \
      #  plot_y_key)

    #print(data)

      namex = "Timesteps"
      namey=plot_y_key
      #datax=list(range(TIME_STEP,timesteps*TIME_STEP,TIME_STEP))
      datax=list(range(TIME_STEP,TIME_STEP*(timesteps+1),TIME_STEP))
      datay=errs
      #plot_title="Ideal "+config_file+" "+case_gpu_cpu+", Cells: "+cell_str
      plot_title="Practical "+"test "+cases_gpu_cpu[0]+ \
                 " "+ cases_multicells_onecell[1] +" vs "+ cases_gpu_cpu[0] + \
                 " "+ cases_multicells_onecell[0] +", Cells: "+cell_str
      #plot_title = config_file + ", Timesteps: 720-1400"

      if(len(cases_gpu_cpu)==2 and len(cases_multicells_onecell)==2):
        plot_title="Practical "+"test "+cases_gpu_cpu[1]+\
                   " "+ cases_multicells_onecell[1] +" vs "+ cases_gpu_cpu[0] +\
                   " "+ cases_multicells_onecell[0] +", Cells: "+cell_str


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

      if(plot_y_key=="Normalized timeLS"):
        data, plot_y_key=plot_functions.normalized_timeLS( \
          plot_y_key,cases_multicells_onecell,data, cells[0])

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

  diff_cells="Practical"
  #diff_cells="Ideal"

  mpi="yes"
  #mpi="no"

  divide_cells_load=False

  mpiProcessesList = [1]

  #cells = [100,1000]
  cells = [10]
  cell_str=str(cells[0])

  timesteps = 5#720=12h
  TIME_STEP = 2

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
  #plot_y_key = "counterBCG"
  #plot_y_key = "Average BCG internal iterations per call"
  #plot_y_key = "Average BCG time per call" #This metric makes no sense, one-cell would always be faster because is computing way less cells
  plot_y_key = "Normalized timeLS"

  remove_iters=0#10 #360

  data = {}

  # make the output directory if it doesn't exist
  if not os.path.exists('out'):
    os.makedirs('out')


  #plot_title = config_file + ", cells: " + str(cells[0])
  #plot_title = config_file + ", cells: " + str(cells[0]) + " Diff cells: temp, press and emissions"
  #plot_title = config_file + ", cells: " + str(cells[0]) + " Diff cells: temp and press"
  #plot_title = config_file + ", cells: " + str(cells[0]) + " Ideal case"
  #plot_title = config_file + ", cells: " + str(cells[0]) + ", Timesteps: 0-72"
  #plot_title = config_file + ", cells: " + str(cells[0]) + ", Timesteps: 720-792"
  #plot_title="Practical "+"test "+"Ind. cells "
  plot_title="Practical "+"test "+"Group cells "

  #plot_title+=cases_gpu_cpu[0]+ \
  #             " "+ cases_multicells_onecell[0] +" vs "+ cases_gpu_cpu[0] + \
  #             " "+ cases_multicells_onecell[0] +", Cells: "+cell_str

  cells_init = cells

  its=0

  #print(len(cases_multicells_onecell), len(cases_gpu_cpu))
  if (len(cases_multicells_onecell) == 1 and len(cases_gpu_cpu) == 1):
    print("Only one case to print, check cases_multicells_onecell and cases_gpu_cpu")
    exit()

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

      plot_title+=case_gpu_cpu+" "+case
      if(its==0):
        plot_title+=" vs "
        its+=1

      data_tmp = {}

      file = 'out/'+config_file+'_'+case+'_solver_stats.csv'

      print(exec_str +" " + cell_str + " " + case + " " + str(timesteps)+" "+diff_cells)
      os.system(exec_str +" " + cell_str + " " + case + " " + str(timesteps)+" "+diff_cells)

      plot_functions.read_solver_stats(file, data_tmp)

      if(plot_y_key=="Average BCG internal iterations per call"):
        data_tmp[plot_y_key]=[0.]*len(data_tmp["counterBCG"])
        for i in range(len(data_tmp["counterBCG"])):
          #print(data_tmp["counterBCG"][i],data_tmp["counterLS"][i])
          data_tmp[plot_y_key][i]=data_tmp["counterBCG"][i]/\
                                  data_tmp["counterLS"][i]
      if(plot_y_key=="Average BCG time per call"):
        data_tmp[plot_y_key]=[0.]*len(data_tmp["timeLS"])
        for i in range(len(data_tmp["timeLS"])):
          #print(data_tmp["counterBCG"][i],data_tmp["counterLS"][i])
          data_tmp[plot_y_key][i]=data_tmp["timeLS"][i]/ \
                                data_tmp["counterLS"][i]

      #             " "+ cases_multicells_onecell[0] +" vs "+ cases_gpu_cpu[0] + \
      #             " "+ cases_multicells_onecell[0] +", Cells: "+cell_str

      data[case]=data_tmp

    plot_title+=", Cells: "+cell_str

    #print(data)

  if(plot_y_key=="Normalized timeLS"):
      data, plot_y_key=plot_functions.normalized_timeLS( \
        plot_y_key,cases_multicells_onecell,data, cells[0])

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

  #print(data)

  plot_functions.plot_solver_stats(data,plot_x_key, plot_y_key2, plot_title)




def debug_no_plot():

  #config_file="simple"
  #config_file="monarch_cb05"
  config_file="monarch_binned"

  diff_cells="Practical"
  #diff_cells="Ideal"

  mpi="yes"
  #mpi="no"

  mpiProcessesList = [1]

  #Read file

  #cells = [100,1000]
  cells = [10]

  timesteps = 2

  #cases_multicells_onecell = ["one-cell","multi-cells"]
  #cases_multicells_onecell = ["one-cell"]
  cases_multicells_onecell = ["multi-cells"]

  #SELECT MANUALLY (future:if arch=cpu then select cpu if not gpu)
  #cases_gpu_cpu = ["CPU"]
  cases_gpu_cpu = ["GPU"]
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
        print(exec_str +" " + cell_str + " " + case + " " + str(timesteps)+" "+diff_cells)
        os.system(exec_str +" " + cell_str + " " + case + " " + str(timesteps)+" "+diff_cells)



"""

  if(len(cases_multicells_onecell)==2 and len(cases_gpu_cpu)==2): #Something to compare

    plot_title+=cases_gpu_cpu[1] + " " + cases_multicells_onecell[1] +" vs " + \
                cases_gpu_cpu[0] + " " + cases_multicells_onecell[0]

    for i in range(2):
      data[cases_multicells_onecell[i]]=run(config_file,diff_cells,mpi,mpiProcessesList,
                                        cells,timesteps,cases_gpu_cpu[i],cases_multicells_onecell[i],results_file)

  elif(len(cases_multicells_onecell)==2):

    plot_title+=cases_multicells_onecell[1] + " vs " + cases_multicells_onecell[0]
    for case_multicells_onecell in cases_multicells_onecell:
      data[case_multicells_onecell]=run(config_file,diff_cells,mpi,mpiProcessesList,
                                        cells,timesteps,cases_gpu_cpu[0],case_multicells_onecell,results_file)
    plot_title+=" "+cases_gpu_cpu[0]

  elif(len(cases_gpu_cpu)==2):
    plot_title+=cases_gpu_cpu[1] + " vs " + cases_gpu_cpu[0]
    for case_gpu_cpu in cases_gpu_cpu:
      data[case_gpu_cpu]=run(config_file,diff_cells,mpi,mpiProcessesList,
                                        cells,timesteps,case_gpu_cpu,cases_multicells_onecell[0],results_file)
    plot_title+=" "+cases_multicells_onecell[0]

  else:
    data[cases_gpu_cpu[0]]=run(config_file,diff_cells,mpi,mpiProcessesList,
                           cells,timesteps,cases_gpu_cpu[0],cases_multicells_onecell[0],results_file)
    raise Exception("Only one case to compare, check cases_multicells_onecell and cases_gpu_cpu")

"""

"""
    first_word=""
    second_word=""
    if(cases_gpu_cpu[0]!=cases_gpu_cpu[1]):
      first_word+=cases_gpu_cpu[1] + " "
      second_word+=cases_gpu_cpu[0] + " "
    if(cases_multicells_onecell[0]!=cases_multicells_onecell[1]):
      first_word+=cases_multicells_onecell[1] + " "
      second_word+=cases_multicells_onecell[0] + " "

    plot_title+=first_word + "vs " +second_word

    if(cases_gpu_cpu[0]!=cases_gpu_cpu[1]):
      plot_title+=cases_gpu_cpu[0] + " "
    if(cases_multicells_onecell[0]!=cases_multicells_onecell[1]):
      plot_title+=cases_multicells_onecell[0] + " "

    plot_title+=cases_gpu_cpu[1] + " " + cases_multicells_onecell[1] +" 


    plot_title = cases[1] + " vs " + cases[0]

    cases_words0=cases[0].split()
    for word in cases_words0:
      if word in plot_title:
        plot_title.replace(word+" ", '',1)
      else:
        word_repeated=word
    plot_title+="vs " + word_repeated + " "

"""
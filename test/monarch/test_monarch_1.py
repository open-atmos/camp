import matplotlib as mpl
mpl.use('TkAgg')
import plot_cases
import matplotlib.pyplot as plt
import csv
import sys, getopt
import os
import numpy as np
from pylab import imread,subplot,imshow,show
import plot_functions
import datetime

def write_itsolver_config_file(cases_multicells_onecell):
  file1 = open("itsolver_options.txt","w")

  #print(case_gpu_cpu)

  if("Multi-cells" in cases_multicells_onecell):
    file1.write("USE_MULTICELLS=ON\n")
  else:
    file1.write("USE_MULTICELLS=OFF\n")

  file1.close()

def write_camp_config_file(case_gpu_cpu):
  file1 = open("config_variables_c_solver.txt","w")

  #print(case_gpu_cpu)

  if(case_gpu_cpu=="CPU"):
    file1.write("USE_CPU=ON\n")
  else:
    file1.write("USE_CPU=OFF\n")

  file1.close()

def run(config_file,diff_cells,mpi,mpiProcesses,n_cells,timesteps,
        case_gpu_cpu,case,results_file,plot_y_key,y_key):

  #MPI
  exec_str=""
  if mpi=="yes":
    exec_str+="mpirun -v -np "+str(mpiProcesses)+" --bind-to none "
    #exec_str+="srun -n "+str(mpiProcesses)+" "

  profileCuda=False
  if(case_gpu_cpu=="GPU"):
    profileCuda=False
    #profileCuda=True
  if(profileCuda):
    exec_str+="nvprof --analysis-metrics -f -o "+config_file+case+".nvprof "
  #--print-gpu-summary

  exec_str+="../../mock_monarch config_"+config_file+".json "+"interface_"+config_file \
            +".json "+config_file

  ADD_EMISIONS="OFF"
  if config_file=="monarch_binned":
    ADD_EMISIONS="ON"

  exec_str+=" "+ADD_EMISIONS

  #GPU-CPU
  write_camp_config_file(case_gpu_cpu)

  #Onecell-Multicells itsolver
  write_itsolver_config_file(case)
  if(case_gpu_cpu=="GPU"):
    case="Multi-cells"

  #Onecell-Multicells

  print(exec_str +" " + str(n_cells) + " " + case + " " + str(timesteps)+" "+diff_cells)
  os.system(exec_str +" " + str(n_cells) + " " + case + " " + str(timesteps)+" "+diff_cells)

  data={}
  file = 'out/'+config_file+'_'+case+results_file
  plot_functions.read_solver_stats(file, data)

  if("timeLS" in plot_y_key and "computational" in plot_y_key):
    data=plot_functions.calculate_computational_timeLS( \
      data,"timeBiconjGradMemcpy")

  if("normalized" in plot_y_key):
    if(y_key=="counterBCG" or y_key=="timeLS"):
      data=plot_functions.normalize_by_counterLS_and_cells( \
        data,y_key,n_cells,case)
      #cases[i]
    else:
      raise Exception("Unkown normalized case",y_key)

  if("(Comp.timeLS/counterBCG)" in plot_y_key):
    data=plot_functions.calculate_computational_timeLS( \
      data,"timeBiconjGradMemcpy")
    #print(data)
    if("CPU" in case_gpu_cpu):
      raise Exception("Incompatible (Comp.timeLS/counterBCG) and CPU (not counter BCG), set only GPU cases")
    for i in range(len(data["timeLS"])):
      data["timeLS"][i]=data["timeLS"][i] \
                             /data["counterBCG"][i]

  #if plot_y_key=="Percentages solveCVODEGPU":
  #data=plot_functions.calculate_percentages_solveCVODEGPU(data)

  return data

def run_cell(config_file,diff_cells,mpi,mpiProcessesList,n_cells_aux,timesteps,
             cases,cases_gpu_cpu,cases_multicells_onecell,results_file,plot_y_key):

  y_key_words = plot_y_key.split()
  y_key = y_key_words[-1]
  data={}

  for i in range(len(cases)):

    if len(mpiProcessesList)==2:
    #print("len(mpiProcessesList)==len(cases)",len(cases))
      mpiProcesses=mpiProcessesList[i]
      n_cells = int(n_cells_aux/mpiProcesses)
      #mpiProcesses=mpiProcessesList[i]
      #n_cells=n_cells_aux
    else:
      mpiProcesses=mpiProcessesList[0]
      n_cells=n_cells_aux

    data[cases[i]] = run(config_file,diff_cells,mpi,mpiProcesses,
                 n_cells,timesteps,cases_gpu_cpu[i],cases_multicells_onecell[i],
                 results_file,plot_y_key,y_key)



  #if("(Comp.timeLS/counterBCG)" in plot_y_key):
  #  data=plot_functions.calculate_computational_timeLS( \
  #    data,"timeBiconjGradMemcpy")
  # y_key="timeLS"
    #print(data)
  #  for case in cases:
  #    for i in range(len(data[case][y_key])):
  #      data[case][y_key][i]=data[case][y_key][i] \
  #                  /data[case]["counterBCG"][i]

  #print(data)

  if("(Comp.timeLS/counterBCG)" in plot_y_key):
    y_key="timeLS"

  if plot_y_key!="MAPE" and len(cases)==1:
    dataMAPE={}
    #todo change filename to be like "CPUMulti-cells" to distinguish Multi-cells CPU & GPU
    #for i in range(len(cases)):
    #  case_gpu_cpu=cases_gpu_cpu[i]
    #  if(cases_gpu_cpu[i]=="GPU"):
    #    case_gpu_cpu="Multi-cells"
    #  data_aux={}
    #  file = 'out/'+config_file+'_'+case_gpu_cpu+"_results_all_cells.csv"
    #  plot_functions.read_solver_stats(file, data_aux)
    #  dataMAPE[cases[i]]=data_aux

    #if len(cases)==1:
    print("len(cases)==1")
    case_gpu_cpu="One-cell"
    data_aux={}
    file = 'out/'+config_file+'_'+case_gpu_cpu+"_results_all_cells.csv"
    plot_functions.read_solver_stats(file, data_aux)
    dataMAPE[case_gpu_cpu]=data_aux


    case_gpu_cpu="Multi-cells"
    data_aux={}
    file = 'out/'+config_file+'_'+case_gpu_cpu+"_results_all_cells.csv"
    plot_functions.read_solver_stats(file, data_aux)
    dataMAPE[case_gpu_cpu]=data_aux

    #print(dataMAPE)

    datayMAPE=plot_functions.calculate_MAPE(dataMAPE,timesteps)
    print("MAPE "+case_gpu_cpu+":",datayMAPE)

  if(len(cases)!=2):
    raise Exception("Only one case to compare")

  if plot_y_key== "NRMSE":
    datay=plot_functions.calculate_NMRSE(data,timesteps)
  elif plot_y_key== "MAPE":
    datay=plot_functions.calculate_MAPE(data,timesteps)
  elif(plot_y_key=="SMAPE"):
    datay=plot_functions.calculate_SMAPE(data,timesteps)
  elif "Speedup" in plot_y_key:
    #y_key = plot_y_key.replace('Speedup ', '')
    #y_key_words = plot_y_key.split()
    #y_key = y_key_words[-1]
    #print(y_key)
    datay=plot_functions.calculate_speedup2(data,y_key)
  elif plot_y_key== "% Time data transfers CPU-GPU BCG":
    y_key="timeBiconjGradMemcpy"
    datay=plot_functions.calculate_BCGPercTimeDataTransfers(data,y_key)
  else:
    raise Exception("Not found plot function for plot_y_key")

  return datay


def all_timesteps():

  #config_file="simple"
  config_file="monarch_cb05"
  #config_file="monarch_binned"

  #diff_cells="Practical"
  diff_cells="Ideal"

  mpi="yes"
  #mpi="no"

  mpiProcessesList = [2]
  #mpiProcessesList = [40,1]

  cells = [10]
  #cells = [1,10,100,1000]
  #cells = [1,10,100,1000,10000,100000]

  timesteps = 1#720=12h
  TIME_STEP = 2 #pending send TIME_STEP to mock_monarch

  #cases = ["CPU One-cell"]
  #cases = ["CPU Multi-cells"]
  #cases = ["GPU One-cell"]
  #cases = ["CPU One-cell","CPU Multi-cells"]
  #cases = ["CPU One-cell","GPU Multi-cells"]
  cases = ["CPU One-cell","GPU One-cell"]
  #cases = ["CPU Multi-cells","GPU Multi-cells"]
  #cases = ["CPU Multi-cells","GPU One-cell"]
  #cases = ["GPU One-cell","GPU Multi-cells"]
  #cases = ["GPU One-cell","GPU 2 One-cell"]

  #plot_y_key = "counterBCG"
  #plot_y_key = "Average BCG internal iterations per call"
  #plot_y_key = "Average BCG time per call" #This metric makes no sense, One-cell would always be faster because is computing way less cells
  #plot_y_key = "Speedup normalized timeLS"

  #plot_y_key = "Speedup timeCVode"
  #plot_y_key = "Speedup counterLS"
  #plot_y_key = "Speedup normalized timeLS"
  #plot_y_key = "Speedup normalized computational timeLS"
  #plot_y_key = "Speedup counterBCG"
  #plot_y_key = "Speedup total iterations - counterBCG"
  #plot_y_key = "Speedup normalized counterBCG"
  #plot_y_key = "Speedup BCG iteration (Comp.timeLS/counterBCG)"
  #plot_y_key = "Percentages solveCVODEGPU" #Uncomment function

  #plot_y_key = "% Time data transfers CPU-GPU BCG"
  #plot_y_key="NRMSE"
  plot_y_key="MAPE"
  #plot_y_key="SMAPE"

  SAVE_PLOT=False
  if len(cells) > 3 or timesteps > 10 or cells[0]>1000:
    SAVE_PLOT=True

  #remove_iters=0#10 #360

  if not os.path.exists('out'):
    os.makedirs('out')

  print("WARNING: DEVELOPING CSR")

  if(config_file=="monarch_cb05"):
    diff_cells="Ideal"
    print("WARNING: ENSURE DERIV_CPU_ON_GPU IS OFF")

  if config_file=="monarch_binned":
    print("WARNING: ENSURE DERIV_CPU_ON_GPU IS ON")

  results_file="_solver_stats.csv"
  if(plot_y_key=="NRMSE" or plot_y_key=="MAPE" or plot_y_key=="SMAPE"):
    results_file='_results_all_cells.csv'

  cases_gpu_cpu=[""]*len(cases)
  cases_multicells_onecell=[""]*len(cases)
  for i in range(len(cases)):
    cases_words=cases[i].split()
    cases_gpu_cpu[i]=cases_words[0]
    cases_multicells_onecell[i]=cases_words[1]
    #if("GPU" in cases_gpu_cpu[i] and "One-cell" in cases_multicells_onecell[i])

  if "total" in plot_y_key:
    print("WARNING: Remember to enable solveBcgCuda_sum_it")
  elif "counterBCG" in plot_y_key:
    print("WARNING: Remember to disable solveBcgCuda_sum_it")

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

  #print("Base")
  #print("Optimized")
  print(datay)

  if(mpiProcessesList==2):
    for i in range(len(cases_multicells_onecell)):
      cases_multicells_onecell[i]=str(mpiProcessesList[i])+" "+cases_multicells_onecell[i]

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
  #plot_title+=diff_cells+" test"#+"Group cells"
  plot_title+=diff_cells+" test"#+"Ind. cells"
  if(len(cells)>1):
    #plot_title+=", Timesteps:[0-"+str(TIME_STEP*timesteps)+"]"#str(timesteps)
    #plot_title+=", Timesteps:"+str(timesteps)
    plot_title+=", Mean over "+str(timesteps)+ " timesteps"
    datax=cells
    plot_x_key="Cells"
    #Default metric
    #plot_y_key="Mean "+plot_y_key #Mean over all timesteps
  else:
    plot_title+=", Cells: "+str(cells[0])
    #datax=list(range(TIME_STEP,TIME_STEP*(timesteps+1),TIME_STEP))
    datax=list(range(1,timesteps+1,1))
    plot_x_key = "Timesteps"

  #if "Speedup" in plot_y_key and "counterLS" in plot_y_key:

  namey=plot_y_key #default name
  if plot_y_key=="Speedup counterLS":
    namey="Ratio of calls reduced"
    print(plot_y_key)
  if plot_y_key=="MAPE":
    namey="MAPE [%]"
  if plot_y_key=="Speedup normalized timeLS":
    namey="Normalized speedup"


  namex=plot_x_key

  if plot_y_key=="Percentages solveCVODEGPU":
    plot_functions.plot_percentages_solveCVODEGPU( \
      data,namex,namey,datax,datay,plot_title)

  #plot_functions.plot(namex,namey,datax,datay,plot_title,SAVE_PLOT)


#print(datetime.__file__)
#now = datetime.now()
"""
now = datetime.datetime.now()
dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
save_folder= "a"+" "+"b"
if not os.path.exists(save_folder):
  os.makedirs(save_folder)
save_path=save_folder+"/"+dt_string
print(save_path)
"""
all_timesteps()

#deprecated
#plot_cases.mpi_scalability()
#plot_cases.speedup_cells("Mean")
#plot_cases.speedup_cells("Standard Deviation")
#plot_cases.error_timesteps()
#plot_cases.speedup_timesteps()
#plot_cases.speedup_timesteps_counterBCG()
#plot_cases.debug_no_plot()

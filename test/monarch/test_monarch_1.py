import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import csv
import sys, getopt
import os
import numpy as np
from pylab import imread,subplot,imshow,show
import plot_functions
import datetime
import pandas as pd
import seaborn as sns

def write_itsolver_config_file(cases_multicells_onecell):
  file1 = open("itsolver_options.txt","w")

  #print(case_gpu_cpu)

  if "Multi-cells" in cases_multicells_onecell or \
      "Block-cells(N)" in cases_multicells_onecell:
    #print("USE_MULTICELLS=ON")
    file1.write("USE_MULTICELLS=ON\n")
  else:
    #print ("USE_MULTICELLS=OFF")
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
        case_gpu_cpu,case,results_file):

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
  if(case_gpu_cpu=="GPU" and case!="One-cell"):
    print("case_gpu_cpu==GPU and case!=One-cell")
    case="Multi-cells"

  #Onecell-Multicells

  print(exec_str +" " + str(n_cells) + " " + case + " " + str(timesteps)+" "+diff_cells)
  os.system(exec_str +" " + str(n_cells) + " " + case + " " + str(timesteps)+" "+diff_cells)

  data={}
  file = 'out/'+config_file+'_'+case+results_file
  plot_functions.read_solver_stats(file, data)

  return data

def run_cell(config_file,diff_cells,mpi,mpiProcessesList,n_cells_aux,timesteps,
             cases,cases_gpu_cpu,cases_multicells_onecell,results_file,plot_y_key):

  y_key_words = plot_y_key.split()
  y_key = y_key_words[-1]
  data={}
  dataMAPE={}

  for i in range(len(cases)):

    if len(mpiProcessesList)==len(cases):
      #print("len(mpiProcessesList)==len(cases)",len(cases))
      mpiProcesses=mpiProcessesList[i]
      n_cells = int(n_cells_aux/mpiProcesses)
      if n_cells==0:
        n_cells=1
      #mpiProcesses=mpiProcessesList[i]
      #n_cells=n_cells_aux
    else:
      mpiProcesses=mpiProcessesList[0]
      n_cells=n_cells_aux

    data[cases[i]] = run(config_file,diff_cells,mpi,mpiProcesses,
                         n_cells,timesteps,cases_gpu_cpu[i],cases_multicells_onecell[i],results_file)

    if("timeLS" in plot_y_key and "computational" in plot_y_key):
      data=plot_functions.calculate_computational_timeLS( \
        data,"timeBiconjGradMemcpy",cases[i])

    if("normalized" in plot_y_key):
      if(y_key=="counterBCG" or y_key=="timeLS"):
        data=plot_functions.normalize_by_counterLS_and_cells( \
          data,y_key,n_cells,cases[i])
      else:
        raise Exception("Unkown normalized case",y_key)

  if(len(cases)!=2):
    raise Exception("Only one case to compare, check cases")

  if("(Comp.timeLS/counterBCG)" in plot_y_key):
    data=plot_functions.calculate_computational_timeLS( \
      data,"timeBiconjGradMemcpy")
    y_key="timeLS"
    #print(data)
    for case in cases:
      for i in range(len(data[case][y_key])):
        data[case][y_key][i]=data[case][y_key][i] \
                    /data[case]["counterBCG"][i]

  #print(data)

  if(plot_y_key=="NRMSE"):
    datay=plot_functions.calculate_NMRSE(data,timesteps)
  elif(plot_y_key=="MAPE"):
    datay=plot_functions.calculate_MAPE(data,timesteps)
  elif(plot_y_key=="SMAPE"):
    datay=plot_functions.calculate_SMAPE(data,timesteps)
  elif("Speedup" in plot_y_key):
    #y_key = plot_y_key.replace('Speedup ', '')
    #y_key_words = plot_y_key.split()
    #y_key = y_key_words[-1]
    #print(y_key)
    datay=plot_functions.calculate_speedup2(data,y_key)
  elif(plot_y_key=="% Time data transfers CPU-GPU BCG"):
    y_key="timeBiconjGradMemcpy"
    datay=plot_functions.calculate_BCGPercTimeDataTransfers(data,y_key)
  else:
    raise Exception("Not found plot function for plot_y_key")

  return datay


def all_timesteps():


  #config_file="simple"
  #config_file="monarch_cb05"
  config_file="monarch_binned"

  diff_cells="Realistic"
  #diff_cells="Ideal"

  mpi="yes"
  #mpi="no"

  mpiProcessesList = [1]
  #mpiProcessesList = [40,1]

  #cells = [10]
  cells = [1,5]
  #cells = [100,500,1000,5000,10000]
  #cells = [100,500,1000,5000,10000]
  #cells = [100,1000,10000,100000]

  timesteps = 1#5 #720=24h #30=1h
  TIME_STEP = 2 #pending send TIME_STEP to mock_monarch

  #cases = ["CPU One-cell"]
  #cases = ["CPU Multi-cells"]
  #cases = ["CPU One-cell","CPU Multi-cells"]
  #cases = ["CPU One-cell","GPU Block-cells(N)"]
  #cases = ["CPU One-cell","GPU Block-cells(1)"]
  #cases = ["CPU Multi-cells","GPU Block-cells(N)"]
  #cases = ["CPU Multi-cells","GPU Block-cells(1)"]
  #cases = ["GPU Block-cells(1)","GPU Block-cells(N)"]
  cases = ["CPU One-cell","GPU One-cell"]

  #plot_y_key = "counterBCG"
  #plot_y_key = "Average BCG internal iterations per call"
  #plot_y_key = "Average BCG time per call" #This metric makes no sense, One-cell would always be faster because is computing way less cells
  #plot_y_key = "Speedup normalized timeLS"

  #plot_y_key = "Speedup timeCVode"
  #plot_y_key = "Speedup counterLS"
  plot_y_key = "Speedup normalized timeLS"
  #plot_y_key = "Speedup normalized computational timeLS"
  #plot_y_key = "Speedup counterBCG"
  #plot_y_key = "Speedup total iterations - counterBCG"
  #plot_y_key = "Speedup normalized counterBCG"
  #plot_y_key = "Speedup BCG iteration (Comp.timeLS/counterBCG)"

  #plot_y_key = "% Time data transfers CPU-GPU BCG"
  #plot_y_key="NRMSE"
  #plot_y_key="MAPE"
  #plot_y_key="SMAPE"

  SAVE_PLOT=False
  if len(cells) > 8 or timesteps > 10 or cells[0]>1000:
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

  #ECMWF_measures=False
  ECMWF_measures=True

  if(ECMWF_measures):
    for i in range(len(cases)):
      if cases_multicells_onecell[i]=="Block-cells(N)":
        cases_multicells_onecell[i]="Block-cells (N)"
      if cases_multicells_onecell[i]=="Block-cells(1)":
        cases_multicells_onecell[i]="Block-cells (1)"

  gpus=1
  if(len(mpiProcessesList)==2):
    for i in range(len(cases_multicells_onecell)):
      #cases_multicells_onecell[i]=str(mpiProcessesList[i])+" "+cases_multicells_onecell[i]
      #print(cases_multicells_onecell[i])
      if cases_gpu_cpu[0]=="CPU":
        cases_gpu_cpu[0]=str(mpiProcessesList[0]) + " MPI processes"
      if cases_gpu_cpu[1]=="GPU":
        cases_gpu_cpu[1]=str(gpus) + " GPU"

  #if(len(mpiProcessesList)==2):
  #  for()
  #  if(cases_multicells_onecell[0]=="CPU"):
  #    cases_multicells_onecell[0]=str(mpiProcessesList[0]) + " MPI"
  #  if(cases_multicells_onecell[1]=="GPU"):
  #    cases_multicells_onecell[0]=str(mpiProcessesList[0]) + " MPI"

  plot_title=""
  first_word=""
  second_word=""
  third_word=""

  if ECMWF_measures:
    first_word+= cases_gpu_cpu[1] + " " + cases_multicells_onecell[1] + " "
    second_word+= cases_gpu_cpu[0] + " " + cases_multicells_onecell[0] + " "
  else:
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

    if not ECMWF_measures:
      plot_title+=", Mean over "+str(timesteps)+ " timesteps"
    datax=cells
    plot_x_key="Cells"

  else:
    plot_title+=", Cells: "+str(cells[0])
    #datax=list(range(TIME_STEP,TIME_STEP*(timesteps+1),TIME_STEP))
    datax=list(range(1,timesteps+1,1))
    plot_x_key = "Timesteps"

  #namey=plot_y_key #default name
  namey="Speedup"
  if plot_y_key=="Speedup normalized computational timeLS":
    namey="Speedup without CPU-GPU data transfers"
  if plot_y_key=="Speedup counterLS":
    namey="Proportion of calls reduced"
  if plot_y_key=="MAPE":
    namey="MAPE [%]"
  #if plot_y_key=="Speedup normalized timeLS":
  #  namey="Normalized speedup"

  namex=plot_x_key

  plot_functions.plot(namex,namey,datax,datay,plot_title,SAVE_PLOT)


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


#rs = np.random.RandomState(365)
#values = rs.randn(365, 4).cumsum(axis=0)
#dates = pd.date_range("1 1 2016", periods=365, freq="D")
#data = pd.DataFrame(values, dates, columns=["A", "B", "C", "D"])
#data = data.rolling(7).mean()

def plotsns():

  datax=[1,10]
  datay=[1,2]

  sns.set_style("whitegrid")

  #sns.set(font_scale=2)
  #sns.set_context("paper", rc={"font.size":8,"axes.titlesize":8,"axes.labelsize":5})
  sns.set_context("paper", font_scale=1.25)

  data = pd.DataFrame(datay, datax)

  plt.xlabel("Colors")
  plt.ylabel("Values")
  plt.title("Colors vs Values") # You can comment this line out if you don't need title

  sns.lineplot(data=data, palette="tab10", linewidth=2.5)
  plt.show()

#plotsns()

all_timesteps()



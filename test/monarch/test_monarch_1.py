import matplotlib as mpl
mpl.use('TkAgg')
import plot_cases #todo delete file
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
import time

def write_itsolver_config_file(cases_multicells_onecell):
  file1 = open("itsolver_options.txt","w")

  #print(case_gpu_cpu)

  cells_method_str="CELLS_METHOD="+cases_multicells_onecell
  file1.write(cells_method_str)
  print(cells_method_str)

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
    #print("case_gpu_cpu==GPU and case!=One-cell")
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
      elif(y_key=="timecvStep"):
        data=plot_functions.normalize_by_countercvStep_and_cells( \
        data,"timecvStep",n_cells,cases[i])
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
  elif plot_y_key== "% Time data transfers CPU-GPU BCG":
    y_key="timeBiconjGradMemcpy"
    datay=plot_functions.calculate_BCGPercTimeDataTransfers(data,y_key)
  else:
    raise Exception("Not found plot function for plot_y_key")

  return datay

def run_case(config_file,diff_cells,mpi,mpiProcessesList,cells,timesteps,
             cases,cases_gpu_cpu,cases_multicells_onecell,results_file,plot_y_key):

  #cases=casesList[0]

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

  return datay

def plot_historic(cases_gpu_cpu,cases_multicells_onecell,cells,diff_cells,timesteps,plot_y_key,
              mpiProcessesList,datay,SAVE_PLOT):
  print("plot_historic")

  for i in range(2):
    if cases_multicells_onecell[i]=="Block-cells(N)":
      cases_multicells_onecell[i]="Block-cells (N)"
    if cases_multicells_onecell[i]=="Block-cells(1)":
      cases_multicells_onecell[i]="Block-cells (1)"

  gpus=1
  if(len(mpiProcessesList)==2):
    for i in range(len(cases_multicells_onecell)):
      if cases_gpu_cpu[i]=="CPU":
        cases_gpu_cpu[i]=str(mpiProcessesList[0]) + " MPI processes"
      if cases_gpu_cpu[i]=="GPU":
        cases_gpu_cpu[i]=str(gpus) + " GPU"

  plot_title=""
  first_word=""
  second_word=""

  first_word+= cases_gpu_cpu[1] + " " + cases_multicells_onecell[1] + " "
  second_word+= cases_gpu_cpu[0] + " " + cases_multicells_onecell[0] + " "

  plot_title+=first_word + "vs " + second_word
  plot_title+=diff_cells+" test"

  if(len(cells)>1):

    print_timesteps_title=False
    if print_timesteps_title:
      plot_title+=", Mean over "+str(timesteps)+ " timesteps"
    datax=cells
    plot_x_key="Cells"

  else:
    plot_title+=", Cells: "+str(cells[0])
    datax=list(range(1,timesteps+1,1))
    plot_x_key = "Timesteps"

  namey="Speedup"
  if plot_y_key=="Speedup normalized computational timeLS":
    namey="Speedup without CPU-GPU data transfers"
  if plot_y_key=="Speedup counterLS":
    namey="Proportion of calls reduced"
  if plot_y_key=="MAPE":
    namey="MAPE [%]"

  namex=plot_x_key

  columns=[]

  plot_functions.plot(namex,namey,datax,datay,plot_title,columns,SAVE_PLOT)

def plot_cases(casesList,cases_gpu_cpu2,cases_multicells_onecell2,cells,diff_cells,timesteps,plot_y_key,
                   mpiProcessesList,datacases,SAVE_PLOT):

  plot_title=""
  columns=[]
  first_word=""
  second_word=""
  gpus=1

  namey="Speedup"
  if plot_y_key=="Speedup normalized computational timeLS":
    namey="Speedup without CPU-GPU data transfers"
  if plot_y_key=="Speedup counterLS":
    namey="Proportion of calls reduced"
  if plot_y_key=="MAPE":
    namey="MAPE [%]"

  for j in range(len(casesList)):
    cases=casesList[j]
    cases_gpu_cpu=[""]*len(cases)
    cases_multicells_onecell=[""]*len(cases)
    for i in range(len(cases)):
      cases_words=cases[i].split()
      cases_gpu_cpu[i]=cases_words[0]
      cases_multicells_onecell[i]=cases_words[1]

      if cases_multicells_onecell[i]=="Block-cells(N)":
        cases_multicells_onecell[i]="Block-cells (N)"
      if cases_multicells_onecell[i]=="Block-cells(1)":
        cases_multicells_onecell[i]="Block-cells (1)"

      if(len(mpiProcessesList)==2):
        if cases_gpu_cpu[i]=="CPU":
          cases_gpu_cpu[i]=str(mpiProcessesList[0]) + " MPI processes"
        if cases_gpu_cpu[i]=="GPU":
          cases_gpu_cpu[i]=str(gpus) + " GPU"

    if(len(casesList)>1):

      column=cases_gpu_cpu[1] + " " + cases_multicells_onecell[1]

      columns.append(column)

  if(len(casesList)>1):
    plot_title+="Case vs " + cases_gpu_cpu[0] + " " + cases_multicells_onecell[0] + " "
    plot_title+=diff_cells+" test"

    datay=datacases
  else:
    first_word+= cases_gpu_cpu[1] + " " + cases_multicells_onecell[1] + " "
    second_word+= cases_gpu_cpu[0] + " " + cases_multicells_onecell[0] + " "
    plot_title+=first_word + "vs " + second_word
    plot_title+=diff_cells+" test"

    datay=datacases[0]

  if(len(cells)>1):
    #print_timesteps_title=True
    print_timesteps_title=False
    if print_timesteps_title:
      #plot_title+=", Mean over "+str(timesteps)+ " timesteps"
      plot_title+=", Timesteps: "+str(timesteps)
    datax=cells
    plot_x_key="Cells"

  else:
    plot_title+=", Cells: "+str(cells[0])
    datax=list(range(1,timesteps+1,1))
    plot_x_key = "Timesteps"

  namex=plot_x_key

  print(namey,":",datay)

  plot_functions.plot(namex,namey,datax,datay,plot_title,columns,SAVE_PLOT)


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

  cells = [100,1000]
  #cells = [10,100,1000]
  #cells = [100,500,1000]
  #cells = [1,5,10,50,100]
  #cells = [100,1000,5000,10000]
  #cells = [100,500,1000,5000,10000]
  #cells = [100,1000,10000,100000]

  timesteps = 1#5 #720=24h #30=1h
  TIME_STEP = 2 #pending send TIME_STEP to mock_monarch

  #cases = ["Historic"]
  #cases = ["CPU One-cell"]
  #cases = ["CPU Multi-cells"]
  #cases = ["GPU One-cell"]
  #cases = ["CPU One-cell","CPU Multi-cells"]
  #cases = ["CPU One-cell","GPU Block-cells(N)"]
  cases = ["CPU One-cell","GPU Block-cells(1)"]
  #cases = ["CPU Multi-cells","GPU Block-cells(N)"]
  #cases = ["CPU Multi-cells","GPU Block-cells(1)"]
  #cases = ["GPU Block-cells(1)","GPU Block-cells(N)"]
  #cases = ["CPU One-cell","GPU One-cell"]
  #cases = ["CPU Multi-cells","GPU Multi-cells"]
  #cases = ["GPU Multi-cells","GPU Block-cells(N)"]
  #cases = ["GPU Block-cells(N)","GPU Block-cells(1)"]

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
  #plot_y_key = "Percentages solveCVODEGPU" #Uncomment function
  #plot_y_key = "Speedup timecvStep"
  #plot_y_key = "Speedup normalized timecvStep"#not needed, is always normalized
  #plot_y_key = "Speedup device timecvStep"

  #plot_y_key = "% Time data transfers CPU-GPU BCG"
  #plot_y_key="NRMSE"
  #plot_y_key="MAPE"#todo check old ls option (cvode_gpu)
  #plot_y_key="SMAPE"

  #remove_iters=0#10 #360

  results_file="_solver_stats.csv"
  if(plot_y_key=="NRMSE" or plot_y_key=="MAPE" or plot_y_key=="SMAPE"):
    results_file='_results_all_cells.csv'

  if not os.path.exists('out'):
    os.makedirs('out')

  print("WARNING: DEVELOPING CSR")

  if "total" in plot_y_key:
    print("WARNING: Remember to enable solveBcgCuda_sum_it")
  elif "counterBCG" in plot_y_key:
    print("WARNING: Remember to disable solveBcgCuda_sum_it")

  if(config_file=="monarch_cb05"):
    diff_cells="Ideal"
    print("WARNING: ENSURE DERIV_CPU_ON_GPU IS OFF")

  if config_file=="monarch_binned":
    print("WARNING: ENSURE DERIV_CPU_ON_GPU IS ON")

  SAVE_PLOT=False
  start_time = time.perf_counter()

  casesList=[]
  if(cases[0]=="Historic"):
    if(len(cells)<2):
      print("WARNING: PENDING TEST HISTORIC WITH TIMESTEPS AS AXIS X")

    casesList.append(["CPU One-cell","GPU Block-cells(1)"])
    casesList.append(["CPU One-cell","GPU Block-cells(N)"])
    casesList.append(["CPU One-cell","GPU Multi-cells"])
    casesList.append(["CPU One-cell","CPU Multi-cells"])
    casesList.append(["CPU One-cell","GPU One-cell"])
  else:
    casesList.append(cases)

  datacases=[]
  columns=[]
  for j in range(len(casesList)):
    cases=casesList[j]
    cases_gpu_cpu=[""]*len(cases)
    cases_multicells_onecell=[""]*len(cases)
    for i in range(len(cases)):
      cases_words=cases[i].split()
      cases_gpu_cpu[i]=cases_words[0]
      cases_multicells_onecell[i]=cases_words[1]

    datacase = run_case(config_file,diff_cells,mpi,mpiProcessesList,cells,timesteps,
             cases,cases_gpu_cpu,cases_multicells_onecell,results_file,plot_y_key)
    datacases.append(datacase)

  end_time=time.perf_counter()
  time_s=end_time-start_time
  #print("time_s:",time_s)
  if time_s>60:
    SAVE_PLOT=True

  plot_cases(casesList,cases_gpu_cpu,cases_multicells_onecell,cells,diff_cells,timesteps,plot_y_key,
            mpiProcessesList,datacases,SAVE_PLOT)


"""
"""

def plotsns():

  namex="Cells"
  namey="Speedup"
  plot_title="Test plotsns"

  ncol=4
  #ncol=2
  if(ncol==4):

    datay2=[[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
    datax=[123, 346, 789]
    columns=["GPU Block-cells(1)",
             "GPU Block-cells(2)",
             "GPU Block-cells(3)",
             "GPU Block-cells(4)"]
  else:
    datay2=[[1, 2, 3], [4, 5, 6]]
    datax=[123, 346, 789]
    columns=["GPU Block-cells(1)",
           "GPU Block-cells(2)"]

  #datay=map(list,)

  #datay=datay2
  datay=list(map(list, zip(*datay2))) # short circuits at shortest nested list if table is jagged
  #numpy_array = np.array(datay2)
  #transpose = numpy_array.T
  #datay = transpose.tolist()

  print(datay)
  print(datax)

  #print(sns.__version__)
  sns.set_style("whitegrid")

  #sns.set(font_scale=2)
  #sns.set_context("paper", rc={"font.size":8,"axes.titlesize":8,"axes.labelsize":5})
  sns.set_context("paper", font_scale=1.25)

  #data = pd.DataFrame(datay, datax)
  data = pd.DataFrame(datay, datax, columns=columns)

  fig = plt.figure()
  ax = plt.subplot(111)

  ax.set_xlabel(namex)
  ax.set_ylabel(namey)
  #ax.set_title(plot_title)

  legend=True
  if(legend==True):

    print("WARNING: Increase plot window manually to take better screenshot")

    sns.lineplot(data=data, palette="tab10", linewidth=2.5)


    #ax.set_position([box.x0, box.y0 + box.height * 0.1,
    #               box.width, box.height * 0.9])

    #Legend under the plot
    #box = ax.get_position()
    #ax.set_position([box.x0, box.y0 + box.height * 0.1,
    #             box.width, box.height * 0.75])
    #ax.legend(bbox_to_anchor=(0.5, -0.05), loc='upper center',
    #          labels=columns,ncol=4, mode="expand", borderaxespad=0.)
    #fig.subplots_adjust(bottom=0.35)
    #borderaxespad=1. to move down more the legend

    #Legend up the plot (problem: hide title)
    ax.set_title(plot_title, y=1.06)

    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1),
          ncol=len(columns), labels=columns,frameon=True,shadow=False, borderaxespad=0.)

    #ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1),
    #          ncol=len(columns), labels=columns,frameon=False, shadow=False, borderaxespad=0.)#fine


    #ax.subplots_adjust(top=0.25) #not work
    #fig.subplots_adjust(top=0.25)

    #legend out of the plot at the right (problem: plot feels very small)
    #sns.lineplot(data=data, palette="tab10", linewidth=2.5)
    #box=ax.get_position()
    #ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])
    #ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0,labels=columns)

  else:
    ax.set_title(plot_title)
    sns.lineplot(data=data, palette="tab10", linewidth=2.5, legend=False)
  plt.show()

#rs = np.random.RandomState(365)
#values = rs.randn(365, 4).cumsum(axis=0)
#dates = pd.date_range("1 1 2016", periods=365, freq="D")
#data = pd.DataFrame(values, dates, columns=["A", "B", "C", "D"])
#data = data.rolling(7).mean()

#plotsns()
all_timesteps()



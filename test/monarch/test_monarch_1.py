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
import time

def write_itsolver_config_file(cases_multicells_onecell):
  file1 = open("itsolver_options.txt","w")

  cells_method_str="CELLS_METHOD="+cases_multicells_onecell
  file1.write(cells_method_str)
  #print(cells_method_str)

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
        case_gpu_cpu,cases_multicells_onecell,results_file,profileCuda):

  exec_str=""
  if mpi=="yes":
    exec_str+="mpirun -v -np "+str(mpiProcesses)+" --bind-to none "
    #exec_str+="srun -n "+str(mpiProcesses)+" "

  if(profileCuda and case_gpu_cpu=="GPU"):
    exec_str+="nvprof --analysis-metrics -f -o "+config_file+cases_multicells_onecell+".nvprof "
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
  write_itsolver_config_file(cases_multicells_onecell)
  if(case_gpu_cpu=="GPU" and cases_multicells_onecell!="One-cell"):
    #print("case_gpu_cpu==GPU and case!=One-cell")
    cases_multicells_onecell="Multi-cells"

  #Onecell-Multicells

  print(exec_str +" " + str(n_cells) + " " + cases_multicells_onecell + " " + str(timesteps)+" "+diff_cells)
  os.system(exec_str +" " + str(n_cells) + " " + cases_multicells_onecell + " " + str(timesteps)+" "+diff_cells)

  data={}
  file = 'out/'+config_file+'_'+cases_multicells_onecell+results_file
  plot_functions.read_solver_stats(file, data)

  return data

def run_cell(config_file,diff_cells,mpi,mpiProcessesList,n_cells_aux,timesteps,
             cases,cases_gpu_cpu,cases_multicells_onecell,results_file,plot_y_key,tol,profileCuda):

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
                         n_cells,timesteps,cases_gpu_cpu[i],cases_multicells_onecell[i],results_file,profileCuda)

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
    raise Exception("Cases to compare != 2, check cases")

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
    datay=plot_functions.calculate_MAPE(data,timesteps,tol)
  elif(plot_y_key=="SMAPE"):
    datay=plot_functions.calculate_SMAPE(data,timesteps)
  elif("Speedup" in plot_y_key):
    #y_key = plot_y_key.replace('Speedup ', '')
    #y_key_words = plot_y_key.split()
    #y_key = y_key_words[-1]
    #print(y_key)
    datay=plot_functions.calculate_speedup2(data,y_key)
  elif plot_y_key== "Percentage data transfers CPU-GPU [%]":
    y_key="timeBiconjGradMemcpy"
    print("elif plot_y_key==Time data transfers")
    datay=plot_functions.calculate_BCGPercTimeDataTransfers(data,y_key)
  else:
    raise Exception("Not found plot function for plot_y_key")

  return datay

def run_case(config_file,diff_cells,mpi,mpiProcessesList,cells,timesteps,
             cases,cases_gpu_cpu,cases_multicells_onecell,results_file,plot_y_key,tol,profileCuda):

  datacase=[]
  for i in range(len(cells)):
    datay_cell = run_cell(config_file,diff_cells,mpi,mpiProcessesList,cells[i],timesteps,
                          cases,cases_gpu_cpu,cases_multicells_onecell,results_file,plot_y_key,tol,profileCuda)

    #print(datay_cell)
    if(len(cells)>1):
      #Mean timesteps
      datacase.append(np.mean(datay_cell))
    else:
      datacase=datay_cell

  return datacase

def run_diff_cells(datacolumns,legend,columnHeader,config_file,diff_cells,mpi,mpiProcessesList,cells,timesteps,
             casesL,results_file,plot_y_key,diff_arquiOptim,tol,profileCuda):

  #print("run_diff_cells start run_diff_cells",datacolumns)
  column=columnHeader
  for j in range(len(casesL)):
    cases=casesL[j]
    cases_gpu_cpu=[""]*len(cases)
    cases_multicells_onecell=[""]*len(cases)
    cases_multicells_onecell_name=[""]*len(cases)
    cases_gpu_cpu_name=[""]*len(cases)
    for i in range(len(cases)):
      cases_words=cases[i].split()
      cases_gpu_cpu[i]=cases_words[0]
      cases_multicells_onecell[i]=cases_words[1]

      if cases_multicells_onecell[i]=="Block-cellsN":
        cases_multicells_onecell_name[i]="Block-cells (N)"
      elif cases_multicells_onecell[i]=="Block-cells1":
        cases_multicells_onecell_name[i]="Block-cells (1)"
      else:
        cases_multicells_onecell_name[i]=cases_multicells_onecell[i]

      if(len(mpiProcessesList)==2):
        if cases_gpu_cpu[i]=="CPU":
          cases_gpu_cpu_name[i]=str(mpiProcessesList[i]) + " MPI"
          #print("cases_gpu_cpu[i]==CPU",cases_gpu_cpu_name[i])
        #elif cases_gpu_cpu[i]=="GPU":
        #  cases_gpu_cpu_name[i]=str(gpus) + " GPU" #always 1 GPU, so comment this on the test section
        else:
          cases_gpu_cpu_name[i]=cases_gpu_cpu[i]
      else:
        cases_gpu_cpu_name[i]=cases_gpu_cpu[i]

    #print("cases_gpu_cpu",cases_gpu_cpu)
    if(len(casesL)>1):
      column=columnHeader+cases_multicells_onecell_name[1]
      if(diff_arquiOptim):
        column=columnHeader+cases_gpu_cpu_name[1]+" "+cases_multicells_onecell_name[1]
      #
        #todo add GPU name only if its different

    legend.append(column)

    datacase = run_case(config_file,diff_cells,mpi,mpiProcessesList,cells,timesteps,
                        cases,cases_gpu_cpu,cases_multicells_onecell,results_file,plot_y_key,tol,profileCuda)
    datacolumns.append(datacase)

  plot_title=""
  first_word= cases_gpu_cpu_name[1] + " " + cases_multicells_onecell_name[1]

  if plot_y_key=="Percentage data transfers CPU-GPU [%]":
    second_word=""
  else:#Speedup
    second_word=" vs "+cases_gpu_cpu_name[0] + " " + cases_multicells_onecell_name[0]

  if(len(casesL)>1):
    if(not diff_arquiOptim):
      plot_title+=cases_gpu_cpu_name[1] + " "
    plot_title+="Implementations"+second_word
  else:
    plot_title=first_word+second_word

  return plot_title

def plot_cases(datayL,legend,plot_title,cells,timesteps,
               plot_y_key,SAVE_PLOT):

  namey=plot_y_key
  if plot_y_key=="Speedup normalized computational timeLS":
    namey="Speedup without data transfers CPU-GPU"
  if plot_y_key=="Speedup counterLS":
    namey="Speedup iterations CVODE solving"
  if plot_y_key=="Speedup normalized timeLS":
    namey="Speedup linear solver"
  if plot_y_key=="Speedup timeCVode":
    namey="Speedup CVODE solving"
  if plot_y_key=="MAPE":
    namey="MAPE [%]"

  if(len(datayL)>1):
    datay=datayL
  else:
    datay=datayL[0]

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

  #plot_functions.plot(namex,namey,datax,datay,plot_title,legend,SAVE_PLOT)


def all_timesteps():

  #config_file="simple"
  #config_file="monarch_cb05"
  config_file="monarch_binned"

  diff_cellsL=[]
  diff_cellsL.append("Realistic")
  #diff_cellsL.append("Ideal")

  mpi="yes"
  #mpi="no"

  profileCuda=False
  #profileCuda=True

  mpiProcessesList = [1]
  #mpiProcessesList = [40,1]

  #todo fix cells=1 realistic test
  cells = [100]
  #cells = [5,10]
  #cells = [100,500,1000]
  #cells = [1,5,10,50,100]
  #cells = [100,500,1000,5000,10000]

  timesteps = 1#5 #720=24h #30=1h
  TIME_STEP = 2 #pending send TIME_STEP to mock_monarch

  caseBase="CPU One-cell"
  #caseBase="CPU Multi-cells"
  #caseBase="GPU Block-cellsN"

  casesOptim=[]
  #casesOptim.append("GPU Block-cells1")
  casesOptim.append("GPU Block-cellsN")
  #casesOptim.append("GPU Multi-cells")
  #casesOptim.append("GPU One-cell")
  #casesOptim.append("CPU Multi-cells")

  #cases = ["Historic"]
  #cases = ["CPU One-cell"]
  #cases = ["CPU Multi-cells"]
  #cases = ["GPU One-cell"]

  #plot_y_key = "Speedup timeCVode"
  #plot_y_key = "Speedup counterLS"
  plot_y_key = "Speedup normalized timeLS"
  #plot_y_key = "Speedup normalized computational timeLS"
  #plot_y_key = "Speedup counterBCG"
  #plot_y_key = "Speedup total iterations - counterBCG"
  #plot_y_key = "Speedup normalized counterBCG"
  #plot_y_key = "Speedup BCG iteration (Comp.timeLS/counterBCG)"
  #plot_y_key = "Speedup timecvStep"
  #plot_y_key = "Speedup normalized timecvStep"#not needed, is always normalized
  #plot_y_key = "Speedup device timecvStep"
  #plot_y_key = "Percentage data transfers CPU-GPU [%]"

  #plot_y_key = "Percentages solveCVODEGPU" #Pending function
  #plot_y_key="MAPE"
  #plot_y_key="SMAPE"
  #plot_y_key="NRMSE"
  tol=1.0E-4 #MAPE=0
  #tol=1.0E-6 #MAPE~=0.5


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

  if config_file=="monarch_binned":
    print("WARNING: ENSURE DERIV_CPU_ON_GPU IS ON")

  SAVE_PLOT=False
  start_time = time.perf_counter()

  casesL=[]
  cases=[]
  diff_arquiOptim=False
  cases_words=casesOptim[0].split()
  lastArquiOptim=cases_words[0]
  for caseOptim in casesOptim:
    #cases.append(caseBase)
    #cases.append(caseOptim)
    cases=[caseBase]+[caseOptim]
    casesL.append(cases)

    cases_words=caseOptim.split()
    arqui=cases_words[0]
    if(lastArquiOptim!=arqui):
      diff_arquiOptim=True
    lastArquiOptim=arqui


  if(cases[0]=="Historic"):
    if(len(cells)<2):
      print("WARNING: PENDING TEST HISTORIC WITH TIMESTEPS AS AXIS X")

    casesL.append(["CPU One-cell","GPU Block-cells1"])
    casesL.append(["CPU One-cell","GPU Block-cellsN"])
    casesL.append(["CPU One-cell","GPU Multi-cells"])
    casesL.append(["CPU One-cell","CPU Multi-cells"])
    casesL.append(["CPU One-cell","GPU One-cell"])
  elif(len(casesL)==0):
    print("len(casesL)==0")
    casesL.append(cases)

  datacolumns=[]
  legend=[]
  plot_title=""
  for diff_cells in diff_cellsL:
    if(config_file=="monarch_cb05"):
      diff_cells="Ideal"
      print("WARNING: ENSURE DERIV_CPU_ON_GPU IS OFF")

    column=""
    if(len(diff_cellsL)>1):
      column+=diff_cells+" "

    plot_title=run_diff_cells(datacolumns,legend,column,config_file,diff_cells,mpi,mpiProcessesList,cells,timesteps,
           casesL,results_file,plot_y_key,diff_arquiOptim,tol,profileCuda)

  end_time=time.perf_counter()
  time_s=end_time-start_time
  #print("time_s:",time_s)
  if time_s>60:
    SAVE_PLOT=True

  print("main plot_title",plot_title)

  if(len(diff_cellsL)==1):
    plot_title+=", "+diff_cells+" test"

  plot_cases(datacolumns,legend,plot_title,cells,timesteps,
             plot_y_key,SAVE_PLOT)


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
    legend=["GPU Block-cells(1)",
             "GPU Block-cells(2)",
             "GPU Block-cells(3)",
             "GPU Block-cells(4)"]
  else:
    datay2=[[1, 2, 3], [4, 5, 6]]
    datax=[123, 346, 789]
    legend=["GPU Block-cells(1)",
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
  data = pd.DataFrame(datay, datax, columns=legend)

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
    #          labels=legend,ncol=4, mode="expand", borderaxespad=0.)
    #fig.subplots_adjust(bottom=0.35)
    #borderaxespad=1. to move down more the legend

    #Legend up the plot (problem: hide title)
    ax.set_title(plot_title, y=1.06)

    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1),
          ncol=len(legend), labels=legend,frameon=True,shadow=False, borderaxespad=0.)

    #ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1),
    #          ncol=len(legend), labels=legend,frameon=False, shadow=False, borderaxespad=0.)#fine


    #ax.subplots_adjust(top=0.25) #not work
    #fig.subplots_adjust(top=0.25)

    #legend out of the plot at the right (problem: plot feels very small)
    #sns.lineplot(data=data, palette="tab10", linewidth=2.5)
    #box=ax.get_position()
    #ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])
    #ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0,labels=legend)

  else:
    ax.set_title(plot_title)
    sns.lineplot(data=data, palette="tab10", linewidth=2.5, legend=False)
  plt.show()

#rs = np.random.RandomState(365)
#values = rs.randn(365, 4).cumsum(axis=0)
#dates = pd.date_range("1 1 2016", periods=365, freq="D")
#data = pd.DataFrame(values, dates, legend=["A", "B", "C", "D"])
#data = data.rolling(7).mean()

#plotsns()
all_timesteps()



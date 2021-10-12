import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib
import csv
import numpy as np
import statistics
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
import math
import datetime
import os
import pandas as pd
import seaborn as sns


def get_values_same_timestep(timestep_to_plot,mpiProcessesList, \
                        data,plot_x_key,plot_y_key):

  new_data = {}
  new_data[plot_x_key]=mpiProcessesList
  new_data[plot_y_key] = []
  for i in range(len(mpiProcessesList)):
    #print(base_data[i],new_data[i], base_data[i]/new_data[i])
    #print(len(data["timestep"]), len(data["mpiProcesses"]))
    step_len=int(len(data["timestep"])/len(mpiProcessesList))
    j = i*step_len+timestep_to_plot
    #print("data[new_plot_y_key][j]",data[plot_y_key][j])
    new_data[plot_y_key].append(data[plot_y_key][j])

  return new_data

def calculate_computational_timeLS(data,plot_y_key,case):

  #cases=list(data.keys())

  #gpu_exist=False

  #for case in cases:
  if("GPU" in case):
    data_timeBiconjGradMemcpy=data[case][plot_y_key]
    data_timeLS=data[case]["timeLS"]
    #print(data_timeBiconjGradMemcpy)
    #print(data_timeLS)
    for i in range(len(data_timeLS)):
      data_timeLS[i]=data_timeLS[i]-data_timeBiconjGradMemcpy[i]
    #print(data_timeLS)
    #gpu_exist=True

  #if(gpu_exist==False):
  #  raise Exception("Not GPU case for calculate_computational_timeLS metric")

  #print("calculate_computational_timeLS")
  #print(data_timeLS)

  #for i in range(len(data_timeLS)):
  #  data_timeLS[i]=data_timeLS[i]-data_timeBiconjGradMemcpy[i]

  return data

def calculate_percentages_solveCVODEGPU(din):

  #data_aux={}
  print(din)
  #print(data["timeNewtonIteration"])

  percNum=100#1

  data={}

  data["timeNewtonIteration"]=[]
  data["timeJac"]=[]
  data["timelinsolsetup"]=[]
  data["timecalc_Jac"]=[]
  data["timeRXNJac"]=[]
  data["timef"]=[]
  data["timeguess_helper"]=[]


  for i in range(len(din["timesolveCVODEGPU"])):
    if(din["timesolveCVODEGPU"][i]!=0):
      data["timeNewtonIteration"].append(din["timeNewtonIteration"][i]/din["timesolveCVODEGPU"][i]*percNum)

      data["timeJac"].append(din["timeJac"][i]/din["timesolveCVODEGPU"][i]*percNum)
      data["timelinsolsetup"].append(din["timelinsolsetup"][i]/din["timesolveCVODEGPU"][i]*percNum)
      data["timecalc_Jac"].append(din["timecalc_Jac"][i]/din["timesolveCVODEGPU"][i]*percNum)
      data["timeRXNJac"].append(din["timeRXNJac"][i]/din["timesolveCVODEGPU"][i]*percNum)
      data["timef"].append(din["timef"][i]/din["timesolveCVODEGPU"][i]*percNum)
      data["timeguess_helper"].append(din["timeguess_helper"][i]/din["timesolveCVODEGPU"][i]*percNum)

      #data["timeNewtonIteration"]=din["timeNewtonIteration"][i]/data["timesolveCVODEGPU"][i]*percNum
      #data["timeJac"][i]=din["timeJac"][i]/data["timesolveCVODEGPU"][i]*percNum
      #data["timelinsolsetup"][i]=din["timelinsolsetup"][i]/data["timesolveCVODEGPU"][i]*percNum
      #data["timecalc_Jac"][i]=din["timecalc_Jac"][i]/data["timesolveCVODEGPU"][i]*percNum
      #data["timeRXNJac"][i]=din["timeRXNJac"][i]/data["timesolveCVODEGPU"][i]*percNum
      #data["timef"][i]=din["timef"][i]/data["timesolveCVODEGPU"][i]*percNum
      #data["timeguess_helper"][i]=din["timeguess_helper"][i]/data["timesolveCVODEGPU"][i]*percNum

  #print(data["timeNewtonIteration"])

  print("calculate_percentages_solveCVODEGPU")
  print(data)

  return data

def normalize_by_counterLS_and_cells(data,plot_y_key,cells,case):

  #plot_y_key = "timeLS"
  #new_plot_y_key="Normalized timeLS"

  #print("normalize_by_counterLS_and_cells")
  #print(data[cases[0]][plot_y_key])

  if("One-cell" in case):
    cells_multiply=cells
  elif("Multi-cells" in case or "GPU" in case):
    cells_multiply=1
  else:
    raise Exception("normalize_by_counterLS_and_cells case without One-cell or Multi-cells key name")

  #print(cells_multiply)
  #print(data)

  #data[case][new_plot_y_key] = []
  for i in range(len(data[case][plot_y_key])):
    #print(base_data[i],new_data[i], base_data[i]/new_data[i])
    data[case][plot_y_key][i]=data[case][plot_y_key][i]\
    /data[case]["counterLS"][i]*cells_multiply
    #data[case][new_plot_y_key].append(data[case][plot_y_key][i] \
    #                                  / data[case]["counterLS"][i]*cells_multiply)

  #print(data)

  return data

def normalize_by_countercvStep_and_cells(data,plot_y_key,cells,case):

  #plot_y_key = "timeLS"
  #new_plot_y_key="Normalized timeLS"

  print("normalize_by_countercvStep_and_cells")

  #print(data[cases[0]][plot_y_key])

  if("One-cell" in case):
    print("One-cell")
    cells_multiply=cells
  elif("Multi-cells" in case):
    print("Multi-cells")
    cells_multiply=1
  else:
    raise Exception("normalize_by_counterLS_and_cells case without One-cell or Multi-cells key name")

  #print(cells_multiply)

  #data[case][new_plot_y_key] = []
  for i in range(len(data[plot_y_key])):
    #print(base_data[i],new_data[i], base_data[i]/new_data[i])
    data[plot_y_key][i]=data[plot_y_key][i] \
                        /data["countercvStep"][i]*cells_multiply

  #print(data[cases[0]][plot_y_key])
  #print(data)

  return data

def normalized_timeLS(new_plot_y_key,cases_multicells_onecell,data, cells):

  plot_y_key = "timeLS"
  #new_plot_y_key="Normalized timeLS"

  #base_data=data[cases_multicells_onecell[0]][plot_y_key]
  #new_data=data[cases_multicells_onecell[1]][plot_y_key]

  #print(base_data)

  for case in cases_multicells_onecell:

    if(case=="One-cell"):
      cells_multiply=cells
    else:
      cells_multiply=1

    data[case][new_plot_y_key] = []
    for i in range(len(data[case][plot_y_key])):
      #print(base_data[i],new_data[i], base_data[i]/new_data[i])
      data[case][new_plot_y_key].append(data[case][plot_y_key][i] \
                           / data[case]["counterLS"][i]*cells_multiply)

  #extract timestep: timestep is common in both cases like speedup
  #data["timestep"]=data.get("timestep",[]) \
  #                 + data[cases_multicells_onecell[0]]["timestep"]

  #print(data)

  return data, new_plot_y_key

def calculate_mean_cell(cell,data, \
                       plot_x_key,plot_y_key):

  new_plot_y_key="Mean " + plot_y_key

  data[plot_x_key] = data.get(plot_x_key,[]) + [cell]

  data[new_plot_y_key] = data.get(new_plot_y_key,[]) \
                         + [np.mean(data[cell][plot_y_key])]

  #print(data)
  #print(plot_y_key)
  #print(data[cell][plot_y_key])
  #print(np.std(data[cell][plot_y_key]))

  plot_y_key = new_plot_y_key
  #print(plot_y_key)
  return data,plot_y_key

def calculate_std_cell(cell,data, \
                      plot_x_key,plot_y_key):

  new_plot_y_key="Variance " + plot_y_key

  #data[new_plot_y_key] = statistics.pstdev(data[new_plot_y_key])

  data[plot_x_key] = data.get(plot_x_key,[]) + [cell]

  data[new_plot_y_key] = data.get(new_plot_y_key,[]) \
                         + [np.std(data[cell][plot_y_key])]
   #+ [statistics.pvariance(data[cell][plot_y_key])]
    #+ [np.var(data[cell][plot_y_key])]

  #print(data)
  #print(plot_y_key)
  #print(data[cell][plot_y_key])
  #print(np.std(data[cell][plot_y_key]))

  plot_y_key = new_plot_y_key
  #print(plot_y_key)
  return data,plot_y_key

def check_tolerances(data,timesteps,rel_tol,abs_tol):

  #Extract data

  cases_one_multi_cells=list(data.keys())
  data1=data[cases_one_multi_cells[0]]
  data2=data[cases_one_multi_cells[1]]

  #print(data)

  #Reorganize data

  #species_keys=list(data1.keys())
  species_names=list(data1.keys())
  len_timestep=int(len(data1[species_names[0]])/timesteps)
  for j in range(timesteps):
    for i in data1:
      data1_values = data1[i]
      data2_values = data2[i]
      l=j*len_timestep
      r=len_timestep*(1+j)
      out1=data1_values[l:r]
      out2=data2_values[l:r]

      for k in range(len(out1)):
        if(out1[k]-out2[k]!=0):
          out_abs_tol=abs(out1[k]-out2[k])
          #out_rel_tol=abs(out1[k]-out2[k])/(abs(out1[k])+abs(out2[k]))
          if(out_abs_tol>abs_tol):
            print("Exceeding abs_tol",abs_tol,"at",k)
          #if(out_rel_tol>rel_tol):
          #  print("Exceeding rel_tol",rel_tol,"at",k)
          #  print(out1[k],out2[k])

def calculate_NMRSE(data,timesteps):

  #Extract data

  cases_one_multi_cells=list(data.keys())
  data1=data[cases_one_multi_cells[0]]
  data2=data[cases_one_multi_cells[1]]

  #print(data)

  #Reorganize data

  #species_keys=list(data1.keys())
  NRMSEs=[0.]*timesteps
  species_names=list(data1.keys())
  len_timestep=int(len(data1[species_names[0]])/timesteps)
  for j in range(timesteps):
    for i in data1:
      data1_values = data1[i]
      data2_values = data2[i]
      l=j*len_timestep
      r=len_timestep*(1+j)
      out1=data1_values[l:r]
      out2=data2_values[l:r]

      MSE = mean_squared_error(out1, out2)
      RMSE = math.sqrt(MSE)

      aux_out=out1+out2

      #print("aux_out",aux_out)
      #print(RMSE)

      NRMSE=0.
      if(max(aux_out)-min(aux_out)!=0):
        NRMSE=RMSE/(max(aux_out)-min(aux_out))

      if(NRMSEs[j]<NRMSE):
        #print("Concs One-cell:",out1)
        #print("Concs Multi-cells:",out2)
        #print("max",max(aux_out))
        #print("min",min(aux_out))
        #print("RMSE:",RMSE)
        #print("NMRSE:",NRMSE)
        NRMSEs[j]=NRMSE

  #print(NRMSEs)

  return NRMSEs

def calculate_MAPE(data,timesteps):

  #Extract data

  cases_one_multi_cells=list(data.keys())
  species1=data[cases_one_multi_cells[0]]
  species2=data[cases_one_multi_cells[1]]

  #print(data)

  #Reorganize data

  #species_keys=list(species1.keys())
  MAPEs=[0.]*timesteps
  species_names=list(species1.keys())
  len_timestep=int(len(species1[species_names[0]])/timesteps)
  max_err=0.0
  max_tol=1.0E-60
  for j in range(timesteps):
    MAPE=0.0
    #MAPE=1.0E-60
    n=0
    for name in species1:
      data1_values = species1[name]
      data2_values = species2[name]
      l=j*len_timestep
      r=len_timestep*(1+j)
      out1=data1_values[l:r]
      out2=data2_values[l:r]

      for k in range(len(out1)):

        #if(out1[k]<max_tol):
        #  out1[k]=1.0E-60
        #if(out2[k]<max_tol):
        #  out2[k]=1.0E-60

        #Filter low concs
        if abs(out1[k]-out2[k]) < max_tol or (out1[k] == 0):
          err=0.
        else:
          err=abs((out1[k]-out2[k])/out1[k])

        #if(out1[k]==0.0):
        #  out1[k]+=1.0E-60
        #  out2[k]+=1.0E-60
        #  err=abs((out1[k]-out2[k])/out1[k])
          #err=1
        #print(out1[k],out2[k])
        #else:
        #err=abs((out1[k]-out2[k])/out1[k])
        MAPE+=err
        n+=1
        if(err>max_err):
          max_err=err
        #if(err>1):
          #print(name,out1[k],out2[k])
    MAPEs[j]=MAPE/n*100

  print("max_error:"+str(max_err*100)+"%")
  #print(NRMSEs)

  return MAPEs

def calculate_SMAPE(data,timesteps):

  #Extract data

  cases_one_multi_cells=list(data.keys())
  species1=data[cases_one_multi_cells[0]]
  species2=data[cases_one_multi_cells[1]]

  #print(data)

  #Reorganize data

  #species_keys=list(species1.keys())
  SMAPEs=[0.]*timesteps
  species_names=list(species1.keys())
  len_timestep=int(len(species1[species_names[0]])/timesteps)

  for j in range(timesteps):
    num=0.0
    den=0.0
    for key in species1:
      specie1 = species1[key]
      specie2 = species2[key]
      l=j*len_timestep
      r=len_timestep*(1+j)
      out1=specie1[l:r]
      out2=specie2[l:r]

      for k in range(len(out1)):
        try:
          num+=abs(out1[k]-out2[k])
          den+=abs(out1[k])+abs(out2[k])
        except Exception as e:
          print(e,k,l,r,len(out1),len(out2))

    if(den!=0.0):
      SMAPEs[j]=num/den*100

  #print(NRMSEs)

  return SMAPEs

def calculate_BCGPercTimeDataTransfers(data,plot_y_key):

  cases=list(data.keys())

  gpu_exist=False

  for case in cases:
    if("GPU" in case):
      data_timeBiconjGradMemcpy=data[case][plot_y_key]
      data_timeLS=data[case]["timeLS"]
      gpu_exist=True

  if(gpu_exist==False):
    raise Exception("Not GPU case for BCGPercTimeDataTransfers metric")

  datay=[0.]*len(data_timeLS)
  for i in range(len(data_timeLS)):
    datay[i]=data_timeBiconjGradMemcpy[i]/data_timeLS[i]*100

  #print(datay)

  return datay

def calculate_speedup2(data,plot_y_key):

  cases=list(data.keys())

  base_data=data[cases[0]][plot_y_key]
  new_data=data[cases[1]][plot_y_key]

  print(plot_y_key)

  #print(base_data)

  #data[new_plot_y_key] = data.get(new_plot_y_key,[])
  datay=[0.]*len(base_data)
  for i in range(len(base_data)):
    #print(base_data[i],new_data[i], base_data[i]/new_data[i])
    datay[i]=base_data[i]/new_data[i]

  #print(data)

  return datay

def calculate_speedup(cases_multicells_onecell,data,\
                      plot_x_key,plot_y_key):

  new_plot_y_key="Speedup " + plot_y_key

  base_data=data[cases_multicells_onecell[0]][plot_y_key]
  new_data=data[cases_multicells_onecell[1]][plot_y_key]

  #print(base_data)

  data[new_plot_y_key] = data.get(new_plot_y_key,[])
  for i in range(len(base_data)):
    #print(base_data[i],new_data[i], base_data[i]/new_data[i])
    data[new_plot_y_key]=data.get(new_plot_y_key,[]) \
                       + [base_data[i]/new_data[i]]

  #extract timestep: timestep is common in both cases like speedup
  data["timestep"]=data.get("timestep",[]) \
                     + data[cases_multicells_onecell[0]]["timestep"]

  #print(data)

  plot_y_key = new_plot_y_key
  #print(plot_y_key)
  return data,plot_y_key

def plot_speedup_cells(x, y, x_name, y_name, plot_title):

  #print(data)

  #fig = plt.figure(figsize=(7, 4.25))
  fig = plt.figure()
  spec2 = mpl.gridspec.GridSpec(ncols=1, nrows=1, wspace=.35,hspace=.1,bottom=.25,top=.85,left=.1,right=.9)
  axes = fig.add_subplot(spec2[0, 0])
  #axes = fig.add_subplot()
  list_colors = ["r","g","b","c","m","y","k","w"]
  list_markers = ["+","x","*","s","s",".","-"]

  i_color=0

  axes.plot(x,y,color=list_colors[i_color], marker=list_markers[i_color])
  axes.set_ylabel(y_name)
  axes.set_xlabel(x_name)

  #axes.set_yscale('log')
  plt.xticks()
  plt.title(plot_title)

  #data[plot_x_key]=data[plot_x_key]+1

  #saveImage=True
  saveImage=False
  if(saveImage==True):
    plt.savefig('out/plot_speedup_cells.png')
  else:
    plt.show()

def plot_solver_stats_mpi(data, plot_x_key, plot_y_key, plot_title):

  #todo generalize function

  #print(data)

  #fig = plt.figure(figsize=(7, 4.25))
  fig = plt.figure()
  spec2 = mpl.gridspec.GridSpec(ncols=1, nrows=1, wspace=.35,hspace=.1,bottom=.25,top=.85,left=.1,right=.9)
  axes = fig.add_subplot(spec2[0, 0])
  #axes = fig.add_subplot()
  list_colors = ["r","g","b","c","m","y","k","w"]
  list_markers = ["+","x","*","s","s",".","-"]

  i_color=0

  axes.plot(data[plot_x_key],data[plot_y_key],color=list_colors[i_color], marker=list_markers[i_color])
  axes.set_ylabel(plot_y_key)
  axes.set_xlabel(plot_x_key + " [min]")

  #axes.set_yscale('log')
  #axes.set_yscale('logit')
  #axes.set_yscale('symlog')
  plt.xticks()
  plt.title(plot_title)

  #data[plot_x_key]=data[plot_x_key]+1

  #print(data)

  plt.show()

def plot_solver_stats(data, plot_x_key, plot_y_key, plot_title):

  #fig = plt.figure(figsize=(7, 4.25))
  fig = plt.figure()
  spec2 = mpl.gridspec.GridSpec(ncols=1, nrows=1, wspace=.35,hspace=.1,bottom=.25,top=.85,left=.1,right=.9)
  axes = fig.add_subplot(spec2[0, 0])
  #axes = fig.add_subplot()
  list_colors = ["r","g","b","c","m","y","k","w"]
  list_markers = ["+","x","*","s","s",".","-"]

  i_color=0

  axes.plot(data[plot_x_key],data[plot_y_key],color=list_colors[i_color], marker=list_markers[i_color])
  axes.set_ylabel(plot_y_key)
  axes.set_xlabel(plot_x_key)

  #axes.set_yscale('log')
  plt.xticks()
  plt.title(plot_title)

  #data[plot_x_key]=data[plot_x_key]+1

  plt.show()

def plotplt(namex, namey, datax, datay, plot_title, SAVE_PLOT):

  #fig = plt.figure(figsize=(7, 4.25))
  fig = plt.figure()
  spec2 = mpl.gridspec.GridSpec(ncols=1, nrows=1, wspace=.35,hspace=.1,bottom=.25,top=.85,left=.1,right=.9)
  axes = fig.add_subplot(spec2[0, 0])
  #axes = fig.add_subplot()
  list_colors = ["r","g","b","c","m","y","k","w"]
  list_markers = ["+","x","*","s","s",".","-"]

  i_color=0

  axes.plot(datax,datay,color=list_colors[i_color], marker=list_markers[i_color])
  axes.set_ylabel(namey)
  axes.set_xlabel(namex)

  #axes.set_yscale('log')
  plt.xticks()
  plt.title(plot_title)

def plotsns(namex, namey, datax, datay, plot_title, columns):

  #print(sns.__version__)
  sns.set_style("whitegrid")

  #sns.set(font_scale=2)
  #sns.set_context("paper", rc={"font.size":8,"axes.titlesize":8,"axes.labelsize":5})
  sns.set_context("paper", font_scale=1.25)

  fig = plt.figure()
  ax = plt.subplot(111)

  ax.set_xlabel(namex)
  ax.set_ylabel(namey)

  if(columns):

    print("WARNING: Increase plot window manually to take better screenshot")

    datay=list(map(list, zip(*datay)))
    #numpy_array = np.array(datay2)
    #transpose = numpy_array.T
    #datay = transpose.tolist()

    #print(datay)
    #print(datax)

    data = pd.DataFrame(datay, datax, columns=columns)
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


    #ax.subplots_adjust(top=0.25) #not work
    #fig.subplots_adjust(top=0.25)

    #legend out of the plot at the right (problem: plot feels very small)
    #sns.lineplot(data=data, palette="tab10", linewidth=2.5)
    #box=ax.get_position()
    #ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])
    #ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0,labels=columns)

  else:
    ax.set_title(plot_title)
    data = pd.DataFrame(datay, datax)
    sns.lineplot(data=data, palette="tab10", linewidth=2.5, legend=False)

def plot(namex, namey, datax, datay, plot_title, columns, SAVE_PLOT):

  #plotplt(namex, namey, datax, datay, plot_title, SAVE_PLOT)
  plotsns(namex, namey, datax, datay, plot_title, columns)

  if SAVE_PLOT:
    now = datetime.datetime.now()
    dt_string = now.strftime("%d-%m-%Y %H:%M:%S")
    #print(dt_string)
    save_folder = "out/plots/"+namey+" "+plot_title
    if not os.path.exists(save_folder):
      os.makedirs(save_folder)
    save_path=save_folder+"/"+dt_string+".png"
    print("Plot saved to", save_path)
    plt.savefig(save_path)

  plt.show()

def read_solver_stats(file, data):

  with open(file) as f:
    csv_reader = csv.reader(f, delimiter=',')

    i_row = 0

    for row in csv_reader:

      if i_row == 0:
        labels=row
        #print(row)
      else:
        for col in range(len(row)):
          #print(labels[col])
          #print(row[col])
          data[labels[col]]=data.get(labels[col],[]) + [float(row[col])]

      i_row += 1

  #print(data)
  #Normalize accumulative timers
  #for key in data:
  #  for i in range(len(data[key])):
  #    if(i != 0):
  #      data[key][i]=data[key][i]-data[key][i-1]
  #print(data)

def plot_species(file):

  fig = plt.figure(figsize=(7, 4.25))
  spec2 = matplotlib.gridspec.GridSpec(ncols=1, nrows=1, wspace=.35,hspace=.1,bottom=.25,top=.85,left=.1,right=.9)
  axes = fig.add_subplot(spec2[0, 0])

  #file = 'out_01/urban_plume_0001_'
  #file = 'out/monarch_cb05_soa_urban_plume_0001.txt'
  #file = 'out/monarch_cb05_urban_plume_0001.txt'
  #todo fix case cb05 (only working cb05_soa)

  print(file)

  """
  try:
    opts, args = getopt.getopt(sys.argv[1:])
  except getopt.GetoptError:
    print 'test.py -i <inputfile> -o <outputfile>'
    sys.exit(2)
  
  file=sys.argv[1]
  print(file")
  
  """

  #public
  plot_case=2
  if(plot_case == 0):
    n_cells = 1
    n_gases = 4
    n_aerosols = 0
    cell_to_plot = 0
  if(plot_case == 2):
    n_cells = 1
    n_gases = 3
    n_aerosols = 2
    cell_to_plot = 0
  if(plot_case == 4):
    #todo plot 4 fix
    n_cells = 1
    n_gases = 5
    n_aerosols = 2
    cell_to_plot = 0

  #private
  n_species = n_gases + n_aerosols
  n_cols = n_species + 2
  header_size = 1
  i_col_time = 0
  list_colors = ["r","g","b","c","m","y","k","w"]
  list_markers = ["+","x","*","s","s",".","-"]

  with open(file) as f:
    reader = csv.reader(f, delimiter=' ')
    n_rows = len(list(reader))
    #print("n_rows",n_rows)

  with open(file) as f:
    csv_reader = csv.reader(f, delimiter=' ')
    n_rows_cell=int((n_rows-1)/n_cells)
    gases = [[[0 for x in range(n_rows_cell)] for y in range(n_gases)] for z in range(n_cells)]
    aerosols = [[[0 for x in range(n_rows_cell)] for y in range(n_aerosols)] for z in range(n_cells)]
    times = [[0 for x in range(n_rows_cell)] for y in range(n_cells)]
    labels = [0 for y in range(n_cols)]
    i_row = 0

    for row in csv_reader:

      if i_row == 0:
        for i_col in range(n_cols):
          #print(f'Column names are {", ".join(row)}')
          labels[i_col] = row[i_col]
          #labels.append[row[i]]
        i_row += 1
      else:
        #print(f'\t column 0: {row[0]} column 1: {row[1]} ,column 2: {row[2]}.')
        i_cell = (i_row-1) % n_cells
        i_row_cell=int((i_row-1)/n_cells)

        #print("i_cell",i_cell,"i_row_cell",i_row_cell )
        times[i_cell][i_row_cell]=float(row[i_col_time])

        for i_gas in range(n_gases):
          gases[i_cell][i_gas][i_row_cell]=float(row[i_gas+2])#/1000

        for i_aerosol in range(n_aerosols):
          aerosols[i_cell][i_aerosol][i_row_cell]=float(row[i_aerosol+n_gases+2])#/1000

        i_row += 1

  #print(f'Processed {i_row} lines.')
  #print(f' ROW 1 {row[1]}.')

"""

  i_color=0
  for i_gas in range(n_gases):
    #print("times",times[cell_to_plot],"axes",gases[cell_to_plot][i_gas], "labels", labels[i_gas+2])
    axes.plot(times[cell_to_plot], gases[cell_to_plot][i_gas], label=labels[i_gas+2], color=list_colors[i_color], marker=list_markers[i_color])
    i_color+=1

  axes.set_ylabel('Gas mixing ratio (ppm)')
  axes.set_xlabel('Time (min)')
  #axes.set_yscale('log')
  #axes.set_xlim([0,1440])
  #axes.set_ylim([1e-20,1])
  plt.xticks()

  # Now the aerosols on the other axes
  ax2 = axes.twinx()
  for i_aerosol in range(n_aerosols):
    #print("times",times[cell_to_plot],"axes",aerosols[cell_to_plot][i_aerosol], "labels", labels[i_aerosol+n_gases+2])
    ax2.plot(times[cell_to_plot], aerosols[cell_to_plot][i_aerosol], label=labels[i_aerosol+n_gases+2], color=list_colors[i_color], marker=list_markers[i_color])#linestyle='None',
    i_color+=1

  ax2.set_ylabel('Aerosol mass concentration (kg m$^{-3}$)');
  ax2.set_yscale('log')
  #ax2.set_ylim([1e-17,1e-7])

  # Add a legend
  legend_x = .5
  legend_y = .08
  h1, l1 = axes.get_legend_handles_labels()
  h2, l2 = ax2.get_legend_handles_labels()
  fig.legend(handles=h1+h2,labels=l1+l2,bbox_to_anchor=(legend_x, legend_y), \
             loc='center',ncol=3)
  #out_filename = 'partmc_case1_soa_one_particle.pdf'
  #fig.savefig(out_filename)

  plt.show()





  for i in data1:
    data1_values = data1[i]
    data2_values = data2[i]

    for j in range(timesteps):
      len_timestep=int(len(data1_values)/timesteps)

      #data1_timestep=data1[0+j*len_timestep:len_timestep+j*len_timestep]
      #data1_timestep=data1_values[j*len_timestep:len_timestep*(1+j)]
      #data2_timestep=data2_values[j*len_timestep:len_timestep*(1+j)]

      l=j*len_timestep
      r=len_timestep*(1+j)

      #d1=np.array(data1_values[l:r])
      #d2=np.array(data2_values[l:r])

      #d1=np.reshape(d1,(-1,1))
      #d2=np.reshape(d2,(-1,1))

      #scaler.fit(d1)
      #scaler.fit(d2)

      #data1_timestep=scaler.transform(d1)
      #data2_timestep=scaler.transform(d2)

      #d1=(d1/(d1.max()-d1.min()))
      #d2=(d2/(d1.max()-d2.min()))

      #out1.append(data1_timestep.tolist())
      #out2.append(data2_timestep.tolist())

      #print("d1",d1)

      #data1_timestep=preprocessing.normalize(d1)
      #data2_timestep=preprocessing.normalize(d2)

      #out1.append(data1_timestep.tolist())
      #out2.append(data2_timestep.tolist())

      #out2.append([data2_timestep[:]/ \
      #            (max(data2_timestep)-min(data2_timestep))])


      out1.append(data1_values[l:r])
      out2.append(data2_values[l:r])

      #Normalize species


      #MSE = mean_squared_error(data1_timestep, data2_timestep)
      #RMSE = math.sqrt(MSE)

      #data_timestep=data1_timestep+data2_timestep

      #NRMSEs_species[i][j]=RMSE/(max(data_timestep)-min(data_timestep))


  #NRMSE among all species

  print("out1",out1)

  NRMSEs=[0]*timesteps
  for i in range(timesteps):

    MSE = mean_squared_error(out1[i], out2[i])
    RMSE = math.sqrt(MSE)

    aux_out=out1[i]+out2[i]

    print("out1[i]",out1[i])

    NRMSEs[i]=RMSE/(max(aux_out)-min(aux_out))


"""


"""
#Jeff code
n_times = 145
times = get_times(file, n_times)
var = "ISOP"
mix_rat = get_gas_mixing_ratio(file, n_times, var) / 1000
axes.plot(times, mix_rat,'r+',label=var)
var = 'ISOP-P1'
mix_rat = get_gas_mixing_ratio(file, n_times, var) / 1000
axes.plot(times, mix_rat,'gx', label=var)
var = 'ISOP-P2'
mix_rat = get_gas_mixing_ratio(file, n_times, var) / 1000
axes.plot(times, mix_rat,'b*',label=var)

axes.set_ylabel('Gas mixing ratio (ppm)')
axes.set_xlabel('Time (min)')
axes.set_xlim([0,1440])
axes.set_yscale('log')
axes.set_ylim([1e-20,1])

# Now the aerosols on the other axes
ax2 = axes.twinx()
var = 'organic_matter.ISOP-P1_aero'
mass_conc = get_mass_conc(file, n_times,var)
ax2.plot(times,mass_conc, 'ms', label=var)
var = 'organic_matter.ISOP-P2_aero'
mass_conc = get_mass_conc(file, n_times,var)
ax2.plot(times,mass_conc, color='cyan', marker='s', label=var)

ax2.set_ylabel('Aerosol mass concentration (kg m$^{-3}$)');
ax2.set_yscale('log')
ax2.set_ylim([1e-17,1e-7])
    
# Add a legend
legend_x = .5
legend_y = .08
h1, l1 = axes.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
fig.legend(handles=h1+h2,labels=l1+l2,bbox_to_anchor=(legend_x, legend_y), \
     loc='center',ncol=3)
out_filename = 'partmc_case1_soa_one_particle.pdf'
fig.savefig(out_filename)
"""


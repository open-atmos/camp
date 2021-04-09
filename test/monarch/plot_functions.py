import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib
import csv
import numpy as np
import statistics


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

def normalized_timeLS(cases_multicells_onecell,data, cells):

  plot_y_key = "timeLS"
  new_plot_y_key="timeLS" +" normalized"

  #base_data=data[cases_multicells_onecell[0]][plot_y_key]
  #new_data=data[cases_multicells_onecell[1]][plot_y_key]

  #print(base_data)

  for case in cases_multicells_onecell:

    if(case=="one-cell"):
      cells_multiply=cells
    elif(case=="multi-cells"):
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
  axes.set_xlabel(plot_x_key)

  #axes.set_yscale('log')
  plt.xticks()
  plt.title(plot_title)

  #data[plot_x_key]=data[plot_x_key]+1

  #print(data)

  plt.show()

def read_solver_stats(file, data):

  with open(file) as f:
    csv_reader = csv.reader(f, delimiter=',')

    i_row = 0

    for row in csv_reader:

      if i_row == 0:
        labels=row

      else:
        for col in range(len(row)):
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
  #print ("hola")

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
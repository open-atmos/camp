import matplotlib.pyplot as plt
import csv
import sys, getopt
import os
import numpy as np
from pylab import imread,subplot,imshow,show

#exec_str="../../mock_monarch config_simple.json interface_simple.json out/simple"
exec_str="../../mock_monarch config_simple.json interface_simple.json simple"

mpi="yes"
#mpi="no"

#test_monarch1


if mpi=="yes":
  exec_str = 'mpirun -v -np 40 --bind-to core:overload-allowed ../../mock_monarch config_simple.json interface_simple.json simple'
else:
  exec_str = '../../mock_monarch config_simple.json interface_simple.json simple'
"""


#test_monarch2
if mpi=="yes":
  exec_str="mpirun -np 40 --bind-to core:overload-allowed ../../mock_monarch config_monarch_cb05.json interface_monarch_cb05.json monarch_cb05"
else:
  exec_str="../../mock_monarch config_monarch_cb05.json interface_monarch_cb05.json monarch_cb05"
  #exec_str = '../../mock_monarch config_simple.json interface_simple.json simple'
"""

#Read file
file = 'out/exported_counters_0.csv'

#cells = [100,1000]
cells = [10]
cells = [str(cell) for cell in cells]
#cases_multicells_onecell = ["one-cell","multi-cells"]
#cases_multicells_onecell = ["one-cell"]
cases_multicells_onecell = ["multi-cells"]

#SELECT MANUALLY (future:if arch=cpu then select cpu if not gpu)
cases_gpu_cpu = ["cpu"]
#cases_gpu_cpu = ["gpu"]

data = {}

for case in cases_multicells_onecell:

  #data[case]={}

  for cell in cells:

    print exec_str + " "+ cell
    os.system(exec_str + " " + cell + " " + case)

    with open(file) as f:
      csv_reader = csv.reader(f, delimiter=' ')

      i_row = 0

      for row in csv_reader:

        data[row[0]] = data.get(row[0],[]) + [row[1]]
        #data[case][row[0]] = data.get(row[0],[]) + [row[1]]

        i_row += 1

  with open(cases_gpu_cpu[0]+"_"+case+".csv", 'w') as file:
    writer = csv.writer(file, delimiter=' ')

    keys=[]
    for key in data.keys():
      keys.append(key)
    writer.writerow(keys)
    for value in data.values():
      writer.writerow(value)

#print(data)

#Read data... right?

"""
fig = plt.figure(figsize=(7, 4.25))
spec2 = matplotlib.gridspec.GridSpec(ncols=1, nrows=1, wspace=.35,hspace=.1,bottom=.25,top=.85,left=.1,right=.9)
axes = fig.add_subplot(spec2[0, 0])
list_colors = ["r","g","b","c","m","y","k","w"]
list_markers = ["+","x","*","s","s",".","-"]

print "hola"
print data["timeCVode"]

i_color=0
axes.plot(data["timeCVode"],data["n_cells"],color=list_colors[i_color], marker=list_markers[i_color])

axes.set_ylabel('Time (s)')
axes.set_xlabel('Number of cells')
#axes.set_yscale('log')

plt.xticks()
"""

np.random.seed(19680801)
data = np.random.randn(2, 100)

fig, axs = plt.subplots(2, 2, figsize=(5, 5))
axs[0, 0].hist(data[0])
axs[1, 0].scatter(data[0], data[1])
axs[0, 1].plot(data[0], data[1])
axs[1, 1].hist2d(data[0], data[1])

#plt.savefig("mygraph.png")
#image = imread('mygraph.png')
#plt.imshow(image)

plt.show()


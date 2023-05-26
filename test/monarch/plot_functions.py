import matplotlib as mpl

mpl.use('TkAgg')
import matplotlib.pyplot as plt
import csv
import numpy as np
from pandas import DataFrame
import seaborn as sns

def plot_speedup_cells(x, y, x_name, y_name, plot_title):
    # print(data)

    # fig = plt.figure(figsize=(7, 4.25))
    fig = plt.figure()
    spec2 = mpl.gridspec.GridSpec(ncols=1, nrows=1, wspace=.35, hspace=.1, bottom=.25, top=.85, left=.1, right=.9)
    axes = fig.add_subplot(spec2[0, 0])
    # axes = fig.add_subplot()
    list_colors = ["r", "g", "b", "c", "m", "y", "k", "w"]
    list_markers = ["+", "x", "*", "s", "s", ".", "-"]

    i_color = 0

    axes.plot(x, y, color=list_colors[i_color], marker=list_markers[i_color])
    axes.set_ylabel(y_name)
    axes.set_xlabel(x_name)

    # axes.set_yscale('log')
    plt.xticks()
    plt.title(plot_title)

    # data[plot_x_key]=data[plot_x_key]+1

    # saveImage=True
    saveImage = False
    if saveImage:
        plt.savefig('out/plot_speedup_cells.png')
    else:
        plt.show()


def plot_solver_stats_mpi(data, plot_x_key, plot_y_key, plot_title):

    # print(data)

    # fig = plt.figure(figsize=(7, 4.25))
    fig = plt.figure()
    spec2 = mpl.gridspec.GridSpec(ncols=1, nrows=1, wspace=.35, hspace=.1, bottom=.25, top=.85, left=.1, right=.9)
    axes = fig.add_subplot(spec2[0, 0])
    # axes = fig.add_subplot()
    list_colors = ["r", "g", "b", "c", "m", "y", "k", "w"]
    list_markers = ["+", "x", "*", "s", "s", ".", "-"]

    i_color = 0

    axes.plot(data[plot_x_key], data[plot_y_key], color=list_colors[i_color], marker=list_markers[i_color])
    axes.set_ylabel(plot_y_key)
    axes.set_xlabel(plot_x_key + " [min]")

    # axes.set_yscale('log')
    # axes.set_yscale('logit')
    # axes.set_yscale('symlog')
    plt.xticks()
    plt.title(plot_title)

    # data[plot_x_key]=data[plot_x_key]+1

    # print(data)

    plt.show()


def plot_solver_stats(data, plot_x_key, plot_y_key, plot_title):
    # fig = plt.figure(figsize=(7, 4.25))
    fig = plt.figure()
    spec2 = mpl.gridspec.GridSpec(ncols=1, nrows=1, wspace=.35, hspace=.1, bottom=.25, top=.85, left=.1, right=.9)
    axes = fig.add_subplot(spec2[0, 0])
    # axes = fig.add_subplot()
    list_colors = ["r", "g", "b", "c", "m", "y", "k", "w"]
    list_markers = ["+", "x", "*", "s", "s", ".", "-"]

    i_color = 0

    axes.plot(data[plot_x_key], data[plot_y_key], color=list_colors[i_color], marker=list_markers[i_color])
    axes.set_ylabel(plot_y_key)
    axes.set_xlabel(plot_x_key)

    # axes.set_yscale('log')
    plt.xticks()
    plt.title(plot_title)

    # data[plot_x_key]=data[plot_x_key]+1

    plt.show()


def plotplt(namex, namey, datax, datay, plot_title):
    # fig = plt.figure(figsize=(7, 4.25))
    fig = plt.figure()
    spec2 = mpl.gridspec.GridSpec(ncols=1, nrows=1, wspace=.35, hspace=.1, bottom=.25, top=.85, left=.1, right=.9)
    axes = fig.add_subplot(spec2[0, 0])
    # axes = fig.add_subplot()
    list_colors = ["r", "g", "b", "c", "m", "y", "k", "w"]
    list_markers = ["+", "x", "*", "s", "s", ".", "-"]

    i_color = 0

    axes.plot(datax, datay, color=list_colors[i_color], marker=list_markers[i_color])
    axes.set_ylabel(namey)
    axes.set_xlabel(namex)

    # axes.set_yscale('log')
    plt.xticks()
    plt.title(plot_title)


def plotsns(namex, namey, datax, datay, std, plot_title, legend):
    # print(sns.__version__)
    sns.set_style("whitegrid")

    # sns.set(font_scale=2)
    # sns.set_context("paper", rc={"font.size":8,"axes.titlesize":8,"axes.labelsize":5})
    sns.set_context("paper", font_scale=1.25)

    fig = plt.figure()
    ax = plt.subplot(111)

    ax.set_xlabel(namex)
    ax.set_ylabel(namey)

    if legend and len(legend) > 1:

        print("WARNING: Increase plot window manually to take better screenshot")

        #print(datay)
        datay = list(map(list, zip(*datay)))
        std = list(map(list, zip(*std)))
        # numpy_array = np.array(datay2)
        # transpose = numpy_array.T
        # datay = transpose.tolist()

        #print("datay zip",datay)
        # print(datax)
        # print(datax)

        data = DataFrame(datay, datax, columns=legend)
        sns.lineplot(data=data, palette="tab10", linewidth=2.5)

        if len(std):
            #print("datay",datay)
            #print("datay",std)

            y1 = [[0 for i in range(len(datay[0]))] for j in range(len(datay))]
            y2 = [[0 for i in range(len(datay[0]))] for j in range(len(datay))]
            #y1 = [[0 for i in range(len(datay))] for j in range(len(datay[0]))]
            #y2 = [[0 for i in range(len(datay))] for j in range(len(datay[0]))]
            for i in range(len(datay)):
                for j in range(len(datay[0])):
                    y1[i][j] = datay[i][j] - std[i][j]
                    y2[i][j] = datay[i][j] + std[i][j]
                #print("y1[i]",y1[i])

            y1Transpose = np.transpose(y1)
            y1 = y1Transpose.tolist()
            y2Transpose = np.transpose(y2)
            y2 = y2Transpose.tolist()

            for i in range(len(y1)):
                ax.fill_between(datax, y1=y1[i],y2=y2[i], alpha=.5)

        # ax.set_position([box.x0, box.y0 + box.height * 0.1,
        #               box.width, box.height * 0.9])
        # Legend under the plot
        # box = ax.get_position()
        # ax.set_position([box.x0, box.y0 + box.height * 0.1,
        #             box.width, box.height * 0.75])
        # ax.legend(bbox_to_anchor=(0.5, -0.05), loc='upper center',
        #          labels=legend,ncol=4, mode="expand", borderaxespad=0.)
        # fig.subplots_adjust(bottom=0.35)
        # borderaxespad=1. to move down more the legend

        # Legend up the plot
        ax.set_title(plot_title, y=1.06)
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1),
                  ncol=len(legend), labels=legend, frameon=True, shadow=False, borderaxespad=0.)

    else:
        ax.set_title(plot_title)
        datay =datay[0]
        data = DataFrame(datay, datax)

        #sns.catplot(data=data,capsize=.2, palette="YlGnBu_d", linewidth=2.5,kind="point", legend=False)
        #sns.pointplot(data=data, palette="tab10", linewidth=2.5, legend=False)

        #print("plot datay",datay)

        sns.lineplot(data=data, palette="tab10", linewidth=2.5, legend=False)

        if len(std):
            std = std[0]
            #print("std,datay",std,datay)
            y1=[0 for i in range(len(datay))]
            y2=[0 for i in range(len(datay))]
            for i in range(len(datay)):
                y1[i] = datay[i] - std[i]
                y2[i] = datay[i] + std[i]
            ax.fill_between(datax, y1=y1,y2=y2, alpha=.5)

    plt.show()

def plot_species(file):
    fig = plt.figure(figsize=(7, 4.25))
    spec2 = plt.gridspec.GridSpec(ncols=1, nrows=1, wspace=.35, hspace=.1, bottom=.25, top=.85, left=.1,
                                         right=.9)
    axes = fig.add_subplot(spec2[0, 0])

    # file = 'out_01/urban_plume_0001_'
    # file = 'out/monarch_cb05_soa_urban_plume_0001.txt'
    # file = 'out/monarch_cb05_urban_plume_0001.txt'

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

    # public
    plot_case = 2
    if (plot_case == 0):
        n_cells = 1
        n_gases = 4
        n_aerosols = 0
        cell_to_plot = 0
    if (plot_case == 2):
        n_cells = 1
        n_gases = 3
        n_aerosols = 2
        cell_to_plot = 0
    if (plot_case == 4):
        # not implemented
        n_cells = 1
        n_gases = 5
        n_aerosols = 2
        cell_to_plot = 0

    # private
    n_species = n_gases + n_aerosols
    n_cols = n_species + 2
    header_size = 1
    i_col_time = 0
    list_colors = ["r", "g", "b", "c", "m", "y", "k", "w"]
    list_markers = ["+", "x", "*", "s", "s", ".", "-"]

    with open(file) as f:
        reader = csv.reader(f, delimiter=' ')
        n_rows = len(list(reader))
        # print("n_rows",n_rows)

    with open(file) as f:
        csv_reader = csv.reader(f, delimiter=' ')
        n_rows_cell = int((n_rows - 1) / n_cells)
        gases = [[[0 for x in range(n_rows_cell)] for y in range(n_gases)] for z in range(n_cells)]
        aerosols = [[[0 for x in range(n_rows_cell)] for y in range(n_aerosols)] for z in range(n_cells)]
        times = [[0 for x in range(n_rows_cell)] for y in range(n_cells)]
        labels = [0 for y in range(n_cols)]
        i_row = 0

        for row in csv_reader:

            if i_row == 0:
                for i_col in range(n_cols):
                    # print(f'Column names are {", ".join(row)}')
                    labels[i_col] = row[i_col]
                    # labels.append[row[i]]
                i_row += 1
            else:
                # print(f'\t column 0: {row[0]} column 1: {row[1]} ,column 2: {row[2]}.')
                i_cell = (i_row - 1) % n_cells
                i_row_cell = int((i_row - 1) / n_cells)

                # print("i_cell",i_cell,"i_row_cell",i_row_cell )
                times[i_cell][i_row_cell] = float(row[i_col_time])

                for i_gas in range(n_gases):
                    gases[i_cell][i_gas][i_row_cell] = float(row[i_gas + 2])  # /1000

                for i_aerosol in range(n_aerosols):
                    aerosols[i_cell][i_aerosol][i_row_cell] = float(row[i_aerosol + n_gases + 2])  # /1000

                i_row += 1

    # print(f'Processed {i_row} lines.')
    # print(f' ROW 1 {row[1]}.')




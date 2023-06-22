import matplotlib as mpl

mpl.use('TkAgg')
import matplotlib.pyplot as plt
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
        # Legend up the plot
        ax.set_title(plot_title, y=1.06)
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1),
                  ncol=len(legend), labels=legend, frameon=True, shadow=False, borderaxespad=0.)

    else:
        ax.set_title(plot_title)
        datay = datay[0]
        data = DataFrame(datay, datax)
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

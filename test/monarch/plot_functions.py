import matplotlib as mpl

mpl.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame
import seaborn as sns

def plotsns(namex, namey, datax, datay, plot_title, legend):
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
        # numpy_array = np.array(datay2)
        # transpose = numpy_array.T
        # datay = transpose.tolist()

        #print("datay zip",datay)
        # print(datax)
        # print(datax)

        data = DataFrame(datay, datax, columns=legend)
        sns.lineplot(data=data, palette="tab10", linewidth=2.5)

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

    plt.show()

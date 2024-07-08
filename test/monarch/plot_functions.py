import matplotlib as mpl

mpl.use('TkAgg')
import matplotlib.pyplot as plt
from pandas import DataFrame
import seaborn as sns


def plotsns(namex, namey, datax, datay, plot_title, legend):
  sns.set_style("whitegrid")
  sns.set_context("paper", font_scale=1.25)
  ax = plt.subplot(111)
  ax.set_xlabel(namex)
  ax.set_ylabel(namey)
  if legend and len(legend) > 1:
    print("WARNING: Increase plot window manually to take better screenshot")
    datay = list(map(list, zip(*datay)))
    data = DataFrame(datay, datax, columns=legend)
    sns.lineplot(data=data, palette="tab10", linewidth=2.5)
    ax.set_title(plot_title, y=1.06)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1),
              ncol=len(legend), labels=legend, frameon=True, shadow=False, borderaxespad=0.)
  else:
    ax.set_title(plot_title)
    sns.lineplot(x=datax,y=datay, palette="tab10", linewidth=2.5, legend=False)
    ax.set_xticks(datax)
    plt.show()

import matplotlib as mpl

mpl.use('TkAgg')

import netCDF4 as nc
import numpy as np
import pandas as pd
import time
import seaborn as sns
import matplotlib.pyplot as plt


def calculate_speedup(file1_path, file2_path):
    try:
        df1 = pd.read_csv(file1_path)
        df2 = pd.read_csv(file2_path)
        timecvStep_values1 = df1["timeCVode"].values
        timecvStep_values2 = df2["timeCVode"].values
        speedup = timecvStep_values1 / timecvStep_values2
    except:
        print("Fail calculate_speedup")
        speedup=-1
    return speedup

def calculate_nrmse(data1, data2):
    rmsd = np.sqrt(np.mean((data1 - data2) ** 2))
    range_data1 = np.max(data1) - np.min(data1)
    nrmse = (rmsd / range_data1) * 100
    return nrmse


def process_variable(dataset1, dataset2, var_name):
    print(f"Processing variable: {var_name}")
    array1 = np.ma.getdata(dataset1.variables[var_name][-1, ...])
    array2 = np.ma.getdata(dataset2.variables[var_name][-1, ...])

    abs_diff = np.abs(array1 - array2)
    max_array = np.maximum(np.abs(array1), np.abs(array2))
    relative_error = np.where(max_array == 0, 0, abs_diff*100 / max_array)

    mean = np.mean(relative_error)
    print(mean)
    if np.isnan(mean) or mean == 0.:
        if mean == 0:
            print(f"Variable difference mean: {var_name} is 0")
        return False
    else:
        nrmse = calculate_nrmse(array1, array2)
        max_error = np.max(relative_error)
        quantile25 = np.percentile(relative_error, 25)
        quantile50 = np.percentile(relative_error, 50)
        quantile75 = np.percentile(relative_error, 75)
        quantile95 = np.percentile(relative_error, 95)
        median = np.median(relative_error)
        std_dev = np.std(relative_error)

    return var_name, nrmse, std_dev,mean, median, quantile25, quantile50,\
    quantile75, quantile95, max_error, relative_error

src_path="/gpfs/scratch/bsc32/bsc032815/monarch_out/"
file1_path_header = "gpu_80coresTstep6/"
file2_path_header = "gpu_80coresTstep6_gpuPerc98/"

file1 = src_path + file1_path_header + "out/stats.csv"
file2 = src_path + file2_path_header + "out/stats.csv"
speedup = calculate_speedup(file1, file2)
print("Speedup:", speedup)

file1 = src_path + file1_path_header + "nmmb_hst_01_nc4_0000h_00m_00.00s.nc"
file2 = src_path + file2_path_header + "nmmb_hst_01_nc4_0000h_00m_00.00s.nc"
dataset1 = nc.Dataset(file1)
dataset2 = nc.Dataset(file2)
variable_names = dataset1.variables.keys()
summary_data = []
start_time = time.time()

for var_name in variable_names:
    variable = dataset1.variables[var_name]
    if len(variable.dimensions) == 4:
        result = process_variable(dataset1, dataset2, var_name)
        if result:
            summary_data.append(result)

print(f"Execution time: {time.time()-start_time:.2f} seconds")
dataset1.close()
dataset2.close()
if not summary_data:
    print("summary_data is empty")
    exit(1)

summary_table = pd.DataFrame(summary_data, columns=[
    'Variable','NRMSE[%]','Std Dev','Mean','Median','Quantiles 25', 'Quantile 50','Quantile 75',
    'Quantile 95','Max','Relative Error'])

worst_variables = summary_table.nlargest(10, 'NRMSE[%]')
data = [row['Relative Error'].reshape(-1) for _, row in worst_variables.iterrows()]
variable_names = [row['Variable'] for _, row in worst_variables.iterrows()]
worst_variables = worst_variables.drop('Relative Error', axis=1)
highest_nrmse_row = worst_variables.iloc[0]
highest_nrmse_variable = highest_nrmse_row['Variable']
highest_nrmse = highest_nrmse_row['NRMSE[%]']
print("worst_variables:")
print(worst_variables)
print("Config:",file1_path_header,"vs",file2_path_header)
print(f"Highest NRMSE[%]: {highest_nrmse:.2f}")
print("Speedup:", speedup)
plot_nrmse = False
if plot_nrmse:
    plt.figure()
    sns.boxplot(data=data, orient='v', showfliers=False)
    plt.ylabel("Relative Error [%]")
    plt.xticks(range(len(variable_names)), variable_names, rotation=90)
    plt.title("Species with highest NRMSE for MONARCH-CAMP") #4 GPUs 480 time-steps
    plt.show()

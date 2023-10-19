import matplotlib as mpl

mpl.use('TkAgg')

import netCDF4 as nc
import numpy as np
import pandas as pd
import time
import seaborn as sns
import matplotlib.pyplot as plt


def calculate_speedup(file1_path, file2_path):
    df1 = pd.read_csv(file1_path)
    df2 = pd.read_csv(file2_path)
    timecvStep_values1 = df1["timecvStep"].values
    timecvStep_values2 = df2["timecvStep"].values
    speedup = timecvStep_values1 / timecvStep_values2
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
    relative_error = np.where(max_array == 0, 0, abs_diff / max_array)

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


def main():
    file1_path_header = "../../../../monarch_out/cpu_tstep479_O3/"
    file2_path_header = "../../../../monarch_out/gpu_tstep479_O3/"

    # Calculate the speedup
    file1 = file1_path_header + "out/stats.csv"
    file2 = file2_path_header + "out/stats.csv"
    #speedup = calculate_speedup(file1, file2)

    #Path to netCDF
    file1 = file1_path_header + "nmmb_hst_01_nc4_0000h_00m_00.00s.nc"
    file2 = file2_path_header + "nmmb_hst_01_nc4_0000h_00m_00.00s.nc"

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
    pd.set_option('display.max_rows', 100)
    pd.set_option('display.max_columns', 100)
    pd.set_option('display.width', 100)
    pd.set_option('display.max_colwidth', 100)
    #print("Summary Table:", summary_table)

    worst_variables = summary_table.nlargest(6, 'NRMSE[%]')
    highest_nrmse_row = worst_variables.iloc[0]
    highest_nrmse_variable = highest_nrmse_row['Variable']
    plt.figure()
    relative_error = highest_nrmse_row['Relative Error'].reshape(-1)
    sns.boxplot(data=relative_error, showfliers=False)
    #sns.violinplot(data=relative_error, inner="quartiles")
    plt.ylabel("Relative Error")
    plt.xlabel(highest_nrmse_variable)
    plt.show()
    raise
    highest_nrmse = highest_nrmse_row['NRMSE[%]']
    print(f"Highest NRMSE[%]: {highest_nrmse:.2f} for variable: {highest_nrmse_variable}")
    #print("Speedup:", speedup)
    #worst_variables.to_csv("exports/summary_table.csv", index=False)




if __name__ == "__main__":
    main()

import matplotlib as mpl

mpl.use('TkAgg')

import netCDF4 as nc
import numpy as np
import pandas as pd
import time
from runtime_stats_monarch import calculate_speedup


def calculate_nrmse(data1, data2):
    rmsd = np.sqrt(np.mean((data1 - data2) ** 2))
    range_data1 = np.max(data1) - np.min(data1)
    nrmse = (rmsd / range_data1) * 100
    return nrmse

def calculate_rmsdiqr(data1, data2):
    diff = data1 - data2
    iqr = np.percentile(diff, 75) - np.percentile(diff, 25)
    rmsdiqr = (np.sqrt(np.mean(diff ** 2)) / iqr) * 100
    return rmsdiqr


def process_variable(dataset1, dataset2, var_name):
    print(f"Processing variable: {var_name}")
    array1 = np.ma.getdata(dataset1.variables[var_name][-1, ...])
    array2 = np.ma.getdata(dataset2.variables[var_name][-1, ...])

    abs_diff = np.abs(array1 - array2)
    max_array = np.maximum(np.abs(array1), np.abs(array2))
    relative_error = np.where(max_array == 0, 0, abs_diff / max_array)

    mean = np.mean(relative_error)
    if np.isnan(mean) or mean == 0.:
        if mean == 0:
            print(f"Variable difference mean: {var_name} is 0")
        return False
    else:
        nrmse = calculate_nrmse(array1, array2)
        max_error = np.max(relative_error)
        quantiles = np.percentile(relative_error, [25, 50, 75, 95])
        quantiles = [f'{q:.2e}' for q in quantiles]
        median = np.median(relative_error)
        std_dev = np.std(relative_error)
    return var_name, max_error, quantiles, median, mean, std_dev, nrmse


def main():
    file1_path_header = "../../../../cpu_tstep20_O3_monarch_out/"
    file2_path_header = "../../../../gpu_tstep20_O3_monarch_out/"

    # Calculate the speedup
    file1 = file1_path_header + "out/stats.csv"
    file2 = file2_path_header + "out/stats.csv"
    speedup = calculate_speedup(file1, file2)
    print("Speedup:", speedup)

    #Path to netCDF
    file1 = file1_path_header + "nmmb_hst_01_nc4_0000h_00m_00.00s.nc"
    file2 = file2_path_header + "nmmb_hst_01_nc4_0000h_00m_00.00s.nc"

    dataset1 = nc.Dataset(file1)
    dataset2 = nc.Dataset(file2)

    variable_names = dataset1.variables.keys()

    summary_data = []
    start_time = time.time()

    start_processing = False  # DEBUG
    processed_count = 0
    start_processing = True # DISABLE DEBUG
    for var_name in variable_names:
        if var_name == "NO2":  # DEBUG
            start_processing = True
        if start_processing and processed_count < 999:
            variable = dataset1.variables[var_name]
            if len(variable.dimensions) == 4:
                processed_count += 1
                result = process_variable(dataset1, dataset2, var_name)
                if result:
                    summary_data.append(result)

    print(f"Total execution time: {time.time()-start_time:.2f} seconds")
    dataset1.close()
    dataset2.close()
    if not summary_data:
        print("summary_data is empty")
        exit(1)

    summary_table = pd.DataFrame(summary_data, columns=['Variable','Max', \
                                                        'Quantiles[25,50,75,95]', 'Median', 'Mean', 'Std Dev', \
                                                        'NRMSE[%]'])
    pd.set_option('display.max_rows', 100)
    pd.set_option('display.max_columns', 100)
    pd.set_option('display.width', 100)
    pd.set_option('display.max_colwidth', 100)
    #print("Summary Table:", summary_table)

    worst_variables = summary_table.nlargest(999, 'NRMSE[%]')
    #print("worst_variables:\n", worst_variables)
    worst_variables.to_csv("exports/summary_table.csv", index=False)
    print("Speedup:", speedup)


if __name__ == "__main__":
    main()

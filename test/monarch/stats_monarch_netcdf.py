import matplotlib as mpl

mpl.use('TkAgg')

import netCDF4 as nc
import numpy as np
import pandas as pd
import time


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
    array1 = np.ma.getdata(dataset1.variables[var_name][:])
    array2 = np.ma.getdata(dataset2.variables[var_name][:])

    abs_diff = np.abs(array1 - array2)
    relative_error = abs_diff / np.maximum(np.abs(array1), np.abs(array2))

    # Calculate statistics
    mean = np.mean(relative_error)
    if np.isnan(mean) or mean == 0.:
        return False
    else:
        nrmse = calculate_nrmse(array1, array2)
        rmsdiqr = calculate_rmsdiqr(array1, array2)
        quantiles = np.percentile(relative_error, [25, 50, 75, 95])
        quantiles = [f'{q:.2e}' for q in quantiles]  # Format quantiles with exponential notation
        median = np.median(relative_error)
        std_dev = np.std(relative_error)
    return var_name, quantiles, median, mean, std_dev, nrmse, rmsdiqr


def main():
    file1 = "exports/nmmb_cpu_hist005_tstep7.nc"
    file2 = "exports/nmmb_gpu_hist005_tstep7.nc"
    dataset1 = nc.Dataset(file1)
    dataset2 = nc.Dataset(file2)

    variable_names = dataset1.variables.keys()  # Get all variable names

    summary_data = []
    start_time = time.time()

    processed_count = 0
    start_processing = False  # DEBUG
    # start_processing = True # DISABLE DEBUG
    for var_name in variable_names:
        if var_name == "NO2":  # DEBUG
            start_processing = True
        if start_processing and processed_count < 9:
            variable = dataset1.variables[var_name]
            if len(variable.dimensions) == 4:
                processed_count += 1
                result = process_variable(dataset1, dataset2, var_name)
                if result:
                    summary_data.append(result)

    print(f"Total execution time: {time.time()-start_time:.2f} seconds")
    dataset1.close()
    dataset2.close()
    summary_table = pd.DataFrame(summary_data, columns=['Variable', \
                                                        'Quantiles[25,50,75,95]', 'Median', 'Mean', 'Std Dev', \
                                                        'NRMSE[%]', 'RMSDIQR[%]'])
    pd.set_option('display.max_rows', 100)
    pd.set_option('display.max_columns', 100)
    pd.set_option('display.width', 100)
    pd.set_option('display.max_colwidth', 100)
    print("Summary Table:", summary_table)
    # numeric_cols = ['Quantiles[25,50,75,95]', 'Median', 'Mean', 'Std Dev', 'NRMSE']
    # summary_table[numeric_cols] = summary_table[numeric_cols].apply(pd.to_numeric, errors='coerce')
    summary_table.to_csv("summary_table.csv", index=False)

    worst_variables = summary_table.nlargest(3, 'NRMSE[%]')  # Modify the number as needed
    print("worst_variables:\n", worst_variables)


if __name__ == "__main__":
    main()

import matplotlib as mpl

mpl.use('TkAgg')

import netCDF4 as nc
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def process_variable(dataset1, dataset2, var_name):
    print(f"Processing variable: {var_name}")
    array1 = dataset1.variables[var_name][:]
    array2 = dataset2.variables[var_name][:]
    # Convert masked arrays to regular NumPy arrays
    array1 = np.ma.getdata(array1)
    array2 = np.ma.getdata(array2)
    #valid_indices = np.logical_and(~np.isnan(array1), ~np.isnan(array2))
    #array1 = array1[valid_indices]
    #array2 = array2[valid_indices]
    abs_diff = abs(abs(array1) - abs(array2))
    relative_error = abs_diff / np.maximum(np.abs(array1), np.abs(array2))  # Calculate relative error
    # Calculate relative error, replacing NaN values with zeros
    relative_error = np.where(np.logical_or(np.isnan(array1), np.isnan(array2)), 0.0,
                              abs_diff / np.maximum(np.abs(array1), np.abs(array2)))

    # Calculate statistics
    mean = np.mean(relative_error)
    if np.isnan(mean): # or mean == 0.
        #return False
        quantiles = np.zeros(4)
        median = 0
        mean = 0
        std_dev = 0
    else:
        quantiles = np.percentile(relative_error, [25, 50, 75, 95])
        median = np.median(relative_error)
        std_dev = np.std(relative_error)
    return [var_name, quantiles, median, mean, std_dev]

def main():
    file1 = "exports/nmmb_cpu_hist005_tstep7.nc"
    file2 = "exports/nmmb_gpu_hist005_tstep7.nc"
    dataset1 = nc.Dataset(file1)
    dataset2 = nc.Dataset(file2)

    variable_names = dataset1.variables.keys()  # Get all variable names

    summary_data = []

    processed_count = 0
    for var_name in variable_names:
        variable = dataset1.variables[var_name]
        if len(variable.dimensions) == 4 and processed_count < 50:
            result = process_variable(dataset1, dataset2, var_name)
            #if result:
            summary_data.append(result)
            processed_count += 1

    # Close the NetCDF datasets
    dataset1.close()
    dataset2.close()
    # Create a summary table
    summary_table = pd.DataFrame(summary_data, columns=['Variable', 'Quantiles[25,50,75,95]', 'Median', 'Mean', 'Std Dev'])
    print("Summary Table:",summary_table)
    # Convert statistics columns to numeric data types
    numeric_cols = ['Quantiles[25,50,75,95]', 'Median', 'Mean', 'Std Dev']
    summary_table[numeric_cols] = summary_table[numeric_cols].apply(pd.to_numeric, errors='coerce')

    # Find the worst variables based on median relative error
    worst_variables = summary_table.nlargest(5, 'Mean')  # Modify the number as needed
    #print(worst_variables\n: worst_variables)

    # Reshape the DataFrame for plotting
    worst_variables_melted = pd.melt(worst_variables, id_vars=['Variable'], value_vars=['Median'], var_name='Statistic')

    # Create a violin plot for the worst variables
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=worst_variables_melted, x='Variable', y='value', inner='quart', palette='Set1')
    plt.title("Violin Plot of Median Relative Error for variables with largest median relative error")
    plt.xlabel("Variable")
    plt.ylabel("Median Relative Error")
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save the summary_table to a CSV file
    summary_table.to_csv("summary_table.csv", index=False)

    # Save the violin plot as an image file
    plt.savefig("violin_plot.png")

    # Display the plot
    plt.show()

if __name__ == "__main__":
    main()

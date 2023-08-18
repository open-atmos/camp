import matplotlib as mpl

mpl.use('TkAgg')

import netCDF4 as nc
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd

def main():
  variable_names = ['O']  # Add more variable names as needed
  # Load the NetCDF files
  file1 = "exports/nmmb_cpu_hist005_tstep7.nc"
  file2 = "exports/nmmb_gpu_hist005_tstep7.nc"
  # Initialize data array to hold differences for each variable
  all_diffs = []
  summary_data = []
  # Iterate through each variable
  for var_name in variable_names:
    print(f"Processing variable: {var_name}")
    # Load data from file 1
    dataset1 = nc.Dataset(file1)
    array1 = dataset1.variables[var_name][:]
    # Load data from file 2
    dataset2 = nc.Dataset(file2)
    array2 = dataset2.variables[var_name][:]
    abs_diff = abs(array1 - array2)
    # Add differences to the data array
    all_diffs.append(abs_diff.flatten())
    # Calculate statistics
    quantiles = np.percentile(abs_diff, [25, 50, 75, 95])
    median = np.median(abs_diff)
    mean = np.mean(abs_diff)
    std_dev = np.std(abs_diff)
    # Add statistics to summary data
    summary_data.append([var_name, quantiles, median, mean, std_dev])
    # Close the NetCDF datasets
    dataset1.close()
    dataset2.close()
  # Create a summary table
  summary_table = pd.DataFrame(summary_data, columns=['Variable', 'Quantiles[25,50,75,95])', 'Median', 'Mean', 'Std Dev'])
  print("\nSummary Table:")
  print(summary_table)
  # Create a violin plot for all variables side by side
  plt.figure(figsize=(12, 6))
  plt.violinplot(all_diffs, showmedians=True)
  plt.title("Difference Array Violin Plot")
  plt.ylabel("Difference Values")
  plt.xticks(np.arange(1, len(variable_names) + 1), variable_names)
  plt.xlabel("Variables")
  #plt.show()

if __name__ == "__main__":
  main()

# Compute the difference between the arrays
#dimension_names1 = dataset1['O'].dimensions
#dimension_names2 = dataset2['O'].dimensions
#print("Array 1 dimension names:", dimension_names1)
#print("Array 2 dimension names:", dimension_names2)
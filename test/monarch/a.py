import matplotlib as mpl

mpl.use('TkAgg')

import netCDF4 as nc
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def calculate_statistics(data):
  mean = np.mean(data)
  median = np.median(data)
  quantiles = np.percentile(data, [25, 50, 75])
  return mean, median, quantiles

def main():
  # Open the first NetCDF file
  file_path1 = 'exports/nmmb_cpu_hist005_tstep7.nc'
  dataset1 = nc.Dataset(file_path1)
  array1 = dataset1['O'][:]  # Replace 'variable_name' with the actual variable name

  # Open the second NetCDF file
  file_path2 = 'exports/nmmb_gpu_hist005_tstep7.nc'
  dataset2 = nc.Dataset(file_path2)
  array2 = dataset2['O'][:]  # Replace 'variable_name' with the actual variable name

  # Calculate the difference between the two arrays
  difference = array1 - array2

  # Calculate statistics for the difference array
  mean, median, quantiles = calculate_statistics(difference)

  # Print statistics
  print("Statistics for the Difference Array:")
  print(f"Mean: {mean}")
  print(f"Median: {median}")
  print(f"Quantiles (25th, 50th, 75th percentiles): {quantiles}")

  # Create a violin plot
  sns.set(style="whitegrid")
  plt.figure(figsize=(8, 6))
  sns.violinplot(data=difference.flatten(), color="skyblue", inner="quartile")
  plt.title("Violin Plot of Difference Array")
  plt.xlabel("Difference")
  plt.ylabel("Value")
  plt.show()

  # Close the NetCDF datasets
  dataset1.close()
  dataset2.close()

if __name__ == "__main__":
  main()

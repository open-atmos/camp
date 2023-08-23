import numpy as np
import pandas as pd


def calculate_speedup(file1_path, file2_path):
    # Read the CSV files
    df1 = pd.read_csv(file1_path)
    df2 = pd.read_csv(file2_path)

    # Get the "timecvStep" column values as numpy arrays
    timecvStep_values1 = df1["timecvStep"].values
    timecvStep_values2 = df2["timecvStep"].values

    print(timecvStep_values1)
    print(timecvStep_values2)

    # Calculate the mean using numpy
    mean_time1 = np.mean(timecvStep_values1)
    mean_time2 = np.mean(timecvStep_values2)

    # Calculate the speedup
    speedup = mean_time1 / mean_time2

    return speedup


if __name__ == "__main__":
    # Paths to the CSV files
    file1 = "../../../../11_cpu_tstep6_O2_monarch_out/out/stats.csv"
    file2 = "../../../../gpu_tstep6_O2_monarch_out/out/stats.csv"

    # Calculate the speedup
    speedup = calculate_speedup(file1, file2)
    print("Speedup:", speedup)

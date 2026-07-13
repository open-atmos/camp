exec(open('./pyeslib.py').read())
#from pyeslib.py import ComputeStatistics

import matplotlib as mpl

mpl.use("TkAgg")

import netCDF4 as nc
import numpy as np
import pandas as pd
import time
import matplotlib as plt
import os


def search_timeChem(file_path):
    last_value = None
    with open(file_path, 'r') as file:
        for line in file:
            index = line.find("chemistry=")
            if index != -1:
                # Extract the substring starting after "chemistry="
                after_str = line[index + len("chemistry="):].strip()
                # Split the string by whitespace and get the first number
                parts = after_str.split()
                if parts:
                    try:
                        last_value = float(
                            parts[0]
                        )  # Read the first number after "chemistry="
                    except ValueError:
                        continue  # Ignore lines with invalid formatting
    return last_value  # Return the last found value or None if not found


def calculate_speedup_chem(path_to_all_days1, path_to_all_days2):
    time1 = 0
    time2 = 0
    #day="2016072112"
    for day in os.listdir(path_to_all_days1):
        file1 = path_to_all_days1 + day + "/nmm_rrtm.out_0"
        file2 = path_to_all_days2 + day + "/nmm_rrtm.out_0"
        try:
            time1 += search_timeChem(file1)
            time2 += search_timeChem(file2)
            #print("Day:", day, "Time1:", time1, "Time2:", time2)
        except TypeError:
            pass

    try:
        speedup = time1 / time2
    except ZeroDivisionError:
        speedup = 0
    return speedup


def search_timeMonarch(file_path):
    last_value = None
    with open(file_path, 'r') as file:
        for line in file:
            index = line.find("total_integration_tim=")
            if index != -1:
                # Extract the substring starting after "chemistry="
                after_str = line[index +
                                 len("total_integration_tim="):].strip()
                # Split the string by whitespace and get the first number
                parts = after_str.split()
                if parts:
                    try:
                        last_value = float(
                            parts[0]
                        )  # Read the first number after "chemistry="
                    except ValueError:
                        continue  # Ignore lines with invalid formatting
    return last_value  # Return the last found value or None if not found


def calculate_speedup_monarch(path_to_all_days1, path_to_all_days2):
    time1 = 0
    time2 = 0
    for day in os.listdir(path_to_all_days1):
        file1 = path_to_all_days1 + day + "/nmm_rrtm.out_0"
        file2 = path_to_all_days2 + day + "/nmm_rrtm.out_0"
        try:
            time1 += search_timeMonarch(file1)
            time2 += search_timeMonarch(file2)
            #print("Day:", day, "Time1:", time1, "Time2:", time2)
        except TypeError:
            pass
    try:
        speedup = time1 / time2
    except ZeroDivisionError:
        speedup = 0
    return speedup


def search_timecvode(file_path):
    with open(file_path, 'r') as file:
        for line in file:
            index = line.find("timeCVode:")
            if index != -1:
                # Extract substring from the position after "timeCVode:" to the end of the line
                result = line[index + len("timeCVode:"):].strip()
                return float(result)
    return None  # Return None if no match is found


def calculate_speedup_CAMPSolver(path_to_all_days1, path_to_all_days2):
    time1 = 0
    time2 = 0
    for day in os.listdir(path_to_all_days1):
        file1 = path_to_all_days1 + day + "/nmm_rrtm.out_0"
        file2 = path_to_all_days2 + day + "/nmm_rrtm.out_0"
        try:
            time1 += search_timecvode(file1)
            time2 += search_timecvode(file2)
        except TypeError:
            continue
    try:
        speedup = time1 / time2
    except ZeroDivisionError:
        speedup = 0
    return speedup


def calculate_speedup_one_day(header1, header2):
    file1 = header1 + "nmm_rrtm.out_0"
    file2 = header2 + "nmm_rrtm.out_0"
    match1 = search_timecvode(file1)
    match2 = search_timecvode(file2)
    try:
        speedup = match1 / match2
    except TypeError:
        speedup = 0
    return speedup


def calculate_speedups(path_to_all_days1, path_to_all_days2, header1, header2,
                       day):
    try:
        speedup_chem = calculate_speedup_chem(path_to_all_days1,
                                              path_to_all_days2)
        print("Speedup CHEM:", speedup_chem)

        speedup_MONARCH = calculate_speedup_monarch(path_to_all_days1,
                                                    path_to_all_days2)
        print("Speedup MONARCH:", speedup_MONARCH)

        speedup = calculate_speedup_CAMPSolver(path_to_all_days1,
                                               path_to_all_days2)
        print("Speedup CAMP:", speedup)
    except FileNotFoundError as e:
        speedup_chem = 0
        speedup_MONARCH = 0
        speedup = 0
        print(e)

    speedup_one_day = calculate_speedup_one_day(header1, header2)
    print("Speedup CAMP Day:", day, speedup_one_day)

    return speedup, speedup_MONARCH, speedup_chem


def get_all_cells(dataset1, dataset2, var_name):
    array1 = np.ma.getdata(dataset1.variables[var_name][-1, ...])
    array2 = np.ma.getdata(dataset2.variables[var_name][-1, ...])
    return array1, array2


def get_surface_cells(dataset1, dataset2, var_name):
    #surface are at the bottom of last vertical layer "lm"
    array1 = np.ma.getdata(dataset1.variables[var_name][-1, -1, ...])
    array2 = np.ma.getdata(dataset2.variables[var_name][-1, -1, ...])
    return array1, array2


def process_variable(dataset1, dataset2, var_name):
    try:
        #array1, array2 = get_all_cells(dataset1, dataset2, var_name)
        array1, array2 = get_surface_cells(dataset1, dataset2, var_name)
    except KeyError:
        return (
            "Unknown",
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        )
    metrics = ['MB', 'nMB', 'RMSE', 'nRMSE', 'PCC', 'N', 'slope', 'Mo', 'Mm']
    #print(array1,array1.shape)
    #print(dataset1.variables[var_name].shape)
    #print(dataset1.variables[var_name].dimensions)
    print(var_name)
    try:
        ComputeStatistics(array1.flatten(),
                          array2.flatten(),
                          metrics,
                          fmt='10.2e',
                          print_result=True)
    except Exception as e:
        print("Error processing variable:", var_name)
        print(e)
        return False
    diff = array2 - array1
    # Avoid zero values
    non_zero_mask = array1 != 0
    relative_error = (diff[non_zero_mask] * 100) / array1[non_zero_mask]

    mean = np.mean(relative_error)
    if np.isnan(mean) or mean == 0.0:
        return False
    else:
        rmse = np.sqrt(np.mean(np.square(np.float64(diff))))
        quantile25 = np.percentile(relative_error, 25)
        quantile50 = np.percentile(relative_error, 50)
        quantile75 = np.percentile(relative_error, 75)
        quantile95 = np.percentile(relative_error, 95)
        median = np.median(relative_error)
        std_dev = np.std(relative_error)

    return (
        var_name,
        rmse,
        median,
        mean,
        quantile25,
        quantile50,
        quantile75,
        quantile95,
        std_dev,
        relative_error,
    )


def get_error(day, expid1, expid2, path_to_all_days1, path_to_all_days2):
    header1 = path_to_all_days1 + day + "/"
    header2 = path_to_all_days2 + day + "/"

    speedup = 0
    speedup_MONARCH = 0
    speedup_chem = 0
    highest_rmse = 0
    speedup, speedup_MONARCH, speedup_chem = calculate_speedups(
        path_to_all_days1, path_to_all_days2, header1, header2, day)

    file1 = header1 + "MONARCH_d01_" + day + ".nc"
    file2 = header2 + "MONARCH_d01_" + day + ".nc"
    print("file1", file1)
    print("file2", file2)

    dataset1 = nc.Dataset(file1)
    dataset2 = nc.Dataset(file2)
    variable_names = dataset1.variables.keys()
    summary_data = []

    for var_name in variable_names:
        variable = dataset1.variables[var_name]
        # Print only relevant species
        if len(variable.dimensions) == 4 and (
                var_name == "O3" or var_name == "NO2" or var_name == "HNO3"
                or var_name == "FORM" or var_name == "ALD2" or var_name == "OH"
                or var_name == "H2O2" or var_name == "CO" or var_name == "O1D"
                or var_name == "NO3"):
            result = process_variable(dataset1, dataset2, var_name)
            if result:
                summary_data.append(result)

    dataset1.close()
    dataset2.close()

    if not summary_data:
        print("summary_data is empty")
        return -1

    summary_table = pd.DataFrame(
        summary_data,
        columns=[
            "Variable",
            "RMSE[%]",
            "Median",
            "Mean",
            "Quantile 25",
            "Quantile 50",
            "Quantile 75",
            "Quantile 95",
            "Std Dev",
            "Relative Error",
        ],
    )

    worst_variables = summary_table.nlargest(10, "RMSE[%]")
    data = [
        row["Relative Error"].reshape(-1)
        for _, row in worst_variables.iterrows()
    ]
    variable_names = [row["Variable"] for _, row in worst_variables.iterrows()]
    worst_variables = worst_variables.drop("Relative Error", axis=1)
    highest_rmse_row = worst_variables.iloc[0]
    highest_rmse = highest_rmse_row["RMSE[%]"]
    #print(worst_variables)
    # print("Config:",file1_path_header,"vs",file2_path_header)
    #plot_rmse = False
    #if plot_rmse:
    #    plt.figure()
    #    sns.boxplot(data=data, orient="v", showfliers=False)
    #    plt.ylabel("Relative Error [%]")
    #    plt.xticks(range(len(variable_names)), variable_names, rotation=90)
    #    plt.title("Species with highest RMSE for MONARCH-CAMP"
    #              )  # 4 GPUs 480 time-steps
    #    plt.show()
    print(
        "%s vs %s, error: %.2e, Speedup CAMP: %.1f, Speedup CHEM: %.1f, Speedup MONARCH: %.1f"
        %
        (expid1, expid2, highest_rmse, speedup, speedup_chem, speedup_MONARCH))


def get_errors():
    expid1 = "gpu_a9lg"  #monarch-camp
    expid2 = "a9mb"
    #    expid2 = "ebi_preDry_preMulti"
    #a86c: #monarch-ebi
    #a5lc: #GPU branch
    path_to_all_days1 = "/gpfs/scratch/bsc32/bsc032815/" + expid1 + "/nmmb-monarch/ARCHIVE/000/"  # Temporal files, read from alogin
    path_to_all_days2 = "/gpfs/scratch/bsc32/bsc032815/" + expid2 + "/nmmb-monarch/ARCHIVE/000/"  # Temporal files, read from alogin
    #path_to_all_days1 = "/esarchive/exp/monarch/" + expid1 + "/original_files/000/" # Archived files, read from hub
    #path_to_all_days2 = "/esarchive/exp/monarch/" + expid2 + "/original_files/000/" # Archived files, read from hub
    folders = sorted(os.listdir(path_to_all_days1))
    day = "2016072112"
    #2016072112: first day
    #2016080812: last day
    print("Last folder:", day)
    get_error(day, expid1, expid2, path_to_all_days1, path_to_all_days2)


get_errors()

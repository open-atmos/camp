import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt
import pandas as pd

def get_values_same_timestep(timestep_to_plot, mpiProcessesList, \
                             data, plot_x_key, plot_y_key):
    new_data = {}
    new_data[plot_x_key] = mpiProcessesList
    new_data[plot_y_key] = []
    for i in range(len(mpiProcessesList)):
        # print(base_data[i],new_data[i], base_data[i]/new_data[i])
        # print(len(data["timestep"]), len(data["mpiProcesses"]))
        step_len = int(len(data["timestep"]) / len(mpiProcessesList))
        j = i * step_len + timestep_to_plot
        # print("data[new_plot_y_key][j]",data[plot_y_key][j])
        new_data[plot_y_key].append(data[plot_y_key][j])

    return new_data


def calculate_computational_timeLS2(data, plot_y_key):
    data_timeBiconjGradMemcpy = data[plot_y_key]
    data_timeLS = data["timeLS"]
    for i in range(len(data_timeLS)):
        data_timeLS[i] = data_timeLS[i] - data_timeBiconjGradMemcpy[i]

    return data


def calculate_computational_timeLS(data, plot_y_key, case):
    if ("GPU" in case):
        data_timeBiconjGradMemcpy = data[case][plot_y_key]
        data_timeLS = data[case]["timeLS"]
        for i in range(len(data_timeLS)):
            data_timeLS[i] = data_timeLS[i] - data_timeBiconjGradMemcpy[i]

    return data


def calculate_percentages_solveCVODEGPU(din):
    # data_aux={}
    print(din)
    # print(data["timeNewtonIteration"])

    percNum = 100  # 1

    data = {}

    data["timeNewtonIteration"] = []
    data["timeJac"] = []
    data["timelinsolsetup"] = []
    data["timecalc_Jac"] = []
    data["timeRXNJac"] = []
    data["timef"] = []
    data["timeguess_helper"] = []

    for i in range(len(din["timecvStep"])):
        if (din["timecvStep"][i] != 0):
            data["timeNewtonIteration"].append(din["timeNewtonIteration"][i] / din["timecvStep"][i] * percNum)

            data["timeJac"].append(din["timeJac"][i] / din["timecvStep"][i] * percNum)
            data["timelinsolsetup"].append(din["timelinsolsetup"][i] / din["timecvStep"][i] * percNum)
            data["timecalc_Jac"].append(din["timecalc_Jac"][i] / din["timecvStep"][i] * percNum)
            data["timeRXNJac"].append(din["timeRXNJac"][i] / din["timecvStep"][i] * percNum)
            data["timef"].append(din["timef"][i] / din["timecvStep"][i] * percNum)
            data["timeguess_helper"].append(din["timeguess_helper"][i] / din["timecvStep"][i] * percNum)


    print("calculate_percentages_solveCVODEGPU")

    return data


def normalize_by_countercvStep_and_cells(data, plot_y_key, cells, case):
    if ("One-cell" in case):
        print("One-cell")
        cells_multiply = cells
    elif ("Multi-cells" in case):
        print("Multi-cells")
        cells_multiply = 1
    else:
        raise Exception("normalize_by_countercvStep_and_cells case without One-cell or Multi-cells key name")

    for i in range(len(data[plot_y_key])):
        data[plot_y_key][i] = data[plot_y_key][i] \
                              / data["countercvStep"][i] * cells_multiply

    return data


def normalized_timeLS(new_plot_y_key, cases_multicells_onecell, data, cells):
    plot_y_key = "timeLS"

    # base_data=data[cases_multicells_onecell[0]][plot_y_key]
    # new_data=data[cases_multicells_onecell[1]][plot_y_key]

    # print(base_data)

    for case in cases_multicells_onecell:

        if case == "One-cell":
            cells_multiply = cells
        else:
            cells_multiply = 1

        data[case][new_plot_y_key] = []
        for i in range(len(data[case][plot_y_key])):
            # print(base_data[i],new_data[i], base_data[i]/new_data[i])
            data[case][new_plot_y_key].append(data[case][plot_y_key][i] \
                                              / data[case]["counterLS"][i] * cells_multiply)

    # extract timestep: timestep is common in both cases like speedup
    # data["timestep"]=data.get("timestep",[]) \
    #                 + data[cases_multicells_onecell[0]]["timestep"]

    return data, new_plot_y_key


def calculate_mean_cell(cell, data, \
                        plot_x_key, plot_y_key):
    new_plot_y_key = "Mean " + plot_y_key

    data[plot_x_key] = data.get(plot_x_key, []) + [cell]

    data[new_plot_y_key] = data.get(new_plot_y_key, []) \
                           + [np.mean(data[cell][plot_y_key])]
    # print(plot_y_key)
    # print(data[cell][plot_y_key])
    # print(np.std(data[cell][plot_y_key]))
    plot_y_key = new_plot_y_key
    # print(plot_y_key)
    return data, plot_y_key


def calculate_std_cell(cell, data, \
                       plot_x_key, plot_y_key):
    new_plot_y_key = "Variance " + plot_y_key

    # data[new_plot_y_key] = statistics.pstdev(data[new_plot_y_key])

    data[plot_x_key] = data.get(plot_x_key, []) + [cell]

    data[new_plot_y_key] = data.get(new_plot_y_key, []) \
                           + [np.std(data[cell][plot_y_key])]
    # + [statistics.pvariance(data[cell][plot_y_key])]
    # + [np.var(data[cell][plot_y_key])]

    # print(data)
    # print(plot_y_key)
    # print(data[cell][plot_y_key])
    # print(np.std(data[cell][plot_y_key]))

    plot_y_key = new_plot_y_key
    # print(plot_y_key)
    return data, plot_y_key


def check_tolerances(data, timesteps, rel_tol, abs_tol):
    # Extract data

    cases_one_multi_cells = list(data.keys())
    data1 = data[cases_one_multi_cells[0]]
    data2 = data[cases_one_multi_cells[1]]

    # Reorganize data

    # species_keys=list(data1.keys())
    species_names = list(data1.keys())
    len_timestep = int(len(data1[species_names[0]]) / timesteps)
    for j in range(timesteps):
        for i in data1:
            data1_values = data1[i]
            data2_values = data2[i]
            l = j * len_timestep
            r = len_timestep * (1 + j)
            out1 = data1_values[l:r]
            out2 = data2_values[l:r]

            for k in range(len(out1)):
                if (out1[k] - out2[k] != 0):
                    out_abs_tol = abs(out1[k] - out2[k])
                    # out_rel_tol=abs(out1[k]-out2[k])/(abs(out1[k])+abs(out2[k]))
                    if (out_abs_tol > abs_tol):
                        print("Exceeding abs_tol", abs_tol, "at", k)
                    # if(out_rel_tol>rel_tol):
                    #  print("Exceeding rel_tol",rel_tol,"at",k)
                    #  print(out1[k],out2[k])


def calculate_NMRSE(data, timesteps):
    # Extract data

    cases_one_multi_cells = list(data.keys())
    data1 = data[cases_one_multi_cells[0]]
    data2 = data[cases_one_multi_cells[1]]

    # Reorganize data

    # species_keys=list(data1.keys())
    NRMSEs = [0.] * timesteps
    species_names = list(data1.keys())
    len_timestep = int(len(data1[species_names[0]]) / timesteps)
    for j in range(timesteps):
        for i in data1:
            data1_values = data1[i]
            data2_values = data2[i]
            l = j * len_timestep
            r = len_timestep * (1 + j)
            out1 = data1_values[l:r]
            out2 = data2_values[l:r]

            MSE = mean_squared_error(out1, out2)
            RMSE = sqrt(MSE)

            aux_out = out1 + out2

            # print("aux_out",aux_out)
            # print(RMSE)

            NRMSE = 0.
            if (max(aux_out) - min(aux_out) != 0):
                NRMSE = RMSE / (max(aux_out) - min(aux_out))

            if (NRMSEs[j] < NRMSE):
                # print("Concs One-cell:",out1)
                # print("Concs Multi-cells:",out2)
                # print("max",max(aux_out))
                # print("min",min(aux_out))
                # print("RMSE:",RMSE)
                # print("NMRSE:",NRMSE)
                NRMSEs[j] = NRMSE

    # print(NRMSEs)

    return NRMSEs


def calculate_MAPE(data, timesteps, max_tol):
    # Extract data

    cases_one_multi_cells = list(data.keys())
    species1 = data[cases_one_multi_cells[0]]
    species2 = data[cases_one_multi_cells[1]]

    # Reorganize data
    # species_keys=list(species1.keys())
    MAPEs = [0.] * timesteps
    species_names = list(species1.keys())
    len_timestep = int(len(species1[species_names[0]]) / timesteps)
    #print("species1[species_names[0]]",species1[species_names[0]])
    #print("len_timestep",len_timestep,"timesteps",timesteps)
    max_err = 0.0
    max_err_name = ""
    max_err_i_name = 0
    max_err_abs = 0.0
    max_err_name_abs = ""
    max_err_i_name_abs = 0
    concs_above_tol = 0
    concs_below_tol = 0
    concs_are_zero = 0
    concs_are_equal = 0
    for j in range(timesteps):
        MAPE = 0.0
        # MAPE=1.0E-60
        n = 0
        for i_name, name in enumerate(species1):
            data1_values = species1[name]
            data2_values = species2[name]
            l = j * len_timestep
            r = len_timestep * (1 + j)
            out1 = data1_values[l:r]
            out2 = data2_values[l:r]
            for k in range(len(out1)):
                err = 0.
                err_abs = abs(out1[k] - out2[k])
                if err_abs < max_tol:
                    concs_below_tol = concs_below_tol + 1
                    if out1[k] == out2[k]:
                        concs_are_equal = concs_are_equal +1
                    else:
                        if out1[k]!=0:
                            err = abs((out1[k] - out2[k]) / out1[k])
                        #print("out1[k]",out1[k],"out2[k]",out1[k],"abs",abs(out1[k] - out2[k]),
                        #      "k",k,"out1",out1,"out2",out2)
                elif out1[k] == 0:
                    concs_are_zero = concs_are_zero + 1
                else:
                    concs_above_tol = concs_above_tol + 1
                    err = abs((out1[k] - out2[k]) / out1[k])
                    MAPE += err
                n += 1
                if err > max_err:
                    max_err = err
                    max_err_name = name
                    max_err_i_name = i_name
                if err_abs > max_err_abs:
                    max_err_abs = err_abs
                    max_err_name_abs = name
                    max_err_i_name_abs = i_name
        MAPEs[j] = MAPE / n * 100

    if concs_are_zero > concs_below_tol + concs_above_tol:
        print("Error: More concs are zero than real values, check for errors")
        raise
    max_err=format(max_err*100, '.2e')
    max_err_abs=format(max_err_abs, '.2e')
    print("relative max_error:" + str(max_err) + "% at: "
          + max_err_name + " with id: " + str(max_err_i_name) +
          " absolute max_error:" + str(max_err_abs) + " at: "
          + max_err_name_abs + " with id: " + str(max_err_i_name_abs)
          , "concs_above_tol", concs_above_tol, "concs_below_tol", concs_below_tol
          , "concs_are_equal", concs_are_equal,  "concs_are_zero", concs_are_zero)
    return MAPEs


def calculate_SMAPE(data, timesteps):
    # Extract data

    cases_one_multi_cells = list(data.keys())
    species1 = data[cases_one_multi_cells[0]]
    species2 = data[cases_one_multi_cells[1]]

    # print(data)

    # Reorganize data

    # species_keys=list(species1.keys())
    SMAPEs = [0.] * timesteps
    species_names = list(species1.keys())
    len_timestep = int(len(species1[species_names[0]]) / timesteps)

    for j in range(timesteps):
        num = 0.0
        den = 0.0
        for key in species1:
            specie1 = species1[key]
            specie2 = species2[key]
            l = j * len_timestep
            r = len_timestep * (1 + j)
            out1 = specie1[l:r]
            out2 = specie2[l:r]

            for k in range(len(out1)):
                try:
                    num += abs(out1[k] - out2[k])
                    den += abs(out1[k]) + abs(out2[k])
                except Exception as e:
                    print(e, k, l, r, len(out1), len(out2))

        if (den != 0.0):
            SMAPEs[j] = num / den * 100

    # print(NRMSEs)

    return SMAPEs


def calculate_BCGPercTimeDataTransfers(data, plot_y_key):
    cases = list(data.keys())

    gpu_exist = False

    for case in cases:
        print("WARNING: Deprecated checking in calculate_BCGPercTimeDataTransfers")
        if ("GPU" in case):
            data_timeBiconjGradMemcpy = data[case][plot_y_key]
            data_timeLS = data[case]["timeLS"]
            gpu_exist = True

    if (gpu_exist == False):
        raise Exception("Not GPU case for BCGPercTimeDataTransfers metric")

    datay = [0.] * len(data_timeLS)
    for i in range(len(data_timeLS)):
        datay[i] = data_timeBiconjGradMemcpy[i] / data_timeLS[i] * 100

    # print(datay)

    return datay


def calculate_speedup(data, plot_y_key):
    cases = list(data.keys())

    base_data = data[cases[0]][plot_y_key]
    new_data = data[cases[1]][plot_y_key]

    # print("calculate_speedup start",data)
    # print(plot_y_key)

    # data[new_plot_y_key] = data.get(new_plot_y_key,[])
    datay = [0.] * len(base_data)
    for i in range(len(base_data)):
        # print(base_data[i],new_data[i], base_data[i]/new_data[i])
        datay[i] = base_data[i] / new_data[i]

    # print(datay)

    return datay

def read_solver_stats(file, nrows):
    #start = time.time()
    df = pd.read_csv(file, nrows=nrows)
    data = df.to_dict('list')
    #print("time", time.time() - start)
    return data
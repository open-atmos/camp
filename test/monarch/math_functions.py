import csv
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt

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

    for i in range(len(din["timesolveCVODEGPU"])):
        if (din["timesolveCVODEGPU"][i] != 0):
            data["timeNewtonIteration"].append(din["timeNewtonIteration"][i] / din["timesolveCVODEGPU"][i] * percNum)

            data["timeJac"].append(din["timeJac"][i] / din["timesolveCVODEGPU"][i] * percNum)
            data["timelinsolsetup"].append(din["timelinsolsetup"][i] / din["timesolveCVODEGPU"][i] * percNum)
            data["timecalc_Jac"].append(din["timecalc_Jac"][i] / din["timesolveCVODEGPU"][i] * percNum)
            data["timeRXNJac"].append(din["timeRXNJac"][i] / din["timesolveCVODEGPU"][i] * percNum)
            data["timef"].append(din["timef"][i] / din["timesolveCVODEGPU"][i] * percNum)
            data["timeguess_helper"].append(din["timeguess_helper"][i] / din["timesolveCVODEGPU"][i] * percNum)

            # data["timeNewtonIteration"]=din["timeNewtonIteration"][i]/data["timesolveCVODEGPU"][i]*percNum
            # data["timeJac"][i]=din["timeJac"][i]/data["timesolveCVODEGPU"][i]*percNum
            # data["timelinsolsetup"][i]=din["timelinsolsetup"][i]/data["timesolveCVODEGPU"][i]*percNum
            # data["timecalc_Jac"][i]=din["timecalc_Jac"][i]/data["timesolveCVODEGPU"][i]*percNum
            # data["timeRXNJac"][i]=din["timeRXNJac"][i]/data["timesolveCVODEGPU"][i]*percNum
            # data["timef"][i]=din["timef"][i]/data["timesolveCVODEGPU"][i]*percNum
            # data["timeguess_helper"][i]=din["timeguess_helper"][i]/data["timesolveCVODEGPU"][i]*percNum

    # print(data["timeNewtonIteration"])

    print("calculate_percentages_solveCVODEGPU")
    print(data)

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

    # print(data)

    return data, new_plot_y_key


def calculate_mean_cell(cell, data, \
                        plot_x_key, plot_y_key):
    new_plot_y_key = "Mean " + plot_y_key

    data[plot_x_key] = data.get(plot_x_key, []) + [cell]

    data[new_plot_y_key] = data.get(new_plot_y_key, []) \
                           + [np.mean(data[cell][plot_y_key])]

    # print(data)
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

    # print(data)

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

    # print(data)

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

    # print(data)

    # Reorganize data

    # species_keys=list(species1.keys())
    MAPEs = [0.] * timesteps
    species_names = list(species1.keys())
    len_timestep = int(len(species1[species_names[0]]) / timesteps)
    max_err = 0.0
    max_err_name = ""
    max_err_k = 0
    # max_tol=1.0E-60
    concs_above_tol = 0
    concs_below_tol = 0
    concs_are_zero = 0

    for j in range(timesteps):
        MAPE = 0.0
        # MAPE=1.0E-60
        n = 0
        for name in species1:
            data1_values = species1[name]
            data2_values = species2[name]
            l = j * len_timestep
            r = len_timestep * (1 + j)
            out1 = data1_values[l:r]
            out2 = data2_values[l:r]

            for k in range(len(out1)):
                err = 0.
                # Filter low concs
                if abs(out1[k] - out2[k]) < max_tol:
                    concs_below_tol = concs_below_tol + 1
                elif out1[k] == 0:
                    concs_are_zero = concs_are_zero + 1
                else:
                    concs_above_tol = concs_above_tol + 1
                    err = abs((out1[k] - out2[k]) / out1[k])

                # if(out1[k]==0.0):
                #  out1[k]+=1.0E-60
                #  out2[k]+=1.0E-60
                #  err=abs((out1[k]-out2[k])/out1[k])
                # err=1
                # print(out1[k],out2[k])
                # else:
                # err=abs((out1[k]-out2[k])/out1[k])
                MAPE += err
                n += 1
                if err > max_err:
                    max_err = err
                    max_err_name = name
                    max_err_k = k
                # if(err>1):
                #print(name,out1[k],out2[k])
        MAPEs[j] = MAPE / n * 100

    if concs_are_zero > concs_below_tol + concs_above_tol:
        print ("Error: More concs are zero than real values, check for errors")
        raise

    print("max_error:" + str(max_err * 100) + "%" + " at species and id: " + max_err_name + " " + str(max_err_k)
          , "concs_above_tol", concs_above_tol,
          "concs_below_tol", concs_below_tol , "concs_are_zero", concs_are_zero)
    # print(NRMSEs)

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

    #print("calculate_speedup start",data)
    #print(plot_y_key)

    # data[new_plot_y_key] = data.get(new_plot_y_key,[])
    datay = [0.] * len(base_data)
    for i in range(len(base_data)):
        # print(base_data[i],new_data[i], base_data[i]/new_data[i])
        datay[i] = base_data[i] / new_data[i]

    #print(datay)

    return datay

def read_solver_stats_all(file, data):
    with open(file) as f:
        csv_reader = csv.reader(f, delimiter=',')
        i_row = 0
        for row in csv_reader:
            if i_row == 0:
                labels = row
                # print(row)
            else:
                for col in range(len(row)):
                    # print(labels[col])
                    # print(row[col])
                    data[labels[col]] = data.get(labels[col], []) + [float(row[col])]
            i_row += 1


def read_solver_stats(file, data, nrows):
    with open(file) as f:
        csv_reader = csv.reader(f, delimiter=',')
        i_row = 0
        for row in csv_reader:
            if i_row == 0:
                labels = row
                # print(row)
            else:
                for col in range(len(row)):
                    # print(labels[col])
                    # print(row[col])
                    data[labels[col]] = data.get(labels[col], []) + [float(row[col])]
            if i_row >= nrows:
                break
            i_row += 1
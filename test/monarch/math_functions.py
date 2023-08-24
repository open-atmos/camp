import numpy as np
from math import sqrt
from pandas import read_csv as pd_read_csv


# import netCDF4 as nc


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

def acalculate_NRMSE(data, n_time_steps, nCellsProcesses,use_monarch, max_abs_tol):
  cases_one_multi_cells = list(data.keys())
  species1 = data[cases_one_multi_cells[0]]
  species2 = data[cases_one_multi_cells[1]]
  n_state = len(species1)
  n_cells = sum(nCellsProcesses)
  n_species = n_state // (n_time_steps * n_cells)
  n_species_monarch = 140

  if use_monarch and n_species_monarch != n_species:
      print("n_species_monarch != n_species calculated by nCellsProcesses")
      print("n_species_monarch", n_species_monarch,
            "n_species", n_species, "n_cells", n_cells)
      raise ValueError("n_species_monarch mismatch")

  NRMSEs = []

  for i in range(n_time_steps):
      NRMSEs_species = [0.0] * n_species
      max_y = [0.0] * n_species
      min_y = [float("inf")] * n_species

      for i3, n_cellsProcess in enumerate(nCellsProcesses):
          i2_base = i3 + i * len(nCellsProcesses)
          i2_range = range(i2_base * n_cellsProcess, (i2_base + 1) * n_cellsProcess)

          for j2 in i2_range:
              for k2 in range(j2 * n_species, (j2 + 1) * n_species):
                  y1 = species1[k2]
                  y2 = species2[k2]
                  NRMSEs_species[k2 % n_species] += (y1 - y2) ** 2

                  max_y[k2 % n_species] = max(max_y[k2 % n_species], y1)
                  min_y[k2 % n_species] = min(min_y[k2 % n_species], y1)

      for k in range(n_species):
          if max_y[k] != min_y[k]:
              NRMSEs_species[k] = (sqrt(NRMSEs_species[k] / n_cells)) / (max_y[k] - min_y[k])

      NRMSEs.append(max(NRMSEs_species) * 100)

  max_err_rel = max_err_rel_process = max_err_rel_timestep = 0.0
  err_rel_at_max_abs = max_err_abs = err_abs_at_max_rel = 0.0
  max_err_rel_specie = max_err_rel_cell = max_err_abs_specie = 0
  max_err_abs_cell = max_err_abs_process = max_err_abs_timestep = 0
  concs_above_tol = concs_below_tol = concs_are_zero = concs_are_equal = 0

  for i, nrmse in enumerate(NRMSEs):
      if nrmse > max_err_rel:
          max_err_rel = nrmse
          err_rel_at_max_abs = NRMSEs[i] / 100
          max_err_rel_timestep = i

      if nrmse > max_err_abs:
          max_err_abs = nrmse
          err_abs_at_max_rel = NRMSEs[i] / 100
          max_err_abs_timestep = i

  print("relative max_error:", format(max_err_rel, ".2e"),
        "% with absolute error",format(err_abs_at_max_rel, ".2e"),
        "at specie id:", max_err_rel_specie,
        "cell:", max_err_rel_cell,
        "process:", max_err_rel_process,
        "timestep:", max_err_rel_timestep,
        "absolute max_error:", format(max_err_abs, ".2e"),
        "with relative error:", format(err_rel_at_max_abs, ".2e"),
        "% at specie id:", max_err_abs_specie,
        "cell:", max_err_abs_cell,
        "process:", max_err_abs_process,
        "timestep:", max_err_abs_timestep,
        "concs_above_tol", concs_above_tol, "concs_below_tol", concs_below_tol,
        "concs_are_equal", concs_are_equal, "concs_are_zero", concs_are_zero)
  return NRMSEs

def calculate_NRMSE(data, n_time_steps, nCellsProcesses,use_monarch, max_abs_tol):
  cases_one_multi_cells = list(data.keys())
  species1 = data[cases_one_multi_cells[0]]
  species2 = data[cases_one_multi_cells[1]]
  n_state = int(len(species1))
  n_cells=sum(nCellsProcesses)
  n_species = int((n_state / n_time_steps) / n_cells)
  n_species_monarch = 140
  if use_monarch and n_species_monarch != n_species:
    print("n_species_monarch ! = n_species calculated by nCellsProcesses")
    print("n_species_monarch",n_species_monarch,
          "n_species",n_species,"n_cells",n_cells)
    raise
  NRMSEs_species = [0.] * n_species
  NRMSEs = [0.] * n_time_steps
  max_y = [0.] * n_species
  min_y = [float("inf")] * n_species
  max_NRMSEs_species = 0.
  max_err_rel = 0.
  err_rel_at_max_abs = 0.
  max_err_rel_specie = 0
  max_err_rel_cell = 0
  max_err_abs_process = 0
  max_err_rel_timestep = 0
  max_err_abs = 0.
  err_abs_at_max_rel = 0.
  max_err_abs_specie = 0
  max_err_abs_cell = 0
  max_err_rel_process = 0
  max_err_abs_timestep = 0
  concs_above_tol = 0
  concs_below_tol = 0
  concs_are_zero = 0
  concs_are_equal = 0
  for i in range(n_time_steps):
    for i3 in range(len(nCellsProcesses)):
      i2 = i3 + i * len(nCellsProcesses)
      n_cellsProcess=nCellsProcesses[i3]
      for j in range(n_cellsProcess):
        j2 = j + i2 * n_cellsProcess
        for k in range(n_species):
          k2 = k+j2*n_species
          y1 = species1[k2]
          y2 = species2[k2]
          NRMSEs_species[k] += (y1 - y2) ** 2
          #if NRMSEs_species[k] == 1.0558509181163416e-18:
            #print(y1,y2,y1 - y2)
          if y1 > max_y[k]:
            max_y[k] = y1
          if y1 < min_y[k]:
            min_y[k] = y1
          err_abs = abs(y1 - y2)
          if y1 != 0:
            err_rel = abs((y1 - y2) / y1)
          else:
            err_rel = 0.
          if err_abs > max_abs_tol:
            concs_above_tol = concs_above_tol + 1
          elif y1 == 0:
            concs_are_zero = concs_are_zero + 1
          else:
            concs_below_tol = concs_below_tol + 1
            if y1 == y2:
              concs_are_equal = concs_are_equal + 1
          if err_abs > max_err_abs:
            max_err_abs = err_abs
            err_rel_at_max_abs = err_rel
            max_err_abs_specie = k
            max_err_abs_cell = j
            max_err_abs_process = i3
            max_err_abs_timestep = i
          if err_rel > max_err_rel:
            max_err_rel = err_rel
            err_abs_at_max_rel = err_abs
            max_err_rel_specie = k
            max_err_rel_cell = j
            max_err_rel_process = i3
            max_err_rel_timestep = i
    for k in range(n_species):
      if max_y[k] != min_y[k]:
        #if max_y[k] == 9.739288159150639e-06:
          #print(sqrt(NRMSEs_species[k]))
          #num=sqrt(NRMSEs_species[k])
          #den=max_y[k] - min_y[k]
           #print("num,den",num,den)
        NRMSEs_species[k] = (sqrt(NRMSEs_species[k] / n_cells)) / (max_y[k] - min_y[k])
      if NRMSEs_species[k] > max_NRMSEs_species:
        max_NRMSEs_species = NRMSEs_species[k]
        #if max_NRMSEs_species > 0.99:
          #print(max_NRMSEs_species,max_y[k] - min_y[k],max_y[k],min_y[k])
      NRMSEs_species[k] = 0.
      max_y[k] = 0.
      min_y[k] = float("inf")
    NRMSEs[i] = max_NRMSEs_species*100
    max_NRMSEs_species = 0.
  max_err_rel = format(max_err_rel * 100, '.2e')
  err_rel_at_max_abs = format(err_rel_at_max_abs * 100, '.2e')
  max_err_abs = format(max_err_abs, '.2e')
  err_abs_at_max_rel = format(err_abs_at_max_rel, '.2e')
  print("relative max_error:", max_err_rel,
        "% with absolute error",err_abs_at_max_rel,
        "at specie id:", max_err_rel_specie,
        "cell:", max_err_rel_cell,
        "process:", max_err_rel_process,
        "timestep:", max_err_rel_timestep,
        "absolute max_error:", max_err_abs,
        "with relative error:", err_rel_at_max_abs,
        "% at specie id:", max_err_abs_specie,
        "cell:", max_err_abs_cell,
        "process:", max_err_abs_process,
        "timestep:", max_err_abs_timestep,
        "concs_above_tol", concs_above_tol, "concs_below_tol", concs_below_tol,
        "concs_are_equal", concs_are_equal, "concs_are_zero", concs_are_zero)
  return NRMSEs


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
  df = pd_read_csv(file, nrows=nrows)
  data = df.to_dict('list')
  return data


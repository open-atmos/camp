from math import sqrt
from pandas import read_csv as pd_read_csv


def calculate_NRMSE(data, n_time_steps, nCellsProcesses,use_monarch):
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
          if y1 > max_y[k]:
            max_y[k] = y1
          if y1 < min_y[k]:
            min_y[k] = y1
          err_abs = abs(y1 - y2)
          if y1 != 0:
            err_rel = abs((y1 - y2) / y1)
          else:
            err_rel = 0.
          if err_abs > 1.0E-4:
            concs_above_tol = concs_above_tol + 1
          elif y1 == 0:
            concs_are_zero = concs_are_zero + 1
          else:
            concs_below_tol = concs_below_tol + 1
            if y1 == y2:
              concs_are_equal = concs_are_equal + 1
          if i == n_time_steps-1:
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
      if max_y[k] - min_y[k] > 1.0e-30:
        NRMSEs_species[k] = (sqrt(NRMSEs_species[k] / n_cells)) / (max_y[k] - min_y[k])
      if NRMSEs_species[k] > max_NRMSEs_species:
        max_NRMSEs_species = NRMSEs_species[k]
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


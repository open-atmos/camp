from math import sqrt
from pandas import read_csv as pd_read_csv


def check_NRMSE(species1, species2, n_time_steps, nCellsProcesses):
  n_state = int(len(species1))
  n_cells = sum(nCellsProcesses)
  n_species = int((n_state / n_time_steps) / n_cells)
  NRMSEs_species = [0.] * n_species
  NRMSEs = [0.] * n_time_steps
  max_y = [0.] * n_species
  min_y = [float("inf")] * n_species
  max_NRMSEs_species = 0.
  for i in range(n_time_steps):
    for i3 in range(len(nCellsProcesses)):
      i2 = i3 + i * len(nCellsProcesses)
      n_cellsProcess = nCellsProcesses[i3]
      for j in range(n_cellsProcess):
        j2 = j + i2 * n_cellsProcess
        for k in range(n_species):
          k2 = k + j2 * n_species
          y1 = species1[k2]
          y2 = species2[k2]
          NRMSEs_species[k] += (y1 - y2) ** 2
          if y1 > max_y[k]:
            max_y[k] = y1
          if y1 < min_y[k]:
            min_y[k] = y1
    for k in range(n_species):
      if max_y[k] - min_y[k] > 1.0e-30:
        NRMSEs_species[k] = (sqrt(NRMSEs_species[k] / n_cells)) / (max_y[k] - min_y[k])
      if NRMSEs_species[k] > max_NRMSEs_species:
        max_NRMSEs_species = NRMSEs_species[k]
      NRMSEs_species[k] = 0.
      max_y[k] = 0.
      min_y[k] = float("inf")
    NRMSEs[i] = max_NRMSEs_species * 100
    max_NRMSEs_species = 0.
    tolerance = 1.0  # 1% error
    if NRMSEs[i] > tolerance:
      raise Exception("ERROR: NMRSE > tolerance; NMRSE:", NRMSEs[i], "tolerance:", tolerance
                      ,"check debug utilities like debug.camp.diff.sh")
  print("NRMSE:",NRMSEs)


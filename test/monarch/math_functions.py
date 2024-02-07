from math import sqrt


def check_NRMSE(species1, species2, nCells):
  n_state = int(len(species1))
  n_species = int(n_state / nCells)
  NRMSEs_species = [0.] * n_species
  max_y = [0.] * n_species
  min_y = [float("inf")] * n_species
  max_NRMSEs_species = 0.
  for j2 in range(nCells):
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
      NRMSEs_species[k] = (sqrt(NRMSEs_species[k] / nCells)) / (max_y[k] - min_y[k])

    if NRMSEs_species[k] > max_NRMSEs_species:
      max_NRMSEs_species = NRMSEs_species[k]
    NRMSEs_species[k] = 0.
    max_y[k] = 0.
    min_y[k] = float("inf")
  NRMSE = max_NRMSEs_species * 100
  tolerance = 1.  # % error
  if NRMSE > tolerance:
    raise Exception("ERROR: NMRSE > tolerance; NMRSE:", NRMSE, "tolerance:", tolerance
                    ,"check debug utilities like debug.camp.diff.sh")
  
  print("NRMSE:",NRMSE)


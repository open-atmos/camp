import netCDF4 as nc
# from netCDF4 import Dataset
import os


class Conf:
  def __init__(self):
    self.n_time_steps = 1
    self.n_cells = 1
    self.n_ranks = 1


def read_netcdf(n_cells, n_ranks, n_time_steps):
  data = [0.] * n_cells * n_ranks
  for i in range(n_time_steps):
    for j in range(n_cells):
      j2 = j + i * n_cells
      for k in range(n_ranks):
        k2 = k + j2 * n_ranks
        file_name = "cell_" + str(j) + "timestep_" + str(i) + "mpirank_" + str(k) + ".nc"
        ncfile = nc.Dataset(file_name)
        state = ncfile.variables["state"][:]
        for z in range(len(state)):
          z2 = z + k2 * state
          data[z2] = state[z]
        # ncfile.close()
  print("data", data)


def get_config(conf):
  print("start translate_netcdf")
  file_list = os.listdir("out/netcdf/")
  print("file_list", file_list)


if __name__ == "__main__":
  conf = Conf()
  get_config(conf)
  read_netcdf(conf.n_cells, conf.n_ranks, conf.n_time_steps)

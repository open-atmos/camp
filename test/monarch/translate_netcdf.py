import netCDF4 as nc
import os
import pandas


class Conf:
  def __init__(self):
    self.n_time_steps = 1
    self.n_cells = 1
    self.n_ranks = 1

def read_netcdf():
  file_name = "out/state.nc"
  ncfile = nc.Dataset(file_name)
  state = ncfile.variables["state"][:].tolist()
  #print("ncfile.variables",ncfile.variables)
  #print("read_netcdf state",state)
  with open("out/state.csv", 'w') as f:
    for z in range(len(state)):
      f.write(str(state[z])+"\n")
      #print(state[z])
      #print(str(state[z]))


def old_read_netcdf(n_cells, n_ranks, n_time_steps):
  file_name = "out/cell_0timestep_0mpirank_0.nc"
  ncfile = nc.Dataset(file_name)
  n_species = len(ncfile.variables["state"][:])
  #data = [0.] * n_species * n_time_steps * n_cells * n_ranks
  print("TODO: Use MPI_GATHER to create a single file since the start")
  with open("out/state.csv", 'w') as f:
    f.write("0")
    for z in range(1, n_species):
      f.write("," + str(z))
    f.write("\n")
    for i in range(n_time_steps):
      for j in range(n_cells):
        j2 = j + i * n_cells
        for k in range(n_ranks):
          k2 = k + j2 * n_ranks
          file_name = "out/cell_" + str(j) + "timestep_" + str(i) + "mpirank_" + str(k) + ".nc"
          ncfile = nc.Dataset(file_name)
          state = ncfile.variables["state"][:].tolist()
          f.write(str(state[0]))
          for z in range(1, n_species):
            f.write("," + str(state[z]))
            #z2 = z + k2 * len(state)
            #data[z2] = state[z]
          f.write("\n")
          # ncfile.close()

  # print("data", data)


def get_config(conf):
  print("start translate_netcdf")
  dir_path = os.path.dirname(os.path.realpath(__file__))
  print("dir_path", dir_path)
  file_list = os.listdir("out/")
  print("file_list", file_list)


if __name__ == "__main__":
  conf = Conf()
  #get_config(conf)
  read_netcdf()

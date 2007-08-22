run_type mc                     # Monte Carlo run
output_file out/urban_plume_summary.d # name of output file
state_prefix out/urban_plume_state # prefix of state files
n_loop 1                        # number of Monte Carlo loops
n_part 100                      # total number of particles
kernel brown                    # coagulation kernel

t_max 86400                     # total simulation time (s)
del_t 300                       # timestep (s)
t_output 3600                   # output interval (0 disables) (s)
t_state 3600                    # state output interval (0 disables) (s)
t_progress 600                  # progress printing interval (0 disables) (s)

n_bin 160                       # number of bins
r_min 1e-8                      # minimum radius (m)
r_max 1e-3                      # maximum radius (m)

gas_data gas_data.dat           # file containing gas data
gas_init gas_init.dat           # initial gas concentrations

aerosol_data aero_data.dat      # file containing aerosol data
aerosol_init aero_init_dist.dat # aerosol initial condition file

temp_profile temp.dat           # temperature profile file
height_profile height_profile.dat       # height profile file
gas_emissions gas_emit_profile.dat      # gas emissions file
gas_background gas_back.dat     # background gas concentrations file
aero_emissions aero_emit_profile.dat    # aerosol emissions file
aero_background aero_back.dat   # aerosol background file

rel_humidity 0.85               # initial relative humidity (1)
pressure 1e5                    # initial pressure (Pa)
latitude 40                     # latitude (degrees, -90 to 90)
longitude 0                     # longitude (degrees, -180 to 180)
altitude 0                      # altitude (m)
start_time 14400                # start time (s since 00:00 UTC)
start_day 200                   # start day of year (UTC)

rand_init 0                     # random initialization (0 to use time)
do_coagulation yes              # whether to do coagulation (yes/no)
allow_double yes                # whether to allow doubling (yes/no)
do_condensation no              # whether to do condensation (yes/no)
do_mosaic yes                   # whether to do MOSAIC (yes/no)
do_restart no                   # whether to restart from stored state (yes/no)
restart_name XXXX.d             # filename to restart from

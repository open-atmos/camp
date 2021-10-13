#!/usr/bin/env python

import scipy.io
import sys
import numpy as np
import matplotlib
matplotlib.use("PDF")
import matplotlib.pyplot as plt
sys.path.append("../../tool")
import camp

def make_plot(in_dir, in_filename, out_filename):
    print in_filename
    ncf = scipy.io.netcdf.netcdf_file(in_dir+in_filename, 'r')
    particles = camp.aero_particle_array_t(ncf)
    ncf.close()

    x_axis = camp.log_grid(min=1e-10,max=1e-4,n_bin=100)
    x_centers = x_axis.centers() 

    dry_diameters = particles.dry_diameters()

    hist = camp.histogram_1d(dry_diameters, x_axis, weights = 1 / particles.comp_vols)

    plt.clf()
    plt.loglog(x_axis.centers(), hist)
    plt.axis([1e-10, 1e-4, 1e7, 1e15])
    plt.xlabel("dry diameter (m)")
    plt.ylabel("number density (m^{-3})")
    fig = plt.gcf()
    fig.savefig(out_filename)

dir_name = "../../scenarios/4_nucleate/out_wei-1/"

filename_in = "nucleate_wc_0001_00000001.nc"
filename_out = "figs/1d_wc_num_001_wei-1.pdf"
make_plot(dir_name, filename_in, filename_out)

filename_in = "nucleate_wc_0001_00000030.nc"
filename_out = "figs/1d_wc_num_030_wei-1.pdf"
make_plot(dir_name, filename_in, filename_out)

filename_in = "nucleate_wc_0001_00000078.nc"
filename_out = "figs/1d_wc_num_078_wei-1.pdf"
make_plot(dir_name, filename_in, filename_out)

filename_in = "nucleate_wc_0001_00000096.nc"
filename_out = "figs/1d_wc_num_096_wei-1.pdf"
make_plot(dir_name, filename_in, filename_out)

filename_in = "nucleate_wc_0001_00000108.nc"
filename_out = "figs/1d_wc_num_108_wei-1.pdf"
make_plot(dir_name, filename_in, filename_out)

filename_in = "nucleate_wc_0001_00000144.nc"
filename_out = "figs/1d_wc_num_144_wei-1.pdf"
make_plot(dir_name, filename_in, filename_out)

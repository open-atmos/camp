#
# Copyright (C) 2022 Barcelona Supercomputing Center and University of
# Illinois at Urbana-Champaign
# SPDX-License-Identifier: MIT
#

from mainMonarch import *


def all_timesteps():
    conf = TestMonarch()
    conf.timeSteps = 10
    conf.loads_gpu = [100]  # e.g. 0: CPU-Only 100: GPU-Only 1-99: CPU+GPU
    conf.load_balance = 0  # 0: fixed, 1: automatic in runtime
    conf.cells = [100]
    conf.mpiProcessesCaseBase = 1
    conf.caseBase = "CPU"
    # conf.caseBase = "GPU BDF"
    conf.mpiProcessesCaseOptimList = [1]
    conf.casesOptim = ["GPU"]
    # conf.is_import = True
    # conf.is_import_base = True
    # conf.profileCuda = "ncu"
    # conf.profileCuda = "nsys"
    # conf.profileExtrae = True
    datay = run_main(conf)
    plot_cases(conf, datay)


if __name__ == "__main__":
    all_timesteps()

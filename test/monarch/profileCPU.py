#
# Copyright (C) 2022 Barcelona Supercomputing Center and University of
# Illinois at Urbana-Champaign
# SPDX-License-Identifier: MIT
#

from mainMonarch import *


def all_timesteps():
    conf = TestMonarch()
    conf.mpiProcessesCaseBase = 2
    conf.cells = [10000]
    conf.timeSteps = 1
    run_main(conf)


if __name__ == "__main__":
    all_timesteps()

#
# Copyright (C) 2022 Barcelona Supercomputing Center and University of
# Illinois at Urbana-Champaign
# SPDX-License-Identifier: MIT
#

from mainMonarch import *


def checkGPU():
    conf = TestMonarch()
    conf.chemFile = "cb05_paperV2"
    conf.mpiProcessesCaseBase = 1
    conf.mpiProcessesCaseOptimList.append(1)
    conf.cells = [10]
    conf.timeSteps = 3
    conf.timeStepsDt = 2
    conf.caseBase = "CPU"
    conf.casesOptim = []
    conf.casesOptim.append("GPU")
    """END OF CONFIGURATION VARIABLES"""
    run_main(conf)


if __name__ == "__main__":
    checkGPU()

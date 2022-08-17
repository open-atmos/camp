/* Copyright (C) 2021 Barcelona Supercomputing Center and University of
 * Illinois at Urbana-Champaign
 * SPDX-License-Identifier: MIT
 */

#ifndef CVODE_gpu_SOLVER_H_
#define CVODE_gpu_SOLVER_H_

#include <cuda.h>
#include "../camp_common.h"
#include "cvode_ls_gpu.h"

void constructor_cvode_gpu(CVodeMem cv_mem, SolverData *sd);
int cudaCVode(void *cvode_mem, realtype tout, N_Vector yout,
               realtype *tret, int itask, SolverData *sd);

void solver_get_statistics_gpu(SolverData *sd);
void solver_reset_statistics_gpu(SolverData *sd);

#endif

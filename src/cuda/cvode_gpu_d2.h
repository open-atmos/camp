/* Copyright (C) 2021 Barcelona Supercomputing Center and University of
 * Illinois at Urbana-Champaign
 * SPDX-License-Identifier: MIT
 */

#ifndef CVODE_gpu_d2_H_
#define CVODE_gpu_d2_H_

#include <cuda.h>
#include "../camp_common.h"
#include "cvode_ls_gpu.h"

void constructor_cvode_cuda_d2(CVodeMem cv_mem, SolverData *sd);
int cudaCVode_d2(void *cvode_mem, realtype tout, N_Vector yout,
               realtype *tret, int itask, SolverData *sd);
void set_jac_data_cuda_d2(SolverData *sd, double *J);
void camp_solver_update_model_state_cuda_d2(N_Vector solver_state, SolverData *sd,
                                       double threshhold, double replacement_value);
void solver_get_statistics_cuda_d2(SolverData *sd);
void solver_reset_statistics_cuda_d2(SolverData *sd);
void free_gpu_cu_d2(SolverData *sd);

#endif
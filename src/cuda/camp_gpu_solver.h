/* Copyright (C) 2021 Barcelona Supercomputing Center and University of
 * Illinois at Urbana-Champaign
 * SPDX-License-Identifier: MIT
 */

#ifndef CAMP_GPU_SOLVER_H_
#define CAMP_GPU_SOLVER_H_
#include <cuda.h>

#include <stdio.h>
#include <stdlib.h>
//extern "C" {
#include "../camp_common.h"
//}

//Value to consider data size too big -> Memory optimization will change below and under the limit

//Functions to debug cuda errors
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

void set_jac_data_gpu(SolverData *sd, double *J);
void camp_solver_update_model_state_gpu(N_Vector solver_state, SolverData *sd);
int rxn_calc_deriv_gpu(SolverData *sd, N_Vector y, N_Vector deriv, double time_step);
void rxn_calc_deriv_aux(ModelData *model_data, double *deriv_data, double time_step);
void rxn_fusion_deriv_gpu(ModelData *model_data, N_Vector deriv);
void free_gpu_cu(SolverData *sd);
void print_gpu_specs();

#endif

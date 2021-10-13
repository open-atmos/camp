/* Copyright (C) 2019 Christian Guzman
 * Licensed under the GNU General Public License version 1 or (at your
 * option) any later version. See the file COPYING for details.
 *
 * Header file for solver functions
 *
 */

#ifndef CAMP_GPU_SOLVER_H_
#define CAMP_GPU_SOLVER_H_
#include <cuda.h>

//#include <cusolverSp.h>
//#include <cuda_runtime_api.h>

#include <stdio.h>
#include <stdlib.h>
//extern "C" {
#include "../camp_common.h"
//}
#include "time_derivative_gpu.h"
#include "Jacobian_gpu.h"
//#include "itsolver_gpu.h"
//#include "../debug_and_stats/camp_debug_2.h"

//Value to consider data size too big -> Memory optimization will change below and under the limit
#define DATA_SIZE_LIMIT_OPT 2000

//Functions to debug cuda errors
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))
#define HANDLE_ERROR2( ) (HandleError2( __FILE__, __LINE__ ))

void solver_new_gpu_cu(SolverData *sd, int n_dep_var, int n_state_var, int n_rxn,
     int n_rxn_int_param, int n_rxn_float_param, int n_rxn_env_param, int n_cells);
void solver_init_int_double_gpu(SolverData *sd);
void init_jac_gpu(SolverData *sd, double *J);
void set_jac_data_gpu(SolverData *sd, double *J);
void rxn_update_env_state_gpu(SolverData *sd);
int camp_solver_check_model_state_gpu(N_Vector solver_state, SolverData *sd,
                                       double threshold, double replacement_value);
void camp_solver_update_model_state_gpu(N_Vector solver_state, SolverData *sd,
                                       double threshold, double replacement_value);
/*
__device__
void cudaDevicef0(
#ifdef CAMP_DEBUG_GPU
        int counterDeriv2,
#endif
        //check_model_state
        double threshhold, double replacement_value, int *flag,
        //f_gpu
        double time_step, int deriv_length_cell, int state_size_cell,
        int n_cells,
        int i_kernel, int threads_block, int n_shr_empty, double *y,
        ModelDataGPU md_object
); //Interface CPU/GPU
*/

int rxn_calc_deriv_gpu(SolverData *sd, N_Vector y, N_Vector deriv, double time_step);
void rxn_calc_deriv_aux(ModelData *model_data, double *deriv_data, double time_step);
void rxn_fusion_deriv_gpu(ModelData *model_data, N_Vector deriv);
int rxn_calc_jac_gpu(SolverData *sd, SUNMatrix jac, double time_step, N_Vector deriv);
void free_gpu_cu(SolverData *sd);
void bubble_sort_gpu(unsigned int *n_zeros, unsigned int *rxn_position, int n_rxn);
void print_gpu_specs();

#endif

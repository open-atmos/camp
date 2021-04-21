/* Copyright (C) 2019 Christian Guzman
 * Licensed under the GNU General Public License version 1 or (at your
 * option) any later version. See the file COPYING for details.
 *
 * Header file for aero_phase_gpu_solver.c
 *
 */
/** \file
 * \brief Header file for aerosol phase functions
 */
#ifndef AERO_PHASE_SOLVER_H
#define AERO_PHASE_SOLVER_H
#include "../camp_gpu_solver.h"

/* Public aerosol phase functions*/

/* Solver functions */
#ifdef __CUDA_ARCH__
__host__ __device__
#endif
int aero_phase_gpu_get_used_jac_elem(ModelDataGPU *model_data, int aero_phase_idx,
                                 int state_var_id, bool *jac_struct);
#ifdef __CUDA_ARCH__
__host__ __device__
#endif
void aero_phase_gpu_get_mass__kg_m3(ModelDataGPU *model_data, int aero_phase_idx,
                                double *state_var, double *mass, double *MW,
                                double *jac_elem_mass, double *jac_elem_MW);
#ifdef __CUDA_ARCH__
__host__ __device__
#endif
void aero_phase_gpu_get_volume__m3_m3(ModelDataGPU *model_data, int aero_phase_idx,
                                  double *state_var, double *volume,
                                  double *jac_elem);

void aero_phase_gpu_print_data(void *solver_data);

/* Setup functions */
#ifdef __CUDA_ARCH__
__host__ __device__
#endif
void aero_phase_gpu_add_condensed_data(int n_int_param, int n_float_param,
                                   int *int_param, double *float_param,
                                   void *solver_data);

#endif

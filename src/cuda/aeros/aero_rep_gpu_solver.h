/* Copyright (C) 2019 Christian Guzman
 * Licensed under the GNU General Public License version 1 or (at your
 * option) any later version. See the file COPYING for details.
 *
 * Header for aerosol representations functions
 */
/** \file
 * \brief Header file for abstract aerosol representation functions
 */
#ifndef aero_rep_gpu_SOLVER_H
#define aero_rep_gpu_SOLVER_H
#include "../camp_gpu_solver.h"

/** Public aerosol representation functions **/

/* Solver functions */
#ifdef __CUDA_ARCH__
__host__ __device__
#endif
int aero_rep_gpu_get_used_jac_elem(ModelDataGPU *model_data, int aero_rep_idx,
                               int aero_phase_idx, bool *jac_struct);
#ifdef __CUDA_ARCH__
__host__ __device__
#endif
void aero_rep_gpu_get_dependencies(ModelDataGPU *model_data, bool *state_flags);
#ifdef __CUDA_ARCH__
__host__ __device__
#endif
void aero_rep_gpu_update_env_state(ModelDataGPU *model_data);
#ifdef __CUDA_ARCH__
__host__ __device__
#endif
void aero_rep_gpu_update_state(ModelDataGPU *model_data);
#ifdef __CUDA_ARCH__
__host__ __device__
#endif
void aero_rep_gpu_get_effective_radius__m(ModelDataGPU *model_data, int aero_rep_idx,
                                      int aero_phase_idx, double *radius,
                                      double *partial_deriv);
#ifdef __CUDA_ARCH__
__host__ __device__
#endif
void aero_rep_gpu_get_number_conc__n_m3(ModelDataGPU *model_data, int aero_rep_idx,
                                    int aero_phase_idx, double *number_conc,
                                    double *partial_deriv);
#ifdef __CUDA_ARCH__
__host__ __device__
#endif
int aero_rep_gpu_get_aero_conc_type(ModelDataGPU *model_data, int aero_rep_idx,
                                int aero_phase_idx);
#ifdef __CUDA_ARCH__
__host__ __device__
#endif
void aero_rep_gpu_get_aero_phase_gpu_mass__kg_m3(ModelDataGPU *model_data,
                                         int aero_rep_idx, int aero_phase_idx,
                                         double *aero_phase_gpu_mass,
                                         double *partial_deriv);
#ifdef __CUDA_ARCH__
__host__ __device__
#endif
void aero_rep_gpu_get_aero_phase_gpu_avg_MW__kg_mol(ModelDataGPU *model_data,
                                            int aero_rep_idx,
                                            int aero_phase_idx,
                                            double *aero_phase_gpu_avg_MW,
                                            double *partial_deriv);
#ifdef __CUDA_ARCH__
__host__
#endif
void aero_rep_gpu_print_data(void *solver_data);

/* Setup functions */
#ifdef __CUDA_ARCH__
__host__ __device__
#endif
void aero_rep_gpu_add_condensed_data(int aero_rep_gpu_type, int n_int_param,
                                 int n_float_param, int n_env_param,
                                 int *int_param, double *float_param,
                                 void *solver_data);

/* Update data functions */
#ifdef __CUDA_ARCH__
__host__ __device__
#endif
void aero_rep_gpu_update_data(int cell_id, int *aero_rep_id,
                          int update_aero_rep_gpu_type, void *update_data,
                          void *solver_data);
void aero_rep_gpu_free_update_data(void *update_data);

#endif
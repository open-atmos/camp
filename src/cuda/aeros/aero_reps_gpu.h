/* Copyright (C) 2019 Christian Guzman
 * Licensed under the GNU General Public License version 1 or (at your
 * option) any later version. See the file COPYING for details.
 *
 * Header for aerosol representations functions
 *
 */
/** \file
 * \brief Header file for aerosol representations functions
 */
#ifndef AERO_REPS_H_
#define AERO_REPS_H_
#include "../camp_gpu_solver.h"

// binned/modal mass
#ifdef __CUDA_ARCH__
__host__ __device__
#endif
int aero_rep_gpu_modal_binned_mass_get_used_jac_elem(ModelDataGPU *model_data,
                                                 int aero_phase_idx,
                                                 int *aero_rep_int_data,
                                                 double *aero_rep_float_data,
                                                 bool *jac_struct);
#ifdef __CUDA_ARCH__
__host__ __device__
#endif
void aero_rep_gpu_modal_binned_mass_get_dependencies(int *aero_rep_int_data,
                                                 double *aero_rep_float_data,
                                                 bool *state_flags);
#ifdef __CUDA_ARCH__
__host__ __device__
#endif
void aero_rep_gpu_modal_binned_mass_update_env_state(ModelDataGPU *model_data,
                                                 int *aero_rep_int_data,
                                                 double *aero_rep_float_data,
                                                 double *aero_rep_env_data);
#ifdef __CUDA_ARCH__
__host__ __device__
#endif
void aero_rep_gpu_modal_binned_mass_update_state(ModelDataGPU *model_data,
                                             int *aero_rep_int_data,
                                             double *aero_rep_float_data,
                                             double *aero_rep_env_data);
#ifdef __CUDA_ARCH__
__host__ __device__
#endif
void aero_rep_gpu_modal_binned_mass_get_effective_radius__m(
        ModelDataGPU *model_data, int aero_phase_idx, double *radius,
        double *partial_deriv, int *aero_rep_int_data, double *aero_rep_float_data,
        double *aero_rep_env_data);
#ifdef __CUDA_ARCH__
__host__ __device__
#endif
void aero_rep_gpu_modal_binned_mass_get_number_conc__n_m3(
        ModelDataGPU *model_data, int aero_phase_idx, double *number_conc,
        double *partial_deriv, int *aero_rep_int_data, double *aero_rep_float_data,
        double *aero_rep_env_data);
#ifdef __CUDA_ARCH__
__host__ __device__
#endif
void aero_rep_gpu_modal_binned_mass_get_aero_conc_type(int aero_phase_idx,
                                                   int *aero_conc_type,
                                                   int *aero_rep_int_data,
                                                   double *aero_rep_float_data,
                                                   double *aero_rep_env_data);
#ifdef __CUDA_ARCH__
__host__ __device__
#endif
void aero_rep_gpu_modal_binned_mass_get_aero_phase_gpu_mass__kg_m3(
        ModelDataGPU *model_data, int aero_phase_idx, double *aero_phase_gpu_mass,
        double *partial_deriv, int *aero_rep_int_data, double *aero_rep_float_data,
        double *aero_rep_env_data);
#ifdef __CUDA_ARCH__
__host__ __device__
#endif
void aero_rep_gpu_modal_binned_mass_get_aero_phase_gpu_avg_MW__kg_mol(
        ModelDataGPU *model_data, int aero_phase_idx, double *aero_phase_gpu_avg_MW,
        double *partial_deriv, int *aero_rep_int_data, double *aero_rep_float_data,
        double *aero_rep_env_data);
#ifdef __CUDA_ARCH__
__host__ __device__
#endif
bool aero_rep_gpu_modal_binned_mass_update_data(void *update_data,
                                            int *aero_rep_int_data,
                                            double *aero_rep_float_data,
                                            double *aero_rep_env_data);
#ifdef __CUDA_ARCH__
__host__ __device__
#endif
void aero_rep_gpu_modal_binned_mass_print(int *aero_rep_int_data,
                                      double *aero_rep_float_data);
#ifdef __CUDA_ARCH__
__host__ __device__
#endif
void *aero_rep_gpu_modal_binned_mass_create_gmd_update_data();
#ifdef __CUDA_ARCH__
__host__ __device__
#endif
void aero_rep_gpu_modal_binned_mass_set_gmd_update_data(void *update_data,
                                                    int aero_rep_id,
                                                    int section_id, double gmd);
#ifdef __CUDA_ARCH__
__host__ __device__
#endif
void *aero_rep_gpu_modal_binned_mass_create_gsd_update_data();
#ifdef __CUDA_ARCH__
__host__ __device__
#endif
void aero_rep_gpu_modal_binned_mass_set_gsd_update_data(void *update_data,
                                                    int aero_rep_id,
                                                    int section_id, double gsd);
#ifdef __CUDA_ARCH__
__host__ __device__
#endif
// single particle
int aero_rep_gpu_single_particle_get_used_jac_elem(ModelDataGPU *model_data,
                                               int aero_phase_idx,
                                               int *aero_rep_int_data,
                                               double *aero_rep_float_data,
                                               bool *jac_struct);
#ifdef __CUDA_ARCH__
__host__ __device__
#endif
void aero_rep_gpu_single_particle_get_dependencies(int *aero_rep_int_data,
                                               double *aero_rep_float_data,
                                               bool *state_flags);
#ifdef __CUDA_ARCH__
__host__ __device__
#endif
void aero_rep_gpu_single_particle_update_env_state(ModelDataGPU *model_data,
                                               int *aero_rep_int_data,
                                               double *aero_rep_float_data,
                                               double *aero_rep_env_data);
#ifdef __CUDA_ARCH__
__host__ __device__
#endif
void aero_rep_gpu_single_particle_update_state(ModelDataGPU *model_data,
                                           int *aero_rep_int_data,
                                           double *aero_rep_float_data,
                                           double *aero_rep_env_data);
#ifdef __CUDA_ARCH__
__host__ __device__
#endif
void aero_rep_gpu_single_particle_get_effective_radius__m(
        ModelDataGPU *model_data, int aero_phase_idx, double *radius,
        double *partial_deriv, int *aero_rep_int_data, double *aero_rep_float_data,
        double *aero_rep_env_data);
#ifdef __CUDA_ARCH__
__host__ __device__
#endif
void aero_rep_gpu_single_particle_get_number_conc__n_m3(
        ModelDataGPU *model_data, int aero_phase_idx, double *number_conc,
        double *partial_deriv, int *aero_rep_int_data, double *aero_rep_float_data,
        double *aero_rep_env_data);
#ifdef __CUDA_ARCH__
__host__ __device__
#endif
void aero_rep_gpu_single_particle_get_aero_conc_type(int aero_phase_idx,
                                                 int *aero_conc_type,
                                                 int *aero_rep_int_data,
                                                 double *aero_rep_float_data,
                                                 double *aero_rep_env_data);
#ifdef __CUDA_ARCH__
__host__ __device__
#endif
void aero_rep_gpu_single_particle_get_aero_phase_gpu_mass__kg_m3(
        ModelDataGPU *model_data, int aero_phase_idx, double *aero_phase_gpu_mass,
        double *partial_deriv, int *aero_rep_int_data, double *aero_rep_float_data,
        double *aero_rep_env_data);
#ifdef __CUDA_ARCH__
__host__ __device__
#endif
void aero_rep_gpu_single_particle_get_aero_phase_gpu_avg_MW__kg_mol(
        ModelDataGPU *model_data, int aero_phase_idx, double *aero_phase_gpu_avg_MW,
        double *partial_deriv, int *aero_rep_int_data, double *aero_rep_float_data,
        double *aero_rep_env_data);
#ifdef __CUDA_ARCH__
__host__ __device__
#endif
bool aero_rep_gpu_single_particle_update_data(void *update_data,
                                          int *aero_rep_int_data,
                                          double *aero_rep_float_data,
                                          double *aero_rep_env_data);
#ifdef __CUDA_ARCH__
__host__ __device__
#endif
void aero_rep_gpu_single_particle_print(int *aero_rep_int_data,
                                    double *aero_rep_float_data);
void *aero_rep_gpu_single_particle_create_number_update_data();
#ifdef __CUDA_ARCH__
__host__ __device__
#endif
void aero_rep_gpu_single_particle_set_number_update_data__n_m3(void *update_data,
                                                           int aero_rep_id,
                                                           int particle_id,
                                                           double number_conc);

#endif


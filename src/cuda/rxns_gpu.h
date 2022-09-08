/* Copyright (C) 2021 Barcelona Supercomputing Center and University of
 * Illinois at Urbana-Champaign
 * SPDX-License-Identifier: MIT
 */

#ifndef RXNS_H_
#define RXNS_H_
#include "camp_gpu_solver.h"

// aqueous_equilibrium
#ifdef CAMP_USE_SUNDIALS
//__device__ double rxn_aqueous_equilibrium_calc_overall_rate(int *rxn_data,
//     double *rxn_double_gpu, double *rxn_env_data, double *state,
//     double react_fact, double prod_fact, double water, int i_phase, int n_rxn2)
void rxn_cpu_aqueous_equilibrium_calc_deriv_contrib(double *rxn_env_data, double *state,
          double *deriv, void *rxn_data, double * rxn_double_gpu, double time_step, int n_rxn);
#ifdef __CUDA_ARCH__
__host__ __device__
#endif
void rxn_gpu_aqueous_equilibrium_calc_deriv_contrib(
          ModelDataGPU *model_data, realtype *deriv, int *rxn_int_data,
          double *rxn_float_data, double *rxn_env_data, double time_step);
#ifdef __CUDA_ARCH__
__host__ __device__
#endif
void rxn_gpu_aqueous_equilibrium_calc_jac_contrib(ModelDataGPU *model_data, JacobianGPU jac, int *rxn_int_data,
          double *rxn_float_data, double *rxn_env_data, double time_step);
#endif

// arrhenius
#ifdef CAMP_USE_SUNDIALS
void rxn_cpu_arrhenius_calc_deriv_contrib(double *rxn_env_data, double *state,
          double *deriv, void *rxn_data, double * rxn_double_gpu, double time_step, int n_rxn);
#ifdef __CUDA_ARCH__
__host__ __device__
#endif
void rxn_gpu_arrhenius_calc_deriv_contrib(ModelDataGPU *model_data, TimeDerivativeGPU time_deriv,
                                      int *rxn_int_data, double *rxn_float_data,
                                      double *rxn_env_data, double time_step);
#ifdef __CUDA_ARCH__
__host__ __device__
#endif
void rxn_gpu_arrhenius_calc_jac_contrib(ModelDataGPU *model_data, JacobianGPU jac, int *rxn_int_data,
          double *rxn_float_data, double *rxn_env_data, double time_step);
void rxn_arrhenius_get_jac_indices(ModelData *model_data, Jacobian jac,
                                    int *rxn_int_data, double *rxn_float_data,
                                    double *rxn_env_data, int *iA, int *jA);
#endif

// CMAQ_H2O2
#ifdef CAMP_USE_SUNDIALS
#ifdef __CUDA_ARCH__
__host__ __device__
#endif
void rxn_gpu_CMAQ_H2O2_calc_deriv_contrib(
          ModelDataGPU *model_data, TimeDerivativeGPU time_deriv, int *rxn_int_data,
          double *rxn_float_data, double *rxn_env_data, double time_step);
#ifdef __CUDA_ARCH__
__host__ __device__
#endif
void rxn_gpu_CMAQ_H2O2_calc_jac_contrib(ModelDataGPU *model_data, JacobianGPU jac, int *rxn_int_data,
          double *rxn_float_data, double *rxn_env_data, double time_step);
#endif

// CMAQ_OH_HNO3
#ifdef CAMP_USE_SUNDIALS
void rxn_cpu_CMAQ_OH_HNO3_calc_deriv_contrib(double *rxn_env_data, double *state,
          double *deriv, void *rxn_data, double * rxn_double_gpu, double time_step, int n_rxn);
#ifdef __CUDA_ARCH__
__host__ __device__
#endif
void rxn_gpu_CMAQ_OH_HNO3_calc_deriv_contrib(
          ModelDataGPU *model_data, TimeDerivativeGPU time_deriv, int *rxn_int_data,
          double *rxn_float_data, double *rxn_env_data, double time_step);
#ifdef __CUDA_ARCH__
__host__ __device__
#endif
void rxn_gpu_CMAQ_OH_HNO3_calc_jac_contrib(ModelDataGPU *model_data, JacobianGPU jac, int *rxn_int_data,
          double *rxn_float_data, double *rxn_env_data, double time_step);
#endif

// condensed_phase_arrhenius
#ifdef CAMP_USE_SUNDIALS
void rxn_cpu_condensed_phase_arrhenius_calc_deriv_contrib(double *rxn_env_data, double *state,
          double *deriv, void *rxn_data, double * rxn_double_gpu, double time_step, int n_rxn);
#ifdef __CUDA_ARCH__
__host__ __device__
#endif
void rxn_gpu_condensed_phase_arrhenius_calc_deriv_contrib(
          ModelDataGPU *model_data, realtype *deriv, int *rxn_int_data,
          double *rxn_float_data, double *rxn_env_data, double time_step);
#ifdef __CUDA_ARCH__
__host__ __device__
#endif
void rxn_gpu_condensed_phase_arrhenius_calc_jac_contrib(ModelDataGPU *model_data, JacobianGPU jac, int *rxn_int_data,
          double *rxn_float_data, double *rxn_env_data, double time_step);
#endif

// emission
#ifdef CAMP_USE_SUNDIALS
void rxn_cpu_emission_calc_deriv_contrib(double *rxn_env_data, double *state,
          double *deriv, void *rxn_data, double * rxn_double_gpu, double time_step, int n_rxn);
#ifdef __CUDA_ARCH__
__host__ __device__
#endif
void rxn_gpu_emission_calc_deriv_contrib(
          ModelDataGPU *model_data, realtype *deriv, int *rxn_int_data,
          double *rxn_float_data, double *rxn_env_data, double time_step);
#ifdef __CUDA_ARCH__
__host__ __device__
#endif
void rxn_gpu_emission_calc_jac_contrib(ModelDataGPU *model_data, JacobianGPU jac, int *rxn_int_data,
          double *rxn_float_data, double *rxn_env_data, double time_step);
#endif

// first_order_loss
#ifdef CAMP_USE_SUNDIALS
void rxn_cpu_first_order_loss_calc_deriv_contrib(double *rxn_env_data, double *state,
          double *deriv, void *rxn_data, double * rxn_double_gpu, double time_step, int n_rxn);
#ifdef __CUDA_ARCH__
__host__ __device__
#endif
void rxn_gpu_first_order_loss_calc_deriv_contrib(
          ModelDataGPU *model_data, realtype *deriv, int *rxn_int_data,
          double *rxn_float_data, double *rxn_env_data, double time_step);
#ifdef __CUDA_ARCH__
__host__ __device__
#endif
void rxn_gpu_first_order_loss_calc_jac_contrib(ModelDataGPU *model_data, JacobianGPU jac, int *rxn_int_data,
          double *rxn_float_data, double *rxn_env_data, double time_step);
#endif

// HL_phase_transfer
#ifdef CAMP_USE_SUNDIALS
#ifdef __CUDA_ARCH__
__host__ __device__
#endif
void rxn_gpu_HL_phase_transfer_calc_deriv_contrib(
        ModelDataGPU *model_data, realtype *deriv, int *rxn_int_data,
          double *rxn_float_data, double *rxn_env_data, double time_step);
#ifdef __CUDA_ARCH__
__host__ __device__
#endif
void rxn_gpu_HL_phase_transfer_calc_jac_contrib(ModelDataGPU *model_data, JacobianGPU jac, int *rxn_int_data,
          double *rxn_float_data, double *rxn_env_data, double time_step);
#endif

// photolysis
#ifdef CAMP_USE_SUNDIALS
void rxn_cpu_photolysis_calc_deriv_contrib(double *rxn_env_data, double *state,
          double *deriv, void *rxn_data, double * rxn_double_gpu, double time_step, int n_rxn);
#ifdef __CUDA_ARCH__
__host__ __device__
#endif
void rxn_gpu_photolysis_calc_deriv_contrib(
          ModelDataGPU *model_data, TimeDerivativeGPU time_deriv, int *rxn_int_data,
          double *rxn_float_data, double *rxn_env_data, double time_step);
#ifdef __CUDA_ARCH__
__host__ __device__
#endif
void rxn_gpu_photolysis_calc_jac_contrib(ModelDataGPU *model_data, JacobianGPU jac, int *rxn_int_data,
          double *rxn_float_data, double *rxn_env_data, double time_step);
#endif

// SIMPOL_phase_transfer
#ifdef CAMP_USE_SUNDIALS
#ifdef __CUDA_ARCH__
__host__ __device__
#endif
void rxn_gpu_SIMPOL_phase_transfer_calc_deriv_contrib(
        ModelDataGPU *model_data, TimeDerivativeGPU time_deriv, int *rxn_int_data,
          double *rxn_float_data, double *rxn_env_data, double time_step);
#ifdef __CUDA_ARCH__
__host__ __device__
#endif
void rxn_gpu_SIMPOL_phase_transfer_calc_jac_contrib(ModelDataGPU *model_data, JacobianGPU jac, int *rxn_int_data,
          double *rxn_float_data, double *rxn_env_data, double time_step);
#endif

// troe
#ifdef CAMP_USE_SUNDIALS
void rxn_cpu_troe_calc_deriv_contrib(double *rxn_env_data, double *state, double *deriv,
          void *rxn_data, double * rxn_double_gpu, double time_step, int n_rxn);
#ifdef __CUDA_ARCH__
__host__ __device__
#endif
void rxn_gpu_troe_calc_deriv_contrib(
          ModelDataGPU *model_data, TimeDerivativeGPU time_deriv, int *rxn_int_data,
          double *rxn_float_data, double *rxn_env_data, double time_step);
#ifdef __CUDA_ARCH__
__host__ __device__
#endif
void rxn_gpu_troe_calc_jac_contrib(ModelDataGPU *model_data, JacobianGPU jac, int *rxn_int_data,
          double *rxn_float_data, double *rxn_env_data, double time_step);
#endif

// wet_deposition
#ifdef CAMP_USE_SUNDIALS
void rxn_cpu_wet_deposition_calc_deriv_contrib(double *rxn_env_data, double *state,
          double *deriv, void *rxn_data, double * rxn_double_gpu, double time_step, int n_rxn);
#ifdef __CUDA_ARCH__
__host__ __device__
#endif
void rxn_gpu_wet_deposition_calc_deriv_contrib(
          ModelDataGPU *model_data, realtype *deriv, int *rxn_int_data,
          double *rxn_float_data, double *rxn_env_data, double time_step);
#ifdef __CUDA_ARCH__
__host__ __device__
#endif
void rxn_gpu_wet_deposition_calc_jac_contrib(ModelDataGPU *model_data, JacobianGPU jac, int *rxn_int_data,
          double *rxn_float_data, double *rxn_env_data, double time_step);
#endif

#endif

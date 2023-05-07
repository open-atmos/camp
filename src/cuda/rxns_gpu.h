/* Copyright (C) 2021 Barcelona Supercomputing Center and University of
 * Illinois at Urbana-Champaign
 * SPDX-License-Identifier: MIT
 */

#ifndef RXNS_H_
#define RXNS_H_
#include "camp_gpu_solver.h"

#ifdef __CUDA_ARCH__
__device__
#endif
void rxn_gpu_aqueous_equilibrium_calc_deriv_contrib(
          ModelDataGPU *model_data, realtype *deriv, int *rxn_int_data,
          double *rxn_float_data, double *rxn_env_data, double time_step);
#ifdef __CUDA_ARCH__
__device__
#endif
void rxn_gpu_aqueous_equilibrium_calc_jac_contrib(ModelDataGPU *model_data, JacobianGPU jac, int *rxn_int_data,
          double *rxn_float_data, double *rxn_env_data, double time_step);

#ifdef __CUDA_ARCH__
__device__
#endif
void rxn_gpu_arrhenius_calc_deriv_contrib(ModelDataGPU *model_data, TimeDerivativeGPU time_deriv,
                                      int *rxn_int_data, double *rxn_float_data,
                                      double *rxn_env_data, double time_step);
#ifdef __CUDA_ARCH__
__device__
#endif
void rxn_gpu_arrhenius_calc_jac_contrib(ModelDataGPU *model_data, JacobianGPU jac, int *rxn_int_data,
          double *rxn_float_data, double *rxn_env_data, double time_step);
#ifdef __CUDA_ARCH__
__device__
#endif
void rxn_gpu_CMAQ_H2O2_calc_deriv_contrib(
          ModelDataGPU *model_data, TimeDerivativeGPU time_deriv, int *rxn_int_data,
          double *rxn_float_data, double *rxn_env_data, double time_step);
#ifdef __CUDA_ARCH__
__device__
#endif
void rxn_gpu_CMAQ_H2O2_calc_jac_contrib(ModelDataGPU *model_data, JacobianGPU jac, int *rxn_int_data,
          double *rxn_float_data, double *rxn_env_data, double time_step);

#ifdef __CUDA_ARCH__
__device__
#endif
void rxn_gpu_CMAQ_OH_HNO3_calc_deriv_contrib(
          ModelDataGPU *model_data, TimeDerivativeGPU time_deriv, int *rxn_int_data,
          double *rxn_float_data, double *rxn_env_data, double time_step);
#ifdef __CUDA_ARCH__
__device__
#endif
void rxn_gpu_CMAQ_OH_HNO3_calc_jac_contrib(ModelDataGPU *model_data, JacobianGPU jac, int *rxn_int_data,
          double *rxn_float_data, double *rxn_env_data, double time_step);

#ifdef __CUDA_ARCH__
__device__
#endif
void rxn_gpu_condensed_phase_arrhenius_calc_deriv_contrib(
          ModelDataGPU *model_data, realtype *deriv, int *rxn_int_data,
          double *rxn_float_data, double *rxn_env_data, double time_step);
#ifdef __CUDA_ARCH__
__device__
#endif
void rxn_gpu_condensed_phase_arrhenius_calc_jac_contrib(ModelDataGPU *model_data, JacobianGPU jac, int *rxn_int_data,
          double *rxn_float_data, double *rxn_env_data, double time_step);

#ifdef __CUDA_ARCH__
__device__
#endif
void rxn_gpu_emission_calc_deriv_contrib(
          ModelDataGPU *model_data, realtype *deriv, int *rxn_int_data,
          double *rxn_float_data, double *rxn_env_data, double time_step);
#ifdef __CUDA_ARCH__
__device__
#endif
void rxn_gpu_emission_calc_jac_contrib(ModelDataGPU *model_data, JacobianGPU jac, int *rxn_int_data,
          double *rxn_float_data, double *rxn_env_data, double time_step);

#ifdef __CUDA_ARCH__
__device__
#endif
void rxn_gpu_first_order_loss_calc_deriv_contrib(
          ModelDataGPU *model_data, TimeDerivativeGPU time_deriv, int *rxn_int_data,
          double *rxn_float_data, double *rxn_env_data, double time_step);
#ifdef __CUDA_ARCH__
__device__
#endif
void rxn_gpu_first_order_loss_calc_jac_contrib(ModelDataGPU *model_data, JacobianGPU jac, int *rxn_int_data,
          double *rxn_float_data, double *rxn_env_data, double time_step);

#ifdef __CUDA_ARCH__
__device__
#endif
void rxn_gpu_HL_phase_transfer_calc_deriv_contrib(
        ModelDataGPU *model_data, realtype *deriv, int *rxn_int_data,
          double *rxn_float_data, double *rxn_env_data, double time_step);
#ifdef __CUDA_ARCH__
__device__
#endif
void rxn_gpu_HL_phase_transfer_calc_jac_contrib(ModelDataGPU *model_data, JacobianGPU jac, int *rxn_int_data,
          double *rxn_float_data, double *rxn_env_data, double time_step);

#ifdef __CUDA_ARCH__
__device__
#endif
void rxn_gpu_photolysis_calc_deriv_contrib(
          ModelDataGPU *model_data, TimeDerivativeGPU time_deriv, int *rxn_int_data,
          double *rxn_float_data, double *rxn_env_data, double time_step);
#ifdef __CUDA_ARCH__
__device__
#endif
void rxn_gpu_photolysis_calc_jac_contrib(ModelDataGPU *model_data, JacobianGPU jac, int *rxn_int_data,
          double *rxn_float_data, double *rxn_env_data, double time_step);

#ifdef __CUDA_ARCH__
__device__
#endif
void rxn_gpu_SIMPOL_phase_transfer_calc_deriv_contrib(
        ModelDataGPU *model_data, TimeDerivativeGPU time_deriv, int *rxn_int_data,
          double *rxn_float_data, double *rxn_env_data, double time_step);
#ifdef __CUDA_ARCH__
__device__
#endif
void rxn_gpu_SIMPOL_phase_transfer_calc_jac_contrib(ModelDataGPU *model_data, JacobianGPU jac, int *rxn_int_data,
          double *rxn_float_data, double *rxn_env_data, double time_step);

#ifdef __CUDA_ARCH__
__device__
#endif
void rxn_gpu_troe_calc_deriv_contrib(
          ModelDataGPU *model_data, TimeDerivativeGPU time_deriv, int *rxn_int_data,
          double *rxn_float_data, double *rxn_env_data, double time_step);
#ifdef __CUDA_ARCH__
__device__
#endif
void rxn_gpu_troe_calc_jac_contrib(ModelDataGPU *model_data, JacobianGPU jac, int *rxn_int_data,
          double *rxn_float_data, double *rxn_env_data, double time_step);

#ifdef __CUDA_ARCH__
__device__
#endif
void rxn_gpu_wet_deposition_calc_deriv_contrib(
          ModelDataGPU *model_data, realtype *deriv, int *rxn_int_data,
          double *rxn_float_data, double *rxn_env_data, double time_step);
#ifdef __CUDA_ARCH__
__device__
#endif
void rxn_gpu_wet_deposition_calc_jac_contrib(ModelDataGPU *model_data, JacobianGPU jac, int *rxn_int_data,
          double *rxn_float_data, double *rxn_env_data, double time_step);

#endif

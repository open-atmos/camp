/* Copyright (C) 2021 Barcelona Supercomputing Center and University of
 * Illinois at Urbana-Champaign
 * SPDX-License-Identifier: MIT
 */

// Equivalent to CPU solver using CAMP and CVODE library. Both components has
// comments around the equivalent code. The function names are equivalent to the
// CVODE library by adding the prefix "cudaDevice", e.g. cvNewtonIteration ->
// cudaDeviceNewtonIteration

#include "cvode_cuda.h"

// Auxiliar methods

/**
 * @brief Computes the power of a base raised to an integer exponent.
 *
 * @param base The base value of type realtype.
 * @param exponent The exponent value of type int.
 * @return The result of base raised to the power of exponent.
 *
 * SUNRpowerI returns the value of base^exponent, where base is of type
 * realtype and exponent is of type int.
 */
__device__ double dSUNRpowerR(double base, double exponent) {
  if (base <= ZERO) return (ZERO);
#ifdef EQUALLIZE_CPU_CUDA_POW
  // The pow function results from CUDA differs from the pow results from CPU
  // This code can be used to equallize the results
  if (exponent == (1. / 2)) return sqrt(base);
  if (exponent == (1. / 3)) return sqrt(sqrt(base));
  if (exponent == (1. / 4)) return sqrt(sqrt(base));
#endif
  return (pow(base, (double)(exponent)));
}

/**
 * @brief Computes the power of a base raised to an integer exponent.
 *
 * @param base The base value of type realtype.
 * @param exponent The exponent value of type int.
 * @return The result of base raised to the power of exponent.
 *
 * SUNRpowerI returns the value of base^exponent, where base is of type
 * realtype and exponent is of type int.
 */
__device__ double dSUNRpowerI(double base, int exponent) {
  int i, expt;
  double prod;
  prod = ONE;
  expt = abs(exponent);
  for (i = 1; i <= expt; i++) prod *= base;
  if (exponent < 0) prod = ONE / prod;
  return (prod);
}

// Solving chemical reactions for a specific time (f(y,t) and Jacobian(y,t))

// Debug mode: remove atomic operations, useful to see that atomicAdd generate
// random results, slightly varying the accuracy of the results
#ifdef IS_DEBUG_MODE_removeAtomic

__device__ void time_derivative_add_value_gpu(TimeDerivativeGPU time_deriv,
                                              unsigned int spec_id,
                                              double rate_contribution) {
  if (rate_contribution > 0.0) {
    time_deriv.production_rates[spec_id] += rate_contribution;
  } else {
    time_deriv.loss_rates[spec_id] += -rate_contribution;
  }
}

__device__ void jacobian_add_value_gpu(JacobianGPU jac, unsigned int elem_id,
                                       int prod_or_loss,
                                       double jac_contribution) {
  if (prod_or_loss == JACOBIAN_PRODUCTION) {
    jac.production_partials[elem_id] += jac_contribution;
  } else {
    jac.loss_partials[elem_id] += jac_contribution;
  }
}

#else

/**
 * @brief Adds a value to the time derivative of a specific species on the GPU.
 *
 * This function adds a rate contribution to the time derivative of a specific
 * species on the GPU. It uses atomic operations to ensure correct
 * synchronization when multiple threads are accessing the same memory location.
 *
 * @param time_deriv The time derivative structure containing the production and
 * loss rates.
 * @param spec_id The ID of the species.
 * @param rate_contribution The rate contribution to be added.
 */
__device__ void time_derivative_add_value_gpu(TimeDerivativeGPU time_deriv,
                                              unsigned int spec_id,
                                              double rate_contribution) {
  // WARNING: Atomicadd is not desirable, because it leads to small deviations
  // in the results, even when scaling the number of data computed in the GPU It
  // would be desirable to remove it
  if (rate_contribution > 0.0) {
    atomicAdd_block(&(time_deriv.production_rates[spec_id]), rate_contribution);
  } else {
    atomicAdd_block(&(time_deriv.loss_rates[spec_id]), -rate_contribution);
  }
}

/**
 * @brief Adds a value to the Jacobian matrix on the GPU.
 *
 * This function adds a contribution to the Jacobian matrix on the GPU for a
 * specific element. The contribution can be either a production or a loss,
 * determined by the `prod_or_loss` parameter. The value of the contribution is
 * specified by the `jac_contribution` parameter.
 *
 * @param jac The Jacobian matrix on the GPU.
 * @param elem_id The ID of the element in the Jacobian matrix.
 * @param prod_or_loss Specifies whether the contribution is a production or a
 * loss. Use the `JACOBIAN_PRODUCTION` constant for production and the
 * `JACOBIAN_LOSS` constant for loss.
 * @param jac_contribution The value of the contribution to be added to the
 * Jacobian matrix.
 */
__device__ void jacobian_add_value_gpu(JacobianGPU jac, unsigned int elem_id,
                                       int prod_or_loss,
                                       double jac_contribution) {
  if (prod_or_loss == JACOBIAN_PRODUCTION) {
    atomicAdd_block(&(jac.production_partials[elem_id]), jac_contribution);
  } else {
    atomicAdd_block(&(jac.loss_partials[elem_id]), jac_contribution);
  }
}

#endif

/**
 * Calculates the derivative contribution from this reaction.
 *
 * @param sc The ModelDataVariable object containing the grid cell state.
 * @param time_deriv The TimeDerivativeGPU object to store the calculated
 * derivative values.
 * @param rxn_int_data An array of integer data for the reaction.
 * @param rxn_float_data An array of floating-point data for the reaction.
 * @param rxn_env_data An array of environmental data for the reaction.
 * @param time_step The time step for the calculation.
 */
__device__ void rxn_gpu_first_order_loss_calc_deriv_contrib(
    ModelDataVariable *sc, TimeDerivativeGPU time_deriv, int *rxn_int_data,
    double *rxn_float_data, double *rxn_env_data, double time_step) {
  int *int_data = rxn_int_data;
  double rate = rxn_env_data[0] * sc->grid_cell_state[int_data[1] - 1];
  if (int_data[2] >= 0)
    time_derivative_add_value_gpu(time_deriv, int_data[2], -rate);
}

/**
 * Calculates the derivative contribution from this reaction.
 *
 * @param sc The ModelDataVariable object containing the grid cell state.
 * @param time_deriv The TimeDerivativeGPU object to store the calculated
 * derivative values.
 * @param rxn_int_data An array of integer data for the reaction.
 * @param rxn_float_data An array of floating-point data for the reaction.
 * @param rxn_env_data An array of environmental data for the reaction.
 * @param time_step The time step for the calculation.
 */
__device__ void rxn_gpu_CMAQ_H2O2_calc_deriv_contrib(
    ModelDataVariable *sc, TimeDerivativeGPU time_deriv, int *rxn_int_data,
    double *rxn_float_data, double *rxn_env_data, double time_step) {
  int *int_data = rxn_int_data;
  double *float_data = rxn_float_data;
  double rate = rxn_env_data[0];
  for (int i_spec = 0; i_spec < int_data[0]; i_spec++)
    rate *= sc->grid_cell_state[int_data[(2 + i_spec)] - 1];
  if (rate != 0.) {
    int i_dep_var = 0;
    for (int i_spec = 0; i_spec < int_data[0]; i_spec++, i_dep_var++) {
      if (int_data[(2 + int_data[0] + int_data[1] + i_dep_var)] < 0) continue;
      time_derivative_add_value_gpu(
          time_deriv, int_data[(2 + int_data[0] + int_data[1] + i_dep_var)],
          -rate);
    }
    for (int i_spec = 0; i_spec < int_data[1]; i_spec++, i_dep_var++) {
      if (int_data[(2 + int_data[0] + int_data[1] + i_dep_var)] < 0) continue;
      if (-rate * float_data[(7 + i_spec)] * time_step <=
          sc->grid_cell_state[int_data[(2 + int_data[0] + i_spec)] - 1]) {
        time_derivative_add_value_gpu(
            time_deriv, int_data[(2 + int_data[0] + int_data[1] + i_dep_var)],
            rate * float_data[(7 + i_spec)]);
      }
    }
  }
}

/**
 * Calculates the derivative contribution from this reaction.
 *
 * @param sc The ModelDataVariable object containing the grid cell state.
 * @param time_deriv The TimeDerivativeGPU object to store the calculated
 * derivative values.
 * @param rxn_int_data An array of integer data for the reaction.
 * @param rxn_float_data An array of floating-point data for the reaction.
 * @param rxn_env_data An array of environmental data for the reaction.
 * @param time_step The time step for the calculation.
 */
__device__ void rxn_gpu_CMAQ_OH_HNO3_calc_deriv_contrib(
    ModelDataVariable *sc, TimeDerivativeGPU time_deriv, int *rxn_int_data,
    double *rxn_float_data, double *rxn_env_data, double time_step) {
  int *int_data = rxn_int_data;
  double *float_data = rxn_float_data;
  double rate = rxn_env_data[0];
  for (int i_spec = 0; i_spec < int_data[0]; i_spec++)
    rate *= sc->grid_cell_state[int_data[(2 + i_spec)] - 1];
  if (rate != 0.) {
    int i_dep_var = 0;
    for (int i_spec = 0; i_spec < int_data[0]; i_spec++, i_dep_var++) {
      if (int_data[(2 + int_data[0] + int_data[1] + i_dep_var)] < 0) continue;
      time_derivative_add_value_gpu(
          time_deriv, int_data[(2 + int_data[0] + int_data[1] + i_dep_var)],
          -rate);
    }
    for (int i_spec = 0; i_spec < int_data[1]; i_spec++, i_dep_var++) {
      if (int_data[(2 + int_data[0] + int_data[1] + i_dep_var)] < 0) continue;
      if (-rate * float_data[(11 + i_spec)] * time_step <=
          sc->grid_cell_state[int_data[(2 + int_data[0] + i_spec)] - 1]) {
        time_derivative_add_value_gpu(
            time_deriv, int_data[(2 + int_data[0] + int_data[1] + i_dep_var)],
            rate * float_data[(11 + i_spec)]);
      }
    }
  }
}

/**
 * Calculates the derivative contribution from this reaction.
 *
 * @param sc The ModelDataVariable object containing the grid cell state.
 * @param time_deriv The TimeDerivativeGPU object to store the calculated
 * derivative values.
 * @param rxn_int_data An array of integer data for the reaction.
 * @param rxn_float_data An array of floating-point data for the reaction.
 * @param rxn_env_data An array of environmental data for the reaction.
 * @param time_step The time step for the calculation.
 */
__device__ void rxn_gpu_arrhenius_calc_deriv_contrib(
    ModelDataVariable *sc, TimeDerivativeGPU time_deriv, int *rxn_int_data,
    double *rxn_float_data, double *rxn_env_data, double time_step) {
  int *int_data = rxn_int_data;
  double *float_data = rxn_float_data;
  double rate = rxn_env_data[0];
  for (int i_spec = 0; i_spec < int_data[0]; i_spec++)
    rate *= sc->grid_cell_state[int_data[(2 + i_spec)] - 1];
  if (rate != 0.) {
    int i_dep_var = 0;
    for (int i_spec = 0; i_spec < int_data[0]; i_spec++, i_dep_var++) {
      if (int_data[2 + int_data[0] + int_data[1] + i_dep_var] < 0) continue;
      time_derivative_add_value_gpu(
          time_deriv, int_data[2 + int_data[0] + int_data[1] + i_dep_var],
          -rate);
    }
    for (int i_spec = 0; i_spec < int_data[1]; i_spec++, i_dep_var++) {
      if (int_data[2 + int_data[0] + int_data[1] + i_dep_var] < 0) continue;
      if (-rate * float_data[6 + i_spec] * time_step <=
          sc->grid_cell_state[int_data[(2 + int_data[0] + i_spec)] - 1]) {
        time_derivative_add_value_gpu(
            time_deriv, int_data[2 + int_data[0] + int_data[1] + i_dep_var],
            rate * float_data[6 + i_spec]);
      }
    }
  }
}

/**
 * Calculates the derivative contribution from this reaction.
 *
 * @param sc The ModelDataVariable object containing the grid cell state.
 * @param time_deriv The TimeDerivativeGPU object to store the calculated
 * derivative values.
 * @param rxn_int_data An array of integer data for the reaction.
 * @param rxn_float_data An array of floating-point data for the reaction.
 * @param rxn_env_data An array of environmental data for the reaction.
 * @param time_step The time step for the calculation.
 */
__device__ void rxn_gpu_troe_calc_deriv_contrib(
    ModelDataVariable *sc, TimeDerivativeGPU time_deriv, int *rxn_int_data,
    double *rxn_float_data, double *rxn_env_data, double time_step) {
  int *int_data = rxn_int_data;
  double *float_data = rxn_float_data;
  double rate = rxn_env_data[0];
  for (int i_spec = 0; i_spec < int_data[0]; i_spec++)
    rate *= sc->grid_cell_state[int_data[(2 + i_spec)] - 1];
  if (rate != ZERO) {
    int i_dep_var = 0;
    for (int i_spec = 0; i_spec < int_data[0]; i_spec++, i_dep_var++) {
      if (int_data[(2 + int_data[0] + int_data[1] + i_dep_var)] < 0) continue;
      time_derivative_add_value_gpu(
          time_deriv, int_data[(2 + int_data[0] + int_data[1] + i_dep_var)],
          -rate);
    }
    for (int i_spec = 0; i_spec < int_data[1]; i_spec++, i_dep_var++) {
      if (int_data[(2 + int_data[0] + int_data[1] + i_dep_var)] < 0) continue;
      if (-rate * float_data[(10 + i_spec)] * time_step <=
          sc->grid_cell_state[int_data[(2 + int_data[0] + i_spec)] - 1]) {
        time_derivative_add_value_gpu(
            time_deriv, int_data[(2 + int_data[0] + int_data[1] + i_dep_var)],
            rate * float_data[(10 + i_spec)]);
      }
    }
  }
}

/**
 * Calculates the derivative contribution from this reaction.
 *
 * @param sc The ModelDataVariable object containing the grid cell state.
 * @param time_deriv The TimeDerivativeGPU object to store the calculated
 * derivative values.
 * @param rxn_int_data An array of integer data for the reaction.
 * @param rxn_float_data An array of floating-point data for the reaction.
 * @param rxn_env_data An array of environmental data for the reaction.
 * @param time_step The time step for the calculation.
 */
__device__ void rxn_gpu_photolysis_calc_deriv_contrib(
    ModelDataVariable *sc, TimeDerivativeGPU time_deriv, int *rxn_int_data,
    double *rxn_float_data, double *rxn_env_data, double time_step) {
  int *int_data = rxn_int_data;
  double *float_data = rxn_float_data;
  double rate = rxn_env_data[0];
  for (int i_spec = 0; i_spec < int_data[0]; i_spec++)
    rate *= sc->grid_cell_state[int_data[(3 + i_spec)] - 1];
  if (rate != ZERO) {
    int i_dep_var = 0;
    for (int i_spec = 0; i_spec < int_data[0]; i_spec++, i_dep_var++) {
      if (int_data[(3 + int_data[0] + int_data[1] + i_dep_var)] < 0) continue;
      time_derivative_add_value_gpu(
          time_deriv, int_data[(3 + int_data[0] + int_data[1] + i_dep_var)],
          -rate);
    }
    for (int i_spec = 0; i_spec < int_data[1]; i_spec++, i_dep_var++) {
      if (int_data[(3 + int_data[0] + int_data[1] + i_dep_var)] < 0) continue;
      if (-rate * float_data[(1 + i_spec)] * time_step <=
          sc->grid_cell_state[int_data[(3 + int_data[0] + i_spec)] - 1]) {
        time_derivative_add_value_gpu(
            time_deriv, int_data[(3 + int_data[0] + int_data[1] + i_dep_var)],
            rate * float_data[(1 + i_spec)]);
      }
    }
  }
}

/**
 * Calculate contributions to the Jacobian from this reaction
 *
 * @param sc The model data variable.
 * @param jac The Jacobian matrix.
 * @param rxn_int_data The integer data for the reaction.
 * @param rxn_float_data The float data for the reaction.
 * @param rxn_env_data The environmental data for the reaction.
 * @param time_step The time step.
 */
__device__ void rxn_gpu_first_order_loss_calc_jac_contrib(
    ModelDataVariable *sc, JacobianGPU jac, int *rxn_int_data,
    double *rxn_float_data, double *rxn_env_data, double time_step) {
  int *int_data = rxn_int_data;
  if (int_data[3] >= 0)
    jacobian_add_value_gpu(jac, int_data[3], JACOBIAN_LOSS, rxn_env_data[0]);
}

/**
 * Calculate contributions to the Jacobian from this reaction
 *
 * @param sc The model data variable.
 * @param jac The Jacobian matrix.
 * @param rxn_int_data The integer data for the reaction.
 * @param rxn_float_data The float data for the reaction.
 * @param rxn_env_data The environmental data for the reaction.
 * @param time_step The time step.
 */
__device__ void rxn_gpu_CMAQ_H2O2_calc_jac_contrib(
    ModelDataVariable *sc, JacobianGPU jac, int *rxn_int_data,
    double *rxn_float_data, double *rxn_env_data, double time_step) {
  int *int_data = rxn_int_data;
  double *float_data = rxn_float_data;
  int i_elem = 0;
  for (int i_ind = 0; i_ind < int_data[0]; i_ind++) {
    double rate = rxn_env_data[0];
    for (int i_spec = 0; i_spec < int_data[0]; i_spec++)
      if (i_ind != i_spec)
        rate *= sc->grid_cell_state[int_data[(2 + i_spec)] - 1];
    for (int i_dep = 0; i_dep < int_data[0]; i_dep++, i_elem++) {
      if (int_data[(2 + 2 * (int_data[0] + int_data[1]) + i_elem)] < 0)
        continue;
      jacobian_add_value_gpu(
          jac, int_data[(2 + 2 * (int_data[0] + int_data[1]) + i_elem)],
          JACOBIAN_LOSS, rate);
    }
    for (int i_dep = 0; i_dep < int_data[1]; i_dep++, i_elem++) {
      if (int_data[(2 + 2 * (int_data[0] + int_data[1]) + i_elem)] < 0)
        continue;
      if (-rate * sc->grid_cell_state[int_data[(2 + i_ind)] - 1] *
              float_data[(7 + i_dep)] * time_step <=
          sc->grid_cell_state[int_data[(2 + int_data[0] + i_dep)] - 1]) {
        jacobian_add_value_gpu(
            jac, int_data[(2 + 2 * (int_data[0] + int_data[1]) + i_elem)],
            JACOBIAN_PRODUCTION, float_data[(7 + i_dep)] * rate);
      }
    }
  }
}

/**
 * Calculate contributions to the Jacobian from this reaction
 *
 * @param sc The model data variable.
 * @param jac The Jacobian matrix.
 * @param rxn_int_data The integer data for the reaction.
 * @param rxn_float_data The float data for the reaction.
 * @param rxn_env_data The environmental data for the reaction.
 * @param time_step The time step.
 */
__device__ void rxn_gpu_CMAQ_OH_HNO3_calc_jac_contrib(
    ModelDataVariable *sc, JacobianGPU jac, int *rxn_int_data,
    double *rxn_float_data, double *rxn_env_data, double time_step) {
  int *int_data = rxn_int_data;
  double *float_data = rxn_float_data;
  int i_elem = 0;
  for (int i_ind = 0; i_ind < int_data[0]; i_ind++) {
    double rate = rxn_env_data[0];
    for (int i_spec = 0; i_spec < int_data[0]; i_spec++)
      if (i_ind != i_spec)
        rate *= sc->grid_cell_state[int_data[(2 + i_spec)] - 1];
    for (int i_dep = 0; i_dep < int_data[0]; i_dep++, i_elem++) {
      if (int_data[(2 + 2 * (int_data[0] + int_data[1]) + i_elem)] < 0)
        continue;
      jacobian_add_value_gpu(
          jac, int_data[(2 + 2 * (int_data[0] + int_data[1]) + i_elem)],
          JACOBIAN_LOSS, rate);
    }
    for (int i_dep = 0; i_dep < int_data[1]; i_dep++, i_elem++) {
      if (int_data[(2 + 2 * (int_data[0] + int_data[1]) + i_elem)] < 0)
        continue;
      if (-rate * sc->grid_cell_state[int_data[(2 + i_ind)] - 1] *
              float_data[(11 + i_dep)] * time_step <=
          sc->grid_cell_state[int_data[(2 + int_data[0] + i_dep)] - 1]) {
        jacobian_add_value_gpu(
            jac, int_data[(2 + 2 * (int_data[0] + int_data[1]) + i_elem)],
            JACOBIAN_PRODUCTION, float_data[(11 + i_dep)] * rate);
      }
    }
  }
}

/**
 * Calculate contributions to the Jacobian from this reaction
 *
 * @param sc The model data variable.
 * @param jac The Jacobian matrix.
 * @param rxn_int_data The integer data for the reaction.
 * @param rxn_float_data The float data for the reaction.
 * @param rxn_env_data The environmental data for the reaction.
 * @param time_step The time step.
 */
__device__ void rxn_gpu_arrhenius_calc_jac_contrib(
    ModelDataVariable *sc, JacobianGPU jac, int *rxn_int_data,
    double *rxn_float_data, double *rxn_env_data, double time_step) {
  int *int_data = rxn_int_data;
  double *float_data = rxn_float_data;
  int i_elem = 0;
  for (int i_ind = 0; i_ind < int_data[0]; i_ind++) {
    double rate = rxn_env_data[0];
    for (int i_spec = 0; i_spec < int_data[0]; i_spec++)
      if (i_spec != i_ind)
        rate *= sc->grid_cell_state[int_data[(2 + i_spec)] - 1];
    for (int i_dep = 0; i_dep < int_data[0]; i_dep++, i_elem++) {
      if (int_data[2 + 2 * (int_data[0] + int_data[1]) + i_elem] < 0) continue;
      jacobian_add_value_gpu(
          jac, int_data[2 + 2 * (int_data[0] + int_data[1]) + i_elem],
          JACOBIAN_LOSS, rate);
    }
    for (int i_dep = 0; i_dep < int_data[1]; i_dep++, i_elem++) {
      if (int_data[2 + 2 * (int_data[0] + int_data[1]) + i_elem] < 0) continue;
      if (-rate * sc->grid_cell_state[int_data[(2 + i_ind)] - 1] *
              float_data[6 + i_dep] * time_step <=
          sc->grid_cell_state[int_data[(2 + int_data[0] + i_dep)] - 1]) {
        jacobian_add_value_gpu(
            jac, int_data[2 + 2 * (int_data[0] + int_data[1]) + i_elem],
            JACOBIAN_PRODUCTION, float_data[6 + i_dep] * rate);
      }
    }
  }
}

/**
 * Calculate contributions to the Jacobian from this reaction
 *
 * @param sc The model data variable.
 * @param jac The Jacobian matrix.
 * @param rxn_int_data The integer data for the reaction.
 * @param rxn_float_data The float data for the reaction.
 * @param rxn_env_data The environmental data for the reaction.
 * @param time_step The time step.
 */
__device__ void rxn_gpu_troe_calc_jac_contrib(
    ModelDataVariable *sc, JacobianGPU jac, int *rxn_int_data,
    double *rxn_float_data, double *rxn_env_data, double time_step) {
  int *int_data = rxn_int_data;
  double *float_data = rxn_float_data;
  int i_elem = 0;
  for (int i_ind = 0; i_ind < int_data[0]; i_ind++) {
    double rate = rxn_env_data[0];
    for (int i_spec = 0; i_spec < int_data[0]; i_spec++)
      if (i_ind != i_spec)
        rate *= sc->grid_cell_state[int_data[(2 + i_spec)] - 1];
    for (int i_dep = 0; i_dep < int_data[0]; i_dep++, i_elem++) {
      if (int_data[(2 + 2 * (int_data[0] + int_data[1]) + i_elem)] < 0)
        continue;
      jacobian_add_value_gpu(
          jac, int_data[(2 + 2 * (int_data[0] + int_data[1]) + i_elem)],
          JACOBIAN_LOSS, rate);
    }
    for (int i_dep = 0; i_dep < int_data[1]; i_dep++, i_elem++) {
      if (int_data[(2 + 2 * (int_data[0] + int_data[1]) + i_elem)] < 0)
        continue;
      if (-rate * sc->grid_cell_state[int_data[(2 + i_ind)] - 1] *
              float_data[(10 + i_dep)] * time_step <=
          sc->grid_cell_state[int_data[(2 + int_data[0] + i_dep)] - 1]) {
        jacobian_add_value_gpu(
            jac, int_data[(2 + 2 * (int_data[0] + int_data[1]) + i_elem)],
            JACOBIAN_PRODUCTION, float_data[(10 + i_dep)] * rate);
      }
    }
  }
}

/**
 * Calculate contributions to the Jacobian from this reaction
 *
 * @param sc The model data variable.
 * @param jac The Jacobian matrix.
 * @param rxn_int_data The integer data for the reaction.
 * @param rxn_float_data The float data for the reaction.
 * @param rxn_env_data The environmental data for the reaction.
 * @param time_step The time step.
 */
__device__ void rxn_gpu_photolysis_calc_jac_contrib(
    ModelDataVariable *sc, JacobianGPU jac, int *rxn_int_data,
    double *rxn_float_data, double *rxn_env_data, double time_step) {
  int *int_data = rxn_int_data;
  double *float_data = rxn_float_data;
  int i_elem = 0;
  for (int i_ind = 0; i_ind < int_data[0]; i_ind++) {
    double rate = rxn_env_data[0];
    for (int i_spec = 0; i_spec < int_data[0]; i_spec++)
      if (i_spec != i_ind)
        rate *= sc->grid_cell_state[int_data[(3 + i_spec)] - 1];
    for (int i_dep = 0; i_dep < int_data[0]; i_dep++, i_elem++) {
      if (int_data[(3 + 2 * (int_data[0] + int_data[1]) + i_elem)] < 0)
        continue;
      jacobian_add_value_gpu(
          jac, int_data[(3 + 2 * (int_data[0] + int_data[1]) + i_elem)],
          JACOBIAN_LOSS, rate);
    }
    for (int i_dep = 0; i_dep < int_data[1]; i_dep++, i_elem++) {
      if (int_data[(3 + 2 * (int_data[0] + int_data[1]) + i_elem)] < 0)
        continue;
      if (-rate * sc->grid_cell_state[int_data[(3 + i_ind)] - 1] *
              float_data[(1 + i_dep)] * time_step <=
          sc->grid_cell_state[int_data[(3 + int_data[0] + i_dep)] - 1]) {
        jacobian_add_value_gpu(
            jac, int_data[(3 + 2 * (int_data[0] + int_data[1]) + i_elem)],
            JACOBIAN_PRODUCTION, float_data[(1 + i_dep)] * rate);
      }
    }
  }
}

// Auxiliar functions for the GPU ODE solver

/**
 * @brief Performs reduction operation to find the minimum value in an array on
 * the device.
 *
 * This function is executed on the device and performs a reduction operation to
 * find the minimum value in an array. It uses shared memory to store
 * intermediate results and performs a binary reduction algorithm.
 *
 * @param g_odata Pointer to the global memory location where the minimum value
 * will be stored.
 * @param in The input value to be compared for finding the minimum.
 * @param sdata Pointer to the shared memory array used for intermediate
 * results.
 * @param n_shr_empty The number of empty elements in the shared memory array.
 */
__device__ void cudaDevicemin(double *g_odata, double in,
                              volatile double *sdata, int n_shr_empty) {
  unsigned int tid = threadIdx.x;
  __syncthreads();
  sdata[tid] = in;
  if (tid < n_shr_empty) sdata[tid + blockDim.x] = sdata[tid];
  __syncthreads();
  for (unsigned int s = (blockDim.x + n_shr_empty) / 2; s > 0; s >>= 1) {
    if (tid < s) {
      if (sdata[tid + s] < sdata[tid]) sdata[tid] = sdata[tid + s];
    }
    __syncthreads();
  }
  *g_odata = sdata[0];
  __syncthreads();
}

/**
 * @brief Applies a preconditioner to a matrix on the device.
 *
 * This function applies a preconditioner to a matrix represented by
 * the compressed sparse row (CSR) format on the device. The preconditioner
 * modifies the matrix in-place by scaling the diagonal elements and applying an
 * alpha factor to the off-diagonal elements.
 *
 * @param dA     Pointer to the matrix elements in CSR format.
 * @param djA    Pointer to the column indices of the matrix elements in CSR
 * format.
 * @param diA    Pointer to the row offsets of the matrix elements in CSR
 * format.
 * @param ddiag  Pointer to the diagonal elements of the matrix.
 * @param alpha  The alpha factor to be applied to the off-diagonal elements.
 */
__device__ void cudaDeviceBCGprecond_2(double *dA, int *djA, int *diA,
                                       double *ddiag, double alpha) {
  int row = threadIdx.x + blockDim.x * blockIdx.x;
  int nnz = diA[blockDim.x];
  for (int j = diA[threadIdx.x]; j < diA[threadIdx.x + 1]; j++) {
    if (djA[j] == threadIdx.x) {
      dA[j + nnz * blockIdx.x] = 1.0 + alpha * dA[j + nnz * blockIdx.x];
      if (dA[j + nnz * blockIdx.x] != 0.0) {
        ddiag[row] = 1.0 / dA[j + nnz * blockIdx.x];
      } else {
        ddiag[row] = 1.0;
      }
    } else {
      dA[j + nnz * blockIdx.x] *= alpha;
    }
  }
}

/**
 * @brief Performs sparse matrix-vector multiplication on a CUDA device.
 *
 * This function calculates the product of a sparse matrix and a vector on a
 * CUDA device. It uses the Compressed Sparse Row (CSR) format to represent the
 * sparse matrix.
 *
 * @param dx Pointer to the output vector.
 * @param db Pointer to the input vector.
 * @param dA Pointer to the values of the sparse matrix.
 * @param djA Pointer to the column indices of the sparse matrix.
 * @param diA Pointer to the row offsets of the sparse matrix.
 */
__device__ void cudaDeviceSpmv_CSR(double *dx, double *db, double *dA, int *djA,
                                   int *diA) {
  __syncthreads();
  int row = threadIdx.x + blockDim.x * blockIdx.x;
  double sum = 0.0;
  int nnz = diA[blockDim.x];
  for (int j = diA[threadIdx.x]; j < diA[threadIdx.x + 1]; j++) {
    sum += db[djA[j] + blockDim.x * blockIdx.x] * dA[j + nnz * blockIdx.x];
  }
  __syncthreads();
  dx[row] = sum;
}

/**
 * @brief Performs a reduction operation within a warp.
 *
 * This function reduces the values stored in the shared memory array `sdata`
 * within a warp. The reduction is performed by adding adjacent elements in the
 * array in a binary tree fashion. The result is stored in the first element of
 * the array.
 *
 * @param sdata The shared memory array containing the values to be reduced.
 * @param tid The thread ID within the warp.
 */
__device__ void warpReduce_2(volatile double *sdata, unsigned int tid) {
  unsigned int blockSize = blockDim.x;
  if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
  if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
  if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
  if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
  if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
  if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
}

/**
 * @brief Performs dot product of two arrays on the device.
 *
 * This function calculates the dot product of two arrays, `g_idata1` and
 * `g_idata2`, and stores the result in `g_odata`. The dot product is calculated
 * by multiplying corresponding elements of the arrays and summing them up.
 *
 * @param g_idata1 Pointer to the first input array.
 * @param g_idata2 Pointer to the second input array.
 * @param g_odata Pointer to the output array.
 * @param n_shr_empty Number of empty elements in the shared memory.
 */
__device__ void cudaDevicedotxy_2(double *g_idata1, double *g_idata2,
                                  double *g_odata, int n_shr_empty) {
  extern __shared__ double sdata[];
  unsigned int tid = threadIdx.x;
  __syncthreads();
  if (tid < n_shr_empty) sdata[tid + blockDim.x] = 0.;
#ifdef IS_DEBUG_MODE_cudaDevicedotxy_2
  // used for compare with cpu
  sdata[0] = 0.;
  __syncthreads();
  if (tid == 0) {
    for (int j = 0; j < blockDim.x; j++) {
      sdata[0] += g_idata1[j + blockIdx.x * blockDim.x] *
                  g_idata2[j + blockIdx.x * blockDim.x];
    }
  }
#else
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  sdata[tid] = g_idata1[i] * g_idata2[i];
  __syncthreads();
  unsigned int blockSize = blockDim.x + n_shr_empty;
  if ((blockSize >= 1024) && (tid < 512)) {
    sdata[tid] += sdata[tid + 512];
  }
  __syncthreads();
  if ((blockSize >= 512) && (tid < 256)) {
    sdata[tid] += sdata[tid + 256];
  }
  __syncthreads();
  if ((blockSize >= 256) && (tid < 128)) {
    sdata[tid] += sdata[tid + 128];
  }
  __syncthreads();
  if ((blockSize >= 128) && (tid < 64)) {
    sdata[tid] += sdata[tid + 64];
  }
  __syncthreads();
  if (tid < 32) warpReduce_2(sdata, tid);
#endif
  __syncthreads();
  *g_odata = sdata[0];
  __syncthreads();
}

/**
 * @brief Calculates the VWRMS (Vector Weighted Root Mean Square) norm of two
 * arrays on the device.
 *
 * This function calculates the VWRMS norm of two arrays `g_idata1` and
 * `g_idata2` on the device. The result is stored in the `g_odata` array.
 *
 * @param g_idata1 Pointer to the first input array.
 * @param g_idata2 Pointer to the second input array.
 * @param g_odata Pointer to the output array.
 * @param n_shr_empty Number of empty elements in the shared memory.
 */
__device__ void cudaDeviceVWRMS_Norm_2(double *g_idata1, double *g_idata2,
                                       double *g_odata, int n_shr_empty) {
  extern __shared__ double sdata[];
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  __syncthreads();
  if (tid < n_shr_empty) sdata[tid + blockDim.x] = 0.;
  sdata[tid] = g_idata1[i] * g_idata2[i];
  sdata[tid] = sdata[tid] * sdata[tid];
  __syncthreads();
#ifdef IS_DEBUG_MODE_cudaDevicedotxy_2
  // used for compare with cpu
  if (tid == 0) {
    double sum = 0.;
    for (int j = 0; j < blockDim.x; j++) {
      sum += sdata[j];
    }
    sdata[0] = sum;
  }
  __syncthreads();
#else
  for (unsigned int s = (blockDim.x + n_shr_empty) / 2; s > 0; s >>= 1) {
    if (tid < s) sdata[tid] += sdata[tid + s];
    __syncthreads();
  }
#endif
  g_odata[0] = sqrt(sdata[0] / blockDim.x);
  __syncthreads();
}

/**
 * @brief Copies the Jacobian values from input array `Ax` to the output array
 * `Bx` for a specific device.
 *
 * This function copies the elements of the input array `Ax` to the output array
 * `Bx` for a specific device. The number of non-zero elements in each row is
 * specified by the `diA` array.
 *
 * @param diA Pointer to the array containing the number of non-zero elements in
 * each row.
 * @param Ax Pointer to the input array.
 * @param Bx Pointer to the output array.
 */
__device__ void cudaDeviceJacCopy(int *diA, double *Ax, double *Bx) {
  int nnz = diA[blockDim.x];
  for (int j = diA[threadIdx.x]; j < diA[threadIdx.x + 1]; j++) {
    Bx[j + nnz * blockIdx.x] = Ax[j + nnz * blockIdx.x];
  }
}

// Functions equivalent to CAMP CPU solver

/**
 * @brief Update reaction data for new environmental conditions
 *
 * @param md A pointer to the ModelDataGPU structure.
 * @param sc A pointer to the ModelDataVariable structure.
 * @param y A pointer to the array of state variables.
 *
 * @return An integer representing the flag indicating the model state.
 */
__device__ int cudaDevicecamp_solver_update_model_state(ModelDataGPU *md,
                                                        ModelDataVariable *sc,
                                                        double *y) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  extern __shared__ int flag_shr[];
  __syncthreads();
  flag_shr[0] = 0;
  __syncthreads();
  if (y[i] < -SMALL) {
    flag_shr[0] = CAMP_SOLVER_FAIL;
  } else {
    md->state[md->map_state_deriv[threadIdx.x] +
              blockIdx.x * md->n_per_cell_state_var] =
        y[i] <= -SMALL ? TINY : y[i];
  }
  __syncthreads();
  int flag = flag_shr[0];
  __syncthreads();
  return flag;
}

/**
 * @brief Calculate the time derivative \f$f(t,y)\f$
 *
 * The reaction data is accessed from the model data based on the reaction
 * index. The function then switches based on the reaction type and calls the
 * corresponding GPU function to calculate the derivative contribution.
 *
 * @param i_rxn The index of the reaction.
 * @param deriv_data The time derivative data on the GPU.
 * @param time_step The time step.
 * @param md The model data on the GPU.
 * @param sc The model data variables on the GPU.
 */
__device__ void cudaDevicerxn_calc_deriv(int i_rxn,
                                         TimeDerivativeGPU deriv_data,
                                         double time_step, ModelDataGPU *md,
                                         ModelDataVariable *sc) {
  double *rxn_float_data =
      (double *)&(md->rxn_double[md->rxn_float_indices[i_rxn]]);
  int *int_data = (int *)&(md->rxn_int[md->rxn_int_indices[i_rxn]]);
  int *rxn_int_data = (int *)&(int_data[1]);
  double *rxn_env_data = &(md->rxn_env_data[md->n_rxn_env_data * blockIdx.x +
                                            md->rxn_env_idx[i_rxn]]);
  switch (int_data[0]) {
    case RXN_ARRHENIUS:
      rxn_gpu_arrhenius_calc_deriv_contrib(sc, deriv_data, rxn_int_data,
                                           rxn_float_data, rxn_env_data,
                                           time_step);
      break;
    case RXN_CMAQ_H2O2:
      rxn_gpu_CMAQ_H2O2_calc_deriv_contrib(sc, deriv_data, rxn_int_data,
                                           rxn_float_data, rxn_env_data,
                                           time_step);
      break;
    case RXN_CMAQ_OH_HNO3:
      rxn_gpu_CMAQ_OH_HNO3_calc_deriv_contrib(sc, deriv_data, rxn_int_data,
                                              rxn_float_data, rxn_env_data,
                                              time_step);
      break;
    case RXN_FIRST_ORDER_LOSS:
      rxn_gpu_first_order_loss_calc_deriv_contrib(sc, deriv_data, rxn_int_data,
                                                  rxn_float_data, rxn_env_data,
                                                  time_step);
      break;
    case RXN_PHOTOLYSIS:
      rxn_gpu_photolysis_calc_deriv_contrib(sc, deriv_data, rxn_int_data,
                                            rxn_float_data, rxn_env_data,
                                            time_step);
      break;
    case RXN_TROE:
      rxn_gpu_troe_calc_deriv_contrib(sc, deriv_data, rxn_int_data,
                                      rxn_float_data, rxn_env_data, time_step);
      break;
  }
}

/** \brief Compute the time derivative f(t,y)
 *
 * \param time_step Current model time (s)
 * \param y Dependent variable array
 * \param yout Vector f(t,y) to calculate
 * \param use_deriv_est Flag to use an scale factor on f(t,y)
 * \param md Global data
 * \param sc Block data
 * \return Status code
 */
__device__ int cudaDevicef(double time_step, double *y, double *yout,
                           bool use_deriv_est, ModelDataGPU *md,
                           ModelDataVariable *sc) {
#ifdef CAMP_PROFILE_DEVICE_FUNCTIONS
  int clock_khz = md->clock_khz;
  clock_t start;
  start = clock();
#endif
  // On the first call to f(), the time step hasn't been set yet, so use the
  // default value
  time_step = sc->cv_next_h;
  time_step = time_step > 0. ? time_step : md->init_time_step;
  // Update the state array with the current dependent variable values.
  int checkflag = cudaDevicecamp_solver_update_model_state(md, sc, y);
  if (checkflag == CAMP_SOLVER_FAIL) {
#ifdef CAMP_PROFILE_DEVICE_FUNCTIONS
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadIdx.x == 0)
      sc->timef += ((double)(int)(clock() - start)) / (clock_khz * 1000);
#endif
    return CAMP_SOLVER_FAIL;
  }
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  // Get the Jacobian-estimated derivative
  md->dn0[i] = y[i] - md->J_state[i];
  cudaDeviceSpmv_CSR(md->dy, md->dn0, md->J_solver, md->djA, md->diA);
  md->dn0[i] = md->J_deriv[i] + md->dy[i];
  TimeDerivativeGPU deriv_data;
  // Set production and loss rates to start of global memory
  __syncthreads();
  deriv_data.production_rates = md->production_rates;
  deriv_data.loss_rates = md->loss_rates;
  __syncthreads();
  // Reset production and loss rates
  deriv_data.production_rates[i] = 0.0;
  deriv_data.loss_rates[i] = 0.0;
  __syncthreads();
  // Get pointers to each cell
  deriv_data.production_rates =
      &(md->production_rates[blockDim.x * blockIdx.x]);
  deriv_data.loss_rates = &(md->loss_rates[blockDim.x * blockIdx.x]);
  sc->grid_cell_state = &(md->state[md->n_per_cell_state_var * blockIdx.x]);
  __syncthreads();
  // Calculate the time derivative f(t,y)
#ifdef IS_DEBUG_MODE_removeAtomic
  if (threadIdx.x == 0) {
    for (int j = 0; j < md->n_rxn; j++) {
      cudaDevicerxn_calc_deriv(j, deriv_data, time_step, md, sc);
    }
  }
#else
  // Assign reactions to each threads evenly
  if (threadIdx.x < md->n_rxn) {  // Avoid case of less threads than reactions,
                                  // where thread would access non-existent data
    int n_iters = md->n_rxn / blockDim.x;
    for (int j = 0; j < n_iters; j++) {
      int i_rxn = threadIdx.x + j * blockDim.x;
      cudaDevicerxn_calc_deriv(i_rxn, deriv_data, time_step, md, sc);
    }
    // In case of non-even division of reactions, assign the remaining
    // reactions
    int residual = md->n_rxn % blockDim.x;
    if (threadIdx.x < residual) {
      int i_rxn = threadIdx.x + blockDim.x * n_iters;
      cudaDevicerxn_calc_deriv(i_rxn, deriv_data, time_step, md, sc);
    }
  }
#endif
  __syncthreads();
  // Reset pointers to global data
  deriv_data.production_rates = md->production_rates;
  deriv_data.loss_rates = md->loss_rates;
  __syncthreads();
  // Update output
  double *r_p = deriv_data.production_rates;
  double *r_l = deriv_data.loss_rates;
  // Avoid division by zero
  if (r_p[i] + r_l[i] != 0.0) {
    if (use_deriv_est) {
      double scale_fact = 1.0 / (r_p[i] + r_l[i]) /
                          (1.0 / (r_p[i] + r_l[i]) +
                           MAX_PRECISION_LOSS / fabs(r_p[i] - r_l[i]));
      yout[i] =
          scale_fact * (r_p[i] - r_l[i]) + (1.0 - scale_fact) * (md->dn0[i]);
    } else {
      yout[i] = r_p[i] - r_l[i];
    }
  } else {
    yout[i] = 0.0;
  }

#ifdef CAMP_PROFILE_DEVICE_FUNCTIONS
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (threadIdx.x == 0)
    sc->timef += ((double)(int)(clock() - start)) / (clock_khz * 1000);
#endif
  return 0;
}

/** \brief Try to improve guesses of y sent to the linear solver
 *
 * This function checks if there are any negative guessed concentrations,
 * and if there are it calculates a set of initial corrections to the
 * guessed state using the state at time \f$t_{n-1}\f$ and the derivative
 * \f$f_{n-1}\f$ and advancing the state according to:
 * \f[
 *   y_n = y_{n-1} + \sum_{j=1}^m h_j * f_j
 * \f]
 * where \f$h_j\f$ is the largest timestep possible where
 * \f[
 *   y_{j-1} + h_j * f_j > 0
 * \f]
 * and
 * \f[
 *   t_n = t_{n-1} + \sum_{j=1}^m h_j
 * \f]
 *
 * \param t_n Current time [s]
 * \param h_n Current time step size [s] If this is set to zero, the change hf
 *            is assumed to be an adjustment where y_n = y_n1 + hf
 * \param y_n Current guess for \f$y(t_n)\f$
 * \param y_n1 \f$y(t_{n-1})\f$
 * \param hf Current guess for change in \f$y\f$ from \f$t_{n-1}\f$ to
 *            \f$t_n\f$ [input/output]
 * \param atmp1 Temporary vector for calculations
 * \param acorr Vector of calculated adjustments to \f$y(t_n)\f$ [output]
 * \param md Global data
 * \param sc Block data
 * \return 1 if corrections were calculated, 0 if not, -1 if error
 */
__device__ int CudaDeviceguess_helper(double t_n, double h_n, double *y_n,
                                      double *y_n1, double *hf, double *atmp1,
                                      double *acorr, ModelDataGPU *md,
                                      ModelDataVariable *sc) {
  extern __shared__ double sdata[];
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  // Only try improvements when negative concentrations are predicted
  double min;
  cudaDevicemin(&min, y_n[i], sdata, md->n_shr_empty);
  if (min > -SMALL) {
    return 0;
  }
#ifdef CAMP_PROFILE_DEVICE_FUNCTIONS
  int clock_khz = md->clock_khz;
  clock_t start;
  start = clock();
#endif
  // Copy \f$y(t_{n-1})\f$ to working array
  atmp1[i] = y_n1[i];
  // Get  \f$f(t_{n-1})\f$
  if (h_n > 0.) {
    acorr[i] = (1. / h_n) * hf[i];
  } else {
    acorr[i] = hf[i];
  }
  // Advance state interatively
  double t_0 = h_n > 0. ? t_n - h_n : t_n - 1.;
  double t_j = 0.;
  for (int iter = 0; iter < GUESS_MAX_ITER && t_0 + t_j < t_n; iter++) {
    // Calculate \f$h_j\f$
    double h_j = t_n - (t_0 + t_j);
#ifdef IS_DEBUG_MODE_CudaDeviceguess_helper
    if (threadIdx.x == 0) {
      for (int j = 0; j < blockDim.x; j++) {
        double t_star = -atmp1[j + blockIdx.x * blockDim.x] /
                        acorr[j + blockIdx.x * blockDim.x];
        if ((t_star > 0. ||
             (t_star == 0. && acorr[j + blockIdx.x * blockDim.x] < 0.)) &&
            t_star < h_j) {
          h_j = t_star;
        }
      }
      sdata[0] = h_j;
    }
    __syncthreads();
    h_j = sdata[0];
    __syncthreads();
#else
    double t_star = -atmp1[i] / acorr[i];
    if (t_star < 0. || (t_star == 0. && acorr[i] >= 0.)) {
      t_star = h_j;
    }
    // Scale incomplete jumps
    cudaDevicemin(&min, t_star, sdata, md->n_shr_empty);
    if (min < h_j) {
      h_j = min;
      h_j *= 0.95 + 0.1 * iter / (double)GUESS_MAX_ITER;
    }
#endif
    h_j = t_n < t_0 + t_j + h_j ? t_n - (t_0 + t_j) : h_j;
    // Only make small changes to adjustment vectors used in Newton iteration
    if (h_n == 0. && t_n - (h_j + t_j + t_0) > md->cv_reltol) {
#ifdef CAMP_PROFILE_DEVICE_FUNCTIONS
      if (threadIdx.x == 0)
        sc->timeguess_helper +=
            ((double)(clock() - start)) / (clock_khz * 1000);
#endif
      return -1;
    }
    // Advance the state
    atmp1[i] += h_j * acorr[i];
    // Advance t_j
    t_j += h_j;
    // Recalculate the time derivative \f$f(t_j)\f$
    int fflag = cudaDevicef(t_0 + t_j, atmp1, acorr, true, md, sc);
    if (fflag == CAMP_SOLVER_FAIL) {
      acorr[i] = 0.;
#ifdef CAMP_PROFILE_DEVICE_FUNCTIONS
      if (threadIdx.x == 0)
        sc->timeguess_helper +=
            ((double)(clock() - start)) / (clock_khz * 1000);
#endif
      return -1;
    }
    if (iter == GUESS_MAX_ITER - 1 && t_0 + t_j < t_n) {
      if (h_n == 0.) {
#ifdef CAMP_PROFILE_DEVICE_FUNCTIONS
        if (threadIdx.x == 0)
          sc->timeguess_helper +=
              ((double)(clock() - start)) / (clock_khz * 1000);
#endif
        return -1;
      }
    }
  }
  // Set the correction vector
  acorr[i] = atmp1[i] - y_n[i];
  // Scale the initial corrections
  if (h_n > 0.) acorr[i] = acorr[i] * 0.999;
  // Update the hf vector
  hf[i] = atmp1[i] - y_n1[i];
#ifdef CAMP_PROFILE_DEVICE_FUNCTIONS
  if (threadIdx.x == 0)
    sc->timeguess_helper += ((double)(clock() - start)) / (clock_khz * 1000);
#endif
  return 1;
}

/**
 * @brief Calculates the Jacobian.
 *
 * The reaction data is accessed from the ModelDataGPU object using the reaction
 * index. The function then switches based on the reaction type and calls the
 * corresponding GPU function to calculate the Jacobian contribution.
 *
 * @param i_rxn The index of the reaction.
 * @param jac The JacobianGPU object.
 * @param md The ModelDataGPU object.
 * @param sc The ModelDataVariable object.
 */
__device__ void cudaDevicerxn_calc_Jac(int i_rxn, JacobianGPU jac,
                                       ModelDataGPU *md,
                                       ModelDataVariable *sc) {
  double *rxn_float_data =
      (double *)&(md->rxn_double[md->rxn_float_indices[i_rxn]]);
  int *int_data = (int *)&(md->rxn_int[md->rxn_int_indices[i_rxn]]);
  int *rxn_int_data = (int *)&(int_data[1]);
  double *rxn_env_data = &(md->rxn_env_data[md->n_rxn_env_data * blockIdx.x +
                                            md->rxn_env_idx[i_rxn]]);
  switch (int_data[0]) {
    case RXN_ARRHENIUS:
      rxn_gpu_arrhenius_calc_jac_contrib(sc, jac, rxn_int_data, rxn_float_data,
                                         rxn_env_data, sc->cv_next_h);
      break;
    case RXN_CMAQ_H2O2:
      rxn_gpu_CMAQ_H2O2_calc_jac_contrib(sc, jac, rxn_int_data, rxn_float_data,
                                         rxn_env_data, sc->cv_next_h);
      break;
    case RXN_CMAQ_OH_HNO3:
      rxn_gpu_CMAQ_OH_HNO3_calc_jac_contrib(
          sc, jac, rxn_int_data, rxn_float_data, rxn_env_data, sc->cv_next_h);
      break;
    case RXN_FIRST_ORDER_LOSS:
      rxn_gpu_first_order_loss_calc_jac_contrib(
          sc, jac, rxn_int_data, rxn_float_data, rxn_env_data, sc->cv_next_h);
      break;
    case RXN_PHOTOLYSIS:
      rxn_gpu_photolysis_calc_jac_contrib(sc, jac, rxn_int_data, rxn_float_data,
                                          rxn_env_data, sc->cv_next_h);
      break;
    case RXN_TROE:
      rxn_gpu_troe_calc_jac_contrib(sc, jac, rxn_int_data, rxn_float_data,
                                    rxn_env_data, sc->cv_next_h);
      break;
  }
}

/** \brief Compute the Jacobian
 *
 * \param md Global data
 * \param sc Block data
 * \return Status code
 */
__device__ int cudaDeviceJac(ModelDataGPU *md, ModelDataVariable *sc) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int retval;
#ifdef CAMP_PROFILE_DEVICE_FUNCTIONS
  int clock_khz = md->clock_khz;
  clock_t start;
  start = clock();
#endif
  // Calculate the the derivative for the current state y without
  // the estimated derivative from the last Jacobian calculation
  retval = cudaDevicef(sc->cv_next_h, md->dcv_y, md->dftemp, false, md, sc);
  if (retval == CAMP_SOLVER_FAIL) return CAMP_SOLVER_FAIL;
    // Calculate the reaction Jacobian
#ifdef CAMP_PROFILE_DEVICE_FUNCTIONS
  int clock_khz = md->clock_khz;
  clock_t start;
  start = clock();
#endif
  JacobianGPU *jac = &md->jac;
  JacobianGPU jacBlock;
  __syncthreads();
  // Set production and loss rates to each cell
  jacBlock.num_elem = jac->num_elem;
  jacBlock.production_partials =
      &(jac->production_partials[jacBlock.num_elem[0] * blockIdx.x]);
  jacBlock.loss_partials =
      &(jac->loss_partials[jacBlock.num_elem[0] * blockIdx.x]);
  sc->grid_cell_state = &(md->state[md->n_per_cell_state_var * blockIdx.x]);
  __syncthreads();
#ifdef IS_DEBUG_MODE_removeAtomic
  if (threadIdx.x == 0) {
    for (int j = 0; j < n_rxn; j++) {
      cudaDevicerxn_calc_Jac(j, jacBlock, md, sc);
    }
  }
#else
  if (threadIdx.x < md->n_rxn) {  // Avoid case of less threads than reactions,
                                  // where thread would access non-existent data
    int n_iters = md->n_rxn / blockDim.x;
    for (int j = 0; j < n_iters; j++) {
      int i_rxn = threadIdx.x + j * blockDim.x;
      cudaDevicerxn_calc_Jac(i_rxn, jacBlock, md, sc);
    }
    int residual = md->n_rxn % blockDim.x;
    if (threadIdx.x < residual) {
      int i_rxn = threadIdx.x + blockDim.x * n_iters;
      cudaDevicerxn_calc_Jac(i_rxn, jacBlock, md, sc);
    }
  }
#endif
  __syncthreads();
  // Set the solver Jacobian using the reaction Jacobians
  JacMap *jac_map = md->jac_map;
  int nnz = md->diA[blockDim.x];
  int n_iters = nnz / blockDim.x;
  for (int z = 0; z < n_iters; z++) {
    int j = threadIdx.x + z * blockDim.x;
    md->dA[jac_map[j].solver_id + nnz * blockIdx.x] =
        jacBlock.production_partials[jac_map[j].rxn_id] -
        jacBlock.loss_partials[jac_map[j].rxn_id];
    jacBlock.production_partials[jac_map[j].rxn_id] = 0.0;
    jacBlock.loss_partials[jac_map[j].rxn_id] = 0.0;
  }
  int residual = nnz % blockDim.x;
  if (threadIdx.x < residual) {
    int j = threadIdx.x + n_iters * blockDim.x;
    md->dA[jac_map[j].solver_id + nnz * blockIdx.x] =
        jacBlock.production_partials[jac_map[j].rxn_id] -
        jacBlock.loss_partials[jac_map[j].rxn_id];
    jacBlock.production_partials[jac_map[j].rxn_id] = 0.0;
    jacBlock.loss_partials[jac_map[j].rxn_id] = 0.0;
  }
  __syncthreads();

#ifdef CAMP_PROFILE_DEVICE_FUNCTIONS
  if (threadIdx.x == 0)
    sc->timecalc_Jac += ((double)(clock() - start)) / (clock_khz * 1000);
#endif
  // Save the Jacobian for use with derivative calculations
  nnz = md->diA[blockDim.x];
  n_iters = nnz / blockDim.x;
  for (int z = 0; z < n_iters; z++) {
    int j = threadIdx.x + z * blockDim.x + nnz * blockIdx.x;
    md->J_solver[j] = md->dA[j];
  }
  residual = nnz % blockDim.x;
  if (threadIdx.x < residual) {
    int j = threadIdx.x + n_iters * blockDim.x + nnz * blockIdx.x;
    md->J_solver[j] = md->dA[j];
  }
  md->J_state[i] = md->dcv_y[i];
  md->J_deriv[i] = md->dftemp[i];
#ifdef CAMP_PROFILE_DEVICE_FUNCTIONS
  if (threadIdx.x == 0)
    sc->timeJac += ((double)(clock() - start)) / (clock_khz * 1000);
#endif
  return 0;
}

// Functions equivalent to CVODE CPU solver (BDF method)

/**
 * \brief Determines whether to update a Jacobian matrix or use a stored version
 * based  on heuristics regarding previous convergence issues and the number of
 * time steps since it was last updated. It then creates the system matrix from
 * this, the 'gamma' factor, and the identity matrix.
 *
 * A = I-gamma*J.
 *
 * This routine then calls the LS 'setup' routine with A.
 *
 * \param md Pointer to the ModelDataGPU structure.
 * \param sc Pointer to the ModelDataVariable structure.
 * \param convfail The convergence failure flag.
 * \return Returns 0 on success, -1 if an error occurred during the Jacobi
 * calculation, and 1 if a guess was made.
 */
__device__ int cudaDevicecvDlsSetup(ModelDataGPU *md, ModelDataVariable *sc,
                                    int convfail) {
  extern __shared__ int flag_shr[];
  double dgamma;
  int jbad, jok;
  /* Use nst, gamma/gammap, and convfail to set J eval. flag jok */
  dgamma = fabs((sc->cv_gamma / sc->cv_gammap) - 1.);
  jbad = (sc->cv_nst == 0) || (sc->cv_nst > sc->nstlj + CVD_MSBJ) ||
         ((convfail == CV_FAIL_BAD_J) && (dgamma < CVD_DGMAX)) ||
         (convfail == CV_FAIL_OTHER);
  jok = !jbad;
  /* If jok = TRUE, use saved copy of J */
  if (jok == 1) {
    sc->cv_jcur = 0;
    cudaDeviceJacCopy(md->diA, md->dsavedJ, md->dA);
    /* If jok = SUNFALSE, call jac routine for new J value */
  } else {
    sc->nstlj = sc->cv_nst;
    sc->cv_jcur = 1;
    int guess_flag = cudaDeviceJac(md, sc);
    if (guess_flag < 0) {
      return -1;
    }
    if (guess_flag > 0) {
      return 1;
    }
    cudaDeviceJacCopy(md->diA, md->dA, md->dsavedJ);
  }
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  // Reset vector for linear solver
  md->dx[i] = 0.;
  // Preconditioner linear solver
  cudaDeviceBCGprecond_2(md->dA, md->djA, md->diA, md->ddiag, -sc->cv_gamma);
  return 0;
}

/**
 * @brief Solves a linear system of equations using the Biconjugate Gradient
 * (BCG) algorithm.
 *
 * Details explained in C. Guzman et. al. "Optimized thread-block arrangement in
 * a GPU implementation of a linear solver for atmospheric chemistry
 * mechanisms", Computer Physics Communications 2024
 *
 * @param md A pointer to the ModelDataGPU object.
 * @param sc A pointer to the ModelDataVariable object.
 */
__device__ void solveBcgCudaDeviceCVODE(ModelDataGPU *md,
                                        ModelDataVariable *sc) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  double alpha, rho0, omega0, beta, rho1, temp1, temp2;
  alpha = rho0 = omega0 = beta = rho1 = temp1 = temp2 = 1.0;
  md->dn0[i] = 0.0;
  md->dp0[i] = 0.0;
  cudaDeviceSpmv_CSR(md->dr0, md->dx, md->dA, md->djA, md->diA);
  md->dr0[i] = md->dtempv[i] - md->dr0[i];
  md->dr0h[i] = md->dr0[i];
  int it = 0;
  while (it < 1000 && temp1 > 1.0E-30) {
    cudaDevicedotxy_2(md->dr0, md->dr0h, &rho1, md->n_shr_empty);
    beta = (rho1 / rho0) * (alpha / omega0);
    md->dp0[i] = beta * md->dp0[i] + md->dr0[i] - md->dn0[i] * omega0 * beta;
    md->dy[i] = md->ddiag[i] * md->dp0[i];
    cudaDeviceSpmv_CSR(md->dn0, md->dy, md->dA, md->djA, md->diA);
    cudaDevicedotxy_2(md->dr0h, md->dn0, &temp1, md->n_shr_empty);
    alpha = rho1 / temp1;
    md->ds[i] = md->dr0[i] - alpha * md->dn0[i];
    md->dx[i] += alpha * md->dy[i];
    md->dy[i] = md->ddiag[i] * md->ds[i];
    cudaDeviceSpmv_CSR(md->dt, md->dy, md->dA, md->djA, md->diA);
    md->dr0[i] = md->ddiag[i] * md->dt[i];
    cudaDevicedotxy_2(md->dy, md->dr0, &temp1, md->n_shr_empty);
    cudaDevicedotxy_2(md->dr0, md->dr0, &temp2, md->n_shr_empty);
    omega0 = temp1 / temp2;
    md->dx[i] += omega0 * md->dy[i];
    md->dr0[i] = md->ds[i] - omega0 * md->dt[i];
    md->dt[i] = 0.0;
    cudaDevicedotxy_2(md->dr0, md->dr0, &temp1, md->n_shr_empty);
    temp1 = sqrt(temp1);
    rho0 = rho1;
    it++;
  }
#ifdef CAMP_PROFILE_DEVICE_FUNCTIONS
  if (threadIdx.x == 0) sc->counterBCGInternal += it;
  if (threadIdx.x == 0) sc->counterBCG++;
#endif
}

/**
 * Performs the Newton iteration. If the iteration succeeds,
 * it returns the value CV_SUCCESS. If not, it may signal the cvNlsNewton
 * routine to call lsetup again and reattempt the iteration, by
 * returning the value TRY_AGAIN. (In this case, cvNlsNewton must set
 * convfail to CV_FAIL_BAD_J before calling setup again).
 * Otherwise, this routine returns one of the appropriate values
 * CV_LSOLVE_FAIL, CV_RHSFUNC_FAIL, CONV_FAIL, or RHSFUNC_RECVR back
 * to cvNlsNewton.
 *
 * @param md The ModelDataGPU object
 * @param sc The ModelDataVariable object
 * @return The result of the Newton iteration
 */
__device__ int cudaDevicecvNewtonIteration(ModelDataGPU *md,
                                           ModelDataVariable *sc) {
  extern __shared__ double flag_shr2[];
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  double del, delp, dcon;
  int m = 0;
  /* Initialize delp to avoid compiler warning message */
  del = delp = 0.0;
  int retval;
#ifdef CAMP_PROFILE_DEVICE_FUNCTIONS
  int clock_khz = md->clock_khz;
  clock_t start;
#endif
  /* Looping point for Newton iteration */
  for (;;) {
    /* Evaluate the residual of the nonlinear system */
    md->dtempv[i] = sc->cv_rl1 * md->dzn[1][i] + md->cv_acor[i];
    md->dtempv[i] = sc->cv_gamma * md->dftemp[i] - md->dtempv[i];
#ifdef CAMP_PROFILE_DEVICE_FUNCTIONS
    start = clock();
#endif
    /* Call the linear solver function */
    solveBcgCudaDeviceCVODE(md, sc);
#ifdef CAMP_PROFILE_DEVICE_FUNCTIONS
    if (threadIdx.x == 0)
      sc->timeBCG += ((double)(int)(clock() - start)) / (clock_khz * 1000);
#endif
    md->dtempv[i] = md->dx[i];

    /* Get WRMS norm of correction */
    cudaDeviceVWRMS_Norm_2(md->dx, md->dewt, &del, md->n_shr_empty);
    md->dftemp[i] = md->dcv_y[i] + md->dtempv[i];

    /* Improve guesses for zn(0) */
    int guessflag =
        CudaDeviceguess_helper(sc->cv_tn, 0., md->dftemp, md->dcv_y, md->dtempv,
                               md->dtempv1, md->dp0, md, sc);
    if (guessflag < 0) {
      if (!(sc->cv_jcur)) {
        return TRY_AGAIN;
      } else {
        return RHSFUNC_RECVR;
      }
    }

    /* Check for negative concentrations */
    md->dftemp[i] = md->dcv_y[i] + md->dtempv[i];
    double min;
    cudaDevicemin(&min, md->dftemp[i], flag_shr2, md->n_shr_empty);
    if (min < -CAMP_TINY) {
      return CONV_FAIL;
    }

    /* Add correction to acor and y */
    md->cv_acor[i] += md->dtempv[i];
    md->dcv_y[i] = md->dzn[0][i] + md->cv_acor[i];

    /* Test for convergence.  If m > 0, an estimate of the convergence
   rate constant is stored in crate, and used in the test. */
    if (m > 0) {
      sc->cv_crate = SUNMAX(0.3 * sc->cv_crate, del / delp);
    }
    dcon = del * SUNMIN(1.0, sc->cv_crate) /
           md->cv_tq[4 + blockIdx.x * (NUM_TESTS + 1)];
    __syncthreads();
    flag_shr2[0] = 0;
    __syncthreads();
    if (dcon <= 1.) {
      cudaDeviceVWRMS_Norm_2(md->cv_acor, md->dewt, &sc->cv_acnrm,
                             md->n_shr_empty);
      sc->cv_jcur = 0;
      return CV_SUCCESS;
    }
    m++;

    /* Stop at maxcor iterations or if iter. seems to be diverging.
   If still not converged and Jacobian data is not current,
   signal to try the solution again */
    if ((m == NLS_MAXCOR) || ((m >= 2) && (del > RDIV * delp))) {
      if (!(sc->cv_jcur)) {
        return TRY_AGAIN;
      } else {
        return RHSFUNC_RECVR;
      }
    }

    /* Save norm of correction, evaluate f, and loop again */
    delp = del;
    retval = cudaDevicef(sc->cv_next_h, md->dcv_y, md->dftemp, true, md, sc);
    md->cv_acor[i] = md->dcv_y[i] + md->dzn[0][i];
    if (retval < 0) {
      return CV_RHSFUNC_FAIL;
    }
    if (retval > 0) {
      if (!(sc->cv_jcur)) {
        return TRY_AGAIN;
      } else {
        return RHSFUNC_RECVR;
      }
    }
  } /* end loop */
}

/**
 * cvNlsNewton
 *
 * This routine handles the Newton iteration. It calls lsetup if
 * indicated, calls cvNewtonIteration to perform the iteration, and
 * retries a failed attempt at Newton iteration if that is indicated.
 *
 * Possible return values:
 *
 *   CV_SUCCESS       ---> continue with error test
 *
 *   CV_RHSFUNC_FAIL  -+
 *   CV_LSETUP_FAIL    |-> halt the integration
 *   CV_LSOLVE_FAIL   -+
 *
 *   CONV_FAIL        -+
 *   RHSFUNC_RECVR    -+-> predict again or stop if too many
 *
 * @param nflag The flag indicating the type of call to cvNlsNewton.
 * @param md The ModelDataGPU structure containing the model data on the GPU.
 * @param sc The ModelDataVariable structure containing the model data
 * variables.
 * @return The return value indicating the success or failure of the Newton
 * iteration.
 */
__device__ int cudaDevicecvNlsNewton(int nflag, ModelDataGPU *md,
                                     ModelDataVariable *sc) {
  extern __shared__ int flag_shr[];
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int retval = 0;
#ifdef CAMP_PROFILE_DEVICE_FUNCTIONS
  int clock_khz = md->clock_khz;
  clock_t start;
#endif
  /* Set flag convfail, input to lsetup for its evaluation decision */
  int convfail = ((nflag == FIRST_CALL) || (nflag == PREV_ERR_FAIL))
                     ? CV_NO_FAILURES
                     : CV_FAIL_OTHER;
  int dgamrat = fabs(sc->cv_gamrat - 1.);
  /* Decide whether or not to call setup routine*/
  int callSetup = (nflag == PREV_CONV_FAIL) || (nflag == PREV_ERR_FAIL) ||
                  (sc->cv_nst == 0) || (sc->cv_nst >= sc->cv_nstlp + MSBP) ||
                  (dgamrat > DGMAX);
  md->dftemp[i] = md->dzn[0][i] - md->cv_last_yn[i];
  md->cv_acor_init[i] = 0.;
  /* Improve guesses for zn(0) */
  int guessflag =
      CudaDeviceguess_helper(sc->cv_tn, sc->cv_h, md->dzn[0], md->cv_last_yn,
                             md->dftemp, md->dtempv1, md->cv_acor_init, md, sc);
  if (guessflag < 0) {
    return RHSFUNC_RECVR;
  }
  /* Looping point for the solution of the nonlinear system.
   Evaluate f at the predicted y, call cvDlsSetup if indicated, and
   call cvNewtonIteration for the Newton iteration itself. */
  for (;;) {
    /* Load prediction into y vector */
    md->dcv_y[i] = md->dzn[0][i] + md->cv_acor_init[i];
    retval = cudaDevicef(sc->cv_tn, md->dcv_y, md->dftemp, true, md, sc);
    if (retval < 0) {
      return CV_RHSFUNC_FAIL;
    }
    if (retval > 0) {
      return RHSFUNC_RECVR;
    }
    if (callSetup) {
#ifdef CAMP_PROFILE_DEVICE_FUNCTIONS
      start = clock();
#endif
      int linflag = cudaDevicecvDlsSetup(md, sc, convfail);
#ifdef CAMP_PROFILE_DEVICE_FUNCTIONS
      if (threadIdx.x == 0)
        sc->timelinsolsetup += ((double)(clock() - start)) / (clock_khz * 1000);
#endif
      callSetup = 0;
      sc->cv_gamrat = sc->cv_crate = 1.0;
      sc->cv_gammap = sc->cv_gamma;
      sc->cv_nstlp = sc->cv_nst;
      /* Break if lsetup failed */
      if (linflag < 0) {
        flag_shr[0] = CV_LSETUP_FAIL;
        break;
      }
      if (linflag > 0) {
        flag_shr[0] = CONV_FAIL;
        break;
      }
    }
    /* Set acor to the initial guess for adjustments to the y vector */
    md->cv_acor[i] = md->cv_acor_init[i];
#ifdef CAMP_PROFILE_DEVICE_FUNCTIONS
    start = clock();
#endif
    /* Do the Newton iteration */
    int nItflag = cudaDevicecvNewtonIteration(md, sc);
#ifdef CAMP_PROFILE_DEVICE_FUNCTIONS
    if (threadIdx.x == 0)
      sc->timeNewtonIteration +=
          ((double)(clock() - start)) / (clock_khz * 1000);
#endif
    /* If there is a convergence failure and the Jacobian-related
       data appears not to be current, loop again with a call to cvDlsSetup
       in which convfail=CV_FAIL_BAD_J.  Otherwise return. */
    if (nItflag != TRY_AGAIN) {
      return nItflag;
    }
    callSetup = 1;
    convfail = CV_FAIL_BAD_J;
  }
  return nflag;
}

/**
 * Rescales the Nordsieck array by multiplying the jth column zn[j] by eta^j, j
 * = 1, ..., q. Then the value of h is rescaled by eta, and hscale is reset to
 * h.
 *
 * @param md - Pointer to ModelDataGPU struct
 * @param sc - Pointer to ModelDataVariable struct
 */
__device__ void cudaDevicecvRescale(ModelDataGPU *md, ModelDataVariable *sc) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  double factor;
  factor = sc->cv_eta;
  for (int j = 1; j <= sc->cv_q; j++) {
    md->dzn[j][i] *= factor;
    factor *= sc->cv_eta;
  }
  sc->cv_h = sc->cv_hscale * sc->cv_eta;
  sc->cv_next_h = sc->cv_h;
  sc->cv_hscale = sc->cv_h;
}

/**
 * Restores the value of tn to saved_t and undoes the prediction.
 * After execution of cvRestore, the Nordsieck array zn has the same values as
 * before the call to cvPredict.
 *
 * @param md - Pointer to ModelDataGPU structure.
 * @param sc - Pointer to ModelDataVariable structure.
 * @param saved_t - The value of tn to be restored.
 */
__device__ void cudaDevicecvRestore(ModelDataGPU *md, ModelDataVariable *sc,
                                    double saved_t) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j, k;
  sc->cv_tn = saved_t;
  for (k = 1; k <= sc->cv_q; k++) {
    for (j = sc->cv_q; j >= k; j--) {
      md->dzn[j - 1][i] -= md->dzn[j][i];
    }
  }
  md->dzn[0][i] = md->cv_last_yn[i];
}

/**
 * cvHandleNFlag
 *
 * This routine takes action on the return value nflag = *nflagPtr
 * returned by cvNls, as follows:
 *
 * If cvNls succeeded in solving the nonlinear system, then
 * cvHandleNFlag returns the constant DO_ERROR_TEST, which tells cvStep
 * to perform the error test.
 *
 * If the nonlinear system was not solved successfully, then ncfn and
 * ncf = *ncfPtr are incremented and Nordsieck array zn is restored.
 *
 * If the solution of the nonlinear system failed due to an
 * unrecoverable failure by setup, we return the value CV_LSETUP_FAIL.
 *
 * If it failed due to an unrecoverable failure in solve, then we return
 * the value CV_LSOLVE_FAIL.
 *
 * If it failed due to an unrecoverable failure in rhs, then we return
 * the value CV_RHSFUNC_FAIL.
 *
 * Otherwise, a recoverable failure occurred when solving the
 * nonlinear system (cvNls returned nflag == CONV_FAIL or RHSFUNC_RECVR).
 * In this case, if ncf is now equal to maxncf or |h| = hmin,
 * we return the value CV_CONV_FAILURE (if nflag=CONV_FAIL) or
 * CV_REPTD_RHSFUNC_ERR (if nflag=RHSFUNC_RECVR).
 * If not, we set *nflagPtr = PREV_CONV_FAIL and return the value
 * PREDICT_AGAIN, telling cvStep to reattempt the step.
 *
 * @param md - Pointer to ModelDataGPU struct
 * @param sc - Pointer to ModelDataVariable struct
 * @param nflagPtr - Pointer to the return value nflag
 * @param saved_t - The saved value of t
 * @param ncfPtr - Pointer to the value of ncf
 * @return int - The return value indicating the action to be taken
 */
__device__ int cudaDevicecvHandleNFlag(ModelDataGPU *md, ModelDataVariable *sc,
                                       int *nflagPtr, double saved_t,
                                       int *ncfPtr) {
  extern __shared__ int flag_shr[];
  if (*nflagPtr == CV_SUCCESS) {
    return (DO_ERROR_TEST);
  }

  /* The nonlinear soln. failed; increment ncfn and restore zn */
  cudaDevicecvRestore(md, sc, saved_t);

  /* Return if lsetup, lsolve, or rhs failed unrecoverably */
  if (*nflagPtr == CV_LSETUP_FAIL) return (CV_LSETUP_FAIL);
  if (*nflagPtr == CV_LSOLVE_FAIL) return (CV_LSOLVE_FAIL);
  if (*nflagPtr == CV_RHSFUNC_FAIL) return (CV_RHSFUNC_FAIL);

  /* At this point, nflag = CONV_FAIL or RHSFUNC_RECVR; increment ncf */
  (*ncfPtr)++;
  sc->cv_etamax = 1.;

  /* If we had maxncf failures or |h| = hmin,
   return CV_CONV_FAILURE or CV_REPTD_RHSFUNC_ERR. */
  if ((fabs(sc->cv_h) <= sc->cv_hmin * ONEPSM) ||
      (*ncfPtr == CAMP_SOLVER_DEFAULT_MAX_CONV_FAILS)) {
    if (*nflagPtr == CONV_FAIL) return (CV_CONV_FAILURE);
    if (*nflagPtr == RHSFUNC_RECVR) return (CV_REPTD_RHSFUNC_ERR);
  }

  /* Reduce step size; return to reattempt the step */
  sc->cv_eta = SUNMAX(ETACF, sc->cv_hmin / fabs(sc->cv_h));
  *nflagPtr = PREV_CONV_FAIL;
  cudaDevicecvRescale(md, sc);
  return (PREDICT_AGAIN);
}

/**
 * \brief Sets the test quantity array `tq` in the case `lmm == CV_BDF`.
 *
 * This function sets the test quantity array `tq` based on the given
 * parameters.
 *
 * \param md The pointer to the `ModelDataGPU` structure.
 * \param sc The pointer to the `ModelDataVariable` structure.
 * \param hsum The sum of `h` values.
 * \param alpha0 The value of `alpha0`.
 * \param alpha0_hat The value of `alpha0_hat`.
 * \param xi_inv The inverse of `xi`.
 * \param xistar_inv The inverse of `xistar`.
 */
__device__ void cudaDevicecvSetTqBDF(ModelDataGPU *md, ModelDataVariable *sc,
                                     double hsum, double alpha0,
                                     double alpha0_hat, double xi_inv,
                                     double xistar_inv) {
  extern __shared__ int flag_shr[];
  double A1, A2, A3, A4, A5, A6;
  double C, Cpinv, Cppinv;
  A1 = 1. - alpha0_hat + alpha0;
  A2 = 1. + sc->cv_q * A1;
  md->cv_tq[2 + blockIdx.x * (NUM_TESTS + 1)] = fabs(A1 / (alpha0 * A2));
  md->cv_tq[5 + blockIdx.x * (NUM_TESTS + 1)] = fabs(
      A2 * xistar_inv / (md->cv_l[sc->cv_q + blockIdx.x * L_MAX] * xi_inv));
  if (sc->cv_qwait == 1) {
    if (sc->cv_q > 1) {
      C = xistar_inv / md->cv_l[sc->cv_q + blockIdx.x * L_MAX];
      A3 = alpha0 + 1. / sc->cv_q;
      A4 = alpha0_hat + xi_inv;
      Cpinv = (1. - A4 + A3) / A3;
      md->cv_tq[1 + blockIdx.x * (NUM_TESTS + 1)] = fabs(C * Cpinv);
    } else
      md->cv_tq[1 + blockIdx.x * (NUM_TESTS + 1)] = 1.;
    hsum += md->cv_tau[sc->cv_q + blockIdx.x * (L_MAX + 1)];
    xi_inv = sc->cv_h / hsum;
    A5 = alpha0 - (1. / (sc->cv_q + 1));
    A6 = alpha0_hat - xi_inv;
    Cppinv = (1. - A6 + A5) / A2;
    md->cv_tq[3 + blockIdx.x * (NUM_TESTS + 1)] =
        fabs(Cppinv / (xi_inv * (sc->cv_q + 2) * A5));
  }
  md->cv_tq[4 + blockIdx.x * (NUM_TESTS + 1)] =
      CV_NLSCOEF / md->cv_tq[2 + blockIdx.x * (NUM_TESTS + 1)];
}

/**
 * \brief This routine computes the coefficients l and tq in the case lmm ==
 * CV_BDF.
 *
 * cvSetBDF calls cvSetTqBDF to set the test quantity array tq.
 *
 * The components of the array l are the coefficients of a polynomial Lambda(x)
 * = l_0 + l_1 x + ... + l_q x^q, given by Lambda(x) = (1 + x / xi*_q) * PRODUCT
 * (1 + x / xi_i) , where xi_i = [t_n - t_(n-i)] / h.
 *
 * The array tq is set to test quantities used in the convergence test, the
 * error test, and the selection of h at a new order.
 *
 * \param md The ModelDataGPU pointer.
 * \param sc The ModelDataVariable pointer.
 */
__device__ void cudaDevicecvSetBDF(ModelDataGPU *md, ModelDataVariable *sc) {
  extern __shared__ int flag_shr[];
  double alpha0, alpha0_hat, xi_inv, xistar_inv, hsum;
  int z, j;
  md->cv_l[0 + blockIdx.x * L_MAX] = md->cv_l[1 + blockIdx.x * L_MAX] = xi_inv =
      xistar_inv = 1.;
  for (z = 2; z <= sc->cv_q; z++) md->cv_l[z + blockIdx.x * L_MAX] = 0.;
  alpha0 = alpha0_hat = -1.;
  hsum = sc->cv_h;
  if (sc->cv_q > 1) {
    for (j = 2; j < sc->cv_q; j++) {
      hsum += md->cv_tau[j - 1 + blockIdx.x * (L_MAX + 1)];
      xi_inv = sc->cv_h / hsum;
      alpha0 -= 1. / j;
      for (z = j; z >= 1; z--)
        md->cv_l[z + blockIdx.x * L_MAX] +=
            md->cv_l[z - 1 + blockIdx.x * L_MAX] * xi_inv;
      /* The l[i] are coefficients of product(1 to j) (1 + x/xi_i) */
    }
    alpha0 -= 1. / sc->cv_q;
    xistar_inv = -md->cv_l[1 + blockIdx.x * L_MAX] - alpha0;
    hsum += md->cv_tau[sc->cv_q - 1 + blockIdx.x * (L_MAX + 1)];
    xi_inv = sc->cv_h / hsum;
    alpha0_hat = -md->cv_l[1 + blockIdx.x * L_MAX] - xi_inv;
    for (z = sc->cv_q; z >= 1; z--)
      md->cv_l[z + blockIdx.x * L_MAX] +=
          md->cv_l[z - 1 + blockIdx.x * L_MAX] * xistar_inv;
  }
  cudaDevicecvSetTqBDF(md, sc, hsum, alpha0, alpha0_hat, xi_inv, xistar_inv);
}

/*
 * cvSet
 *
 * This routine is a high level routine which calls cvSetBDF to set the
 * polynomial l, the test quantity array tq, and the related variables rl1,
 * gamma, and gamrat.
 *
 * The array tq is loaded with constants used in the control of estimated
 * local errors and in the nonlinear convergence test.  Specifically, while
 * running at order q, the components of tq are as follows:
 *   tq[1] = a coefficient used to get the est. local error at order q-1
 *   tq[2] = a coefficient used to get the est. local error at order q
 *   tq[3] = a coefficient used to get the est. local error at order q+1
 *   tq[4] = constant used in nonlinear iteration convergence test
 *   tq[5] = coefficient used to get the order q+2 derivative vector used in
 *           the est. local error at order q+1
 */
__device__ void cudaDevicecvSet(ModelDataGPU *md, ModelDataVariable *sc) {
  extern __shared__ int flag_shr[];
  cudaDevicecvSetBDF(md, sc);
  sc->cv_rl1 = 1.0 / md->cv_l[1 + blockIdx.x * L_MAX];
  sc->cv_gamma = sc->cv_h * sc->cv_rl1;
  if (sc->cv_nst == 0) {
    sc->cv_gammap = sc->cv_gamma;
  }
  sc->cv_gamrat = (sc->cv_nst > 0) ? sc->cv_gamma / sc->cv_gammap
                                   : 1.;  // protect x / x != 1.0
}

/**
 * cvPredict
 *
 * This routine advances tn by the tentative step size h, and computes
 * the predicted array z_n(0), which is overwritten on zn. The
 * prediction of zn is done by repeated additions.
 * If tstop is enabled, it is possible for tn + h to be past tstop by roundoff,
 * and in that case, we reset tn (after incrementing by h) to tstop.
 *
 * @param md The pointer to the ModelDataGPU struct.
 * @param sc The pointer to the ModelDataVariable struct.
 */
__device__ void cudaDevicecvPredict(ModelDataGPU *md, ModelDataVariable *sc) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j, k;
  sc->cv_tn += sc->cv_h;
  md->cv_last_yn[i] = md->dzn[0][i];
  for (k = 1; k <= sc->cv_q; k++) {
    for (j = sc->cv_q; j >= k; j--) {
      md->dzn[j - 1][i] += md->dzn[j][i];
    }
  }
}

/**
 * Decreases the history array on a decrease in the order q in the case that lmm
 * == CV_BDF. Each zn[j] is adjusted by a multiple of zn[q]. The coefficients in
 * the adjustment are the coefficients of the polynomial
 * x*x*(x+xi_1)*...*(x+xi_j), where xi_j = [t_n - t_(n-j)]/h.
 *
 * @param md The ModelDataGPU object.
 * @param sc The ModelDataVariable object.
 */
__device__ void cudaDevicecvDecreaseBDF(ModelDataGPU *md,
                                        ModelDataVariable *sc) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  double hsum, xi;
  int z, j;
  for (z = 0; z <= BDF_Q_MAX; z++) md->cv_l[z + blockIdx.x * L_MAX] = 0.;
  md->cv_l[2 + blockIdx.x * L_MAX] = 1.;
  hsum = 0.;
  for (j = 1; j <= sc->cv_q - 2; j++) {
    hsum += md->cv_tau[j + blockIdx.x * (L_MAX + 1)];
    xi = hsum / sc->cv_hscale;
    for (z = j + 2; z >= 2; z--)
      md->cv_l[z + blockIdx.x * L_MAX] = md->cv_l[z + blockIdx.x * L_MAX] * xi +
                                         md->cv_l[z - 1 + blockIdx.x * L_MAX];
  }
  for (j = 2; j < sc->cv_q; j++) {
    md->dzn[j][i] = -md->cv_l[j + blockIdx.x * L_MAX] * md->dzn[sc->cv_q][i] +
                    md->dzn[j][i];
  }
}

/**
 * @brief Performs the local error test.
 *
 * This routine performs the local error test. The weighted local error norm dsm
 * is loaded into *dsmPtr, and the test dsm ?<= 1 is made.
 *
 * If the test passes, cvDoErrorTest returns CV_SUCCESS.
 *
 * If the test fails, we undo the step just taken (call cvRestore) and
 *
 *   - if maxnef error test failures have occurred or if SUNRabs(h) = hmin, we
 * return CV_ERR_FAILURE.
 *
 *   - if more than MXNEF1 error test failures have occurred, an order reduction
 * is forced. If already at order 1, restart by reloading zn from scratch. If
 * f() fails we return either CV_RHSFUNC_FAIL or CV_UNREC_RHSFUNC_ERR (no
 * recovery is possible at this stage).
 *
 *   - otherwise, set *nflagPtr to PREV_ERR_FAIL, and return TRY_AGAIN.
 *
 * @param md The ModelDataGPU object.
 * @param sc The ModelDataVariable object.
 * @param nflagPtr Pointer to the flag indicating the status of the error test.
 * @param saved_t The saved value of t.
 * @param nefPtr Pointer to the number of error test failures.
 * @param dsmPtr Pointer to the weighted local error norm.
 * @return CV_SUCCESS if the test passes, CV_ERR_FAILURE if the test fails and
 * maxnef error test failures have occurred or SUNRabs(h) = hmin, TRY_AGAIN if
 * the test fails and more than MXNEF1 error test failures have occurred,
 * CV_RHSFUNC_FAIL if f() fails, CV_UNREC_RHSFUNC_ERR if f() fails and no
 * recovery is possible at this stage.
 */
__device__ int cudaDevicecvDoErrorTest(ModelDataGPU *md, ModelDataVariable *sc,
                                       int *nflagPtr, double saved_t,
                                       int *nefPtr, double *dsmPtr) {
  extern __shared__ double flag_shr2[];
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  double dsm;
  double min_val;
  int retval;

  /* Find the minimum concentration and if it's small and negative, make it
   * positive */
  md->dftemp[i] = md->cv_l[blockIdx.x * L_MAX] * md->cv_acor[i] + md->dzn[0][i];
  cudaDevicemin(&min_val, md->dftemp[i], flag_shr2, md->n_shr_empty);
  if (min_val < 0. && min_val > -CAMP_TINY) {
    md->dftemp[i] = fabs(md->dftemp[i]);
    md->dzn[0][i] =
        md->dftemp[i] - md->cv_l[0 + blockIdx.x * L_MAX] * md->cv_acor[i];
    min_val = 0.;
  }
  dsm = sc->cv_acnrm * md->cv_tq[2 + blockIdx.x * (NUM_TESTS + 1)];

  /* If est. local error norm dsm passes test and there are no negative values,
   * return CV_SUCCESS */
  *dsmPtr = dsm;
  if (dsm <= 1. && min_val >= 0.) return (CV_SUCCESS);

  /* Test failed; increment counters, set nflag, and restore zn array */
  (*nefPtr)++;
  *nflagPtr = PREV_ERR_FAIL;
  cudaDevicecvRestore(md, sc, saved_t);

  /* At maxnef failures or |h| = hmin, return CV_ERR_FAILURE */
  if ((fabs(sc->cv_h) <= sc->cv_hmin * ONEPSM) ||
      (*nefPtr == CAMP_SOLVER_DEFAULT_MAX_CONV_FAILS))
    return (CV_ERR_FAILURE);

  /* Set etamax = 1 to prevent step size increase at end of this step */
  sc->cv_etamax = 1.;

  /* Set h ratio eta from dsm, rescale, and return for retry of step */
  if (*nefPtr <= MXNEF1) {
    sc->cv_eta = 1. / (dSUNRpowerR(BIAS2 * dsm, 1. / sc->cv_L) + ADDON);
    sc->cv_eta =
        SUNMAX(ETAMIN, SUNMAX(sc->cv_eta, sc->cv_hmin / fabs(sc->cv_h)));
    if (*nefPtr >= SMALL_NEF) sc->cv_eta = SUNMIN(sc->cv_eta, ETAMXF);
    cudaDevicecvRescale(md, sc);
    return (TRY_AGAIN);
  }
  /* After MXNEF1 failures, force an order reduction and retry step */
  if (sc->cv_q > 1) {
    sc->cv_eta = SUNMAX(ETAMIN, sc->cv_hmin / fabs(sc->cv_h));
    cudaDevicecvDecreaseBDF(md, sc);
    sc->cv_L = sc->cv_q;
    sc->cv_q--;
    sc->cv_qwait = sc->cv_L;
    cudaDevicecvRescale(md, sc);
    return (TRY_AGAIN);
  }

  /* If already at order 1, restart: reload zn from scratch */
  sc->cv_eta = SUNMAX(ETAMIN, sc->cv_hmin / fabs(sc->cv_h));
  sc->cv_h *= sc->cv_eta;
  sc->cv_next_h = sc->cv_h;
  sc->cv_hscale = sc->cv_h;
  sc->cv_qwait = 10;
  retval = cudaDevicef(sc->cv_tn, md->dzn[0], md->dtempv, true, md, sc);
  if (retval < 0) return (CV_RHSFUNC_FAIL);
  if (retval > 0) return (CV_UNREC_RHSFUNC_ERR);
  md->dzn[1][i] = sc->cv_h * md->dtempv[i];
  return (TRY_AGAIN);
}

/**
 * \brief This routine performs various update operations when the solution
 * to the nonlinear system has passed the local error test.
 *
 * We increment the step counter nst, record the values hu and qu,
 * update the tau array, and apply the corrections to the zn array.
 * The tau[i] are the last q values of h, with tau[1] the most recent.
 * The counter qwait is decremented, and if qwait == 1 (and q < qmax)
 * we save acor and cv_mem->cv_tq[5] for a possible order increase.
 *
 * \param md Pointer to the ModelDataGPU structure.
 * \param sc Pointer to the ModelDataVariable structure.
 */
__device__ void cudaDevicecvCompleteStep(ModelDataGPU *md,
                                         ModelDataVariable *sc) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int z, j;
  sc->cv_nst++;
  sc->cv_hu = sc->cv_h;
  for (z = sc->cv_q; z >= 2; z--)
    md->cv_tau[z + blockIdx.x * (L_MAX + 1)] =
        md->cv_tau[z - 1 + blockIdx.x * (L_MAX + 1)];
  if ((sc->cv_q == 1) && (sc->cv_nst > 1))
    md->cv_tau[2 + blockIdx.x * (L_MAX + 1)] =
        md->cv_tau[1 + blockIdx.x * (L_MAX + 1)];
  md->cv_tau[1 + blockIdx.x * (L_MAX + 1)] = sc->cv_h;
  /* Apply correction to column j of zn: l_j * Delta_n */
  for (j = 0; j <= sc->cv_q; j++) {
    md->dzn[j][i] += md->cv_l[j + blockIdx.x * L_MAX] * md->cv_acor[i];
  }
  sc->cv_qwait--;
  if ((sc->cv_qwait == 1) && (sc->cv_q != BDF_Q_MAX)) {
    md->dzn[BDF_Q_MAX][i] = md->cv_acor[i];
    sc->cv_saved_tq5 = md->cv_tq[5 + blockIdx.x * (NUM_TESTS + 1)];
  }
}

/**
 * \brief Given etaqm1, etaq, etaqp1 (the values of eta for qprime = q - 1, q,
 * or q + 1, respectively), this routine chooses the maximum eta value, sets eta
 * to that value, and sets qprime to the corresponding value of q.
 *
 * If there is a tie, the preference order is to (1) keep the same order, then
 * (2) decrease the order, and finally (3) increase the order. If the maximum
 * eta value is below the threshhold THRESH, the order is kept unchanged and eta
 * is set to 1.
 *
 * \param cv_etaqp1 The value of eta for qprime = q + 1.
 * \param cv_etaq The value of eta for q.
 * \param cv_etaqm1 The value of eta for qprime = q - 1.
 * \param md Pointer to the ModelDataGPU structure.
 * \param sc Pointer to the ModelDataVariable structure.
 */
__device__ void cudaDevicecvChooseEta(double cv_etaqp1, double cv_etaq,
                                      double cv_etaqm1, ModelDataGPU *md,
                                      ModelDataVariable *sc) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  double etam;
  etam = SUNMAX(cv_etaqm1, SUNMAX(cv_etaq, cv_etaqp1));
  if (etam < THRESH) {
    sc->cv_eta = 1.;
    sc->cv_qprime = sc->cv_q;
    return;
  }
  if (etam == cv_etaq) {
    sc->cv_eta = cv_etaq;
    sc->cv_qprime = sc->cv_q;
  } else if (etam == cv_etaqm1) {
    sc->cv_eta = cv_etaqm1;
    sc->cv_qprime = sc->cv_q - 1;
  } else {
    sc->cv_eta = cv_etaqp1;
    sc->cv_qprime = sc->cv_q + 1;
    /*
     * Store Delta_n in zn[qmax] to be used in order increase
     *
     * This happens at the last step of order q before an increase
     * to order q+1, so it represents Delta_n in the ELTE at q+1
     */
    md->dzn[BDF_Q_MAX][i] = md->cv_acor[i];
  }
}

/**
 * Adjusts the value of eta according to the various heuristic limits and the
 * optional input hmax.
 *
 * @param md The model data on the device.
 * @param sc The model data variable on the device.
 */
__device__ void cudaDevicecvSetEta(ModelDataGPU *md, ModelDataVariable *sc) {
  /* If eta below the threshhold THRESH, reject a change of step size */
  if (sc->cv_eta < THRESH) {
    sc->cv_eta = 1.;
    sc->cv_hprime = sc->cv_h;
  } else {
    /* Limit eta by etamax and hmax, then set hprime */
    sc->cv_eta = SUNMIN(sc->cv_eta, sc->cv_etamax);
    sc->cv_hprime = sc->cv_h * sc->cv_eta;
  }
}

/**
 * \brief This routine handles the setting of stepsize and order for the next
 * step -- hprime and qprime. Along with hprime, it sets the ratio eta =
 * hprime/h. It also updates other state variables related to a change of step
 * size or order.
 *
 * \param md The ModelDataGPU object pointer.
 * \param sc The ModelDataVariable object pointer.
 * \param dsm The value of dsm.
 *
 * \return Returns 0 on success.
 */
__device__ int cudaDevicecvPrepareNextStep(ModelDataGPU *md,
                                           ModelDataVariable *sc, double dsm) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  /* If etamax = 1, defer step size or order changes */
  if (sc->cv_etamax == 1.) {
    sc->cv_qwait = SUNMAX(sc->cv_qwait, 2);
    sc->cv_qprime = sc->cv_q;
    sc->cv_hprime = sc->cv_h;
    sc->cv_eta = 1.;
    return 0;
  }

  /* etaq is the ratio of new to old h at the current order */
  double cv_etaq = 1. / (dSUNRpowerR(BIAS2 * dsm, 1. / sc->cv_L) + ADDON);

  /* If no order change, adjust eta and acor in cvSetEta and return */
  if (sc->cv_qwait != 0) {
    sc->cv_eta = cv_etaq;
    sc->cv_qprime = sc->cv_q;
    cudaDevicecvSetEta(md, sc);
    return 0;
  }
  /* If qwait = 0, consider an order change.   etaqm1 and etaqp1 are
   the ratios of new to old h at orders q-1 and q+1, respectively.
   cvChooseEta selects the largest; cvSetEta adjusts eta and acor */
  sc->cv_qwait = 2;
  double ddn;
  double cv_etaqm1 = 0.;
  if (sc->cv_q > 1) {
    cudaDeviceVWRMS_Norm_2(md->dzn[sc->cv_q], md->dewt, &ddn, md->n_shr_empty);
    ddn *= md->cv_tq[1 + blockIdx.x * (NUM_TESTS + 1)];
    cv_etaqm1 = 1. / (dSUNRpowerR(BIAS1 * ddn, 1. / sc->cv_q) + ADDON);
  }
  double dup, cquot;
  double cv_etaqp1 = 0.;
  if (sc->cv_q != BDF_Q_MAX && sc->cv_saved_tq5 != 0.) {
    cquot = (md->cv_tq[5 + blockIdx.x * (NUM_TESTS + 1)] / sc->cv_saved_tq5) *
            dSUNRpowerI(sc->cv_h / md->cv_tau[2 + blockIdx.x * (L_MAX + 1)],
                        (double)sc->cv_L);
    md->dtempv[i] = md->cv_acor[i] - cquot * md->dzn[BDF_Q_MAX][i];
    cudaDeviceVWRMS_Norm_2(md->dtempv, md->dewt, &dup, md->n_shr_empty);
    dup *= md->cv_tq[3 + blockIdx.x * (NUM_TESTS + 1)];
    cv_etaqp1 = 1. / (dSUNRpowerR(BIAS3 * dup, 1. / (sc->cv_L + 1)) + ADDON);
  }
  cudaDevicecvChooseEta(cv_etaqp1, cv_etaq, cv_etaqm1, md, sc);
  cudaDevicecvSetEta(md, sc);
  return CV_SUCCESS;
}

/**
 * \brief Adjusts the history array on an increase in the order q in the case
 * that lmm == CV_BDF.
 *
 * This routine sets a new column zn[q+1] equal to a multiple of the saved
 * vector (= acor) in zn[indx_acor]. Then each zn[j] is adjusted by a multiple
 * of zn[q+1]. The coefficients in the adjustment are the coefficients of the
 * polynomial x*x*(x+xi_1)*...*(x+xi_j), where xi_j = [t_n - t_(n-j)]/h.
 *
 * \param md Pointer to the ModelDataGPU structure.
 * \param sc Pointer to the ModelDataVariable structure.
 */
__device__ void cudaDevicecvIncreaseBDF(ModelDataGPU *md,
                                        ModelDataVariable *sc) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  double alpha0, alpha1, prod, xi, xiold, hsum, A1;
  int z, j;
  for (z = 0; z <= BDF_Q_MAX; z++) md->cv_l[z + blockIdx.x * L_MAX] = 0.;
  md->cv_l[2 + blockIdx.x * L_MAX] = alpha1 = prod = xiold = 1.;
  alpha0 = -1.;
  hsum = sc->cv_hscale;
  if (sc->cv_q > 1) {
    for (j = 1; j < sc->cv_q; j++) {
      hsum += md->cv_tau[j + 1 + blockIdx.x * (L_MAX + 1)];
      xi = hsum / sc->cv_hscale;
      prod *= xi;
      alpha0 -= 1. / (j + 1);
      alpha1 += 1. / xi;
      for (z = j + 2; z >= 2; z--)
        md->cv_l[z + blockIdx.x * L_MAX] =
            md->cv_l[z + blockIdx.x * L_MAX] * xiold +
            md->cv_l[z - 1 + blockIdx.x * L_MAX];
      xiold = xi;
    }
  }
  A1 = (-alpha0 - alpha1) / prod;
  md->dzn[sc->cv_L][i] = A1 * md->dzn[BDF_Q_MAX][i];
  for (j = 2; j <= sc->cv_q; j++) {
    md->dzn[j][i] += md->cv_l[j + blockIdx.x * L_MAX] * md->dzn[sc->cv_L][i];
  }
}

/**
 * \brief Adjusts the parameters of the history array zn based on a change in
 * step size.
 *
 * This routine is called when a change in step size was decided upon,
 * and it handles the required adjustments to the history array zn.
 * If there is to be a change in order, it calls cvAdjustOrder and resets
 * q, L = q+1, and qwait. Then in any case, it calls cvRescale, which
 * resets h and rescales the Nordsieck array.
 *
 * \param md The model data on the device.
 * \param sc The model data variable on the device.
 */
__device__ void cudaDevicecvAdjustParams(ModelDataGPU *md,
                                         ModelDataVariable *sc) {
  if (sc->cv_qprime != sc->cv_q) {
    int deltaq = sc->cv_qprime - sc->cv_q;
    switch (deltaq) {
      case 1:
        cudaDevicecvIncreaseBDF(md, sc);
        break;
      case -1:
        cudaDevicecvDecreaseBDF(md, sc);
        break;
    }
    sc->cv_q = sc->cv_qprime;
    sc->cv_L = sc->cv_q + 1;
    sc->cv_qwait = sc->cv_L;
  }
  cudaDevicecvRescale(md, sc);
}

/**
 * Performs one internal cvode step, from tn to tn + h.
 * Calls other routines to do all the work.
 *
 * The main operations done here are as follows:
 * - Preliminary adjustments if a new step size was chosen.
 * - Prediction of the Nordsieck history array zn at tn + h.
 * - Setting of multistep method coefficients and test quantities.
 * - Solution of the nonlinear system.
 * - Testing the local error.
 * - Updating zn and other state data if successful.
 * - Resetting stepsize and order for the next step.
 * - If SLDET is on, check for stability, reduce order if necessary.
 * On a failure in the nonlinear system solution or error test, the
 * step may be reattempted, depending on the nature of the failure.
 *
 * @param md The ModelDataGPU object.
 * @param sc The ModelDataVariable object.
 * @return The result of the cvStep operation.
 */
__device__ int cudaDevicecvStep(ModelDataGPU *md, ModelDataVariable *sc) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int ncf = 0;
  int nef = 0;
  int nflag = FIRST_CALL;
  double saved_t = sc->cv_tn;
  double dsm;
  if ((sc->cv_nst > 0) && (sc->cv_hprime != sc->cv_h)) {
    cudaDevicecvAdjustParams(md, sc);
  }

  /* Looping point for attempts to take a step */
  for (;;) {
    cudaDevicecvPredict(md, sc);
    cudaDevicecvSet(md, sc);
    nflag = cudaDevicecvNlsNewton(nflag, md, sc);
    int kflag = cudaDevicecvHandleNFlag(md, sc, &nflag, saved_t, &ncf);

    /* Go back in loop if we need to predict again (nflag=PREV_CONV_FAIL)*/
    if (kflag == PREDICT_AGAIN) {
      continue;
    }

    /* Return if nonlinear solve failed and recovery not possible. */
    if (kflag != DO_ERROR_TEST) {
      return (kflag);
    }

    /* Perform error test (nflag=CV_SUCCESS) */
    int eflag = cudaDevicecvDoErrorTest(md, sc, &nflag, saved_t, &nef, &dsm);

    /* Go back in loop if we need to predict again (nflag=PREV_ERR_FAIL) */
    if (eflag == TRY_AGAIN) {
      continue;
    }

    /* Return if error test failed and recovery not possible. */
    if (eflag != CV_SUCCESS) {
      return (eflag);
    }

    /* Error test passed (eflag=CV_SUCCESS), break from loop */
    break;
  }

  /* Nonlinear system solve and error test were both successful.
   Update data, and consider change of step and/or order.       */
  cudaDevicecvCompleteStep(md, sc);
  cudaDevicecvPrepareNextStep(md, sc, dsm);
  sc->cv_etamax = 10.;

  /*  Finally, we rescale the acor array to be the
    estimated local error vector. */
  md->cv_acor[i] *= md->cv_tq[2 + blockIdx.x * (NUM_TESTS + 1)];
  return (CV_SUCCESS);
}

/**
 * CVodeGetDky
 *
 * This routine computes the k-th derivative of the interpolating
 * polynomial at the time t and stores the result in the vector dky.
 * The formula is:
 *         q
 *  dky = SUM c(j,k) * (t - tn)^(j-k) * h^(-j) * zn[j] ,
 *        j=k
 * where c(j,k) = j*(j-1)*...*(j-k+1), q is the current order, and
 * zn[j] is the j-th column of the Nordsieck history array.
 *
 * This function is called by CVode with k = 0 and t = tout, but
 * may also be called directly by the user.
 *
 * @param md The ModelDataGPU pointer.
 * @param sc The ModelDataVariable pointer.
 * @param t The time at which to compute the derivative.
 * @param k The order of the derivative to compute.
 * @param dky The vector to store the computed derivative.
 * @return CV_SUCCESS if successful, CV_BAD_T if t is outside the valid range.
 */
__device__ int cudaDeviceCVodeGetDky(ModelDataGPU *md, ModelDataVariable *sc,
                                     double t, int k, double *dky) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  double s, c, r;
  double tfuzz, tp, tn1;
  int z, j;

  /* Allow for some slack */
  tfuzz = FUZZ_FACTOR * UNIT_ROUNDOFF * (fabs(sc->cv_tn) + fabs(sc->cv_hu));
  if (sc->cv_hu < 0.) tfuzz = -tfuzz;
  tp = sc->cv_tn - sc->cv_hu - tfuzz;
  tn1 = sc->cv_tn + tfuzz;
  if ((t - tp) * (t - tn1) > 0.) {
    return (CV_BAD_T);
  }

  /* Sum the differentiated interpolating polynomial */
  s = (t - sc->cv_tn) / sc->cv_h;
  for (j = sc->cv_q; j >= k; j--) {
    c = 1.;
    for (z = j; z >= j - k + 1; z--) c *= z;
    if (j == sc->cv_q) {
      dky[i] = c * md->dzn[j][i];
    } else {
      dky[i] = c * md->dzn[j][i] + s * dky[i];
    }
  }
  if (k == 0) return (CV_SUCCESS);
  r = dSUNRpowerI(double(sc->cv_h), double(-k));
  dky[i] = dky[i] * r;
  return (CV_SUCCESS);
}

/**
 * \brief This routine is responsible for setting the error weight vector ewt.
 *
 * This routine sets the error weight vector ewt according to the formula:
 *    ewt[i] = 1 / (reltol * abs(ycur[i]) + abstol[i]), i=0,...,neq-1
 *
 * It tests for non-positive components before inverting.
 * If ewt is successfully set, it returns 0. Otherwise, it returns -1 and ewt is
 * considered undefined.
 *
 * \param md Pointer to the ModelDataGPU structure.
 * \param sc Pointer to the ModelDataVariable structure.
 * \param weight Pointer to the array for storing the error weight vector ewt.
 *
 * \return Returns 0 if ewt is successfully set, -1 otherwise.
 */
__device__ int cudaDevicecvEwtSetSV(ModelDataGPU *md, ModelDataVariable *sc,
                                    double *weight) {
  extern __shared__ double flag_shr2[];
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  md->dtempv[i] = fabs(md->dzn[0][i]);
  double min;
  md->dtempv[i] = md->cv_reltol * md->dtempv[i] + md->cv_Vabstol[threadIdx.x];
  cudaDevicemin(&min, md->dtempv[i], flag_shr2, md->n_shr_empty);
  if (min <= 0.) return (-1);
  weight[i] = 1. / md->dtempv[i];
  return (0);
}

/*
 * CVode
 *
 * This routine is equivalent to the main driver of the CVODE package.
 *
 * It integrates over a time interval defined by the user, by calling
 * cvStep to do internal time steps.
 *
 * It computes a tentative initial step size h.
 *
 * The solver steps until it reaches or passes tout
 * and then interpolates to obtain y(tout).
 */
__device__ int cudaDeviceCVode(ModelDataGPU *md, ModelDataVariable *sc) {
  extern __shared__ int flag_shr[];
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int kflag2, retval;
  sc->cv_h = md->init_time_step;  // CVodeSetInitStep
  // CVodeReInit
  // Set step parameters
  sc->cv_q = 1;
  sc->cv_L = 2;
  sc->cv_qwait = sc->cv_L;
  sc->cv_etamax = ETAMX1;
  sc->cv_next_h = 0.;  // Set other integrator optional output
  /*
   * --------------------------------------------------
   * Looping point for internal steps
   *
   *    1. check for errors (too many steps, too much
   *         accuracy requested, step size too small)
   *    2. take a new step (call cvStep)
   *    3. stop on error
   *    4. perform stop tests:
   *         - check for root in last step
   *         - check if tout was passed
   *         - check if close to tstop
   * --------------------------------------------------
   */
  // Reset and check ewt
  retval = cudaDevicecvEwtSetSV(md, sc, md->dewt);
  if (retval != 0) {
    return (CV_ILL_INPUT);
  }
  retval = cudaDevicef(sc->cv_tn, md->dzn[0], md->dzn[1], true, md, sc);
  md->yout[i] = md->dzn[0][i];
  if (retval != 0) {
    md->yout[i] = md->dzn[0][i];
    return (CV_RHSFUNC_FAIL);
  }
  if (fabs(sc->cv_h) < sc->cv_hmin) {
    sc->cv_h *= sc->cv_hmin / fabs(sc->cv_h);
  }
  sc->cv_hscale = sc->cv_h;
  sc->cv_hprime = sc->cv_h;
  md->dzn[1][i] *= sc->cv_h;
  md->dtempv1[i] = md->dzn[0][i] + md->dzn[1][i];
  CudaDeviceguess_helper(sc->cv_tn + sc->cv_h, sc->cv_h, md->dtempv1,
                         md->dzn[0], md->dzn[1], md->dp0, md->cv_acor_init, md,
                         sc);
  int nstloc = 0;
  sc->nstlj = 0;
  sc->cv_nst = 0;
  sc->cv_nstlp = 0;
  // Most external loop of solving
  for (;;) {
#ifdef CAMP_PROFILE_DEVICE_FUNCTIONS
    if (threadIdx.x == 0) sc->countercvStep++;
#endif
    __syncthreads();
    flag_shr[0] = 0;
    __syncthreads();
    sc->cv_next_h = sc->cv_h;
    /* Reset and check ewt */
    if (sc->cv_nst > 0) {
      int ewtsetOK = cudaDevicecvEwtSetSV(md, sc, md->dewt);
      if (ewtsetOK != 0) {
        md->yout[i] = md->dzn[0][i];
        return CV_ILL_INPUT;
      }
    }
    /* Check for too many steps */
    if ((CAMP_SOLVER_DEFAULT_MAX_STEPS > 0) &&
        (nstloc >= CAMP_SOLVER_DEFAULT_MAX_STEPS)) {
      md->yout[i] = md->dzn[0][i];
      if (i == 0)
        printf(
            "ERROR: CAMP_SOLVER_DEFAULT_MAX_STEPS reached "
            "nstloc %d CAMP_SOLVER_DEFAULT_MAX_STEPS %d\n",
            nstloc, CAMP_SOLVER_DEFAULT_MAX_STEPS);
      return CV_TOO_MUCH_WORK;
    }
    /* Check for too much accuracy requested */
    double nrm;
    cudaDeviceVWRMS_Norm_2(md->dzn[0], md->dewt, &nrm, md->n_shr_empty);
    if (UNIT_ROUNDOFF * nrm > 1.) {
      md->yout[i] = md->dzn[0][i];
      if (i == 0) printf("ERROR: cv_tolsf > 1\n");
      return CV_TOO_MUCH_ACC;
    }
    /* Call cvStep to take a step */
    kflag2 = cudaDevicecvStep(md, sc);
    /* Process failed step cases, and exit loop */
    if (kflag2 != CV_SUCCESS) {
      md->yout[i] = md->dzn[0][i];
      if (i == 0) printf("ERROR: kflag != CV_SUCCESS\n");
      return kflag2;
    }
    nstloc++;
    /*Check if tout reached */
    if ((sc->cv_tn - md->tout) * sc->cv_h >= 0.) {
      cudaDeviceCVodeGetDky(md, sc, md->tout, 0, md->yout);
      return CV_SUCCESS;
    }
  }
}

__global__ void cudaGlobalCVode(double t_initial, ModelDataGPU md_object) {
  ModelDataGPU *md = &md_object;
  extern __shared__ int flag_shr[];
  ModelDataVariable sc_object = *md->sCells;
  ModelDataVariable *sc = &sc_object;
  sc->cv_tn = t_initial;  // Set initial value of "t" to each cell data
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  // Update input
  md->dzn[0][i] = md->state[md->map_state_deriv[threadIdx.x] +
                            blockIdx.x * md->n_per_cell_state_var] > TINY
                      ? md->state[md->map_state_deriv[threadIdx.x] +
                                  blockIdx.x * md->n_per_cell_state_var]
                      : TINY;
#ifdef CAMP_PROFILE_DEVICE_FUNCTIONS
  int clock_khz = md->clock_khz;
  clock_t start;
  start = clock();
#endif
// Solve
#ifdef DEBUG_SOLVER_FAILURES
  int flag = cudaDeviceCVode(md, sc);
  if (threadIdx.x == 0) md->flags[blockIdx.x] = flag;
#else
  cudaDeviceCVode(md, sc);
#endif
  // Update output
  md->state[md->map_state_deriv[threadIdx.x] +
            blockIdx.x * md->n_per_cell_state_var] =
      md->yout[i] > 0. ? md->yout[i] : 0.;
#ifdef CAMP_PROFILE_DEVICE_FUNCTIONS
  if (threadIdx.x == 0)
    sc->timeDeviceCVode +=
        ((double)(int)(clock() - start)) / (clock_khz * 1000);
#endif
#ifdef CAMP_PROFILE_DEVICE_FUNCTIONS
  ModelDataVariable *mdvo = md->mdvo;
  *mdvo = *sc;
#endif
}

/**
 * Calculates the next power of two for a given integer.
 *
 * @param v The input integer.
 * @return The next power of two.
 */
static int nextPowerOfTwo(int v) {
  v--;
  v |= v >> 1;
  v |= v >> 2;
  v |= v >> 4;
  v |= v >> 8;
  v |= v >> 16;
  v++;
  return v;
}

/**
 * Runs the CVode solver on the GPU.
 *
 * @param t_initial The initial time.
 * @param mGPU The pointer to the ModelDataGPU structure.
 * @param blocks The number of blocks for the kernel launch.
 * @param threads_block The number of threads per block for the kernel launch.
 * @param stream The CUDA stream to associate with the kernel launch.
 */
void cvodeRun(double t_initial, ModelDataGPU *mGPU, int blocks,
              int threads_block, cudaStream_t stream) {
  int n_shr_memory = nextPowerOfTwo(threads_block);
  mGPU->n_shr_empty = n_shr_memory - threads_block;
  cudaGlobalCVode<<<blocks, threads_block, n_shr_memory * sizeof(double),
                    stream>>>(t_initial, *mGPU);  // Call to GPU
}
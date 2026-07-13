/* Copyright (C) 2021 Barcelona Supercomputing Center and University of
 * Illinois at Urbana-Champaign
 * SPDX-License-Identifier: MIT
 */

/** \file
 * \brief Gas-phase reaction solver functions
 */
#ifndef GAS_RXNS_DEV_H_
#define GAS_RXNS_DEV_H_

#include "common_dev.h"

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
__device__ void rxn_gpu_emission_calc_deriv_contrib(
    ModelDataVariable *sc, TimeDerivativeGPU time_deriv, int *rxn_int_data,
    double *rxn_float_data, double *rxn_env_data, double time_step) {
  int *int_data = rxn_int_data;
  double rate = rxn_env_data[0];
  if (int_data[2] >= 0)
    time_derivative_add_value_gpu(time_deriv, int_data[2], rate);
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
__device__ void rxn_gpu_emission_calc_jac_contrib(
    ModelDataVariable *sc, JacobianGPU jac, int *rxn_int_data,
    double *rxn_float_data, double *rxn_env_data, double time_step) {
  // No Jacobian contributions from 0th order emissions
  return;
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

#endif // GAS_RXNS_DEV_H_
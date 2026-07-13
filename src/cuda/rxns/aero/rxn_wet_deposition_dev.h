/* Copyright (C) 2021 Barcelona Supercomputing Center and University of
 * Illinois at Urbana-Champaign
 * SPDX-License-Identifier: MIT
 */

/** \file
 * \brief Surface reaction solver functions
 */
#ifndef AERO_RXN_WET_DEPOSITION_DEV_H_
#define AERO_RXN_WET_DEPOSITION_DEV_H_

#include "common_dev.h"

/** \brief Calculate contributions to the time derivative \f$f(t,y)\f$ from
 * this reaction.
 *
 * \param model_data Pointer to the model data, including the state array
 * \param time_deriv TimeDerivative object
 * \param rxn_int_data Pointer to the reaction integer data
 */
__device__ void rxn_gpu_wet_deposition_calc_deriv_contrib(ModelDataVariable *sc, 
    TimeDerivativeGPU time_deriv, int *rxn_int_data, double *rxn_env_data) {
  int *int_data = rxn_int_data;
  double *state = sc->grid_cell_state;

  // Add contributions to the time derivative
  for (int i_spec = 0; i_spec < int_data[1]; i_spec++) {
    if (int_data[2 + int_data[1] + i_spec] >= 0) {
      double rate = rxn_env_data[0] * state[int_data[2 + i_spec] - 1];
      time_derivative_add_value_gpu(time_deriv, int_data[2 + int_data[1] + i_spec], -rate);
    }
  }

  return;
}

/** \brief Calculate contributions to the Jacobian from this reaction
 *
 * \param jac Reaction Jacobian
 * \param rxn_int_data Pointer to the reaction integer data
 */
__device__ void rxn_gpu_wet_deposition_calc_jac_contrib(JacobianGPU jac,
                                         int *rxn_int_data, double *rxn_env_data) {
  int *int_data = rxn_int_data;

  // Add contributions to the Jacobian
  for (int i_spec = 0; i_spec < int_data[1]; i_spec++) {
    if (int_data[2 + 2 * int_data[1] + i_spec] >= 0)
      jacobian_add_value_gpu(jac, (unsigned int)int_data[2 + 2 * int_data[1] + i_spec], JACOBIAN_LOSS,
                         rxn_env_data[0]);
  }

  return;
}

#endif // AERO_RXN_WET_DEPOSITION_DEV_H_
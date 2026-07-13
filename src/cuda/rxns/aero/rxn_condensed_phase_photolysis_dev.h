/* Copyright (C) 2021 Barcelona Supercomputing Center and University of
 * Illinois at Urbana-Champaign
 * SPDX-License-Identifier: MIT
 */

/** \file
 * \brief Surface reaction solver functions
 */
#ifndef AERO_RXN_CONDENSED_PHASE_PHOTOLYSIS_DEV_H_
#define AERO_RXN_CONDENSED_PHASE_PHOTOLYSIS_DEV_H_

#include "common_dev.h"

/** \brief Calculate contributions to the time derivative f(t,y) from this
 * reaction.
 *
 * \param md Pointer to the model data, including the state array
 * \param time_deriv TimeDerivative object
 * \param rxn_int_data Pointer to the reaction integer data
 * \param rxn_float_data Pointer to the reaction floating-point data
 * \param rxn_env_data Pointer to the environment-dependent parameters
 * \param time_step Current time step of the itegrator (s)
 */
__device__ void rxn_gpu_condensed_phase_photolysis_calc_deriv_contrib(
    ModelDataVariable *sc, ModelDataGPU *md, TimeDerivativeGPU time_deriv, int *rxn_int_data,
    double *rxn_float_data, double *rxn_env_data, double time_step) {
  int *int_data = rxn_int_data;
  double *float_data = rxn_float_data;
  double *state = sc->grid_cell_state;

  int intd01 = int_data[0] + int_data[1];

  // Calculate derivative contributions for each aerosol phase
  for (int i_phase = 0, i_deriv = 0; i_phase < int_data[2]; i_phase++) {
    // If this is an aqueous reaction, get the unit conversion from mol/m3 -> M
    double unit_conv = 1.0;
    if (int_data[4 + intd01 * int_data[2] + i_phase] - 1 >= 0) {
      unit_conv = state[int_data[4 + intd01 * int_data[2] + i_phase] -
                        1]; // convert from kg/m3->L/m3

      // For aqueous reactions, if no aerosol water is present, no reaction
      // occurs
      if (unit_conv <= ZERO) {
        i_deriv += intd01;
        continue;
      }
      unit_conv = 1.0 / unit_conv;
    }

    // Calculate the reaction rate rate (M/s or mol/m3/s)
    double rate = rxn_env_data[0];
    for (int i_react = 0; i_react < int_data[0]; i_react++) {
      rate *= state[int_data[4 + (i_phase * int_data[0] + i_react)] - 1] *
              float_data[1 + int_data[1] + i_react] * unit_conv;
    }

    // Reactant change
    for (int i_react = 0; i_react < int_data[0]; i_react++) {
      if (int_data[4 + (intd01 + 1) * int_data[2] + i_deriv] < 0) {
        i_deriv++;
        continue;
      }
      time_derivative_add_value_gpu(
          time_deriv, int_data[4 + (intd01 + 1) * int_data[2] + i_deriv++],
          -rate / (float_data[1 + int_data[1] + i_react] * unit_conv));
    }

    // Products change
    for (int i_prod = 0; i_prod < int_data[1]; i_prod++) {
      if (int_data[4 + (intd01 + 1) * int_data[2] + i_deriv] < 0) {
        i_deriv++;
        continue;
      }
      time_derivative_add_value_gpu(
          time_deriv, int_data[4 + (intd01 + 1) * int_data[2] + i_deriv++],
          rate * float_data[1 + i_prod] /
              (float_data[1 + intd01 + i_prod] * unit_conv));
    }
  }

  return;
}

/** \brief Calculate contributions to the Jacobian from this reaction
 *
 * \param md Pointer to the model data
 * \param jac Reaction Jacobian
 * \param rxn_int_data Pointer to the reaction integer data
 * \param rxn_float_data Pointer to the reaction floating-point data
 * \param rxn_env_data Pointer to the environment-dependent parameters
 * \param time_step Current time step of the itegrator (s)
 */
__device__ void rxn_gpu_condensed_phase_photolysis_calc_jac_contrib(
    ModelDataVariable *sc, ModelDataGPU *md, JacobianGPU jac, int *rxn_int_data, double *rxn_float_data,
    double *rxn_env_data, double time_step) {
  int *int_data = rxn_int_data;
  double *float_data = rxn_float_data;
  double *state = sc->grid_cell_state;

  int intd01 = int_data[0] + int_data[1];
  int int4012 = 4 + (2 * intd01 + 1) * int_data[2];

  // Calculate Jacobian contributions for each aerosol phase
  for (int i_phase = 0, i_jac = 0; i_phase < int_data[2]; i_phase++) {
    // If this is an aqueous reaction, get the unit conversion from mol/m3 -> M
    realtype unit_conv = 1.0;
    if (int_data[4 + intd01 * int_data[2] + i_phase] - 1 >= 0) {
      unit_conv = state[int_data[4 + intd01 * int_data[2] + i_phase] -
                        1]; // convert from kg/m3->L/m3

      // For aqueous reactions, if no aerosol water is present, no reaction
      // occurs
      if (unit_conv <= ZERO) {
        i_jac += intd01 * (int_data[0] + 1);
        continue;
      }
      unit_conv = 1.0 / unit_conv;
    }

    // Add dependence on reactants for reactants and products
    for (int i_react_ind = 0; i_react_ind < int_data[0]; i_react_ind++) {
      // Calculate d_rate / d_react_i
      realtype rate = rxn_env_data[0];
      for (int i_react = 0; i_react < int_data[0]; i_react++) {
        if (i_react == i_react_ind) {
          rate *= float_data[1 + int_data[1] + i_react] * unit_conv;
        } else {
          rate *= state[int_data[4 + (i_phase * int_data[0] + i_react)] - 1] *
                  float_data[1 + int_data[1] + i_react] * unit_conv;
        }
      }

      // Add the Jacobian elements
      //
      // For reactant dependence on reactants
      for (int i_react_dep = 0; i_react_dep < int_data[0]; i_react_dep++) {
        if (int_data[int4012 + i_jac] < 0) {
          i_jac++;
          continue;
        }
        jacobian_add_value_gpu(
            jac, (unsigned int)int_data[int4012 + i_jac++], JACOBIAN_LOSS,
            rate / (float_data[1 + int_data[1] + i_react_dep] * unit_conv));
      }
      // For product dependence on reactants
      for (int i_prod_dep = 0; i_prod_dep < int_data[1]; i_prod_dep++) {
        if (int_data[int4012 + i_jac] < 0) {
          i_jac++;
          continue;
        }
        jacobian_add_value_gpu(
            jac, (unsigned int)int_data[int4012 + i_jac++], JACOBIAN_PRODUCTION,
            rate * float_data[1 + i_prod_dep] /
                (float_data[1 + int_data[1] + (int_data[0] + i_prod_dep)] *
                 unit_conv));
      }
    }

    // Add dependence on aerosol-phase water for reactants and products in
    // aqueous reactions
    if (int_data[4 + intd01 * int_data[2] + i_phase] - 1 < 0) {
      i_jac += intd01;
      continue;
    }

    // Calculate the overall reaction rate (M/s or mol/m3/s)
    realtype rate = rxn_env_data[0];
    for (int i_react = 0; i_react < int_data[0]; i_react++) {
      rate *= state[int_data[4 + (i_phase * int_data[0] + i_react)] - 1] *
              float_data[1 + int_data[1] + i_react] * unit_conv;
    }

    // Dependence of reactants on aerosol-phase water
    for (int i_react_dep = 0; i_react_dep < int_data[0]; i_react_dep++) {
      if (int_data[int4012 + i_jac] < 0) {
        i_jac++;
        continue;
      }
      jacobian_add_value_gpu(jac, (unsigned int)int_data[int4012 + i_jac++],
                             JACOBIAN_LOSS,
                             -(int_data[0] - 1) * rate /
                                 float_data[1 + int_data[1] + i_react_dep]);
    }
    // Dependence of products on aerosol-phase water
    for (int i_prod_dep = 0; i_prod_dep < int_data[1]; i_prod_dep++) {
      if (int_data[int4012 + i_jac] < 0) {
        i_jac++;
        continue;
      }
      jacobian_add_value_gpu(
          jac, (unsigned int)int_data[int4012 + i_jac++], JACOBIAN_PRODUCTION,
          -(int_data[0] - 1) * rate * float_data[1 + i_prod_dep] /
              float_data[1 + intd01 + i_prod_dep]);
    }
  }

  return;
}

#endif // AERO_RXN_CONDENSED_PHASE_PHOTOLYSIS_DEV_H_
/* Copyright (C) 2021 Barcelona Supercomputing Center and University of
 * Illinois at Urbana-Champaign
 * SPDX-License-Identifier: MIT
 */

/** \file
 * \brief Surface reaction solver functions
 */
#ifndef AERO_RXN_AQUEOUS_EQUILIBRIUM_DEV_H_
#define AERO_RXN_AQUEOUS_EQUILIBRIUM_DEV_H_

#include "common_dev.h"

// Factor used to calculate minimum water concentration for aqueous
// phase equilibrium reactions
#define MIN_WATER_ 1.0e-4

/** \brief Calculate the reaction rate for a set of conditions using the
 *         standard equation per mixing ratio of water [M_X/s*kg_H2O/m^3]
 * \param rxn_int_data Pointer to the reaction integer data
 * \param rxn_float_data Pointer to the reaction floating point data
 * \param rxn_env_data Pointer to the environment-dependent parameters
 * \param is_water_partial Flag indicating whether the calculation should
 *                         return the partial derivative d_rate/d_H2O
 * \param rate_forward [output] calculated forward rate
 * \param rate_reverse [output] calculated reverse rate
 * \return reaction rate per mixing ratio of water [M_X/s*kg_H2O/m^3]
 */
__device__ double calc_standard_rate(int *rxn_int_data, double *rxn_float_data,
                                     double *rxn_env_data,
                                     bool is_water_partial,
                                     double *rate_forward,
                                     double *rate_reverse) {
  int *int_data = rxn_int_data;
  double *float_data = rxn_float_data;

  double react_fact, prod_fact;
  double water = float_data[3];
  int intd01 = int_data[0] + int_data[1];

  // Get the product of all reactants
  react_fact = (double)float_data[5 + intd01] * float_data[5];
  for (int i_react = 1; i_react < int_data[0]; i_react++) {
    react_fact *=
        float_data[5 + intd01 + i_react] * float_data[5 + i_react] / water;
  }

  // Get the product of all product
  prod_fact = (double)float_data[5 + 2 * int_data[0] + int_data[1]] *
              float_data[5 + int_data[0]];
  prod_fact *= (double)float_data[4];
  for (int i_prod = 1; i_prod < int_data[1]; i_prod++) {
    prod_fact *= float_data[5 + 2 * int_data[0] + int_data[1] + i_prod] *
                 float_data[5 + int_data[0] + i_prod] / water;
  }

  *rate_forward = rxn_env_data[0] * react_fact;
  *rate_reverse = float_data[2] * prod_fact;

  if (is_water_partial) {
    return *rate_forward * (int_data[0] - 1) -
           *rate_reverse * (int_data[1] - 1);
  } else {
    return *rate_forward - *rate_reverse;
  }
}

/** \brief Calculate contributions to the time derivative \f$f(t,y)\f$ from
 * this reaction.
 *
 * \param model_data Pointer to the model data, including the state array
 * \param time_deriv TimeDerivative object
 * \param rxn_int_data Pointer to the reaction integer data
 * \param rxn_float_data Pointer to the reaction floating-point data
 * \param rxn_env_data Pointer to the environment-dependent parameters
 * \param time_step Current time step of the itegrator (s)
 */
__device__ void rxn_gpu_aqueous_equilibrium_calc_deriv_contrib(
    ModelDataVariable *sc, ModelDataGPU *md, TimeDerivativeGPU time_deriv, 
    int *rxn_int_data, double *rxn_float_data, double *rxn_env_data, double time_step) {
  int *int_data = rxn_int_data;
  double *float_data = rxn_float_data;
  double *state = sc->grid_cell_state;

  int intd01 = int_data[0] + int_data[1];

  // Calculate derivative contributions for each aerosol phase
  for (int i_phase = 0, i_deriv = 0; i_phase < int_data[2]; i_phase++) {
    // If no aerosol water is present, no reaction occurs
    double water = state[int_data[3 + intd01 * int_data[2] + i_phase] - 1];
    if (water <
        MIN_WATER_ *
            float_data[5 + 2 * int_data[0] + 2 * int_data[1] + i_phase]) {
      i_deriv += intd01;
      continue;
    }

    // Set the concentrations for all species and the activity coefficient
    for (int i_react = 0; i_react < int_data[0]; ++i_react)
      float_data[5 + intd01 + i_react] =
          state[int_data[3 + i_phase * int_data[0] + i_react] - 1];
    for (int i_prod = 0; i_prod < int_data[1]; ++i_prod)
      float_data[5 + 2 * int_data[0] + int_data[1] + i_prod] =
          state[int_data[3 + int_data[0] * int_data[2] + i_phase * int_data[1] +
                         i_prod] -
                1];
    float_data[3] = state[int_data[3 + intd01 * int_data[2] + i_phase] - 1];
    if (int_data[3 + (intd01 + 1) * int_data[2] + i_phase] - 1 >= 0) {
      float_data[4] =
          state[int_data[3 + (intd01 + 1) * int_data[2] + i_phase] - 1];
    } else {
      float_data[4] = 1.0;
    }

    // Get the rate using the standard calculation
    double rate_forward, rate_reverse;
    double rate = calc_standard_rate(rxn_int_data, rxn_float_data, rxn_env_data,
                                     false, &rate_forward, &rate_reverse);
    if (rate == ZERO) {
      i_deriv += intd01;
      continue;
    }

    // Reactants change as (reverse - forward) (kg/m3/s)
    for (int i_react = 0; i_react < int_data[0]; i_react++) {
      if (int_data[3 + (intd01 + 2) * int_data[2] + i_deriv] < 0) {
        i_deriv++;
        continue;
      }
      time_derivative_add_value_gpu(
          time_deriv, int_data[3 + (intd01 + 2) * int_data[2] + i_deriv],
          -rate_forward / float_data[5 + i_react]);
      time_derivative_add_value_gpu(
          time_deriv, int_data[3 + (intd01 + 2) * int_data[2] + i_deriv++],
          rate_reverse / float_data[5 + i_react]);
    }

    // Products change as (forward - reverse) (kg/m3/s)
    for (int i_prod = 0; i_prod < int_data[1]; i_prod++) {
      if (int_data[3 + (intd01 + 2) * int_data[2] + i_deriv] < 0) {
        i_deriv++;
        continue;
      }
      time_derivative_add_value_gpu(
          time_deriv, int_data[3 + (intd01 + 2) * int_data[2] + i_deriv],
          rate_forward / float_data[5 + int_data[0] + i_prod]);
      time_derivative_add_value_gpu(
          time_deriv, int_data[3 + (intd01 + 2) * int_data[2] + i_deriv++],
          -rate_reverse / float_data[5 + int_data[0] + i_prod]);
    }
  }

  return;
}

/** \brief Calculate contributions to the Jacobian from this reaction
 *
 * \param model_data Pointer to the model data
 * \param jac Reaction Jacobian
 * \param rxn_int_data Pointer to the reaction integer data
 * \param rxn_float_data Pointer to the reaction floating-point data
 * \param rxn_env_data Pointer to the environment-dependent parameters
 * \param time_step Current time step of the itegrator (s)
 */
__device__ void rxn_gpu_aqueous_equilibrium_calc_jac_contrib(
    ModelDataVariable *sc, ModelDataGPU *md, JacobianGPU jac, int *rxn_int_data,
    double *rxn_float_data, double *rxn_env_data, double time_step) {
  int *int_data = rxn_int_data;
  double *float_data = rxn_float_data;
  double *state = sc->grid_cell_state;

  int intd01 = int_data[0] + int_data[1];
  int int3012 = 3 + (2 * intd01 + 2) * int_data[2];

  // Calculate Jacobian contributions for each aerosol phase
  for (int i_phase = 0, i_jac = 0; i_phase < int_data[2]; i_phase++) {
    // If not aerosol water is present, no reaction occurs
    double water = state[int_data[3 + intd01 * int_data[2] + i_phase] - 1];
    if (water <
        MIN_WATER_ *
            float_data[5 + 2 * int_data[0] + 2 * int_data[1] + i_phase]) {
      i_jac += intd01 * (intd01 + 2);
      continue;
    }

    // Calculate the forward rate (M/s)
    double forward_rate = rxn_env_data[0];
    for (int i_react = 0; i_react < int_data[0]; i_react++) {
      forward_rate *= state[int_data[3 + i_phase * int_data[0] + i_react] - 1] *
                      float_data[5 + i_react] / water;
    }

    // Calculate the reverse rate (M/s)
    double reverse_rate = float_data[2];
    for (int i_prod = 0; i_prod < int_data[1]; i_prod++) {
      reverse_rate *= state[int_data[3 + int_data[0] * int_data[2] +
                                     i_phase * int_data[1] + i_prod] -
                            1] *
                      float_data[5 + int_data[0] + i_prod] / water;
    }
    if (int_data[3 + (intd01 + 1) * int_data[2] + i_phase] - 1 >= 0)
      reverse_rate *=
          state[int_data[3 + (intd01 + 1) * int_data[2] + i_phase] - 1];

    // Add dependence on reactants for reactants and products (forward reaction)
    for (int i_react_ind = 0; i_react_ind < int_data[0]; i_react_ind++) {
      for (int i_react_dep = 0; i_react_dep < int_data[0]; i_react_dep++) {
        if (int_data[3 + (2 * intd01 + 2) * int_data[2] + i_jac] < 0 ||
            forward_rate == 0.0) {
          i_jac++;
          continue;
        }
        jacobian_add_value_gpu(
            jac, (unsigned int)int_data[int3012 + i_jac++], JACOBIAN_LOSS,
            forward_rate /
                state[int_data[3 + i_phase * int_data[0] + i_react_ind] - 1] /
                float_data[5 + i_react_dep] * water);
      }
      for (int i_prod_dep = 0; i_prod_dep < int_data[1]; i_prod_dep++) {
        if (int_data[int3012 + i_jac] < 0 || forward_rate == 0.0) {
          i_jac++;
          continue;
        }
        jacobian_add_value_gpu(
            jac, (unsigned int)int_data[int3012 + i_jac++], JACOBIAN_PRODUCTION,
            forward_rate /
                state[int_data[3 + i_phase * int_data[0] + i_react_ind] - 1] /
                float_data[5 + int_data[0] + i_prod_dep] * water);
      }
    }

    // Add dependence on products for reactants and products (reverse reaction)
    for (int i_prod_ind = 0; i_prod_ind < int_data[1]; i_prod_ind++) {
      for (int i_react_dep = 0; i_react_dep < int_data[0]; i_react_dep++) {
        if (int_data[int3012 + i_jac] < 0 || reverse_rate == 0.0) {
          i_jac++;
          continue;
        }
        jacobian_add_value_gpu(
            jac, (unsigned int)int_data[int3012 + i_jac++], JACOBIAN_PRODUCTION,
            reverse_rate /
                state[int_data[3 + int_data[0] * int_data[2] +
                               i_phase * int_data[1] + i_prod_ind] -
                      1] /
                float_data[5 + i_react_dep] * water);
      }
      for (int i_prod_dep = 0; i_prod_dep < int_data[1]; i_prod_dep++) {
        if (int_data[int3012 + i_jac] < 0 || reverse_rate == 0.0) {
          i_jac++;
          continue;
        }
        jacobian_add_value_gpu(
            jac, (unsigned int)int_data[int3012 + i_jac++], JACOBIAN_LOSS,
            reverse_rate /
                state[int_data[3 + int_data[0] * int_data[2] +
                               i_phase * int_data[1] + i_prod_ind] -
                      1] /
                float_data[5 + int_data[0] + i_prod_dep] * water);
      }
    }

    // Add dependence on aerosol-phase water for reactants and products
    for (int i_react_dep = 0; i_react_dep < int_data[0]; i_react_dep++) {
      if (int_data[int3012 + i_jac] < 0) {
        i_jac++;
        continue;
      }
      jacobian_add_value_gpu(
          jac, (unsigned int)int_data[int3012 + i_jac], JACOBIAN_LOSS,
          -forward_rate / float_data[5 + i_react_dep] * (int_data[0] - 1));
      jacobian_add_value_gpu(
          jac, (unsigned int)int_data[int3012 + i_jac++], JACOBIAN_PRODUCTION,
          -reverse_rate / float_data[5 + i_react_dep] * (int_data[1] - 1));
    }
    for (int i_prod_dep = 0; i_prod_dep < int_data[1]; i_prod_dep++) {
      if (int_data[int3012 + i_jac] < 0) {
        i_jac++;
        continue;
      }
      jacobian_add_value_gpu(
          jac, (unsigned int)int_data[int3012 + i_jac], JACOBIAN_PRODUCTION,
          -forward_rate / float_data[5 + int_data[0] + i_prod_dep] *
              (int_data[0] - 1));
      jacobian_add_value_gpu(
          jac, (unsigned int)int_data[int3012 + i_jac++], JACOBIAN_LOSS,
          -reverse_rate / float_data[5 + int_data[0] + i_prod_dep] *
              (int_data[1] - 1));
    }

    // Add dependence on activity coefficients for reactants and products
    if (int_data[3 + (intd01 + 1) * int_data[2] + i_phase] - 1 < 0) {
      i_jac += intd01;
      continue;
    }
    for (int i_react_dep = 0; i_react_dep < int_data[0]; i_react_dep++) {
      if (int_data[int3012 + i_jac] < 0) {
        i_jac++;
        continue;
      }
      jacobian_add_value_gpu(
          jac, (unsigned int)int_data[int3012 + i_jac++], JACOBIAN_PRODUCTION,
          reverse_rate /
              state[int_data[3 + (intd01 + 1) * int_data[2] + i_phase] - 1] /
              float_data[5 + i_react_dep] * water);
    }
    for (int i_prod_dep = 0; i_prod_dep < int_data[1]; i_prod_dep++) {
      if (int_data[int3012 + i_jac] < 0) {
        i_jac++;
        continue;
      }
      jacobian_add_value_gpu(
          jac, (unsigned int)int_data[int3012 + i_jac++], JACOBIAN_LOSS,
          reverse_rate /
              state[int_data[3 + (intd01 + 1) * int_data[2] + i_phase] - 1] /
              float_data[5 + int_data[0] + i_prod_dep] * water);
    }
  }

  return;
}

#endif // AERO_RXN_AQUEOUS_EQUILIBRIUM_DEV_H_
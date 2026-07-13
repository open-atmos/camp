/* Copyright (C) 2021 Barcelona Supercomputing Center and University of
 * Illinois at Urbana-Champaign
 * SPDX-License-Identifier: MIT
 */

/** \file
 * \brief Surface reaction solver functions
 */
#ifndef AERO_RXN_SURFACE_DEV_H_
#define AERO_RXN_SURFACE_DEV_H_

#include "aero_solver_dev.h"

/** \brief Calculate contributions to the time derivative \f$f(t,y)\f$ from
 * this reaction.
 *
 * \param sc Pointer to the structure containing the grid_cell data state array
 * \param md Pointer to the GPU model data
 * \param time_deriv TimeDerivative object
 * \param rxn_int_data Pointer to the reaction integer data
 * \param rxn_float_data Pointer to the reaction floating-point data
 * \param rxn_env_data Pointer to the environment-dependent parameters
 * \param time_step Current time step being computed (s)
 */
__device__ void
rxn_gpu_surface_calc_deriv_contrib(ModelDataVariable *sc, ModelDataGPU *md,
                                   TimeDerivativeGPU time_deriv,
                                   int *rxn_int_data, double *rxn_float_data,
                                   double *rxn_env_data, double time_step) {
  int *int_data = rxn_int_data;
  double *float_data = rxn_float_data;
  double *state = sc->grid_cell_state;

  // Calculate derivative contributions for each aerosol phase
  for (int i_phase = 0; i_phase < int_data[0]; i_phase++) {
    int aero_rep_id = int_data[int_data[5 + 3 * int_data[2] + i_phase]] - 1;
    int aero_phase_id =
        int_data[int_data[5 + 3 * int_data[2] + i_phase] - 1] - 1;

    int *aero_rep_int_data =
        &(md->aero_rep_int_data[md->aero_rep_int_indices[aero_rep_id]]);
    double *aero_rep_float_data =
        &(md->aero_rep_float_data[md->aero_rep_float_indices[aero_rep_id]]);
    double *aero_rep_env_data =
        &(md->aero_rep_env_data[md->aero_rep_env_idx[aero_rep_id]]);

    // Get the particle effective radius (m)
    double radius;
    aero_rep_modal_binned_mass_get_effective_radius__m(
        aero_phase_id, &radius, NULL, aero_rep_int_data, aero_rep_float_data);

    // Get the particle number concentration (#/m3)
    double number_conc;
    aero_rep_modal_binned_mass_get_number_conc__n_m3(
        md, aero_phase_id, state, &number_conc, NULL, aero_rep_int_data,
        aero_rep_float_data, aero_rep_env_data);

    // Calculate the rate constant for diffusion limited mass transfer to the
    // aerosol phase (1/s)
    double cond_rate =
        state[int_data[1] - 1] * number_conc *
        gpu_gas_aerosol_continuum_rxn_rate_constant(
            float_data[0], rxn_env_data[0], radius, float_data[1]);

    // Loss of the reactant
    if (int_data[3 + int_data[2]] >= 0) {
      time_derivative_add_value_gpu(time_deriv, int_data[3 + int_data[2]],
                                    -cond_rate);
    }
    // Gain of each product
    for (int i_prod = 0; i_prod < int_data[2]; ++i_prod) {
      if (int_data[4 + int_data[2] + i_prod] >= 0) {
        time_derivative_add_value_gpu(time_deriv,
                                      int_data[4 + int_data[2] + i_prod],
                                      float_data[3 + i_prod] * cond_rate);
      }
    }
  }
}

/** \brief Calculate contributions to the Jacobian from this reaction
 *
 * \param sc Pointer to the structure containing the grid_cell data state array
 * \param md Pointer to the GPU model data
 * \param jac Reaction Jacobian
 * \param rxn_int_data Pointer to the reaction integer data
 * \param rxn_float_data Pointer to the reaction floating-point data
 * \param rxn_env_data Pointer to the environment-dependent parameters
 * \param time_step Current time step being calculated (s)
 */
__device__ void rxn_gpu_surface_calc_jac_contrib(
    ModelDataVariable *sc, ModelDataGPU *md, JacobianGPU jac, int *rxn_int_data,
    double *rxn_float_data, double *rxn_env_data, double time_step) {
  int *int_data = rxn_int_data;
  double *float_data = rxn_float_data;
  double *state = sc->grid_cell_state;

  // Calculate derivative contributions for each aerosol phase
  for (int i_phase = 0; i_phase < int_data[0]; i_phase++) {
    int aero_sub_id = int_data[5 + 3 * int_data[2] + i_phase];
    int aero_rep_id = int_data[aero_sub_id] - 1;
    int aero_phase_id = int_data[aero_sub_id - 1] - 1;

    int *aero_rep_int_data =
        &(md->aero_rep_int_data[md->aero_rep_int_indices[aero_rep_id]]);
    double *aero_rep_float_data =
        &(md->aero_rep_float_data[md->aero_rep_float_indices[aero_rep_id]]);
    double *aero_rep_env_data =
        &(md->aero_rep_env_data[md->aero_rep_env_idx[aero_rep_id]]);

    // Get the particle effective radius (m)
    double radius;
    aero_rep_modal_binned_mass_get_effective_radius__m(
        aero_phase_id, &radius,
        &float_data[int_data[5 + 3 * int_data[2] + int_data[0] + i_phase] - 1],
        aero_rep_int_data, aero_rep_float_data);

    // Get the particle number concentration (#/m3)
    double number_conc;
    aero_rep_modal_binned_mass_get_number_conc__n_m3(
        md, aero_phase_id, state, &number_conc,
        &float_data[int_data[5 + 3 * int_data[2] + int_data[0] + i_phase] - 1 +
                    int_data[aero_sub_id + 1]],
        aero_rep_int_data, aero_rep_float_data, aero_rep_env_data);

    // Calculate the rate constant for diffusion limited mass transfer to the
    // aerosol phase (1/s)
    double rate_const = gpu_gas_aerosol_continuum_rxn_rate_constant(
        float_data[0], rxn_env_data[0], radius, float_data[1]);

    // Dependence on the reactant
    if (int_data[4 + 2 * int_data[2]] >= 0) {
      jacobian_add_value_gpu(jac, (unsigned int)int_data[4 + 2 * int_data[2]],
                             JACOBIAN_LOSS, number_conc * rate_const);
    }
    for (int i_prod = 0; i_prod < int_data[2]; ++i_prod) {
      if (int_data[5 + 2 * int_data[2] + i_prod] >= 0) {
        jacobian_add_value_gpu(
            jac, (unsigned int)int_data[5 + 2 * int_data[2] + i_prod],
            JACOBIAN_PRODUCTION,
            float_data[3 + i_prod] * number_conc * rate_const);
      }
    }

    // Calculate d_rate/d_effective_radius and d_rate/d_number_concentration
    double d_rate_d_radius =
        state[int_data[1] - 1] * number_conc *
        gpu_d_gas_aerosol_continuum_rxn_rate_constant_d_radius(
            float_data[0], rxn_env_data[0], radius, float_data[1]);
    double d_rate_d_number = state[int_data[1] - 1] * rate_const;

    // Loop through aerosol dependencies
    for (int i_elem = 0; i_elem < int_data[aero_sub_id + 1]; ++i_elem) {
      int eff_jac_id =
          int_data[5 + 3 * int_data[2] + int_data[0] + i_phase] - 1 + i_elem;
      // Reactant dependencies
      if (int_data[aero_sub_id + 2 + i_elem] > 0) {
        // Dependence on effective radius
        jacobian_add_value_gpu(
            jac, (unsigned int)int_data[aero_sub_id + 2 + i_elem],
            JACOBIAN_LOSS, d_rate_d_radius * float_data[eff_jac_id]);
        // Dependence on number concentration
        jacobian_add_value_gpu(
            jac, (unsigned int)int_data[aero_sub_id + 2 + i_elem],
            JACOBIAN_LOSS,
            d_rate_d_number *
                float_data[eff_jac_id + int_data[aero_sub_id + 1]]);
      }
      // Product dependencies
      for (int i_prod = 0; i_prod < int_data[2]; ++i_prod) {
        if (int_data[aero_sub_id + 2 +
                     (i_prod + 1) * int_data[aero_sub_id + 1] + i_elem] > 0) {
          // Dependence on effective radius
          jacobian_add_value_gpu(
              jac,
              (unsigned int)
                  int_data[aero_sub_id + 2 +
                           (i_prod + 1) * int_data[aero_sub_id + 1] + i_elem],
              JACOBIAN_PRODUCTION,
              float_data[3 + i_prod] * d_rate_d_radius *
                  float_data[eff_jac_id]);
          // Dependence on number concentration
          jacobian_add_value_gpu(
              jac,
              (unsigned int)
                  int_data[aero_sub_id + 2 +
                           (i_prod + 1) * int_data[aero_sub_id + 1] + i_elem],
              JACOBIAN_PRODUCTION,
              float_data[3 + i_prod] * d_rate_d_number *
                  float_data[eff_jac_id + int_data[aero_sub_id + 1]]);
        }
      }
    }
  }
}

#endif // AERO_RXN_SURFACE_DEV_H_
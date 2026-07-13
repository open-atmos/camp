/* Copyright (C) 2021 Barcelona Supercomputing Center and University of
 * Illinois at Urbana-Champaign
 * SPDX-License-Identifier: MIT
 */

/** \file
 * \brief Raoult Phase Transfer reaction solver functions
 */
#ifndef AERO_RXN_RAOULT_PT_DEV_H_
#define AERO_RXN_RAOULT_PT_DEV_H_

#include "aero_solver_dev.h"

#define DIFF_COEFF_ float_data[2]
#define PRE_C_AVG_ float_data[3]
#define CONV_ float_data[8]
#define MW_ float_data[9]
#define GAS_SPEC_ (int_data[1] - 1)
#define MFP_M_ rxn_env_data[0]
#define ALPHA_ rxn_env_data[1]
#define EQUIL_CONST_ rxn_env_data[2]
#define KGM3_TO_PPM_ rxn_env_data[3]

/** \brief Calculate contributions to the time derivative \f$f(t,y)\f$ from
 * this reaction.
 *
 * \param model_data Pointer to the model data, including the state array
 * \param time_deriv TimeDerivative object
 * \param rxn_int_data Pointer to the reaction integer data
 * \param rxn_float_data Pointer to the reaction floating-point data
 * \param rxn_env_data Pointer to the environment-dependent parameters
 * \param time_step Current time step being computed (s)
 */
__device__ void rxn_gpu_raoult_phase_transfer_calc_deriv_contrib(
    ModelDataVariable *sc, ModelDataGPU *md, TimeDerivativeGPU time_deriv, int *rxn_int_data,
    double *rxn_float_data, double *rxn_env_data, double time_step) {
  int *int_data = rxn_int_data;
  double *float_data = rxn_float_data;
  double *state = sc->grid_cell_state;

  // Calculate derivative contributions for each aerosol phase
  for (int i_phase = 0; i_phase < int_data[0]; i_phase++) {
    int *aero_rep_int_data =
        &(md->aero_rep_int_data[md->aero_rep_int_indices
                                    [int_data[2 + 3 * (int_data[0]) + i_phase] -
                                     1]]) +
        1;
    double *aero_rep_float_data =
        &(float_data[md->aero_rep_float_indices
                         [int_data[2 + 3 * (int_data[0]) + i_phase] - 1]]);
    double *aero_rep_env_data =
        &(md->aero_rep_env_data
              [md->aero_rep_env_idx[int_data[2 + 3 * (int_data[0]) + i_phase] -
                                    1]]);

    // Get the particle effective radius (m)
    double radius;
    aero_rep_modal_binned_mass_get_effective_radius__m(
        int_data[2 + 2 * (int_data[0]) + i_phase] - 1, &radius, NULL,
        aero_rep_int_data, aero_rep_float_data);

    // Get the particle number concentration (#/m3)
    double number_conc;
    aero_rep_modal_binned_mass_get_number_conc__n_m3(
        md, int_data[2 + 2 * (int_data[0]) + i_phase] - 1, state, &number_conc,
        NULL, aero_rep_int_data, aero_rep_float_data, aero_rep_env_data);

    // Get the total mass of the aerosol phase (kg/m3)
    double aero_phase_mass;
    aero_rep_modal_binned_mass_get_aero_phase_mass__kg_m3(
        md, int_data[2 + 2 * (int_data[0]) + i_phase] - 1, state,
        &aero_phase_mass, NULL, aero_rep_int_data, aero_rep_float_data);

    // Get the total mass of the aerosol phase (kg/mol)
    double aero_phase_avg_MW;
    aero_rep_modal_binned_mass_get_aero_phase_avg_MW__kg_mol(
        md, int_data[2 + 2 * (int_data[0]) + i_phase] - 1, state,
        &aero_phase_avg_MW, NULL, aero_rep_int_data, aero_rep_float_data);

    // Calculate the rate constant for diffusion limited mass transfer to the
    // aerosol phase (m3/#/s)
    double cond_rate = gpu_gas_aerosol_transition_rxn_rate_constant(
        DIFF_COEFF_, MFP_M_, radius, ALPHA_);

    // Calculate the evaporation rate constant (ppm_x*m^3/kg_x/s)
    double evap_rate =
        cond_rate * (EQUIL_CONST_ * aero_phase_avg_MW / aero_phase_mass);

    // Get the activity coefficient (if one exists)
    double act_coeff = 1.0;
    if (int_data[2 + int_data[0] + i_phase] > 0) {
      act_coeff = state[int_data[2 + int_data[0] + i_phase] - 1];
    }

    // Calculate aerosol-phase evaporation rate (ppm/s)
    evap_rate *= act_coeff;

    // Calculate the evaporation and condensation rates
    cond_rate *= state[int_data[1] - 1];
    evap_rate *= state[int_data[2 + i_phase] - 1];

    // Change in the gas-phase is evaporation - condensation (ppm/s)
    if (int_data[2 + 4 * (int_data[0])] >= 0) {
      time_derivative_add_value_gpu(time_deriv, int_data[2 + 4 * (int_data[0])],
                                    number_conc * evap_rate);
      time_derivative_add_value_gpu(time_deriv, int_data[2 + 4 * (int_data[0])],
                                    -number_conc * cond_rate);
    }

    // Change in the aerosol-phase species is condensation - evaporation
    // (kg/m^3/s)
    if (int_data[3 + 4 * (int_data[0]) + i_phase] >= 0) {
      time_derivative_add_value_gpu(time_deriv,
                                    int_data[3 + 4 * (int_data[0]) + i_phase],
                                    -number_conc * evap_rate / KGM3_TO_PPM_);
      time_derivative_add_value_gpu(time_deriv,
                                    int_data[3 + 4 * (int_data[0]) + i_phase],
                                    number_conc * cond_rate / KGM3_TO_PPM_);
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
 * \param time_step Current time step being calculated (s)
 */
__device__ void rxn_gpu_raoult_phase_transfer_calc_jac_contrib(
    ModelDataVariable *sc, ModelDataGPU *md, JacobianGPU jac, int *rxn_int_data,
    double *rxn_float_data, double *rxn_env_data, double time_step) {
  int *int_data = rxn_int_data;
  double *float_data = rxn_float_data;
  double *state = sc->grid_cell_state;

  // Calculate derivative contributions for each aerosol phase
  for (int i_phase = 0; i_phase < int_data[0]; i_phase++) {
    int *aero_rep_int_data =
        &(md->aero_rep_int_data[md->aero_rep_int_indices
                                    [int_data[2 + 3 * (int_data[0]) + i_phase] -
                                     1]]) +
        1;
    double *aero_rep_float_data =
        &(float_data[md->aero_rep_float_indices
                         [int_data[2 + 3 * (int_data[0]) + i_phase] - 1]]);
    double *aero_rep_env_data =
        &(md->aero_rep_env_data
              [md->aero_rep_env_idx[int_data[2 + 3 * (int_data[0]) + i_phase] -
                                    1]]);

    // Get the particle effective radius (m)
    double radius;
    aero_rep_modal_binned_mass_get_effective_radius__m(
        int_data[2 + 2 * (int_data[0]) + i_phase] - 1, &radius, NULL,
        aero_rep_int_data, aero_rep_float_data);

    // Get the particle number concentration (#/m3)
    double number_conc;
    aero_rep_modal_binned_mass_get_number_conc__n_m3(
        md, int_data[2 + 2 * (int_data[0]) + i_phase] - 1, state, &number_conc,
        NULL, aero_rep_int_data, aero_rep_float_data, aero_rep_env_data);

    // Get the total mass of the aerosol phase (kg/m3)
    double aero_phase_mass;
    aero_rep_modal_binned_mass_get_aero_phase_mass__kg_m3(
        md, int_data[2 + 2 * (int_data[0]) + i_phase] - 1, state,
        &aero_phase_mass, NULL, aero_rep_int_data, aero_rep_float_data);

    // Get the total mass of the aerosol phase (kg/mol)
    double aero_phase_avg_MW;
    aero_rep_modal_binned_mass_get_aero_phase_avg_MW__kg_mol(
        md, int_data[2 + 2 * (int_data[0]) + i_phase] - 1, state,
        &aero_phase_avg_MW, NULL, aero_rep_int_data, aero_rep_float_data);

    // Calculate the rate constant for diffusion limited mass transfer to the
    // aerosol phase (m3/#/s)
    double cond_rate = gpu_gas_aerosol_transition_rxn_rate_constant(
        DIFF_COEFF_, MFP_M_, radius, ALPHA_);

    // Calculate the evaporation rate constant (ppm_x*m^3/kg_x/s)
    double evap_rate =
        cond_rate * (EQUIL_CONST_ * aero_phase_avg_MW / aero_phase_mass);

    // Get the activity coefficient (if one exists)
    double act_coeff = 1.0;
    if (int_data[2 + int_data[0] + i_phase] - 1 > -1) {
      act_coeff = state[int_data[2 + int_data[0] + i_phase] - 1];
    }

    // total-particle mass concentrations
    // Change in the gas-phase is evaporation - condensation (ppm/s)
    if (int_data[5 + 7 * (int_data[0]) + i_phase * 3] >= 0) {
      jacobian_add_value_gpu(
          jac, (unsigned int)int_data[5 + 7 * (int_data[0]) + i_phase * 3],
          JACOBIAN_PRODUCTION, number_conc * evap_rate * act_coeff);
    }
    if (int_data[3 + 7 * (int_data[0])] >= 0) {
      jacobian_add_value_gpu(jac, (unsigned int)int_data[3 + 7 * (int_data[0])],
                             JACOBIAN_LOSS, number_conc * cond_rate);
    }

    // Change in the aerosol-phase species is condensation - evaporation
    // (kg/m^3/s)
    if (int_data[4 + 7 * (int_data[0]) + i_phase * 3] >= 0) {
      jacobian_add_value_gpu(
          jac, (unsigned int)int_data[4 + 7 * (int_data[0]) + i_phase * 3],
          JACOBIAN_PRODUCTION, number_conc * cond_rate / KGM3_TO_PPM_);
    }
    if (int_data[6 + 7 * (int_data[0]) + i_phase * 3] >= 0) {
      jacobian_add_value_gpu(
          jac, (unsigned int)int_data[6 + 7 * (int_data[0]) + i_phase * 3],
          JACOBIAN_LOSS, number_conc * evap_rate * act_coeff / KGM3_TO_PPM_);
    }

    // Activity coefficient contributions
    if (int_data[3 + 5 * (int_data[0]) + i_phase] > 0) {
      jacobian_add_value_gpu(
          jac, (unsigned int)int_data[3 + 5 * (int_data[0]) + i_phase],
          JACOBIAN_PRODUCTION,
          number_conc * evap_rate * state[int_data[2 + i_phase] - 1]);
    }
    if (int_data[3 + 6 * (int_data[0]) + i_phase] > 0) {
      jacobian_add_value_gpu(
          jac, (unsigned int)int_data[3 + 6 * (int_data[0]) + i_phase],
          JACOBIAN_LOSS,
          number_conc * evap_rate / KGM3_TO_PPM_ *
              state[int_data[2 + i_phase] - 1]);
    }

    // Get the overall rates
    evap_rate *= act_coeff;
    cond_rate *= state[int_data[1] - 1];
    evap_rate *= state[int_data[2 + i_phase] - 1];

    // Calculate partial derivatives
    double d_cond_d_radius =
        gpu_d_gas_aerosol_transition_rxn_rate_constant_d_radius(DIFF_COEFF_, MFP_M_,
                                                            radius, ALPHA_) *
        state[int_data[1] - 1];
    double d_evap_d_radius = d_cond_d_radius / state[int_data[1] - 1] *
                             EQUIL_CONST_ * aero_phase_avg_MW /
                             aero_phase_mass * state[int_data[2 + i_phase] - 1];
    double d_evap_d_mass = -evap_rate / aero_phase_mass;
    double d_evap_d_MW = evap_rate / aero_phase_avg_MW;

    // Loop through Jac elements and update
    for (int i_elem = 0;
         i_elem < int_data[int_data[4 + 10 * (int_data[0]) + i_phase] - 1];
         ++i_elem) {
      // Gas-phase species dependencies
      if (int_data[int_data[4 + 10 * int_data[0] + i_phase] +
                   (0) *
                       int_data[int_data[4 + 10 * int_data[0] + i_phase] - 1] +
                   i_elem] > 0) {
        // species involved in effective radius calculations
        jacobian_add_value_gpu(
            jac,
            (unsigned int)int_data
                [int_data[4 + 10 * int_data[0] + i_phase] +
                 (0) * int_data[int_data[4 + 10 * int_data[0] + i_phase] - 1] +
                 i_elem],
            JACOBIAN_PRODUCTION,
            number_conc * d_evap_d_radius *
                float_data[int_data[4 + 11 * (int_data[0]) + i_phase] - 1 +
                           i_elem]);
        jacobian_add_value_gpu(
            jac,
            (unsigned int)int_data
                [int_data[4 + 10 * int_data[0] + i_phase] +
                 (0) * int_data[int_data[4 + 10 * int_data[0] + i_phase] - 1] +
                 i_elem],
            JACOBIAN_LOSS,
            number_conc * d_cond_d_radius *
                float_data[int_data[4 + 11 * (int_data[0]) + i_phase] - 1 +
                           i_elem]);

        // species involved in number concentration
        jacobian_add_value_gpu(
            jac,
            (unsigned int)int_data
                [int_data[4 + 10 * int_data[0] + i_phase] +
                 (0) * int_data[int_data[4 + 10 * int_data[0] + i_phase] - 1] +
                 i_elem],
            JACOBIAN_PRODUCTION,
            evap_rate *
                float_data[int_data[4 + 11 * (int_data[0]) + i_phase] - 1 +
                           int_data[int_data[4 + 10 * int_data[0] + i_phase] -
                                    1] +
                           i_elem]);
        jacobian_add_value_gpu(
            jac,
            (unsigned int)int_data
                [int_data[4 + 10 * int_data[0] + i_phase] +
                 (0) * int_data[int_data[4 + 10 * int_data[0] + i_phase] - 1] +
                 i_elem],
            JACOBIAN_LOSS,
            cond_rate *
                float_data[int_data[4 + 11 * (int_data[0]) + i_phase] - 1 +
                           int_data[int_data[4 + 10 * int_data[0] + i_phase] -
                                    1] +
                           i_elem]);

        // species involved in mass calculations
        jacobian_add_value_gpu(
            jac,
            (unsigned int)int_data
                [int_data[4 + 10 * int_data[0] + i_phase] +
                 (0) * int_data[int_data[4 + 10 * int_data[0] + i_phase] - 1] +
                 i_elem],
            JACOBIAN_PRODUCTION,
            number_conc * d_evap_d_mass *
                float_data
                    [int_data[4 + 11 * (int_data[0]) + i_phase] - 1 +
                     2 * int_data[int_data[4 + 10 * int_data[0] + i_phase] -
                                  1] +
                     i_elem]);

        // species involved in average MW calculations
        jacobian_add_value_gpu(
            jac,
            (unsigned int)int_data
                [int_data[4 + 10 * int_data[0] + i_phase] +
                 (0) * int_data[int_data[4 + 10 * int_data[0] + i_phase] - 1] +
                 i_elem],
            JACOBIAN_PRODUCTION,
            number_conc * d_evap_d_MW *
                float_data
                    [int_data[4 + 11 * (int_data[0]) + i_phase] - 1 +
                     3 * int_data[int_data[4 + 10 * int_data[0] + i_phase] -
                                  1] +
                     i_elem]);
      }

      // Aerosol-phase species dependencies
      if (int_data[int_data[4 + 10 * int_data[0] + i_phase] +
                   (1) *
                       int_data[int_data[4 + 10 * int_data[0] + i_phase] - 1] +
                   i_elem] > 0) {
        // species involved in effective radius calculations
        jacobian_add_value_gpu(
            jac,
            (unsigned int)int_data
                [int_data[4 + 10 * int_data[0] + i_phase] +
                 (1) * int_data[int_data[4 + 10 * int_data[0] + i_phase] - 1] +
                 i_elem],
            JACOBIAN_LOSS,
            number_conc * d_evap_d_radius / KGM3_TO_PPM_ *
                float_data[int_data[4 + 11 * (int_data[0]) + i_phase] - 1 +
                           i_elem]);
        jacobian_add_value_gpu(
            jac,
            (unsigned int)int_data
                [int_data[4 + 10 * int_data[0] + i_phase] +
                 (1) * int_data[int_data[4 + 10 * int_data[0] + i_phase] - 1] +
                 i_elem],
            JACOBIAN_PRODUCTION,
            number_conc * d_cond_d_radius / KGM3_TO_PPM_ *
                float_data[int_data[4 + 11 * (int_data[0]) + i_phase] - 1 +
                           i_elem]);

        // species involved in number concentration
        jacobian_add_value_gpu(
            jac,
            (unsigned int)int_data
                [int_data[4 + 10 * int_data[0] + i_phase] +
                 (1) * int_data[int_data[4 + 10 * int_data[0] + i_phase] - 1] +
                 i_elem],
            JACOBIAN_LOSS,
            evap_rate / KGM3_TO_PPM_ *
                float_data[int_data[4 + 11 * (int_data[0]) + i_phase] - 1 +
                           int_data[int_data[4 + 10 * int_data[0] + i_phase] -
                                    1] +
                           i_elem]);
        jacobian_add_value_gpu(
            jac,
            (unsigned int)int_data
                [int_data[4 + 10 * int_data[0] + i_phase] +
                 (1) * int_data[int_data[4 + 10 * int_data[0] + i_phase] - 1] +
                 i_elem],
            JACOBIAN_PRODUCTION,
            cond_rate / KGM3_TO_PPM_ *
                float_data[int_data[4 + 11 * (int_data[0]) + i_phase] - 1 +
                           int_data[int_data[4 + 10 * int_data[0] + i_phase] -
                                    1] +
                           i_elem]);

        // species involved in mass calculations
        jacobian_add_value_gpu(
            jac,
            (unsigned int)int_data
                [int_data[4 + 10 * int_data[0] + i_phase] +
                 (1) * int_data[int_data[4 + 10 * int_data[0] + i_phase] - 1] +
                 i_elem],
            JACOBIAN_LOSS,
            number_conc * d_evap_d_mass / KGM3_TO_PPM_ *
                float_data
                    [int_data[4 + 11 * (int_data[0]) + i_phase] - 1 +
                     2 * int_data[int_data[4 + 10 * int_data[0] + i_phase] -
                                  1] +
                     i_elem]);

        // species involved in average MW calculations
        jacobian_add_value_gpu(
            jac,
            (unsigned int)int_data
                [int_data[4 + 10 * int_data[0] + i_phase] +
                 (1) * int_data[int_data[4 + 10 * int_data[0] + i_phase] - 1] +
                 i_elem],
            JACOBIAN_LOSS,
            number_conc * d_evap_d_MW / KGM3_TO_PPM_ *
                float_data
                    [int_data[4 + 11 * (int_data[0]) + i_phase] - 1 +
                     3 * int_data[int_data[4 + 10 * int_data[0] + i_phase] -
                                  1] +
                     i_elem]);
      }
    }
  }
  return;
}

#endif // AERO_RXN_RAOULT_PT_DEV_H_
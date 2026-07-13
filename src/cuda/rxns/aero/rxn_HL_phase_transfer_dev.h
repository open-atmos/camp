/* Copyright (C) 2021 Barcelona Supercomputing Center and University of
 * Illinois at Urbana-Champaign
 * SPDX-License-Identifier: MIT
 */

/** \file
 * \brief HL Phase Transfer reaction solver functions
 */
#ifndef AERO_RXN_HL_PT_DEV_H_
#define AERO_RXN_HL_PT_DEV_H_

#include "aero_solver_dev.h"

/** \brief Calculate contributions to the time derivative \f$f(t,y)\f$ from
 * this reaction.
 *
 * \bug this does not work for modal/binned aero reps. Needs update following
 *      the logic in the SIMPOL partitioning reaction
 *
 * \param sc Pointer to the structure containing the grid_cell data state array
 * \param mc Pointer to the GPU model data
 * \param time_deriv TimeDerivative object
 * \param rxn_int_data Pointer to the reaction integer data
 * \param rxn_float_data Pointer to the reaction floating-point data
 * \param rxn_env_data Pointer to the environment-dependent parameters
 * \param time_step Current time step being computed (s)
 */
__device__ void rxn_gpu_HL_phase_transfer_calc_deriv_contrib(
    ModelDataVariable *sc, ModelDataGPU *md, TimeDerivativeGPU time_deriv, int *rxn_int_data,
    double *rxn_float_data, double *rxn_env_data, double time_step) {
  int *int_data = rxn_int_data;
  double *float_data = rxn_float_data;
  double *state = sc->grid_cell_state;

  // Calculate derivative contributions for each aerosol phase
  for (int i_phase = 0; i_phase < int_data[0]; i_phase++) {
    int aero_off = int_data[4 + 6 * int_data[0] + i_phase];
    int aero_rep_id = int_data[aero_off + 2] - 1;
    int aero_phase_id = int_data[aero_off + 1] - 1;

    int *aero_rep_int_data =
        &(md->aero_rep_int_data[md->aero_rep_int_indices[aero_rep_id]]);
    double *aero_rep_float_data =
        &(md->aero_rep_float_data[md->aero_rep_float_indices[aero_rep_id]]);

    // Get the particle effective radius (m)
    double radius;
    aero_rep_modal_binned_mass_get_effective_radius__m(
        aero_phase_id, &radius, NULL, aero_rep_int_data, aero_rep_float_data);

    // int aero_rep_type = *(aero_rep_int_data++);

    // int aero_conc_type = aero_rep_get_aero_conc_type(){
    // switch(aero_rep_type)
    // case (AERO_REP_MODAL_BINNED_MASS (2)){ return 1 }
    // case (AERO_REP_SINGLE_PARTICLE (1)) { return 0 }}

    // Get the particle number concentration (#/m3) for per-particle mass
    // concentrations; otherwise set to 1
    double number_conc = 1.0;

    // Calculate the rate constant for diffusion limited mass transfer to the
    // aerosol phase (1/s)
    double cond_rate = gpu_gas_aerosol_transition_rxn_rate_constant(
        float_data[2], rxn_env_data[0], radius, rxn_env_data[1]);

    // Calculate the evaporation rate constant (1/s)
    double evap_rate = cond_rate / rxn_env_data[2];

    // Calculate the evaporation and condensation rates (ppm/s)
    cond_rate *= state[int_data[1] - 1];
    evap_rate *=
        state[int_data[aero_off - 1] - 1] / state[int_data[aero_off] - 1];

    // Change in the gas-phase is evaporation - condensation (ppm/s)
    if (int_data[2] >= 0) {
      time_derivative_add_value_gpu(time_deriv, int_data[2],
                                    number_conc * evap_rate);
      time_derivative_add_value_gpu(time_deriv, int_data[2],
                                    -number_conc * cond_rate);
    }

    // Change in the aerosol-phase species is condensation - evaporation
    // (kg/m^3/s)
    if (int_data[3 + i_phase] >= 0) {
      time_derivative_add_value_gpu(time_deriv, int_data[3 + i_phase],
                                    -evap_rate / rxn_env_data[3]);
      time_derivative_add_value_gpu(time_deriv, int_data[3 + i_phase],
                                    cond_rate / rxn_env_data[3]);
    }
  }
}

/** \brief Calculate contributions to the Jacobian from this reaction
 *
 * \param sc Pointer to the structure containing the grid_cell data state array
 * \param mc Pointer to the GPU model data
 * \param jac Reaction Jacobian
 * \param rxn_int_data Pointer to the reaction integer data
 * \param rxn_float_data Pointer to the reaction floating-point data
 * \param rxn_env_data Pointer to the environment-dependent parameters
 * \param time_step Current time step being calculated (s)
 */
__device__ void rxn_gpu_HL_phase_transfer_calc_jac_contrib(
    ModelDataVariable *sc, ModelDataGPU *md, JacobianGPU jac, int *rxn_int_data,
    double *rxn_float_data, double *rxn_env_data, double time_step) {
  int *int_data = rxn_int_data;
  double *float_data = rxn_float_data;
  double *state = sc->grid_cell_state;

  // Calculate derivative contributions for each aerosol phase
  for (int i_phase = 0; i_phase < int_data[0]; i_phase++) {
    int aero_off = int_data[4 + 6 * int_data[0] + i_phase];
    int aero_rep_id = int_data[aero_off + 2] - 1;
    int aero_phase_id = int_data[aero_off + 1] - 1;
    int num_aero_jac = int_data[aero_off + 3];

    int *aero_rep_int_data =
        &(md->aero_rep_int_data[md->aero_rep_int_indices[aero_rep_id]]);
    double *aero_rep_float_data =
        &(md->aero_rep_float_data[md->aero_rep_float_indices[aero_rep_id]]);

    // Get the particle effective radius (m)
    double radius;
    aero_rep_modal_binned_mass_get_effective_radius__m(
        aero_phase_id, &radius,
        &float_data[int_data[4 + 7 * int_data[0] + i_phase]], aero_rep_int_data,
        aero_rep_float_data);

    // Get the particle number concentration (#/m3) for per-particle
    // concentrations
    double number_conc = 1.0;
    for (int i_elem = 0; i_elem < num_aero_jac; ++i_elem)
      float_data[int_data[4 + 7 * int_data[0] + i_phase] + i_elem + num_aero_jac] = 0.0;

    // Calculate the rate constant for diffusion limited mass transfer to the
    // aerosol phase (1/s)
    double cond_rate = gpu_gas_aerosol_transition_rxn_rate_constant(
        float_data[2], rxn_env_data[0], radius, rxn_env_data[1]);

    // Calculate the evaporation rate constant (1/s)
    double evap_rate = cond_rate / rxn_env_data[2];

    // Change in the gas-phase is evaporation - condensation (ppm/s)
    if (int_data[3 + int_data[0]] >= 0)
      jacobian_add_value_gpu(jac, (unsigned int)int_data[3 + int_data[0]],
                             JACOBIAN_LOSS, number_conc * cond_rate);
    if (int_data[5 + int_data[0] + i_phase * 5] >= 0)
      jacobian_add_value_gpu(
          jac, (unsigned int)int_data[5 + int_data[0] + i_phase * 5],
          JACOBIAN_PRODUCTION,
          number_conc * evap_rate / state[int_data[aero_off] - 1]);
    if (int_data[7 + int_data[0] + i_phase * 5] >= 0)
      jacobian_add_value_gpu(
          jac, (unsigned int)int_data[7 + int_data[0] + i_phase * 5],
          JACOBIAN_PRODUCTION,
          -number_conc * evap_rate * state[int_data[aero_off - 1] - 1] /
              state[int_data[aero_off] - 1] / state[int_data[aero_off] - 1]);

    // Change in the aerosol-phase species is condensation - evaporation
    // (kg/m^3/s)
    if (int_data[4 + int_data[0] + i_phase * 5] >= 0)
      jacobian_add_value_gpu(
          jac, (unsigned int)int_data[4 + int_data[0] + i_phase * 5],
          JACOBIAN_PRODUCTION, cond_rate / rxn_env_data[3]);
    if (int_data[6 + int_data[0] + i_phase * 5] >= 0)
      jacobian_add_value_gpu(
          jac, (unsigned int)int_data[6 + int_data[0] + i_phase * 5],
          JACOBIAN_LOSS,
          evap_rate / state[int_data[aero_off] - 1] / rxn_env_data[3]);
    if (int_data[8 + int_data[0] + i_phase * 5] >= 0)
      jacobian_add_value_gpu(
          jac, (unsigned int)int_data[8 + int_data[0] + i_phase * 5],
          JACOBIAN_LOSS,
          -evap_rate * state[int_data[aero_off - 1] - 1] / rxn_env_data[3] /
              state[int_data[aero_off] - 1] / state[int_data[aero_off] - 1]);

    // Calculate the condensation and evaporation rates (ppm/s)
    cond_rate *= state[int_data[1] - 1];
    evap_rate *=
        state[int_data[aero_off - 1] - 1] / state[int_data[aero_off] - 1];

    // Add contributions from species used in aerosol property calculations

    // Calculate d_rate/d_effecive_radius and d_rate/d_number_concentration
    // ( This was replaced with transition-regime rate equation. )
    double d_cond_d_radius =
        gpu_d_gas_aerosol_transition_rxn_rate_constant_d_radius(
            float_data[2], rxn_env_data[0], radius, rxn_env_data[1]) *
        state[int_data[1] - 1];
    double d_evap_d_radius =
        d_cond_d_radius / state[int_data[1] - 1] / rxn_env_data[2] *
        state[int_data[aero_off - 1] - 1] / state[int_data[aero_off] - 1];

    // Loop through Jac elements and update
    for (int i_elem = 0; i_elem < num_aero_jac; ++i_elem) {
      int eff_jac_id = int_data[4 + 7 * int_data[0] + i_phase] + i_elem;
      // Gas-phase species dependencies
      if (int_data[aero_off + 4 + i_elem] > 0) {
        // species involved in effective radius calculation
        jacobian_add_value_gpu(
            jac, (unsigned int)int_data[aero_off + 4 + i_elem],
            JACOBIAN_PRODUCTION,
            number_conc * d_evap_d_radius * float_data[eff_jac_id]);
        jacobian_add_value_gpu(
            jac, (unsigned int)int_data[aero_off + 4 + i_elem], JACOBIAN_LOSS,
            number_conc * d_cond_d_radius * float_data[eff_jac_id]);

        // species involved in number concentration
        jacobian_add_value_gpu(
            jac, (unsigned int)int_data[aero_off + 4 + i_elem],
            JACOBIAN_PRODUCTION,
            evap_rate * float_data[eff_jac_id + num_aero_jac]);
        jacobian_add_value_gpu(
            jac, (unsigned int)int_data[aero_off + 4 + i_elem], JACOBIAN_LOSS,
            cond_rate * float_data[eff_jac_id + num_aero_jac]);
      }

      // Aerosol-phase species dependencies
      if (int_data[aero_off + 4 + num_aero_jac + i_elem] > 0) {
        // species involved in effective radius calculation
        jacobian_add_value_gpu(
            jac, (unsigned int)int_data[aero_off + 4 + num_aero_jac + i_elem],
            JACOBIAN_LOSS,
            d_evap_d_radius / rxn_env_data[3] * float_data[eff_jac_id]);
        jacobian_add_value_gpu(
            jac, (unsigned int)int_data[aero_off + 4 + num_aero_jac + i_elem],
            JACOBIAN_PRODUCTION,
            d_cond_d_radius / rxn_env_data[3] * float_data[eff_jac_id]);
      }
    }
  }
}
#endif  // AERO_RXN_HL_PT_DEV_H_
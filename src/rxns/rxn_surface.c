/* Copyright (C) 2023 Barcelona Supercomputing Center, University of
 * Illinois at Urbana-Champaign, and National Center for Atmospheric Research
 * SPDX-License-Identifier: MIT
 *
 * Surface reaction solver functions
 *
 */
/** \file
 * \brief Surface reaction solver functions
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "../aero_rep_solver.h"
#include "../rxns.h"
#include "../util.h"

// TODO Lookup environmental indices during initialization
#define TEMPERATURE_K_ env_data[0]
#define PRESSURE_PA_ env_data[1]

#define DIFF_COEFF_ float_data[0]
#define PRE_C_AVG_ float_data[1]
#define GAMMA_ float_data[2]
#define MW_ float_data[3]
#define NUM_AERO_PHASE_ int_data[0]
#define GAS_SPEC_ (int_data[1] - 1)
#define NUM_PROD_ int_data[2]
#define MFP_M_ rxn_env_data[0]
#define NUM_INT_PROP_ 3
#define NUM_FLOAT_PROP_ 4
#define NUM_ENV_PARAM_ 1
#define DERIV_ID_(x) int_data[NUM_INT_PROP_ + x]
#define JAC_ID_(x) int_data[NUM_INT_PROP_ + 1 + NUM_PROD_ + x]
#define PHASE_INT_LOC_(x) \
  (int_data[NUM_INT_PROP_ + 2 + 2 * NUM_PROD_ + x] - 1)
#define PHASE_REAL_LOC_(x) \
  (int_data[NUM_INT_PROP_ + 2 + 2 * NUM_PROD_ + NUM_AERO_PHASE_ + x] - 1)
#define AERO_PHASE_ID_(x) (int_data[PHASE_INT_LOC_(x)] - 1)
#define AERO_REP_ID_(x) (int_data[PHASE_INT_LOC_(x) + 1] - 1)
#define NUM_AERO_PHASE_JAC_ELEM_(x) (int_data[PHASE_INT_LOC_(x) + 2])
#define PHASE_JAC_ID_(x, s, e) \
  int_data[PHASE_INT_LOC_(x) + 2 + s * NUM_AERO_PHASE_JAC_ELEM_(x) + e]
#define YIELD_(x) float_data[NUM_REAL_PROP_ + x]
#define EFF_RAD_JAC_ELEM_(x, e) float_data[PHASE_REAL_LOC_(x) + e]
#define NUM_CONC_JAC_ELEM_(x, e) \
  float_data[PHASE_REAL_LOC_(x) + NUM_AERO_PHASE_JAC_ELEM_(x) + e]

/** \brief Flag Jacobian elements used by this reaction
 *
 * \param model_data Pointer to the model data
 * \param rxn_int_data Pointer to the reaction integer data
 * \param rxn_float_data Pointer to the reaction floating-point data
 * \param jac Jacobian
 */
void rxn_surface_get_used_jac_elem(ModelData *model_data,
                                             int *rxn_int_data,
                                             double *rxn_float_data,
                                             Jacobian *jac) {
  int *int_data = rxn_int_data;
  double *float_data = rxn_float_data;

  bool *aero_jac_elem =
      (bool *)malloc(sizeof(bool) * model_data->n_per_cell_state_var);
  if (aero_jac_elem == NULL) {
    printf(
        "\n\nERROR allocating space for 1D jacobian structure array for "
        "surface reaction\n\n");
    exit(1);
  }

  jacobian_register_element(jac, GAS_SPEC_, GAS_SPEC_);

  // Finish

  free(aero_jac_elem);

  return;
}

/** \brief Update the time derivative and Jacbobian array indices
 *
 * \param model_data Pointer to the model data
 * \param deriv_ids Id of each state variable in the derivative array
 * \param jac Jacobian
 * \param rxn_int_data Pointer to the reaction integer data
 * \param rxn_float_data Pointer to the reaction floating-point data
 */
void rxn_surface_update_ids(ModelData *model_data, int *deriv_ids,
                                      Jacobian jac, int *rxn_int_data,
                                      double *rxn_float_data) {
  int *int_data = rxn_int_data;
  double *float_data = rxn_float_data;

  return;
}

/** \brief Update reaction data for new environmental conditions
 *
 * For surface reactions this only involves recalculating the rate
 * constant.
 *
 * \param model_data Pointer to the model data
 * \param rxn_int_data Pointer to the reaction integer data
 * \param rxn_float_data Pointer to the reaction floating-point data
 * \param rxn_env_data Pointer to the environment-dependent parameters
 */
void rxn_surface_update_env_state(ModelData *model_data,
                                            int *rxn_int_data,
                                            double *rxn_float_data,
                                            double *rxn_env_data) {
  int *int_data = rxn_int_data;
  double *float_data = rxn_float_data;
  double *env_data = model_data->grid_cell_env;

  // save the mean free path [m] for calculating condensation rates
  MFP_M_ = mean_free_path__m(DIFF_COEFF_, TEMPERATURE_K_, GAMMA_);

  return;
}

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
#ifdef CAMP_USE_SUNDIALS
void rxn_surface_calc_deriv_contrib(
    ModelData *model_data, TimeDerivative time_deriv, int *rxn_int_data,
    double *rxn_float_data, double *rxn_env_data, realtype time_step) {
  int *int_data = rxn_int_data;
  double *float_data = rxn_float_data;
  double *state = model_data->grid_cell_state;
  double *env_data = model_data->grid_cell_env;

  // Calculate derivative contributions for each aerosol phase
  for (int i_phase = 0; i_phase < NUM_AERO_PHASE_; i_phase++) {
    // Get the particle effective radius (m)
    realtype radius;
    aero_rep_get_effective_radius__m(
        model_data,               // model data
        AERO_REP_ID_(i_phase),    // aerosol representation index
        AERO_PHASE_ID_(i_phase),  // aerosol phase index
        &radius,                  // particle effective radius (m)
        NULL);                    // partial derivative

    // Check the aerosol concentration type (per-particle or total per-phase
    // mass)
    int aero_conc_type = aero_rep_get_aero_conc_type(
        model_data,                // model data
        AERO_REP_ID_(i_phase),     // aerosol representation index
        AERO_PHASE_ID_(i_phase));  // aerosol phase index

    // Get the particle number concentration (#/m3) for per-particle mass
    // concentrations; otherwise set to 1
    realtype number_conc = ONE;
    if (aero_conc_type == 0) {
      aero_rep_get_number_conc__n_m3(
          model_data,               // model data
          AERO_REP_ID_(i_phase),    // aerosol representation index
          AERO_PHASE_ID_(i_phase),  // aerosol phase index
          &number_conc,             // particle number concentration
                                    // (#/m3)
          NULL);                    // partial derivative
    }

    // Calculate the rate constant for diffusion limited mass transfer to the
    // aerosol phase (1/s)

    // Finish
  }
  return;
}
#endif

/** \brief Calculate contributions to the Jacobian from this reaction
 *
 * \param model_data Pointer to the model data
 * \param jac Reaction Jacobian
 * \param rxn_int_data Pointer to the reaction integer data
 * \param rxn_float_data Pointer to the reaction floating-point data
 * \param rxn_env_data Pointer to the environment-dependent parameters
 * \param time_step Current time step being calculated (s)
 */
#ifdef CAMP_USE_SUNDIALS
void rxn_surface_calc_jac_contrib(ModelData *model_data, Jacobian jac,
                                            int *rxn_int_data,
                                            double *rxn_float_data,
                                            double *rxn_env_data,
                                            realtype time_step) {
  int *int_data = rxn_int_data;
  double *float_data = rxn_float_data;
  double *state = model_data->grid_cell_state;
  double *env_data = model_data->grid_cell_env;

  // Calculate derivative contributions for each aerosol phase
  for (int i_phase = 0; i_phase < NUM_AERO_PHASE_; i_phase++) {
    // Get the particle effective radius (m)
    realtype radius;
    aero_rep_get_effective_radius__m(
        model_data,                         // model data
        AERO_REP_ID_(i_phase),              // aerosol representation index
        AERO_PHASE_ID_(i_phase),            // aerosol phase index
        &radius,                            // particle effective radius (m)
        &(EFF_RAD_JAC_ELEM_(i_phase, 0)));  // partial derivative

    // Check the aerosol concentration type (per-particle or total per-phase
    // mass)
    int aero_conc_type = aero_rep_get_aero_conc_type(
        model_data,                // model data
        AERO_REP_ID_(i_phase),     // aerosol representation index
        AERO_PHASE_ID_(i_phase));  // aerosol phase index

    // Get the particle number concentration (#/m3) for per-particle
    // concentrations
    realtype number_conc = ONE;
    if (aero_conc_type == 0) {
      aero_rep_get_number_conc__n_m3(
          model_data,                          // model data
          AERO_REP_ID_(i_phase),               // aerosol representation index
          AERO_PHASE_ID_(i_phase),             // aerosol phase index
          &number_conc,                        // particle number concentration
                                               // (#/m3)
          &(NUM_CONC_JAC_ELEM_(i_phase, 0)));  // partial derivative
    } else {
      for (int i_elem = 0; i_elem < NUM_AERO_PHASE_JAC_ELEM_(i_phase); ++i_elem)
        NUM_CONC_JAC_ELEM_(i_phase, i_elem) = ZERO;
    }

    // Calculate the rate constant for diffusion limited mass transfer to the
    // aerosol phase (1/s)

    // Finish

  }
  return;
}
#endif

/** \brief Print the surface reaction parameters
 *
 * \param rxn_int_data Pointer to the reaction integer data
 * \param rxn_float_data Pointer to the reaction floating-point data
 */
void rxn_surface_print(int *rxn_int_data, double *rxn_float_data) {
  int *int_data = rxn_int_data;
  double *float_data = rxn_float_data;

  printf("\n\nsurface reaction\n");

  return;
}

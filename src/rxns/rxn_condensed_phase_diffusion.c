/* Copyright (C) 2021 Barcelona Supercomputing Center and University of
 * Illinois at Urbana-Champaign
 * SPDX-License-Identifier: MIT
 *
 * Condensed phase diffusion reaction solver functions
 *
 */
/** \file
 * \brief Condensed phase diffusion reaction solver functions
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <camp/aero_rep_solver.h>
#include <camp/aero_phase_solver.h>
#include <camp/rxns.h>
#include <camp/sub_model_solver.h>
#include <camp/util.h>

// TODO Lookup environmental indices during initialization
//#define TEMPERATURE_K_ env_data[0]
//#define PRESSURE_PA_ env_data[1]

#define NUM_ADJACENT_PAIRS_ int_data[0]

#define NUM_INT_PROP_ 1
#define NUM_FLOAT_PROP_ 0
#define NUM_ENV_PARAM_ 0

#define DIFF_COEFF_INNER_(x) (float_data[(NUM_FLOAT_PROP_) + (x)])
#define DIFF_COEFF_OUTER_(x) (float_data[(NUM_FLOAT_PROP_) + (NUM_ADJACENT_PAIRS_) + (x)])
#define PHASE_ID_INNER_(x) (int_data[(NUM_INT_PROP_) + (x)]-1)
#define PHASE_ID_OUTER_(x) (int_data[(NUM_INT_PROP_) + (NUM_ADJACENT_PAIRS_) + (x)]-1)
#define AERO_SPEC_INNER_(x) (int_data[(NUM_INT_PROP_) + (2*NUM_ADJACENT_PAIRS_) + (x)]-1)
#define AERO_SPEC_OUTER_(x) (int_data[(NUM_INT_PROP_) + (3*NUM_ADJACENT_PAIRS_) + (x)]-1)
#define AERO_REP_ID_(x) (int_data[(NUM_INT_PROP_) + (4*NUM_ADJACENT_PAIRS_) + (x)]-1)

#define DERIV_ID_INNER_(x) (int_data[(NUM_INT_PROP_) + (5*NUM_ADJACENT_PAIRS_) + (x)])
#define DERIV_ID_OUTER_(x) (int_data[(NUM_INT_PROP_) + (6*NUM_ADJACENT_PAIRS_) + (x)])
//#define JAC_ID_(x) (int_data[4*BLOCK_SIZE_ + x]-1)
//#define PHASE_INT_LOC_(x) (int_data[5*BLOCK_SIZE_ + x]-1)
//#define PHASE_FLOAT_LOC_(x) (int_data[9*BLOCK_SIZE_ + x]-1)
//#define NUM_AERO_PHASE_JAC_ELEM_INNER_(x) (int_data[10*BLOCK_SIZE + x]-1)
//#define NUM_AERO_PHASE_JAC_ELEM_OUTER_(x) (int_data[11*BLOCK_SIZE + x]-1)
//#define PHASE_JAC_ID_(x) (int_data[12*BLOCK_SIZE + x]-1)
//#define NUM_CONC_JAC_ELEM_(x) (int_data[13*BLOCK_SIZE + x]-1)
//#define MASS_JAC_ELEM_(x) (int_data[14*BLOCK_SIZE + x]-1)

/** \brief Flag Jacobian elements used by this reaction
 *
 * \param model_data Pointer to the model data
 * \param rxn_int_data Pointer to the reaction integer data
 * \param rxn_float_data Pointer to the reaction floating-point data
 * \param jac Jacobian
 */
void rxn_condensed_phase_diffusion_get_used_jac_elem(ModelData *model_data,
                                                 int *rxn_int_data,
                                                 double *rxn_float_data,
                                                 Jacobian *jac) {
  int *int_data = rxn_int_data;
  double *float_data = rxn_float_data;

  bool *aero_jac_elem =
      (bool *)malloc(sizeof(bool) * model_data->n_per_cell_state_var);
  if (aero_jac_elem == NULL) {
    printf(
        "\n\nERROR allocating space for 1D Jacobian structure array for "
        "condensed phase diffusion reaction\n\n");
    exit(1);
  }

  for (int i_adj_pairs = 0; i_adj_pairs < NUM_ADJACENT_PAIRS_; ++i_adj_pairs) {
     jacobian_register_element(jac, AERO_SPEC_INNER_(i_adj_pairs),
                               AERO_SPEC_INNER_(i_adj_pairs));
     jacobian_register_element(jac, AERO_SPEC_INNER_(i_adj_pairs),
                               AERO_SPEC_OUTER_(i_adj_pairs));
     jacobian_register_element(jac, AERO_SPEC_OUTER_(i_adj_pairs),
                               AERO_SPEC_INNER_(i_adj_pairs));
     jacobian_register_element(jac, AERO_SPEC_OUTER_(i_adj_pairs),
                               AERO_SPEC_OUTER_(i_adj_pairs));
  }
  free(aero_jac_elem);

}

/** \brief Update the time derivative and Jacbobian array indices
 *
 * \param model_data Pointer to the model data for finding sub model ids
 * \param deriv_ids Id of each state variable in the derivative array
 * \param jac Jacobian
 * \param rxn_int_data Pointer to the reaction integer data
 * \param rxn_float_data Pointer to the reaction floating-point data
 */
void rxn_condensed_phase_diffusion_update_ids(ModelData *model_data, int *deriv_ids,
                                          Jacobian jac, int *rxn_int_data,
                                          double *rxn_float_data) {
  int *int_data = rxn_int_data;
  double *float_data = rxn_float_data;

  for (int i_adj_pairs = 0; i_adj_pairs < NUM_ADJACENT_PAIRS_; ++i_adj_pairs) {
     DERIV_ID_INNER_(i_adj_pairs) = deriv_ids[AERO_SPEC_INNER_(i_adj_pairs)];
     DERIV_ID_OUTER_(i_adj_pairs) = deriv_ids[AERO_SPEC_OUTER_(i_adj_pairs)];
  }
}

/** \brief Update reaction data for new environmental conditions
 *
 *
 * \param model_data Pointer to the model data
 * \param rxn_int_data Pointer to the reaction integer data
 * \param rxn_float_data Pointer to the reaction floating-point data
 * \param rxn_env_data Pointer to the environment-dependent parameters
 */
void rxn_condensed_phase_diffusion_update_env_state(ModelData *model_data,
                                                int *rxn_int_data,
                                                double *rxn_float_data,
                                                double *rxn_env_data) {
  int *int_data = rxn_int_data;
  double *float_data = rxn_float_data;
  double *env_data = model_data->grid_cell_env;

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
void rxn_condensed_phase_diffusion_calc_deriv_contrib(
    ModelData *model_data, TimeDerivative time_deriv, int *rxn_int_data,
    double *rxn_float_data, double *rxn_env_data, realtype time_step) {
  int *int_data = rxn_int_data;
  double *float_data = rxn_float_data;
  double *state = model_data->grid_cell_state;
  double *env_data = model_data->grid_cell_env;

  // Calculate derivative contributions for each aerosol phase
  for (int i_adj_pairs = 0, i_deriv = 0; i_adj_pairs < NUM_ADJACENT_PAIRS_; i_adj_pairs++) {

    /* Get the layer thickness for inner phase id (m) */
    realtype layer_thickness_inner;
    aero_rep_get_layer_thickness__m(
      model_data, //model data 
      AERO_REP_ID_(i_adj_pairs), // aerosol representation index
      PHASE_ID_INNER_(i_adj_pairs), // inner phase id
      &layer_thickness_inner, // layer thickness 
      NULL); // partial derivative

    // Get the layer thickness for outer phase id (m)
    realtype layer_thickness_outer;
    aero_rep_get_layer_thickness__m(
      model_data, //model data 
      AERO_REP_ID_(i_adj_pairs), // aerosol representation index
      PHASE_ID_OUTER_(i_adj_pairs), // outer phase id
      &layer_thickness_outer, // layer thickness 
      NULL); // partial derivative

    // Get the interface surface area (m2)
    realtype eff_sa;
    aero_rep_get_interface_surface_area__m2(
        model_data, //model data 
        AERO_REP_ID_(i_adj_pairs), // aerosol representation index
        PHASE_ID_INNER_(i_adj_pairs), // inner phase id
        PHASE_ID_OUTER_(i_adj_pairs), // outer phase id
        &eff_sa, // interface surface area 
        NULL); // partial derivative

    // Get the volume of the inner phase
    realtype volume_phase_inner;
    aero_rep_get_phase_volume__m3_m3(
        model_data, //model data
        AERO_REP_ID_(i_adj_pairs), // aerosol representation index
        PHASE_ID_INNER_(i_adj_pairs), // inner phase id
        &volume_phase_inner, // volume of inner phase
        NULL); // partial derivative

    // Get the volume of the outer phase
    realtype volume_phase_outer;
    aero_rep_get_phase_volume__m3_m3(
        model_data, //model data
        AERO_REP_ID_(i_adj_pairs), // aerosol representation index
        PHASE_ID_OUTER_(i_adj_pairs), // outer phase id
        &volume_phase_outer, // volume of outer phase
        NULL); // partial derivative

    // Calculate the rate constant for diffusion limited mass transfer between
    // particle layers
    double rate_inner = (double)(eff_sa / volume_phase_inner);
    double rate_outer = (double)(eff_sa / volume_phase_outer);

    rate_inner *= ((-DIFF_COEFF_INNER_(i_adj_pairs) / layer_thickness_inner) 
                    * state[AERO_SPEC_INNER_(i_adj_pairs)] +
                    (DIFF_COEFF_OUTER_(i_adj_pairs) / layer_thickness_outer) 
                    * state[AERO_SPEC_OUTER_(i_adj_pairs)]); 
    rate_outer *= ((DIFF_COEFF_INNER_(i_adj_pairs) / layer_thickness_inner)
                    * state[AERO_SPEC_INNER_(i_adj_pairs)] -
                    (DIFF_COEFF_OUTER_(i_adj_pairs) / layer_thickness_outer)
                    * state[AERO_SPEC_OUTER_(i_adj_pairs)]);
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
void rxn_condensed_phase_diffusion_calc_jac_contrib(ModelData *model_data,
                                                Jacobian jac, int *rxn_int_data,
                                                double *rxn_float_data,
                                                double *rxn_env_data,
                                                realtype time_step) {
  //int *int_data = rxn_int_data;
  //double *float_data = rxn_float_data;
  //double *state = model_data->grid_cell_state;
  //double *env_data = model_data->grid_cell_env;
}
#endif

/** \brief Print the Phase Transfer reaction parameters
 *
 * \param rxn_int_data Pointer to the reaction integer data
 * \param rxn_float_data Pointer to the reaction floating-point data
 */
void rxn_condensed_phase_diffusion_print(int *rxn_int_data,
                                     double *rxn_float_data) {
  int *int_data = rxn_int_data;
  double *float_data = rxn_float_data;

  printf("\n\nCondensed Phase Diffusion reaction\n");
  for (int i = 0; i < NUM_ADJACENT_PAIRS_; ++i) {
    printf("\n  Diffusion coefficient inner: %g", DIFF_COEFF_INNER_(i));
    printf("\n  Diffusion coefficient outer: %g", DIFF_COEFF_OUTER_(i));
    printf("\n  Aerosol phase id inner: %d", PHASE_ID_INNER_(i));
    printf("\n  Aerosol phase id outer: %d", PHASE_ID_OUTER_(i));
    printf("\n  Aerosol species id inner: %d", AERO_SPEC_INNER_(i));
    printf("\n  Aerosol species id outer: %d", AERO_SPEC_OUTER_(i));
    printf("\n  Aerosol representation id: %d", AERO_REP_ID_(i));
}

  return;
}

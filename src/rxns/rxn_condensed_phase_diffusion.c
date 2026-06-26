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

#define DIFF_COEFF_FIRST_(x) (float_data[(NUM_FLOAT_PROP_) + (x)])
#define DIFF_COEFF_SECOND_(x) (float_data[(NUM_FLOAT_PROP_) + (NUM_ADJACENT_PAIRS_) + (x)])
#define PHASE_ID_FIRST_(x) (int_data[(NUM_INT_PROP_) + (x)]-1)
#define PHASE_ID_SECOND_(x) (int_data[(NUM_INT_PROP_) + (NUM_ADJACENT_PAIRS_) + (x)]-1)
#define AERO_SPEC_FIRST_(x) (int_data[(NUM_INT_PROP_) + (2*NUM_ADJACENT_PAIRS_) + (x)]-1)
#define AERO_SPEC_SECOND_(x) (int_data[(NUM_INT_PROP_) + (3*NUM_ADJACENT_PAIRS_) + (x)]-1)
#define AERO_REP_ID_(x) (int_data[(NUM_INT_PROP_) + (4*NUM_ADJACENT_PAIRS_) + (x)]-1)

#define DERIV_ID_FIRST_(x) (int_data[(NUM_INT_PROP_) + (5*NUM_ADJACENT_PAIRS_) + (x)])
#define DERIV_ID_SECOND_(x) (int_data[(NUM_INT_PROP_) + (6*NUM_ADJACENT_PAIRS_) + (x)])
//#define JAC_ID_(x) (int_data[4*BLOCK_SIZE_ + x]-1)
//#define PHASE_INT_LOC_(x) (int_data[5*BLOCK_SIZE_ + x]-1)
//#define PHASE_FLOAT_LOC_(x) (int_data[9*BLOCK_SIZE_ + x]-1)
//#define NUM_AERO_PHASE_JAC_ELEM_FIRST_(x) (int_data[10*BLOCK_SIZE + x]-1)
//#define NUM_AERO_PHASE_JAC_ELEM_SECOND_(x) (int_data[11*BLOCK_SIZE + x]-1)
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
}

/** \brief Update reaction data for new environmental conditions
 *
 * For Phase Transfer reaction this only involves recalculating the rate
 * constant.
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

    /* Get the layer thickness for first phase id (m) */
    realtype layer_thickness_first;
    aero_rep_get_layer_thickness__m(
      model_data, //model data 
      AERO_REP_ID_(i_adj_pairs), // aerosol representation index
      PHASE_ID_FIRST_(i_adj_pairs), // first phase id
      &layer_thickness_first, // layer thickness 
      NULL); // partial derivative

    // Get the layer thickness for second phase id (m)
    realtype layer_thickness_second;
    aero_rep_get_layer_thickness__m(
      model_data, //model data 
      AERO_REP_ID_(i_adj_pairs), // aerosol representation index
      PHASE_ID_SECOND_(i_adj_pairs), // second phase id
      &layer_thickness_second, // layer thickness 
      NULL); // partial derivative

    // Get the interface surface area (m2)
    realtype eff_sa;
    aero_rep_get_interface_surface_area__m2(
        model_data, //model data 
        AERO_REP_ID_(i_adj_pairs), // aerosol representation index
        PHASE_ID_FIRST_(i_adj_pairs), // first phase id
        PHASE_ID_SECOND_(i_adj_pairs), // second phase id
        &eff_sa, // interface surface area 
        NULL); // partial derivative

    // Get the volume of the first phase
    realtype volume_phase_first;
    aero_rep_get_phase_volume__m3_m3(
        model_data, //model data
        AERO_REP_ID_(i_adj_pairs), // aerosol representation index
        PHASE_ID_FIRST_(i_adj_pairs), // first phase id
        &volume_phase_first, // volume of first phase
        NULL); // partial derivative

    // Get the volume of the second phase
    realtype volume_phase_second;
    aero_rep_get_phase_volume__m3_m3(
        model_data, //model data
        AERO_REP_ID_(i_adj_pairs), // aerosol representation index
        PHASE_ID_SECOND_(i_adj_pairs), // second phase id
        &volume_phase_second, // volume of second phase
        NULL); // partial derivative

    // Calculate the rate constant for diffusion limited mass transfer between
    // particle layers
    double rate_first = (double)(eff_sa / volume_phase_first);
    double rate_second = (double)(eff_sa / volume_phase_second);

    rate_first *= ((-DIFF_COEFF_FIRST_(i_adj_pairs) / layer_thickness_first) 
                    * state[AERO_SPEC_FIRST_(i_adj_pairs)] +
                    (DIFF_COEFF_SECOND_(i_adj_pairs) / layer_thickness_second) 
                    * state[AERO_SPEC_SECOND_(i_adj_pairs)]); 
    rate_second *= ((DIFF_COEFF_FIRST_(i_adj_pairs) / layer_thickness_first)
                    * state[AERO_SPEC_FIRST_(i_adj_pairs)] -
                    (DIFF_COEFF_SECOND_(i_adj_pairs) / layer_thickness_second)
                    * state[AERO_SPEC_SECOND_(i_adj_pairs)]);
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
    printf("\n  Diffusion coefficient first: %g", DIFF_COEFF_FIRST_(i));
    printf("\n  Diffusion coefficient second: %g", DIFF_COEFF_SECOND_(i));
    printf("\n  Aerosol phase id first: %d", PHASE_ID_FIRST_(i));
    printf("\n  Aerosol phase id second: %d", PHASE_ID_SECOND_(i));
    printf("\n  Aerosol species id first: %d", AERO_SPEC_FIRST_(i));
    printf("\n  Aerosol species id second: %d", AERO_SPEC_SECOND_(i));
    printf("\n  Aerosol representation id: %d", AERO_REP_ID_(i));
}

  return;
}

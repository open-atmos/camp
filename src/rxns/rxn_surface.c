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
#define GAMMA_ float_data[1]
#define MW_ float_data[2]
#define NUM_AERO_PHASE_ int_data[0]
#define REACT_ID_ (int_data[1] - 1)
#define NUM_PROD_ int_data[2]
#define MEAN_SPEED_MS_ rxn_env_data[0]
#define NUM_INT_PROP_ 3
#define NUM_FLOAT_PROP_ 3
#define NUM_ENV_PARAM_ 1
#define PROD_ID_(x) (int_data[NUM_INT_PROP_ + x] - 1)
#define DERIV_ID_(x) int_data[NUM_INT_PROP_ + NUM_PROD_ + x]
#define JAC_ID_(x) int_data[NUM_INT_PROP_ + 1 + 2 * NUM_PROD_ + x]
#define PHASE_INT_LOC_(x) \
  (int_data[NUM_INT_PROP_ + 2 + 3 * NUM_PROD_ + x] - 1)
#define PHASE_FLOAT_LOC_(x) \
  (int_data[NUM_INT_PROP_ + 2 + 3 * NUM_PROD_ + NUM_AERO_PHASE_ + x] - 1)
#define AERO_PHASE_ID_(x) (int_data[PHASE_INT_LOC_(x)] - 1)
#define AERO_REP_ID_(x) (int_data[PHASE_INT_LOC_(x) + 1] - 1)
#define NUM_AERO_PHASE_JAC_ELEM_(x) (int_data[PHASE_INT_LOC_(x) + 2])
#define PHASE_JAC_ID_(x, s, e) \
  int_data[PHASE_INT_LOC_(x) + 3 + (s) * NUM_AERO_PHASE_JAC_ELEM_(x) + e]
#define YIELD_(x) float_data[NUM_FLOAT_PROP_ + x]
#define EFF_RAD_JAC_ELEM_(x, e) float_data[PHASE_FLOAT_LOC_(x) + e]
#define NUM_CONC_JAC_ELEM_(x, e) \
  float_data[PHASE_FLOAT_LOC_(x) + NUM_AERO_PHASE_JAC_ELEM_(x) + e]

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

  jacobian_register_element(jac, REACT_ID_, REACT_ID_);
  for (int i_prod = 0; i_prod < NUM_PROD_; ++i_prod) {
    jacobian_register_element(jac, PROD_ID_(i_prod), REACT_ID_);
  }

  for (int i_aero_phase = 0; i_aero_phase < NUM_AERO_PHASE_; ++i_aero_phase) {
    for (int i_elem = 0; i_elem < model_data->n_per_cell_state_var; ++i_elem) {
      aero_jac_elem[i_elem] = false;
    }
    int n_jac_elem =
        aero_rep_get_used_jac_elem(model_data, AERO_REP_ID_(i_aero_phase),
                                   AERO_PHASE_ID_(i_aero_phase), aero_jac_elem);
    if (n_jac_elem > NUM_AERO_PHASE_JAC_ELEM_(i_aero_phase)) {
      printf(
          "\n\nERROR Received more Jacobian elements than expected for surface "
          "reaction. Got %d, expected <= %d",
          n_jac_elem, NUM_AERO_PHASE_JAC_ELEM_(i_aero_phase));
      exit(1);
    }
    int i_used_elem = 0;
    for (int i_elem = 0; i_elem < model_data->n_per_cell_state_var; ++i_elem) {
      if (aero_jac_elem[i_elem] == true) {
        jacobian_register_element(jac, REACT_ID_, i_elem);
        PHASE_JAC_ID_(i_aero_phase, 0, i_used_elem) = i_elem;
        for (int i_prod = 0; i_prod < NUM_PROD_; ++i_prod) {
          jacobian_register_element(jac, PROD_ID_(i_prod), i_elem);
          PHASE_JAC_ID_(i_aero_phase, i_prod + 1, i_used_elem) = i_elem;
        }
        ++i_used_elem;
      }
    }
    for (; i_used_elem < NUM_AERO_PHASE_JAC_ELEM_(i_aero_phase);
         ++i_used_elem) {
      for (int i_spec = 0; i_spec < NUM_PROD_ + 1; ++i_spec) {
        PHASE_JAC_ID_(i_aero_phase, i_spec, i_used_elem) = -1;
      }
    }
    if (i_used_elem != n_jac_elem) {
      printf(
          "\n\nERROR Error setting used Jacobian elements in surface "
          "reaction %d %d\n\n",
          i_used_elem, n_jac_elem);
      rxn_surface_print(rxn_int_data, rxn_float_data);
      exit(1);
    }
  }

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

  // Update the time derivative ids
  DERIV_ID_(0) = deriv_ids[REACT_ID_];
  for (int i_prod = 0; i_prod < NUM_PROD_; ++i_prod) {
    DERIV_ID_(i_prod + 1) = deriv_ids[PROD_ID_(i_prod)];
  }

  // Update the Jacobian element ids
  int i_jac = 0;
  JAC_ID_(i_jac++) = jacobian_get_element_id(jac, REACT_ID_, REACT_ID_);
  for (int i_prod = 0; i_prod < NUM_PROD_; ++i_prod) {
    JAC_ID_(i_jac++) = jacobian_get_element_id(jac, PROD_ID_(i_prod), REACT_ID_);
  }
  for (int i_aero_phase = 0; i_aero_phase < NUM_AERO_PHASE_; ++i_aero_phase) {
    for (int i_elem = 0; i_elem < NUM_AERO_PHASE_JAC_ELEM_(i_aero_phase); ++i_elem) {
      if (PHASE_JAC_ID_(i_aero_phase, 0, i_elem) > 0) {
        PHASE_JAC_ID_(i_aero_phase, 0, i_elem) = jacobian_get_element_id(
            jac, REACT_ID_, PHASE_JAC_ID_(i_aero_phase, 0, i_elem));
      }
      for (int i_prod = 0; i_prod < NUM_PROD_; ++i_prod) {
        if (PHASE_JAC_ID_(i_aero_phase, i_prod + 1, i_elem) > 0) {
          PHASE_JAC_ID_(i_aero_phase, i_prod + 1, i_elem) =
              jacobian_get_element_id(jac, PROD_ID_(i_prod),
                                      PHASE_JAC_ID_(i_aero_phase, i_prod + 1, i_elem));
        }
      }
    }
  }
  return;
}

/** \brief Update reaction data for new environmental conditions
 *
 * For surface reactions this only involves calculating the mean
 * speed of the reacting species
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

  // save the mean speed [m s-1] for calculating condensation rates
  MEAN_SPEED_MS_ = mean_speed__m_s(TEMPERATURE_K_, MW_);

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

    // Get the particle number concentration (#/m3)
    realtype number_conc;
    aero_rep_get_number_conc__n_m3(
        model_data,               // model data
        AERO_REP_ID_(i_phase),    // aerosol representation index
        AERO_PHASE_ID_(i_phase),  // aerosol phase index
        &number_conc,             // particle number concentration
                                  // (#/m3)
        NULL);                    // partial derivative

    // Calculate the rate constant for diffusion limited mass transfer to the
    // aerosol phase (1/s)
    double cond_rate = state[REACT_ID_] * number_conc *
        gas_aerosol_continuum_rxn_rate_constant(DIFF_COEFF_, MEAN_SPEED_MS_,
                                                radius, GAMMA_);

    // Loss of the reactant
    if (DERIV_ID_(0) >= 0) {
      time_derivative_add_value(time_deriv, DERIV_ID_(0), -cond_rate);
    }
    // Gain of each product
    for (int i_prod = 0; i_prod < NUM_PROD_; ++i_prod) {
      if (DERIV_ID_(i_prod + 1) >= 0) {
        time_derivative_add_value(time_deriv, DERIV_ID_(i_prod + 1),
                                  YIELD_(i_prod) * cond_rate);
      }
    }
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

    // Get the particle number concentration (#/m3)
    realtype number_conc;
    aero_rep_get_number_conc__n_m3(
        model_data,                          // model data
        AERO_REP_ID_(i_phase),               // aerosol representation index
        AERO_PHASE_ID_(i_phase),             // aerosol phase index
        &number_conc,                        // particle number concentration
                                             // (#/m3)
        &(NUM_CONC_JAC_ELEM_(i_phase, 0)));  // partial derivative

    // Calculate the rate constant for diffusion limited mass transfer to the
    // aerosol phase (1/s)
    double rate_const =
        gas_aerosol_continuum_rxn_rate_constant(DIFF_COEFF_, MEAN_SPEED_MS_,
                                                radius, GAMMA_);
    double cond_rate = state[REACT_ID_] * number_conc * rate_const;

    // Dependence on the reactant
    if (JAC_ID_(0) >=0) {
      jacobian_add_value(jac, (unsigned int)JAC_ID_(0), JACOBIAN_LOSS,
                         number_conc * rate_const);
    }
    for (int i_prod = 0; i_prod < NUM_PROD_; ++i_prod) {
      if (JAC_ID_(i_prod + 1) >= 0) {
        jacobian_add_value(jac, (unsigned int)JAC_ID_(i_prod + 1), JACOBIAN_PRODUCTION,
                           YIELD_(i_prod) * number_conc * rate_const);
      }
    }

    // Calculate d_rate/d_effective_radius and d_rate/d_number_concentration
    double d_rate_d_radius = state[REACT_ID_] * number_conc *
        d_gas_aerosol_continuum_rxn_rate_constant_d_radius(DIFF_COEFF_, MEAN_SPEED_MS_,
                                                           radius, GAMMA_);
    double d_rate_d_number = state[REACT_ID_] * rate_const;

    // Loop through aerosol dependencies
    for (int i_elem = 0; i_elem < NUM_AERO_PHASE_JAC_ELEM_(i_phase); ++i_elem) {
      // Reactant dependencies
      if (PHASE_JAC_ID_(i_phase, 0, i_elem) > 0) {
        // Dependence on effective radius
        jacobian_add_value(
            jac, (unsigned int)PHASE_JAC_ID_(i_phase, 0, i_elem),
            JACOBIAN_LOSS,
            d_rate_d_radius * EFF_RAD_JAC_ELEM_(i_phase, i_elem));
        // Dependence on number concentration
        jacobian_add_value(
            jac, (unsigned int)PHASE_JAC_ID_(i_phase, 0, i_elem),
            JACOBIAN_LOSS,
            d_rate_d_number * NUM_CONC_JAC_ELEM_(i_phase, i_elem));
      }
      // Product dependencies
      for (int i_prod = 0; i_prod < NUM_PROD_; ++i_prod) {
        if (PHASE_JAC_ID_(i_phase, i_prod + 1, i_elem) > 0) {
          // Dependence on effective radius
          jacobian_add_value(
              jac, (unsigned int)PHASE_JAC_ID_(i_phase, i_prod + 1, i_elem),
              JACOBIAN_PRODUCTION,
              YIELD_(i_prod) * d_rate_d_radius *
                  EFF_RAD_JAC_ELEM_(i_phase, i_elem));
          // Dependence on number concentration
          jacobian_add_value(
              jac, (unsigned int)PHASE_JAC_ID_(i_phase, i_prod + 1, i_elem),
              JACOBIAN_PRODUCTION,
              YIELD_(i_prod) * d_rate_d_number *
                  NUM_CONC_JAC_ELEM_(i_phase, i_elem));
        }
      }
    }
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

  printf("\n\nSurface reaction\n");
  printf("\ndiffusion coefficient: %lg gamma: %lg, MW: %lg", DIFF_COEFF_,
         GAMMA_, MW_);
  printf("\nnumber of aerosol phases: %d", NUM_AERO_PHASE_);
  printf("\nreactant state id: %d", REACT_ID_);
  printf("\nnumber of products: %d", NUM_PROD_);
  for (int i_prod = 0; i_prod < NUM_PROD_; ++i_prod) {
    printf("\n  product %d id: %d", i_prod, PROD_ID_(i_prod));
  }
  printf("\nreactant derivative id: %d", DERIV_ID_(0));
  printf("\nd_reactant/d_reactant Jacobian id %d", JAC_ID_(0));
  for (int i_prod = 0; i_prod < NUM_PROD_; ++i_prod) {
    printf("\n  product %d derivative id: %d", i_prod, DERIV_ID_(i_prod+1));
    printf("\n  d_product_%d/d_reactant Jacobian id: %d", i_prod,
           JAC_ID_(i_prod+1));
  }
  for (int i_phase = 0; i_phase < NUM_AERO_PHASE_; ++i_phase) {
    printf("\nPhase %d start indices int: %d float: %d", i_phase,
           PHASE_INT_LOC_(i_phase), PHASE_FLOAT_LOC_(i_phase));
    printf("\n  phase id %d; aerosol representation id %d",
           AERO_PHASE_ID_(i_phase), AERO_REP_ID_(i_phase));
    printf("\n  number of Jacobian elements: %d",
           NUM_AERO_PHASE_JAC_ELEM_(i_phase));
    for (int i_elem = 0; i_elem < NUM_AERO_PHASE_JAC_ELEM_(i_phase); ++i_elem) {
      printf("\n  - d_reactant/d_phase_species_%d Jacobian id %d",
             i_elem, PHASE_JAC_ID_(i_phase,0,i_elem));
      for (int i_prod = 0; i_prod < NUM_PROD_; ++i_prod) {
        printf("\n  - d_product_%d/d_phase_species_%d Jacobian id %d",
               i_prod, i_elem, PHASE_JAC_ID_(i_phase,i_prod+1,i_elem));
      }
      printf("\n  - d_radius/d_phase_species_%d = %le",
             i_elem, EFF_RAD_JAC_ELEM_(i_phase,i_elem));
      printf("\n  - d_number/d_phase_species_%d = %le",
             i_elem, EFF_RAD_JAC_ELEM_(i_phase,i_elem));
    }
  }
  printf("\n *** end surface reaction ***\n\n");
  return;
}

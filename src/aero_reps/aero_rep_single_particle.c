/* Copyright (C) 2021 Barcelona Supercomputing Center and University of
 * Illinois at Urbana-Champaign
 * SPDX-License-Identifier: MIT
 *
 * Single particle aerosol representation functions
 *
 */
/** \file
 * \brief Single particle aerosol representation functions
 */
#include <stdio.h>
#include <stdlib.h>
#include "../aero_phase_solver.h"
#include "../aero_reps.h"
#include "../camp_solver.h"

// TODO Lookup environmental indicies during initialization
#define TEMPERATURE_K_ env_data[0]
#define PRESSURE_PA_ env_data[1]

#define UPDATE_NUMBER 0

#define NUM_LAYERS_ int_data[0]
#define AERO_REP_ID_ int_data[1]
#define MAX_PARTICLES_ int_data[2]
#define PARTICLE_STATE_SIZE_ int_data[3]
#define NUMBER_CONC_(x) aero_rep_env_data[x]
#define NUM_INT_PROP_ 4
#define NUM_FLOAT_PROP_ 0
#define NUM_ENV_PARAM_ MAX_PARTICLES_
#define LAYER_PHASE_START_(l) (int_data[NUM_INT_PROP_+l]-1)
#define LAYER_PHASE_END_(l) (int_data[NUM_INT_PROP_+NUM_LAYERS_+l]-1)
#define TOTAL_NUM_PHASES_ (LAYER_PHASE_END_(NUM_LAYERS_-1)-LAYER_PHASE_START_(0)+1)
#define NUM_PHASES_(l) (LAYER_PHASE_END_(l)-LAYER_PHASE_START_(l)+1)
#define PHASE_STATE_ID_(l,p) (int_data[NUM_INT_PROP_+2*NUM_LAYERS_+LAYER_PHASE_START_(l)+p]-1)
#define PHASE_MODEL_DATA_ID_(l,p) (int_data[NUM_INT_PROP_+2*NUM_LAYERS_+TOTAL_NUM_PHASES_+LAYER_PHASE_START_(l)+p]-1)
#define PHASE_NUM_JAC_ELEM_(l,p) int_data[NUM_INT_PROP_+2*NUM_LAYERS_+2*TOTAL_NUM_PHASES_+LAYER_PHASE_START_(l)+p]

/** \brief Flag Jacobian elements used in calcualtions of mass and volume
 *
 * \param aero_rep_int_data Pointer to the aerosol representation integer data
 * \param aero_rep_float_data Pointer to the aerosol representation
 *                            floating-point data
 * \param model_data Pointer to the model data
 * \param aero_phase_idx Index of the aerosol phase to find elements for
 * \param jac_struct 1D array of flags indicating potentially non-zero
 *                   Jacobian elements. (The dependent variable should have
 *                   been chosen by the calling function.)
 * \return Number of Jacobian elements flagged
 */

int aero_rep_single_particle_get_used_jac_elem(ModelData *model_data,
                                               int aero_phase_idx,
                                               int *aero_rep_int_data,
                                               double *aero_rep_float_data,
                                               bool *jac_struct) {
  int *int_data = aero_rep_int_data;
  double *float_data = aero_rep_float_data;
  int n_jac_elem = 0;
  int i_part = aero_phase_idx / TOTAL_NUM_PHASES_;

  // Each phase in a single particle has the same jac elements
  // (one for each species in each phase in the particle)
  for (int i_layer = 0; i_layer < NUM_LAYERS_; ++i_layer) {
    for (int i_phase = 0; i_phase < NUM_PHASES_(i_layer); ++i_phase) {
      PHASE_NUM_JAC_ELEM_(i_layer,i_phase) = aero_phase_get_used_jac_elem(
          model_data, PHASE_MODEL_DATA_ID_(i_layer,i_phase),
          i_part * PARTICLE_STATE_SIZE_ + PHASE_STATE_ID_(i_layer,i_phase), jac_struct);
      n_jac_elem += PHASE_NUM_JAC_ELEM_(i_layer,i_phase);
     }
  }
  return n_jac_elem;
}

/** \brief Flag elements on the state array used by this aerosol representation
 *
 * The single particle aerosol representation functions do not use state array
 * values
 *
 * \param aero_rep_int_data Pointer to the aerosol representation integer data
 * \param aero_rep_float_data Pointer to the aerosol representation
 *                            floating-point data
 * \param state_flags Array of flags indicating state array elements used
 */

void aero_rep_single_particle_get_dependencies(int *aero_rep_int_data,
                                               double *aero_rep_float_data,
                                               bool *state_flags) {
  int *int_data = aero_rep_int_data;
  double *float_data = aero_rep_float_data;
  
  return;
}

/** \brief Update aerosol representation data for new environmental conditions
 *
 * The single particle aerosol representation does not use environmental
 * conditions
 *
 * \param model_data Pointer to the model data
 * \param aero_rep_int_data Pointer to the aerosol representation integer data
 * \param aero_rep_float_data Pointer to the aerosol representation
 *                            floating-point data
 * \param aero_rep_env_data Pointer to the aerosol representation
 *                          environment-dependent parameters
 */

void aero_rep_single_particle_update_env_state(ModelData *model_data,
                                               int *aero_rep_int_data,
                                               double *aero_rep_float_data,
                                               double *aero_rep_env_data) {
  int *int_data = aero_rep_int_data;
  double *float_data = aero_rep_float_data;
  double *env_data = model_data->grid_cell_env;

  return;
}

/** \brief Update aerosol representation data for a new state
 *
 * \param model_data Pointer to the model data, include the state array
 * \param aero_rep_int_data Pointer to the aerosol representation integer data
 * \param aero_rep_float_data Pointer to the aerosol representation
 *                            floating-point data
 * \param aero_rep_env_data Pointer to the aerosol representation
 *                          environment-dependent parameters
 */

void aero_rep_single_particle_update_state(ModelData *model_data,
                                           int *aero_rep_int_data,
                                           double *aero_rep_float_data,
                                           double *aero_rep_env_data) {
  int *int_data = aero_rep_int_data;
  double *float_data = aero_rep_float_data;

  return;
}

/** \brief Get the effective particle radius \f$r_{eff}\f$ (m)
 *
 * \param model_data Pointer to the model data, including the state array
 * \param aero_phase_idx Index of the aerosol phase within the representation
 * \param radius Effective particle radius (m)
 * \param partial_deriv \f$\frac{\partial r_{eff}}{\partial y}\f$ where \f$y\f$
 *                      are species on the state array
 * \param aero_rep_int_data Pointer to the aerosol representation integer data
 * \param aero_rep_float_data Pointer to the aerosol representation
 *                            floating-point data
 * \param aero_rep_env_data Pointer to the aerosol representation
 *                          environment-dependent parameters
 */

void aero_rep_single_particle_get_effective_radius__m(
    ModelData *model_data, int aero_phase_idx, double *radius,
    double *partial_deriv, int *aero_rep_int_data, double *aero_rep_float_data,
    double *aero_rep_env_data) {

  int *int_data = aero_rep_int_data;
  double *float_data = aero_rep_float_data;
  int i_part = aero_phase_idx / TOTAL_NUM_PHASES_;
  double *curr_partial = NULL;

  *radius = 0.0;
  if (partial_deriv) curr_partial = partial_deriv;
  for (int i_layer = 0; i_layer < NUM_LAYERS_; ++i_layer) {
    for (int i_phase = 0; i_phase < NUM_PHASES_(i_layer); ++i_phase) {
      double *state = (double *)(model_data->grid_cell_state);
      state += i_part * PARTICLE_STATE_SIZE_ + PHASE_STATE_ID_(i_layer,i_phase);
      double volume;
      aero_phase_get_volume__m3_m3(model_data, PHASE_MODEL_DATA_ID_(i_layer,i_phase),
                                   state, &(volume), curr_partial);
      if (partial_deriv) curr_partial += PHASE_NUM_JAC_ELEM_(i_layer,i_phase);
      *radius += volume;
    }
  }
  *radius = pow(((*radius) * 3.0 / 4.0 / 3.14159265359), 1.0 / 3.0);
  if (!partial_deriv) return;
  for (int i_layer = 0; i_layer < NUM_LAYERS_; ++i_layer) {
    for (int i_phase = 0; i_phase < NUM_PHASES_(i_layer); ++i_phase) {
      for (int i_spec = 0; i_spec < PHASE_NUM_JAC_ELEM_(i_layer,i_phase); ++i_spec) {
        *partial_deriv =
            1.0 / 4.0 / 3.14159265359 * pow(*radius, -2.0) * (*partial_deriv);
        ++partial_deriv;
      }
    }
  }
  return;
}

/** \brief Get the surface area of specified particle layer \f$r_{eff}\f$ (m)
 *
 * Solve for the surface area of the interfacial layer that exists between the 
 * two phases considered in aerosol phase mass tranfer between layers. When more
 * than one phase exists in a layer, a "fractional volume overlap" configuration 
 * is applied (see CAMP Github Documentation for details).
 * 
 * \param model_data Pointer to the model data, including the state array
 * \param aero_phase_idx_first Index of the first aerosol phase within the representation
 * \param aero_phase_idx_second Index of the second aerosol phase within the representation
 * \param surface_area Pointer to surface area of inner layer (m^2)
 * \param partial_deriv \f$\frac{\partial r_{eff}}{\partial y}\f$ where \f$y\f$
 *                      are species on the state array
 * \param aero_rep_int_data Pointer to the aerosol representation integer data
 * \param aero_rep_float_data Pointer to the aerosol representation
 *                            floating-point data
 * \param aero_rep_env_data Pointer to the aerosol representation
 *                          environment-dependent parameters
 * \param surf_area_layer Surface area of specified layer (m2)
 */

void aero_rep_single_particle_get_interface_surface_area__m2(
    ModelData *model_data, int aero_phase_idx_first, int aero_phase_idx_second, 
    double *surface_area, double *partial_deriv, 
    int *aero_rep_int_data, double *aero_rep_float_data, double *aero_rep_env_data) {

  int *int_data = aero_rep_int_data;
  double *float_data = aero_rep_float_data;
  double *curr_partial = NULL;
  int layer_first = -1;
  int layer_second = -1;
  int layer_interface = -1;
  int phase_model_data_id_first = -1;
  int phase_model_data_id_second = -1;
  double radius;
  
  int i_phase_count = 0;
  for (int i_layer = 0; i_layer < NUM_LAYERS_; ++i_layer) {
    for (int i_phase = 0; i_phase < NUM_PHASES_(i_layer); ++i_phase) {
      if (LAYER_PHASE_START_(i_layer) <= aero_phase_idx_first && 
          aero_phase_idx_first <= LAYER_PHASE_END_(i_layer) && 
          i_phase_count == aero_phase_idx_first) {
        layer_first = i_layer;
        phase_model_data_id_first = PHASE_MODEL_DATA_ID_(i_layer, i_phase);
      } else if (LAYER_PHASE_START_(i_layer) <= aero_phase_idx_second &&
                 aero_phase_idx_second <= LAYER_PHASE_END_(i_layer) &&
                 i_phase_count == aero_phase_idx_second) {
        layer_second = i_layer;
        phase_model_data_id_second = PHASE_MODEL_DATA_ID_(i_layer, i_phase);
      }
      ++i_phase_count;
    }
  }
  layer_interface = layer_first > layer_second ? layer_second : layer_first;

  double total_volume_layer_first = 0.0;
  double total_volume_layer_second = 0.0;
  double volume_phase_first = 0.0;
  double volume_phase_second = 0.0;
  i_phase_count = 0;
  for (int i_layer = 0; i_layer <= layer_second; ++i_layer) {
    for (int i_phase = 0; i_phase < NUM_PHASES_(i_layer); ++i_phase) {
      double *state = (double *)(model_data->grid_cell_state);
      state += PARTICLE_STATE_SIZE_ + PHASE_STATE_ID_(i_layer,i_phase);
      double volume;
      aero_phase_get_volume__m3_m3(model_data, PHASE_MODEL_DATA_ID_(i_layer,i_phase),
                                   state, &(volume), curr_partial);
      if (i_layer == layer_first) total_volume_layer_first += volume;
      if (i_phase_count == aero_phase_idx_first && 
          PHASE_MODEL_DATA_ID_(i_layer, i_phase) == 
          phase_model_data_id_first) volume_phase_first = volume;
      if (i_layer == layer_second) total_volume_layer_second += volume;
      if (i_phase_count == aero_phase_idx_second && 
          PHASE_MODEL_DATA_ID_(i_layer, i_phase) == 
          phase_model_data_id_second) volume_phase_second = volume;
      ++i_phase_count;
    }
  }
  double f_first = volume_phase_first / total_volume_layer_first;
  double f_second = volume_phase_second / total_volume_layer_second;
  double total_volume = 0.0;
  if (partial_deriv) curr_partial = partial_deriv;
  for (int i_layer = 0; i_layer <= layer_interface; ++i_layer) {
    for (int i_phase = 0; i_phase < NUM_PHASES_(i_layer); ++i_phase) {
      double *state = (double *)(model_data->grid_cell_state);
      state += PARTICLE_STATE_SIZE_ + PHASE_STATE_ID_(i_layer,i_phase);
      double volume;
      aero_phase_get_volume__m3_m3(model_data, PHASE_MODEL_DATA_ID_(i_layer,i_phase),
                                   state, &(volume), curr_partial);
      if (partial_deriv) curr_partial += PHASE_NUM_JAC_ELEM_(i_layer,i_phase);
      total_volume += volume;
    }
  }
  radius = pow(((total_volume) * 3.0 / 4.0 / 3.14159265359), 1.0 / 3.0);
  *surface_area = f_first * f_second * 4 * 3.14159265359 * pow(radius, 2.0);
  if (!partial_deriv) return;
  for (int i_layer = 0; i_layer < NUM_LAYERS_; ++i_layer) {
    for (int i_phase = 0; i_phase < NUM_PHASES_(i_layer); ++i_phase) {
      for (int i_spec = 0; i_spec < PHASE_NUM_JAC_ELEM_(i_layer,i_phase); ++i_spec) {
        if (i_layer <= layer_interface) {
          *partial_deriv =
              2.0 * f_first * f_second * pow(radius, -1.0)  * (*partial_deriv);
          ++partial_deriv;
        }
        else if (i_layer > layer_interface) *(partial_deriv++) = ZERO;
      }
    }
  }
  return;
}

/** \brief Get the particle number concentration \f$n\f$
 * (\f$\mbox{\si{\#\per\cubic\metre}}\f$)
 *
 * This single particle number concentration is set by the aerosol model prior
 * to solving the chemistry. Thus, all \f$\frac{\partial n}{\partial y}\f$ are
 * zero. Also, there is only one set of particles in the single particle
 * representation, so the phase index is not used.
 *
 * \param model_data Pointer to the model data, including the state array
 * \param aero_phase_idx Index of the aerosol phase within the representation
 *                       (not used)
 * \param number_conc Particle number concentration, \f$n\f$
 *                    (\f$\mbox{\si{\#\per\cubic\metre}}\f$)
 * \param partial_deriv \f$\frac{\partial n}{\partial y}\f$ where \f$y\f$ are
 *                      the species on the state array
 * \param aero_rep_int_data Pointer to the aerosol representation integer data
 * \param aero_rep_float_data Pointer to the aerosol representation
 *                            floating-point data
 * \param aero_rep_env_data Pointer to the aerosol representation
 *                          environment-dependent parameters
 */

void aero_rep_single_particle_get_number_conc__n_m3(
    ModelData *model_data, int aero_phase_idx, double *number_conc,
    double *partial_deriv, int *aero_rep_int_data, double *aero_rep_float_data,
    double *aero_rep_env_data) {


  int *int_data = aero_rep_int_data;
  double *float_data = aero_rep_float_data;
  int i_part = aero_phase_idx / TOTAL_NUM_PHASES_;

  *number_conc = NUMBER_CONC_(i_part);

  if (partial_deriv) {
    for (int i_layer = 0; i_layer < NUM_LAYERS_; ++i_layer) {
      for (int i_phase = 0; i_phase < NUM_PHASES_(i_layer); ++i_phase) {
        for (int i_spec = 0; i_spec < PHASE_NUM_JAC_ELEM_(i_layer,i_phase); ++i_spec)
          *(partial_deriv++) = ZERO;
      }
    }
  }
  return;
}

/** \brief Get the type of aerosol concentration used.
 *
 * Single particle concentrations are per-particle.
 *
 * \param aero_phase_idx Index of the aerosol phase within the representation
 * \param aero_conc_type Pointer to int that will hold the concentration type
 *                       code (0 = per particle mass concentrations;
 *                       1 = total particle mass concentrations)
 * \param aero_rep_int_data Pointer to the aerosol representation integer data
 * \param aero_rep_float_data Pointer to the aerosol representation
 *                            floating-point data
 * \param aero_rep_env_data Pointer to the aerosol representation
 *                          environment-dependent parameters
 */

void aero_rep_single_particle_get_aero_conc_type(int aero_phase_idx,
                                                 int *aero_conc_type,
                                                 int *aero_rep_int_data,
                                                 double *aero_rep_float_data,
                                                 double *aero_rep_env_data) {
  int *int_data = aero_rep_int_data;
  double *float_data = aero_rep_float_data;

  *aero_conc_type = 0;

  return;
}

/** \brief Get the total mass in an aerosol phase \f$m\f$
 * (\f$\mbox{\si{\kilogram\per\cubic\metre}}\f$)
 *
 * The single particle mass is set for each new state as the sum of the masses
 * of the aerosol phases that compose the particle
 *
 * \param model_data Pointer to the model data, including the state array
 * \param aero_phase_idx Index of the aerosol phase within the representation
 * \param aero_phase_mass Total mass in the aerosol phase, \f$m\f$
 *                        (\f$\mbox{\si{\kilogram\per\cubic\metre}}\f$)
 * \param partial_deriv \f$\frac{\partial m}{\partial y}\f$ where \f$y\f$ are
 *                      the species on the state array
 * \param aero_rep_int_data Pointer to the aerosol representation integer data
 * \param aero_rep_float_data Pointer to the aerosol representation
 *                            floating-point data
 * \param aero_rep_env_data Pointer to the aerosol representation
 *                          environment-dependent parameters
 */

void aero_rep_single_particle_get_aero_phase_mass__kg_m3(
    ModelData *model_data, int aero_phase_idx, double *aero_phase_mass,
    double *partial_deriv, int *aero_rep_int_data, double *aero_rep_float_data,
    double *aero_rep_env_data) {

  int *int_data = aero_rep_int_data;
  double *float_data = aero_rep_float_data;
  int i_part = aero_phase_idx / TOTAL_NUM_PHASES_;
  aero_phase_idx -= i_part * TOTAL_NUM_PHASES_;

  int i_total_phase = 0; 
  for (int i_layer = 0; i_layer < NUM_LAYERS_; ++i_layer) {
    for (int i_phase = 0; i_phase < NUM_PHASES_(i_layer); ++i_phase) {  
      if ( i_total_phase == aero_phase_idx ) {
        double *state = (double *)(model_data->grid_cell_state);
        state += i_part * PARTICLE_STATE_SIZE_ + PHASE_STATE_ID_(i_layer,i_phase);
        double mw;
        aero_phase_get_mass__kg_m3(model_data, PHASE_MODEL_DATA_ID_(i_layer,i_phase),
                                   state, aero_phase_mass, &mw, partial_deriv, NULL);
        if (partial_deriv) partial_deriv += PHASE_NUM_JAC_ELEM_(i_layer,i_phase);
      } else if (partial_deriv) {
        for (int i_spec = 0; i_spec < PHASE_NUM_JAC_ELEM_(i_layer,i_phase); ++i_spec)
          *(partial_deriv++) = ZERO;
      }
      ++i_total_phase;
    }
  }
  return;
}

/** \brief Get the average molecular weight in an aerosol phase
 **        \f$m\f$ (\f$\mbox{\si{\kilo\gram\per\mol}}\f$)
 *
 * The single particle mass is set for each new state as the sum of the masses
 * of the aerosol phases that compose the particle
 *
 * \param model_data Pointer to the model data, including the state array
 * \param aero_phase_idx Index of the aerosol phase within the representation
 * \param aero_phase_avg_MW Average molecular weight in the aerosol phase
 *                          (\f$\mbox{\si{\kilogram\per\mole}}\f$)
 * \param partial_deriv \f$\frac{\partial m}{\partial y}\f$ where \f$y\f$ are
 *                      the species on the state array
 * \param aero_rep_int_data Pointer to the aerosol representation integer data
 * \param aero_rep_float_data Pointer to the aerosol representation
 *                            floating-point data
 * \param aero_rep_env_data Pointer to the aerosol representation
 *                          environment-dependent parameters
 */

void aero_rep_single_particle_get_aero_phase_avg_MW__kg_mol(
    ModelData *model_data, int aero_phase_idx, double *aero_phase_avg_MW,
    double *partial_deriv, int *aero_rep_int_data, double *aero_rep_float_data,
    double *aero_rep_env_data) {

  int *int_data = aero_rep_int_data;
  double *float_data = aero_rep_float_data;
  int i_part = aero_phase_idx / TOTAL_NUM_PHASES_;
  aero_phase_idx -= i_part * TOTAL_NUM_PHASES_;
  
  int i_total_phase = 0;
  for (int i_layer = 0; i_layer < NUM_LAYERS_; ++i_layer) {
    for (int i_phase = 0; i_phase < NUM_PHASES_(i_layer); ++i_phase) {
      if ( i_total_phase == aero_phase_idx ) {
        double *state = (double *)(model_data->grid_cell_state);
        state += i_part * PARTICLE_STATE_SIZE_ + PHASE_STATE_ID_(i_layer,i_phase);
        double mass;
        aero_phase_get_mass__kg_m3(model_data, PHASE_MODEL_DATA_ID_(i_layer,i_phase),
                                   state, &mass, aero_phase_avg_MW, NULL, partial_deriv);
        if (partial_deriv) partial_deriv += PHASE_NUM_JAC_ELEM_(i_layer,i_phase);
      } else if (partial_deriv) {
        for (int i_spec = 0; i_spec < PHASE_NUM_JAC_ELEM_(i_layer,i_phase); ++i_spec)
          *(partial_deriv++) = ZERO;
      }
      ++i_total_phase;
    }
  }
  return;
}

/** \brief Update aerosol representation data
 *
 * Single particle aerosol representation update data is structured as follows:
 *
 *  - \b int aero_rep_id (Id of one or more aerosol representations set by the
 *       host model using the
 *       camp_aero_rep_single_particle::aero_rep_single_particle_t::set_id
 *       function prior to initializing the solver.)
 *  - \b int update_type (Type of update to perform. Can be UPDATE_NUMBER
 *       only.)
 *  - \b double new_value (Either the new radius (m) or the new number
 *       concentration (\f$\mbox{\si{\#\per\cubic\centi\metre}}\f$).)
 *
 * \param update_data Pointer to the updated aerosol representation data
 * \param aero_rep_int_data Pointer to the aerosol representation integer data
 * \param aero_rep_float_data Pointer to the aerosol representation
 *                            floating-point data
 * \param aero_rep_env_data Pointer to the aerosol representation
 *                          environment-dependent parameters
 * \return Flag indicating whether this is the aerosol representation to update
 */

bool aero_rep_single_particle_update_data(void *update_data,
                                          int *aero_rep_int_data,
                                          double *aero_rep_float_data,
                                          double *aero_rep_env_data) {
  int *int_data = aero_rep_int_data;
  double *float_data = aero_rep_float_data;

  int *aero_rep_id = (int *)update_data;
  int *update_type = (int *)&(aero_rep_id[1]);
  int *particle_id = (int *)&(update_type[1]);
  double *new_value = (double *)&(update_type[2]);

  // Set the new radius or number concentration for matching aerosol
  // representations
  if (*aero_rep_id == AERO_REP_ID_ && AERO_REP_ID_ != 0) {
    if (*update_type == UPDATE_NUMBER) {
      NUMBER_CONC_(*particle_id) = (double)*new_value;
      return true;
    }
  }

  return false;
}

/** \brief Print the Single Particle reaction parameters
 *
 * \param aero_rep_int_data Pointer to the aerosol representation integer data
 * \param aero_rep_float_data Pointer to the aerosol representation
 *                            floating-point data
 */

void aero_rep_single_particle_print(int *aero_rep_int_data,
                                    double *aero_rep_float_data) {
  int *int_data = aero_rep_int_data;
  double *float_data = aero_rep_float_data;

  printf("\n\nSingle particle aerosol representation\n");
  printf("\nNumber of phases: %d", TOTAL_NUM_PHASES_);
  printf("\nAerosol representation id: %d", AERO_REP_ID_);
  printf("\nMax computational particles: %d", MAX_PARTICLES_);
  printf("\nParticle state size: %d", PARTICLE_STATE_SIZE_);
  for(int i_layer = 0; i_layer < NUM_LAYERS_; ++i_layer){
    printf("\nLayer: %d", i_layer);
    printf("\n  Start phase: %d End phase: %d", LAYER_PHASE_START_(i_layer), LAYER_PHASE_END_(i_layer));
    printf("\n  Number of phases: %d", NUM_PHASES_(i_layer));
    printf("\n\n   - Phases -");
    for (int i_phase = 0; i_phase < NUM_PHASES_(i_layer); ++i_phase) {
      printf("\n  state id: %d model data id: %d num Jac elements: %d",
             PHASE_STATE_ID_(i_layer,i_phase), PHASE_MODEL_DATA_ID_(i_layer,i_phase),
             PHASE_NUM_JAC_ELEM_(i_layer,i_phase));
    }
  }  
  printf("\n\nEnd single particle aerosol representation\n");
  return;
}


/** \brief Create update data for new particle number
 *
 * \return Pointer to a new number update data object
 */
void *aero_rep_single_particle_create_number_update_data() {
  int *update_data = (int *)malloc(3 * sizeof(int) + sizeof(double));
  if (update_data == NULL) {
    printf("\n\nERROR allocating space for number update data\n\n");
    exit(1);
  }
  return (void *)update_data;
}

/** \brief Set number update data (#/m3)
 *
 * \param update_data Pointer to an allocated number update data object
 * \param aero_rep_id Id of the aerosol representation(s) to update
 * \param particle_id Id of the computational particle
 * \param number_conc New particle number (#/m3)
 */
void aero_rep_single_particle_set_number_update_data__n_m3(void *update_data,
                                                           int aero_rep_id,
                                                           int particle_id,
                                                           double number_conc) {
  int *new_aero_rep_id = (int *)update_data;
  int *update_type = (int *)&(new_aero_rep_id[1]);
  int *new_particle_id = (int *)&(update_type[1]);
  double *new_number_conc = (double *)&(update_type[2]);
  *new_aero_rep_id = aero_rep_id;
  *update_type = UPDATE_NUMBER;
  *new_particle_id = particle_id;
  *new_number_conc = number_conc;
}


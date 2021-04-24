/* Copyright (C) 2019 Christian Guzman
 * Licensed under the GNU General Public License version 1 or (at your
 * option) any later version. See the file COPYING for details.
 *
 * Phase Transfer reaction solver functions
 *
*/
/** \file
 * \brief Phase Transfer reaction solver functions
*/
extern "C"{
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "../aeros/aero_rep_gpu_solver.h"
#include "../aeros/sub_model_gpu_solver.h"
#include "../../util.h"

#define TEMPERATURE_K_ env_data[0]
#define PRESSURE_PA_ env_data[1]

// Jacobian set indices
#define JAC_GAS 0
#define JAC_AERO 1

// Aerosol mass concentration types
#define PER_PARTICLE_MASS 0
#define TOTAL_PARTICLE_MASS 1

#ifndef REVERSE_INT_FLOAT_MATRIX

#define DELTA_H_ float_data[0*n_rxn]
#define DELTA_S_ float_data[1*n_rxn]
#define DIFF_COEFF_ float_data[2*n_rxn]
#define PRE_C_AVG_ float_data[3*n_rxn]
#define B1_ float_data[4*n_rxn]
#define B2_ float_data[5*n_rxn]
#define B3_ float_data[6*n_rxn]
#define B4_ float_data[7*n_rxn]
#define CONV_ float_data[8*n_rxn]
#define MW_ float_data[9*n_rxn]
#define NUM_AERO_PHASE_ int_data[0*n_rxn]
#define GAS_SPEC_ (int_data[1*n_rxn]-1)
#define MFP_M_ rxn_env_data[0]
#define ALPHA_ rxn_env_data[1]
#define EQUIL_CONST_ rxn_env_data[2]
#define KGM3_TO_PPM_ rxn_env_data[3]
#define NUM_INT_PROP_ 2
#define NUM_FLOAT_PROP_ 10
#define NUM_ENV_PARAM_ 4
#define AERO_SPEC_(x) (int_data[(NUM_INT_PROP_ + x)*n_rxn]-1)
#define AERO_ACT_ID_(x) (int_data[(NUM_INT_PROP_ + NUM_AERO_PHASE_ + x)*n_rxn]-1)
#define AERO_PHASE_ID_(x) (int_data[(NUM_INT_PROP_ + 2*(NUM_AERO_PHASE_) + x)*n_rxn]-1)
#define AERO_REP_ID_(x) (int_data[(NUM_INT_PROP_ + 3*(NUM_AERO_PHASE_) + x)*n_rxn]-1)
#define DERIV_ID_(x) (int_data[(NUM_INT_PROP_ + 4*(NUM_AERO_PHASE_) + x)*n_rxn])
#define GAS_ACT_JAC_ID_(x) int_data[(NUM_INT_PROP_ + 1 + 5*(NUM_AERO_PHASE_) + x)*n_rxn]
#define AERO_ACT_JAC_ID_(x) int_data[(NUM_INT_PROP_ + 1 + 6*(NUM_AERO_PHASE_) + x)*n_rxn]
#define JAC_ID_(x) (int_data[(NUM_INT_PROP_ + 1 + 7*(NUM_AERO_PHASE_) + x)*n_rxn])
#define PHASE_INT_LOC_(x) (int_data[(NUM_INT_PROP_ + 2 + 10*(NUM_AERO_PHASE_) + x)*n_rxn]-1)
#define PHASE_FLOAT_LOC_(x) (int_data[(NUM_INT_PROP_ + 2 + 11*(NUM_AERO_PHASE_) + x)*n_rxn]-1)
#define NUM_AERO_PHASE_JAC_ELEM_(x) (int_data[PHASE_INT_LOC_(x)*n_rxn])
#define PHASE_JAC_ID_(x,s,e) int_data[(PHASE_INT_LOC_(x)+1+s*NUM_AERO_PHASE_JAC_ELEM_(x)+e)*n_rxn]
#define EFF_RAD_JAC_ELEM_(x,e) float_data[(PHASE_FLOAT_LOC_(x)+e]
#define NUM_CONC_JAC_ELEM_(x,e) float_data[(PHASE_FLOAT_LOC_(x)+NUM_AERO_PHASE_JAC_ELEM_(x)+e)*n_rxn]
#define MASS_JAC_ELEM_(x,e) float_data[(PHASE_FLOAT_LOC_(x)+2*NUM_AERO_PHASE_JAC_ELEM_(x)+e)*n_rxn]
#define MW_JAC_ELEM_(x,e) float_data[(PHASE_FLOAT_LOC_(x)+3*NUM_AERO_PHASE_JAC_ELEM_(x)+e)*n_rxn]

#else

#define DELTA_H_ float_data[0]
#define DELTA_S_ float_data[1]
#define DIFF_COEFF_ float_data[2]
#define PRE_C_AVG_ float_data[3]
#define B1_ float_data[4]
#define B2_ float_data[5]
#define B3_ float_data[6]
#define B4_ float_data[7]
#define CONV_ float_data[8]
#define MW_ float_data[9]
#define NUM_AERO_PHASE_ int_data[0]
#define GAS_SPEC_ (int_data[1]-1)
#define MFP_M_ rxn_env_data[0]
#define ALPHA_ rxn_env_data[1]
#define EQUIL_CONST_ rxn_env_data[2]
#define KGM3_TO_PPM_ rxn_env_data[3]
#define NUM_INT_PROP_ 2
#define NUM_FLOAT_PROP_ 10
#define NUM_ENV_PARAM_ 4
#define AERO_SPEC_(x) (int_data[(NUM_INT_PROP_ + x)]-1)
#define AERO_ACT_ID_(x) (int_data[(NUM_INT_PROP_ + NUM_AERO_PHASE_ + x)]-1)
#define AERO_PHASE_ID_(x) (int_data[(NUM_INT_PROP_ + 2*(NUM_AERO_PHASE_) + x)]-1)
#define AERO_REP_ID_(x) (int_data[(NUM_INT_PROP_ + 3*(NUM_AERO_PHASE_) + x)]-1)
#define DERIV_ID_(x) (int_data[(NUM_INT_PROP_ + 4*(NUM_AERO_PHASE_) + x)])
#define GAS_ACT_JAC_ID_(x) int_data[(NUM_INT_PROP_ + 1 + 5*(NUM_AERO_PHASE_) + x)]
#define AERO_ACT_JAC_ID_(x) int_data[(NUM_INT_PROP_ + 1 + 6*(NUM_AERO_PHASE_) + x)]
#define JAC_ID_(x) (int_data[(NUM_INT_PROP_ + 1 + 7*(NUM_AERO_PHASE_) + x)])
#define PHASE_INT_LOC_(x) (int_data[(NUM_INT_PROP_ + 2 + 10*(NUM_AERO_PHASE_) + x)]-1)
#define PHASE_FLOAT_LOC_(x) (int_data[(NUM_INT_PROP_ + 2 + 11*(NUM_AERO_PHASE_) + x)]-1)
#define NUM_AERO_PHASE_JAC_ELEM_(x) (int_data[PHASE_INT_LOC_(x)])
#define PHASE_JAC_ID_(x,s,e) int_data[(PHASE_INT_LOC_(x)+1+s*NUM_AERO_PHASE_JAC_ELEM_(x)+e)]
#define EFF_RAD_JAC_ELEM_(x,e) float_data[(PHASE_FLOAT_LOC_(x)+e]
#define NUM_CONC_JAC_ELEM_(x,e) float_data[(PHASE_FLOAT_LOC_(x)+NUM_AERO_PHASE_JAC_ELEM_(x)+e)]
#define MASS_JAC_ELEM_(x,e) float_data[(PHASE_FLOAT_LOC_(x)+2*NUM_AERO_PHASE_JAC_ELEM_(x)+e)]
#define MW_JAC_ELEM_(x,e) float_data[(PHASE_FLOAT_LOC_(x)+3*NUM_AERO_PHASE_JAC_ELEM_(x)+e)]


#endif

#ifdef __CUDA_ARCH__
__host__ __device__
#endif
#ifdef BASIC_CALC_DERIV
void rxn_gpu_SIMPOL_phase_calc_deriv_contrib(ModelDataGPU *model_data, realtype *deriv,
                                      int *rxn_int_data, double *rxn_float_data,
                                      double *rxn_env_data, double time_step)
#else
void rxn_gpu_SIMPOL_phase_transfer_calc_deriv_contrib(ModelDataGPU *model_data, TimeDerivativeGPU time_deriv,
                                          int *rxn_int_data, double *rxn_float_data,
                                          double *rxn_env_data, double time_step)
#endif
{
#ifdef __CUDA_ARCH__
  int n_rxn=model_data->n_rxn;
#else
  int n_rxn=1;
#endif
  int *int_data = rxn_int_data;
  double *float_data = rxn_float_data;
  double *state = model_data->grid_cell_state;
  double *env_data = model_data->grid_cell_env;

  // Calculate derivative contributions for each aerosol phase
  for (int i_phase = 0; i_phase < NUM_AERO_PHASE_; i_phase++) {
    // Get the particle effective radius (m)
    double radius;
    aero_rep_gpu_get_effective_radius__m(
            model_data,               // model data
            AERO_REP_ID_(i_phase),    // aerosol representation index
            AERO_PHASE_ID_(i_phase),  // aerosol phase index
            &radius,                  // particle effective radius (m)
            NULL);                    // partial derivative

    // Check the aerosol concentration type (per-particle or total per-phase
    // mass)
    int aero_conc_type = aero_rep_gpu_get_aero_conc_type(
            model_data,                // model data
            AERO_REP_ID_(i_phase),     // aerosol representation index
            AERO_PHASE_ID_(i_phase));  // aerosol phase index

    // Get the particle number concentration (#/m3)
    double number_conc;
    aero_rep_gpu_get_number_conc__n_m3(
            model_data,               // model data
            AERO_REP_ID_(i_phase),    // aerosol representation index
            AERO_PHASE_ID_(i_phase),  // aerosol phase index
            &number_conc,             // particle number conc (#/m3)
            NULL);                    // partial derivative

    // Get the total mass of the aerosol phase (kg/m3)
    double aero_phase_gpu_mass;
    aero_rep_gpu_get_aero_phase_gpu_mass__kg_m3(
            model_data,               // model data
            AERO_REP_ID_(i_phase),    // aerosol representation index
            AERO_PHASE_ID_(i_phase),  // aerosol phase index
            &aero_phase_gpu_mass,         // total aerosol-phase mass (kg/m3)
            NULL);                    // partial derivatives

    // Get the total mass of the aerosol phase (kg/mol)
    double aero_phase_gpu_avg_MW;
    aero_rep_gpu_get_aero_phase_gpu_avg_MW__kg_mol(
            model_data,               // model data
            AERO_REP_ID_(i_phase),    // aerosol representation index
            AERO_PHASE_ID_(i_phase),  // aerosol phase index
            &aero_phase_gpu_avg_MW,       // avg MW in the aerosol phase (kg/mol)
            NULL);                    // partial derivatives

    // Calculate the rate constant for diffusion limited mass transfer to the
    // aerosol phase (m3/#/s)
    double cond_rate =
            gas_aerosol_rxn_rate_constant(DIFF_COEFF_, MFP_M_, radius, ALPHA_);

    // Calculate the evaporation rate constant (ppm_x*m^3/kg_x/s)
    double evap_rate =
            cond_rate * (EQUIL_CONST_ * aero_phase_gpu_avg_MW / aero_phase_gpu_mass);

    // Get the activity coefficient (if one exists)
    double act_coeff = 1.0;
    if (AERO_ACT_ID_(i_phase) > -1) {
      act_coeff = state[AERO_ACT_ID_(i_phase)];
    }

    // Calculate aerosol-phase evaporation rate (ppm/s)
    evap_rate *= act_coeff;

    // Calculate the evaporation and condensation rates
    cond_rate *= state[GAS_SPEC_];
    evap_rate *= state[AERO_SPEC_(i_phase)];

    // per-particle mass concentration rates
    if (aero_conc_type == PER_PARTICLE_MASS) {
      // Change in the gas-phase is evaporation - condensation (ppm/s)
      if (DERIV_ID_(0) >= 0) {
        time_derivative_add_value_gpu(time_deriv, DERIV_ID_(0),
                                  number_conc * evap_rate);
        time_derivative_add_value_gpu(time_deriv, DERIV_ID_(0),
                                  -number_conc * cond_rate);
      }

      // Change in the aerosol-phase species is condensation - evaporation
      // (kg/m^3/s)
      if (DERIV_ID_(1 + i_phase) >= 0) {
        time_derivative_add_value_gpu(time_deriv, DERIV_ID_(1 + i_phase),
                                  -evap_rate / KGM3_TO_PPM_);
        time_derivative_add_value_gpu(time_deriv, DERIV_ID_(1 + i_phase),
                                  cond_rate / KGM3_TO_PPM_);
      }

      // total-aerosol mass concentration rates
    } else {
      // Change in the gas-phase is evaporation - condensation (ppm/s)
      if (DERIV_ID_(0) >= 0) {
        time_derivative_add_value_gpu(time_deriv, DERIV_ID_(0),
                                  number_conc * evap_rate);
        time_derivative_add_value_gpu(time_deriv, DERIV_ID_(0),
                                  -number_conc * cond_rate);
      }

      // Change in the aerosol-phase species is condensation - evaporation
      // (kg/m^3/s)
      if (DERIV_ID_(1 + i_phase) >= 0) {
        time_derivative_add_value_gpu(time_deriv, DERIV_ID_(1 + i_phase),
                                  -number_conc * evap_rate / KGM3_TO_PPM_);
        time_derivative_add_value_gpu(time_deriv, DERIV_ID_(1 + i_phase),
                                  number_conc * cond_rate / KGM3_TO_PPM_);
      }
    }

  }
  return;
}


}
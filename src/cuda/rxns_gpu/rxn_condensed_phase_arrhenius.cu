/* Copyright (C) 2019 Christian Guzman
 * Licensed under the GNU General Public License version 1 or (at your
 * option) any later version. See the file COPYING for details.
 *
 * Condensed Phase Arrhenius reaction solver functions
 *
*/
/** \file
 * \brief Condensed Phase Arrhenius reaction solver functions
*/
extern "C"{
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "../rxns_gpu.h"

 /*

#define TEMPERATURE_K_ env_data[0]
#define PRESSURE_PA_ env_data[1]

// Small number
#define SMALL_NUMBER_ 1.0e-30

#define NUM_REACT_ (int_data[0])
#define NUM_PROD_ (int_data[1])
#define NUM_AERO_PHASE_ (int_data[2])
#define A_ (float_data[0])
#define B_ (float_data[1])
#define C_ (float_data[2])
#define D_ (float_data[3])
#define E_ (float_data[4])
#define RATE_CONSTANT_ rxn_env_data[0]
#define NUM_INT_PROP_ 3
#define NUM_FLOAT_PROP_ 5
#define REACT_(x) (int_data[(NUM_INT_PROP_ + x)]-1)
#define PROD_(x) (int_data[(NUM_INT_PROP_+NUM_REACT_*NUM_AERO_PHASE_+x)]-1)
#define WATER_(x) (int_data[(NUM_INT_PROP_+(NUM_REACT_+NUM_PROD_)*NUM_AERO_PHASE_+x)]-1)
#define DERIV_ID_(x) (int_data[(NUM_INT_PROP_+(NUM_REACT_+NUM_PROD_+1)*NUM_AERO_PHASE_+x)])
#define JAC_ID_(x) (int_data[(NUM_INT_PROP_+(2*(NUM_REACT_+NUM_PROD_)+1)*NUM_AERO_PHASE_+x)])
#define YIELD_(x) (float_data[(NUM_FLOAT_PROP_+x)])
#define UGM3_TO_MOLM3_(x) (float_data[(NUM_FLOAT_PROP_+NUM_PROD_+x)])
#define INT_DATA_SIZE_ (NUM_INT_PROP_+((NUM_REACT_+NUM_PROD_)*(NUM_REACT_+3)+1)*NUM_AERO_PHASE_)
#define FLOAT_DATA_SIZE_ (NUM_FLOAT_PROP_+2*NUM_PROD_+NUM_REACT_)

#ifdef CAMP_USE_SUNDIALS
#ifdef __CUDA_ARCH__
__host__ __device__
#endif
void rxn_gpu_condensed_phase_arrhenius_calc_deriv_contrib(ModelDataGPU *model_data, realtype *deriv, int *rxn_int_data,
          double *rxn_float_data, double *rxn_env_data, double time_step)
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
  for (int i_phase=0, i_deriv = 0; i_phase<NUM_AERO_PHASE_; i_phase++) {

    // If this is an aqueous reaction, get the unit conversion from mol/m3 -> M
    double unit_conv = 1.0;
    if (WATER_(i_phase)>=0) {
      unit_conv = state[WATER_(i_phase)] * 1.0e-9; // convert from ug/m3->L/m3

      // For aqueous reactions, if no aerosol water is present, no reaction
      // occurs
      if (unit_conv < SMALL_NUMBER_) {
        i_deriv += NUM_REACT_ + NUM_PROD_;
        continue;
      }
      unit_conv = 1.0/unit_conv;endif
    }

    // Calculate the reaction rate rate (M/s or mol/m3/s)
    //double rate = RATE_CONSTANT_;
    realtype rate = RATE_CONSTANT_;
    for (int i_react = 0; i_react < NUM_REACT_; i_react++) {
      rate *= state[REACT_(i_phase*NUM_REACT_+i_react)] *
              UGM3_TO_MOLM3_(i_react) * unit_conv;
    }

    // Reactant change
    for (int i_react = 0; i_react < NUM_REACT_; i_react++) {
      if (DERIV_ID_(i_deriv)<0) {i_deriv++; continue;}
#ifdef __CUDA_ARCH__
      atomicAdd((double*)&(deriv[DERIV_ID_(i_deriv++)]), -(rate /
      (UGM3_TO_MOLM3_(i_react) * unit_conv)));
#else
      deriv[DERIV_ID_(i_deriv++)] -=
        rate / (UGM3_TO_MOLM3_(i_react) * unit_conv);
#endif
    }

    // Products change
    for (int i_prod = 0; i_prod < NUM_PROD_; i_prod++) {
      if (DERIV_ID_(i_deriv)<0) {i_deriv++; continue;}
#ifdef __CUDA_ARCH__
      atomicAdd((double*)&(deriv[DERIV_ID_(i_deriv++)]),rate * YIELD_(i_prod) /
	      (UGM3_TO_MOLM3_(NUM_REACT_+i_prod) * unit_conv));
#else
      deriv[DERIV_ID_(i_deriv++)] +=
          rate * YIELD_(i_prod) /
          (UGM3_TO_MOLM3_(NUM_REACT_ + i_prod) * unit_conv);
#endif
    }

  }

}

#ifdef __CUDA_ARCH__
__host__ __device__
#endif
void rxn_gpu_condensed_phase_arrhenius_calc_jac_contrib(ModelDataGPU *model_data, JacobianGPU jac, int *rxn_int_data,
          double *rxn_float_data, double *rxn_env_data, double time_step)
{
#ifdef __CUDA_ARCH__
  int n_rxn=model_data->n_rxn;
#else
  int n_rxn=1;;
#endif
  int *int_data = rxn_int_data;
  double *float_data = rxn_float_data;
  double *state = model_data->grid_cell_state;
  double *env_data = model_data->grid_cell_env;
  double rate = RATE_CONSTANT_;

}
#endif

#undef TEMPERATURE_K_
#undef PRESSURE_PA_

#undef SMALL_NUMBER_

#undef NUM_REACT_
#undef NUM_PROD_
#undef NUM_AERO_PHASE_
#undef A_
#undef B_
#undef C_
#undef D_
#undef E_
#undef RATE_CONSTANT_
#undef NUM_INT_PROP_
#undef NUM_FLOAT_PROP_
#undef REACT_
#undef PROD_
#undef WATER_
#undef DERIV_ID_
#undef JAC_ID_
#undef YIELD_
#undef UGM3_TO_MOLM3_
#undef INT_DATA_SIZE_
#undef FLOAT_DATA_SIZE_
*/
}
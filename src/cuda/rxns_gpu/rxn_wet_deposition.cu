/* Copyright (C) 2019 Christian Guzman
 * Licensed under the GNU General Public License version 1 or (at your
 * option) any later version. See the file COPYING for details.
 *
 * Wet deposition reaction solver functions
 *
*/
/** \file
 * \brief Wet deposition reaction solver functions
*/
extern "C"{
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "../rxns_gpu.h"

  /*
#define TEMPERATURE_K_ env_data[0]
#define PRESSURE_PA_ env_data[1]

#define RXN_ID_ (int_data[0])
#define NUM_SPEC_ (int_data[1])
#define SCALING_ float_data[0]
#define RATE_CONSTANT_ rxn_env_data[0]
#define BASE_RATE_ rxn_env_data[1]
#define NUM_INT_PROP_ 2
#define NUM_FLOAT_PROP_ 1
#define REACT_(s) (int_data[(NUM_INT_PROP_+s)]-1)
#define DERIV_ID_(s) int_data[(NUM_INT_PROP_+NUM_SPEC_+s)]
#define JAC_ID_(s) int_data[(NUM_INT_PROP_+2*NUM_SPEC_+s)]
#define INT_DATA_SIZE_ (NUM_INT_PROP_+3*NUM_SPEC_)
#define FLOAT_DATA_SIZE_ (NUM_FLOAT_PROP_)

#endif

#ifdef CAMP_USE_SUNDIALS
#ifdef __CUDA_ARCH__
__host__ __device__
#endif
void rxn_gpu_wet_deposition_calc_deriv_contrib(ModelDataGPU *model_data, realtype *deriv, int *rxn_int_data,
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

  // Add contributions to the time derivative
  for (int i_spec = 0; i_spec < NUM_SPEC_; i_spec++) {
    if (DERIV_ID_(i_spec) >= 0 )
#ifdef __CUDA_ARCH__
        atomicAdd((double*)&(deriv[DERIV_ID_(i_spec)]),-RATE_CONSTANT_ * state[REACT_(i_spec)]);
#else
        deriv[DERIV_ID_(i_spec)] -= RATE_CONSTANT_ * state[REACT_(i_spec)];;
#endif
  }

}

#ifdef __CUDA_ARCH__
__host__ __device__
#endif
void rxn_gpu_wet_deposition_calc_jac_contrib(ModelDataGPU *model_data, JacobianGPU jac, int *rxn_int_data,
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

  // Add contributions to the Jacobian
  for (int i_spec = 0; i_spec < NUM_SPEC_; i_spec++) {
    if (JAC_ID_(i_spec) >= 0)
      jacobian_add_value_gpu(jac, (unsigned int)JAC_ID_(i_spec), JACOBIAN_LOSS,
                         RATE_CONSTANT_);
  }

}
#endif

#undef TEMPERATURE_K_
#undef PRESSURE_PA_

#undef RXN_ID_
#undef NUM_SPEC_
#undef BASE_RATE_
#undef SCALING_
#undef RATE_CONSTANT_
#undef NUM_INT_PROP_
#undef NUM_FLOAT_PROP_
#undef REACT_
#undef DERIV_ID_
#undef JAC_ID_
#undef INT_DATA_SIZE_
#undef FLOAT_DATA_SIZE_
*/
}
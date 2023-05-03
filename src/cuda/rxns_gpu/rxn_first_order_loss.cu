/* Copyright (C) 2019 Christian Guzman
 * Licensed under the GNU General Public License version 1 or (at your
 * option) any later version. See the file COPYING for details.
 *
 * First-Order loss reaction solver functions
 *
*/
/** \file
 * \brief First-Order loss reaction solver functions
*/
extern "C"{
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "../rxns_gpu.h"

#define TEMPERATURE_K_ env_data[0]
#define PRESSURE_PA_ env_data[1]

#define RXN_ID_ (int_data[0])
#define REACT_ (int_data[1]-1)
#define DERIV_ID_ int_data[2]
#define JAC_ID_ int_data[3]
#define SCALING_ float_data[0]
#define RATE_CONSTANT_ rxn_env_data[0]
#define BASE_RATE_ rxn_env_data[1]
#define NUM_INT_PROP_ 4
#define NUM_FLOAT_PROP_ 1
#define INT_DATA_SIZE_ (NUM_INT_PROP_)
#define FLOAT_DATA_SIZE_ (NUM_FLOAT_PROP_)

#ifdef CAMP_USE_SUNDIALS
#ifdef __CUDA_ARCH__
__device__
#endif
void rxn_gpu_first_order_loss_calc_deriv_contrib(ModelDataGPU *model_data, TimeDerivativeGPU time_deriv, int *rxn_int_data,
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

  realtype rate = RATE_CONSTANT_ * state[REACT_];
  if (DERIV_ID_ >= 0) time_derivative_add_value_gpu(time_deriv, DERIV_ID_, -rate);
}

#ifdef __CUDA_ARCH__
__device__
#endif
void rxn_gpu_first_order_loss_calc_jac_contrib(ModelDataGPU *model_data, JacobianGPU jac, int *rxn_int_data,
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

  if (JAC_ID_ >= 0)
    jacobian_add_value_gpu(jac, (unsigned int)JAC_ID_, JACOBIAN_LOSS,
                       RATE_CONSTANT_);
}
#endif
}
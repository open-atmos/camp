/* Copyright (C) 2019 Christian Guzman
 * Licensed under the GNU General Public License version 1 or (at your
 * option) any later version. See the file COPYING for details.
 *
 * Photolysis reaction solver functions
 *
*/
/** \file
 * \brief Photolysis reaction solver functions
*/
extern "C"{
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "../rxns_gpu.h"

#define TEMPERATURE_K_ env_data[0]
#define PRESSURE_PA_ env_data[1]

#ifdef REVERSE_INT_FLOAT_MATRIX

#define NUM_REACT_ int_data[0*n_rxn]
#define NUM_PROD_ int_data[1*n_rxn]
#define PHOTO_ID_ int_data[2*n_rxn]
#define SCALING_ float_data[0*n_rxn]
#define RATE_CONSTANT_ rxn_env_data[0*n_rxn]
#define BASE_RATE_ rxn_env_data[1*n_rxn]
#define NUM_INT_PROP_ 3
#define NUM_FLOAT_PROP_ 1
#define REACT_(x) (int_data[(NUM_INT_PROP_ + x)*n_rxn]-1)
#define PROD_(x) (int_data[(NUM_INT_PROP_ + NUM_REACT_ + x)*n_rxn]-1)
#define DERIV_ID_(x) int_data[(NUM_INT_PROP_ + NUM_REACT_ + NUM_PROD_ + x)*n_rxn]
#define JAC_ID_(x) int_data[(NUM_INT_PROP_ + 2*(NUM_REACT_+NUM_PROD_) + x)*n_rxn]
#define YIELD_(x) float_data[(NUM_FLOAT_PROP_ + x)*n_rxn]
#define INT_DATA_SIZE_ (NUM_INT_PROP_+(NUM_REACT_+2)*(NUM_REACT_+NUM_PROD_))
#define FLOAT_DATA_SIZE_ (NUM_FLOAT_PROP_+NUM_PROD_)

#else

#define NUM_REACT_ int_data[0]
#define NUM_PROD_ int_data[1]
#define PHOTO_ID_ int_data[2]
#define SCALING_ float_data[0]
#define RATE_CONSTANT_ rxn_env_data[0]
#define BASE_RATE_ rxn_env_data[1]
#define NUM_INT_PROP_ 3
#define NUM_FLOAT_PROP_ 1
#define REACT_(x) (int_data[(NUM_INT_PROP_ + x)]-1)
#define PROD_(x) (int_data[(NUM_INT_PROP_ + NUM_REACT_ + x)]-1)
#define DERIV_ID_(x) int_data[(NUM_INT_PROP_ + NUM_REACT_ + NUM_PROD_ + x)]
#define JAC_ID_(x) int_data[(NUM_INT_PROP_ + 2*(NUM_REACT_+NUM_PROD_) + x)]
#define YIELD_(x) float_data[(NUM_FLOAT_PROP_ + x)]
#define INT_DATA_SIZE_ (NUM_INT_PROP_+(NUM_REACT_+2)*(NUM_REACT_+NUM_PROD_))
#define FLOAT_DATA_SIZE_ (NUM_FLOAT_PROP_+NUM_PROD_)

#endif

/** \brief Calculate contributions to the time derivative \f$f(t,y)\f$ from
 * this reaction.
 *
 * \param model_data Pointer to the model data, including the state array
 * \param deriv Pointer to the time derivative to add contributions to
 * \param rxn_data Pointer to the reaction data
 * \param time_step Current time step being computed (s)
 * \return The rxn_data pointer advanced by the size of the reaction data
 */
#ifdef CAMP_USE_SUNDIALS
#ifdef __CUDA_ARCH__
__host__ __device__
#endif
void rxn_gpu_photolysis_calc_deriv_contrib(ModelDataGPU *model_data, TimeDerivativeGPU time_deriv, int *rxn_int_data,
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

  // Calculate the reaction rate
  realtype rate = RATE_CONSTANT_;
  for (int i_spec=0; i_spec<NUM_REACT_; i_spec++)
          rate *= state[REACT_(i_spec)];

  // Add contributions to the time derivative
  if (rate!=ZERO) {
    int i_dep_var = 0;
    for (int i_spec=0; i_spec<NUM_REACT_; i_spec++, i_dep_var++) {
      if (DERIV_ID_(i_dep_var) < 0) continue;

      time_derivative_add_value_gpu(time_deriv, DERIV_ID_(i_dep_var), -rate);
    }
    for (int i_spec=0; i_spec<NUM_PROD_; i_spec++, i_dep_var++) {
      if (DERIV_ID_(i_dep_var) < 0) continue;
            // Negative yields are allowed, but prevented from causing negative
      // concentrations that lead to solver failures
        if (-rate * YIELD_(i_spec) * time_step <= state[PROD_(i_spec)]){
        time_derivative_add_value_gpu(time_deriv, DERIV_ID_(i_dep_var),rate*YIELD_(i_spec));
      }
    }
  }

}
#endif

/** \brief Calculate contributions to the Jacobian from this reaction
 *
 * \param model_data Pointer to the model data
 * \param J Pointer to the sparse Jacobian matrix to add contributions to
 * \param rxn_data Pointer to the reaction data
 * \param time_step Current time step being calculated (s)
 * \return The rxn_data pointer advanced by the size of the reaction data
 */
#ifdef CAMP_USE_SUNDIALS
#ifdef __CUDA_ARCH__
__host__ __device__
#endif
void rxn_gpu_photolysis_calc_jac_contrib(ModelDataGPU *model_data, JacobianGPU jac, int *rxn_int_data,
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
  int i_elem = 0;
  for (int i_ind = 0; i_ind < NUM_REACT_; i_ind++) {
    // Calculate d_rate / d_i_ind
    double rate = RATE_CONSTANT_;
    for (int i_spec = 0; i_spec < NUM_REACT_; i_spec++)
      if (i_spec != i_ind) rate *= state[REACT_(i_spec)];

    for (int i_dep = 0; i_dep < NUM_REACT_; i_dep++, i_elem++) {
      if (JAC_ID_(i_elem) < 0) continue;
      //jacobian_add_value_gpu(jac, (unsigned int)JAC_ID_(i_elem), JACOBIAN_LOSS,
      //                   RATE_CONSTANT_);
      jacobian_add_value_gpu(jac, (unsigned int)JAC_ID_(i_elem), JACOBIAN_LOSS,
                        rate);
    }
    for (int i_dep = 0; i_dep < NUM_PROD_; i_dep++, i_elem++) {
      if (JAC_ID_(i_elem) < 0) continue;
      // Negative yields are allowed, but prevented from causing negative
      // concentrations that lead to solver failures
      if (-rate * state[REACT_(i_ind)] * YIELD_(i_dep) * time_step <=
          state[PROD_(i_dep)]) {
      //jacobian_add_value_gpu(jac, (unsigned int)JAC_ID_(i_elem),
      //                   JACOBIAN_PRODUCTION, YIELD_(i_dep) * RATE_CONSTANT_);
      jacobian_add_value_gpu(jac, (unsigned int)JAC_ID_(i_elem),
                   JACOBIAN_PRODUCTION, YIELD_(i_dep) * rate);
      }
    }
  }

}
#endif

#undef TEMPERATURE_K_
#undef PRESSURE_PA_

#undef NUM_REACT_
#undef NUM_PROD_
#undef PHOTO_ID_
#undef BASE_RATE_
#undef SCALING_
#undef RATE_CONSTANT_
#undef NUM_INT_PROP_
#undef NUM_FLOAT_PROP_
#undef REACT_
#undef PROD_
#undef DERIV_ID_
#undef JAC_ID_
#undef YIELD_
#undef INT_DATA_SIZE_
#undef FLOAT_DATA_SIZE_
}
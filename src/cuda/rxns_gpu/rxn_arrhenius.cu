/* Copyright (C) 2019 Christian Guzman
 * Licensed under the GNU General Public License version 1 or (at your
 * option) any later version. See the file COPYING for details.
 *
 * Arrhenius reaction solver functions
 *
*/
/** \file
 * \brief Arrhenius reaction solver functions
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
#define A_ float_data[0*n_rxn]
#define B_ float_data[1*n_rxn]
#define C_ float_data[2*n_rxn]
#define D_ float_data[3*n_rxn]
#define E_ float_data[4*n_rxn]
#define CONV_ float_data[5*n_rxn]
#define RATE_CONSTANT_ rxn_env_data[0*n_rxn]
#define NUM_INT_PROP_ 2
#define NUM_FLOAT_PROP_ 6
#define NUM_ENV_PARAM_ 1
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
#define A_ float_data[0]
#define B_ float_data[1]
#define C_ float_data[2]
#define D_ float_data[3]
#define E_ float_data[4]
#define CONV_ float_data[5]
#define RATE_CONSTANT_ rxn_env_data[0]
#define NUM_INT_PROP_ 2
#define NUM_FLOAT_PROP_ 6
#define NUM_ENV_PARAM_ 1
#define REACT_(x) (int_data[(NUM_INT_PROP_ + x)]-1)
#define PROD_(x) (int_data[(NUM_INT_PROP_ + NUM_REACT_ + x)]-1)
#define DERIV_ID_(x) int_data[(NUM_INT_PROP_ + NUM_REACT_ + NUM_PROD_ + x)]
#define JAC_ID_(x) int_data[(NUM_INT_PROP_ + 2*(NUM_REACT_+NUM_PROD_) + x)]
#define YIELD_(x) float_data[(NUM_FLOAT_PROP_ + x)]
#define INT_DATA_SIZE_ (NUM_INT_PROP_+(NUM_REACT_+2)*(NUM_REACT_+NUM_PROD_))
#define FLOAT_DATA_SIZE_ (NUM_FLOAT_PROP_+NUM_PROD_)

#endif

/** \brief Flag Jacobian elements used by this reaction
 *
 * \param rxn_data A pointer to the reaction data
 * \param jac_struct 2D array of flags indicating potentially non-zero
 *                   Jacobian elements
 * \return The rxn_data pointer advanced by the size of the reaction data
 */
void rxn_gpu_arrhenius_get_used_jac_elem(void *rxn_data, bool **jac_struct)
{
  int n_rxn=1;
  int *int_data = (int*) rxn_data;
  double *float_data = (double*) &(int_data[INT_DATA_SIZE_]);

  for (int i_ind = 0; i_ind < NUM_REACT_; i_ind++) {
    for (int i_dep = 0; i_dep < NUM_REACT_; i_dep++) {
      jac_struct[REACT_(i_dep)][REACT_(i_ind)] = true;
    }
    for (int i_dep = 0; i_dep < NUM_PROD_; i_dep++) {
      jac_struct[PROD_(i_dep)][REACT_(i_ind)] = true;
    }
  }

}

/** \brief Update the time derivative and Jacbobian array indices
 *
 * \param model_data Pointer to the model data
 * \param deriv_ids Id of each state variable in the derivative array
 * \param jac_ids Id of each state variable combo in the Jacobian array
 * \param rxn_data Pointer to the reaction data
 * \return The rxn_data pointer advanced by the size of the reaction data
 */
void * rxn_gpu_arrhenius_update_ids(ModelDataGPU *model_data, int *deriv_ids,
                                    int **jac_ids, void *rxn_data)
{
  int n_rxn=1;
  int *int_data = (int*) rxn_data;
  double *float_data = (double*) &(int_data[INT_DATA_SIZE_]);

  // Update the time derivative ids
  for (int i=0; i < NUM_REACT_; i++)
    DERIV_ID_(i) = deriv_ids[REACT_(i)];
  for (int i=0; i < NUM_PROD_; i++)
    DERIV_ID_(i + NUM_REACT_) = deriv_ids[PROD_(i)];

  // Update the Jacobian ids
  int i_jac = 0;
  for (int i_ind = 0; i_ind < NUM_REACT_; i_ind++) {
    for (int i_dep = 0; i_dep < NUM_REACT_; i_dep++) {
      JAC_ID_(i_jac++) = jac_ids[REACT_(i_dep)][REACT_(i_ind)];
    }
    for (int i_dep = 0; i_dep < NUM_PROD_; i_dep++) {
      JAC_ID_(i_jac++) = jac_ids[PROD_(i_dep)][REACT_(i_ind)];
    }
  }
  return (void*) &(float_data[FLOAT_DATA_SIZE_]);
}

/** \brief Do pre-derivative calculations
 *
 * Nothing to do for arrhenius reactions
 *
 * \param model_data Pointer to the model data, including the state array
 * \param rxn_data Pointer to the reaction data
 * \return The rxn_data pointer advanced by the size of the reaction data
 */
void * rxn_gpu_arrhenius_pre_calc(ModelDataGPU *model_data, void *rxn_data)
{
  int n_rxn=1;
  int *int_data = (int*) rxn_data;
  double *float_data = (double*) &(int_data[INT_DATA_SIZE_]);

  return (void*) &(float_data[FLOAT_DATA_SIZE_]);
}

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
void rxn_gpu_arrhenius_calc_deriv_contrib(ModelDataGPU *model_data, TimeDerivativeGPU time_deriv,
                                      int *rxn_int_data, double *rxn_float_data,
                                      double *rxn_env_data, double time_step)
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

  double rate = RATE_CONSTANT_;
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
      if (-rate*YIELD_(i_spec)*time_step <= state[PROD_(i_spec)]) {
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
void rxn_gpu_arrhenius_calc_jac_contrib(ModelDataGPU *model_data, JacobianGPU jac, int *rxn_int_data,
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
      jacobian_add_value_gpu(jac, (unsigned int)JAC_ID_(i_elem), JACOBIAN_LOSS,
                         rate);
      //check_isnanld(&rate,1,"post rxn_arrhenius_calc_jac_contrib rate");
    }
    for (int i_dep = 0; i_dep < NUM_PROD_; i_dep++, i_elem++) {
      if (JAC_ID_(i_elem) < 0) continue;
      // Negative yields are allowed, but prevented from causing negative
      // concentrations that lead to solver failures
      if (-rate * state[REACT_(i_ind)] * YIELD_(i_dep) * time_step <=
          state[PROD_(i_dep)]) {
        jacobian_add_value_gpu(jac, (unsigned int)JAC_ID_(i_elem),
                           JACOBIAN_PRODUCTION, YIELD_(i_dep) * rate);
        //check_isnanld(&YIELD_(i_dep),1,"post rxn_arrhenius_calc_jac_contrib YIELD_(i_dep)");
        //check_isnanld(&rate,1,"post rxn_arrhenius_calc_jac_contrib rate");
      }
    }
  }

}
#endif

}

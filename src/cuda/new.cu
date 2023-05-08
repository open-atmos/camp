/* Copyright (C) 2021 Barcelona Supercomputing Center and University of
 * Illinois at Urbana-Champaign
 * SPDX-License-Identifier: MIT
 */

extern "C" {
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "new.h"

__device__
void jacobian_add_value_gpu(JacobianGPU jac, unsigned int elem_id,
                            unsigned int prod_or_loss,
                            double jac_contribution) {
  if (prod_or_loss == JACOBIAN_PRODUCTION)
    atomicAdd_block(&(jac.production_partials[elem_id]),jac_contribution);
  if (prod_or_loss == JACOBIAN_LOSS)
    atomicAdd_block(&(jac.loss_partials[elem_id]),jac_contribution);
}


__device__
void time_derivative_add_value_gpu(TimeDerivativeGPU time_deriv, unsigned int spec_id,
                               double rate_contribution) {
  if (rate_contribution > 0.0) {
    atomicAdd_block(&(time_deriv.production_rates[spec_id]),rate_contribution);
  } else {
    atomicAdd_block(&(time_deriv.loss_rates[spec_id]),-rate_contribution);
  }
}

__device__
void rxn_gpu_first_order_loss_calc_deriv_contrib(ModelDataGPU *model_data, TimeDerivativeGPU time_deriv, int *rxn_int_data,
          double *rxn_float_data, double *rxn_env_data, double time_step){
  int *int_data = rxn_int_data;
  double *state = model_data->grid_cell_state;
  double rate = rxn_env_data[0] * state[int_data[1]-1];
  if (int_data[2] >= 0) time_derivative_add_value_gpu(time_deriv, int_data[2], -rate);
}

__device__
void rxn_gpu_first_order_loss_calc_jac_contrib(ModelDataGPU *model_data, JacobianGPU jac, int *rxn_int_data,
          double *rxn_float_data, double *rxn_env_data, double time_step){
  int *int_data = rxn_int_data;
  if (int_data[3] >= 0)
    jacobian_add_value_gpu(jac, (unsigned int)int_data[3], JACOBIAN_LOSS,
                       rxn_env_data[0]);
}

__device__
void rxn_gpu_troe_calc_deriv_contrib(ModelDataGPU *model_data, TimeDerivativeGPU time_deriv, int *rxn_int_data,
          double *rxn_float_data, double *rxn_env_data, double time_step){
  int *int_data = rxn_int_data;
  double *float_data = rxn_float_data;
  double *state = model_data->grid_cell_state;
  double rate = rxn_env_data[0];
  for (int i_spec=0; i_spec<int_data[0]; i_spec++)
          rate *= state[int_data[(2 + i_spec)]-1];
  if (rate!=ZERO) {
    int i_dep_var = 0;
    for (int i_spec=0; i_spec<int_data[0]; i_spec++, i_dep_var++) {
      if (int_data[(2 + int_data[0]+int_data[1]+i_dep_var)] < 0) continue;
      time_derivative_add_value_gpu(time_deriv, int_data[(2 + int_data[0]+int_data[1]+i_dep_var)], -rate);
    }
    for (int i_spec=0; i_spec<int_data[1]; i_spec++, i_dep_var++) {
      if (int_data[(2 + int_data[0]+int_data[1]+i_dep_var)] < 0) continue;
      if (-rate * float_data[(10 + i_spec)] * time_step <= state[int_data[(2 + int_data[0]+ i_spec)]-1]) {
        time_derivative_add_value_gpu(time_deriv, int_data[(2 + int_data[0]+int_data[1]+i_dep_var)],rate*float_data[(10 + i_spec)]);
      }
    }
  }
}

__device__
void rxn_gpu_troe_calc_jac_contrib(ModelDataGPU *model_data, JacobianGPU jac, int *rxn_int_data,
          double *rxn_float_data, double *rxn_env_data, double time_step){
  int *int_data = rxn_int_data;
  double *float_data = rxn_float_data;
  double *state = model_data->grid_cell_state;
  int i_elem = 0;
  for (int i_ind = 0; i_ind < int_data[0]; i_ind++) {
    double rate = rxn_env_data[0];
    for (int i_spec = 0; i_spec < int_data[0]; i_spec++)
      if (i_ind != i_spec) rate *= state[int_data[(2 + i_spec)]-1];
    for (int i_dep = 0; i_dep < int_data[0]; i_dep++, i_elem++) {
      if (int_data[(2 + 2*(int_data[0]+int_data[1])+i_elem)] < 0) continue;
      jacobian_add_value_gpu(jac, (unsigned int)int_data[(2 + 2*(int_data[0]+int_data[1])+i_elem)], JACOBIAN_LOSS,
                         rate);
    }
    for (int i_dep = 0; i_dep < int_data[1]; i_dep++, i_elem++) {
      if (int_data[(2 + 2*(int_data[0]+int_data[1])+i_elem)] < 0) continue;
      if (-rate * state[int_data[(2 + i_ind)]-1] * float_data[(10 + i_dep)] * time_step <=
        state[int_data[(2 + int_data[0]+ i_dep)]-1]) {
        int elem_id = (unsigned int) int_data[(2 + 2*(int_data[0]+int_data[1])+i_elem)];
        atomicAdd_block(&(jac.production_partials[elem_id]), float_data[(10 + i_dep)] * rate);
        //jacobian_add_value_gpu(jac, (unsigned int)int_data[(2 + 2*(int_data[0]+int_data[1])+i_elem)],JACOBIAN_PRODUCTION, float_data[(10 + i_dep)] * rate);
      }
    }
  }
}

__device__
void rxn_gpu_photolysis_calc_deriv_contrib(ModelDataGPU *model_data, TimeDerivativeGPU time_deriv, int *rxn_int_data,
          double *rxn_float_data, double *rxn_env_data, double time_step){
  int *int_data = rxn_int_data;
  double *float_data = rxn_float_data;
  double *state = model_data->grid_cell_state;
  double rate = rxn_env_data[0];
  for (int i_spec=0; i_spec<int_data[0]; i_spec++)
          rate *= state[int_data[(3 + i_spec)]-1];
  if (rate!=ZERO) {
    int i_dep_var = 0;
    for (int i_spec=0; i_spec<int_data[0]; i_spec++, i_dep_var++) {
      if (int_data[(3 + int_data[0]+int_data[1]+i_dep_var)] < 0) continue;
      time_derivative_add_value_gpu(time_deriv, int_data[(3 + int_data[0]+int_data[1]+i_dep_var)], -rate);
    }
    for (int i_spec=0; i_spec<int_data[1]; i_spec++, i_dep_var++) {
      if (int_data[(3 + int_data[0]+int_data[1]+i_dep_var)] < 0) continue;
        if (-rate * float_data[(1 + i_spec)] * time_step <= state[int_data[(3 + int_data[0]+ i_spec)]-1]){
        time_derivative_add_value_gpu(time_deriv, int_data[(3 + int_data[0]+int_data[1]+i_dep_var)],rate*float_data[(1 + i_spec)]);
      }
    }
  }
}

__device__
void rxn_gpu_photolysis_calc_jac_contrib(ModelDataGPU *model_data, JacobianGPU jac, int *rxn_int_data,
          double *rxn_float_data, double *rxn_env_data, double time_step){
  int *int_data = rxn_int_data;
  double *float_data = rxn_float_data;
  double *state = model_data->grid_cell_state;
  int i_elem = 0;
  for (int i_ind = 0; i_ind < int_data[0]; i_ind++) {
    double rate = rxn_env_data[0];
    for (int i_spec = 0; i_spec < int_data[0]; i_spec++)
      if (i_spec != i_ind) rate *= state[int_data[(3 + i_spec)]-1];
    for (int i_dep = 0; i_dep < int_data[0]; i_dep++, i_elem++) {
      if (int_data[(3 + 2*(int_data[0]+int_data[1])+i_elem)] < 0) continue;
      jacobian_add_value_gpu(jac, (unsigned int)int_data[(3 + 2*(int_data[0]+int_data[1])+i_elem)], JACOBIAN_LOSS,
                        rate);
    }
    for (int i_dep = 0; i_dep < int_data[1]; i_dep++, i_elem++) {
      if (int_data[(3 + 2*(int_data[0]+int_data[1])+i_elem)] < 0) continue;
      if (-rate * state[int_data[(3 + i_ind)]-1] * float_data[(1 + i_dep)] * time_step <=
          state[int_data[(3 + int_data[0]+ i_dep)]-1]) {
        int elem_id=int_data[(3 + 2*(int_data[0]+int_data[1])+i_elem)];
        //atomicAdd_block(&(jac.production_partials[elem_id]),float_data[(1 + i_dep)] * rate);
        jacobian_add_value_gpu(jac, (unsigned int)int_data[(3 + 2*(int_data[0]+int_data[1])+i_elem)],JACOBIAN_PRODUCTION, float_data[(1 + i_dep)] * rate);
      }
    }
  }
}

__device__
void rxn_gpu_CMAQ_H2O2_calc_deriv_contrib(ModelDataGPU *model_data, TimeDerivativeGPU time_deriv, int *rxn_int_data,
          double *rxn_float_data, double *rxn_env_data, double time_step){
  int *int_data = rxn_int_data;
  double *float_data = rxn_float_data;
  double *state = model_data->grid_cell_state;
  double rate = rxn_env_data[0];
  for (int i_spec=0; i_spec<int_data[0]; i_spec++) rate *= state[int_data[(2 + i_spec)]-1];
  if (rate!=ZERO) {
    int i_dep_var = 0;
    for (int i_spec=0; i_spec<int_data[0]; i_spec++, i_dep_var++) {
      if (int_data[(2 + int_data[0]+int_data[1]+i_dep_var)] < 0) continue;
      time_derivative_add_value_gpu(time_deriv, int_data[(2 + int_data[0]+int_data[1]+i_dep_var)], -rate);
    }
    for (int i_spec=0; i_spec<int_data[1]; i_spec++, i_dep_var++) {
      if (int_data[(2 + int_data[0]+int_data[1]+i_dep_var)] < 0) continue;
      if (-rate*float_data[(7 + i_spec)]*time_step <= state[int_data[(2 + int_data[0]+ i_spec)]-1]) {
        time_derivative_add_value_gpu(time_deriv, int_data[(2 + int_data[0]+int_data[1]+i_dep_var)],rate*float_data[(7 + i_spec)]);
      }
    }
  }
}

__device__
void rxn_gpu_CMAQ_H2O2_calc_jac_contrib(ModelDataGPU *model_data, JacobianGPU jac, int *rxn_int_data,
          double *rxn_float_data, double *rxn_env_data, double time_step){
  int *int_data = rxn_int_data;
  double *float_data = rxn_float_data;
  double *state = model_data->grid_cell_state;
  int i_elem = 0;
  for (int i_ind = 0; i_ind < int_data[0]; i_ind++) {
    double rate = rxn_env_data[0];
    for (int i_spec = 0; i_spec < int_data[0]; i_spec++)
      if (i_ind != i_spec) rate *= state[int_data[(2 + i_spec)]-1];
    for (int i_dep = 0; i_dep < int_data[0]; i_dep++, i_elem++) {
      if (int_data[(2 + 2*(int_data[0]+int_data[1])+i_elem)] < 0) continue;
      jacobian_add_value_gpu(jac, (unsigned int)int_data[(2 + 2*(int_data[0]+int_data[1])+i_elem)], JACOBIAN_LOSS,
                         rate);
    }
    for (int i_dep = 0; i_dep < int_data[1]; i_dep++, i_elem++) {
      if (int_data[(2 + 2*(int_data[0]+int_data[1])+i_elem)] < 0) continue;
      if (-rate * state[int_data[(2 + i_ind)]-1] * float_data[(7 + i_dep)] * time_step <=
          state[int_data[(2 + int_data[0]+ i_dep)]-1]) {
        int elem_id=int_data[(2 + 2*(int_data[0]+int_data[1])+i_elem)];
        //atomicAdd_block(&(jac.production_partials[elem_id]),float_data[(7 + i_dep)] * rate);
        jacobian_add_value_gpu(jac, (unsigned int)int_data[(2 + 2*(int_data[0]+int_data[1])+i_elem)], JACOBIAN_PRODUCTION, float_data[(7 + i_dep)] * rate);
      }
    }
  }
}

__device__
void rxn_gpu_CMAQ_OH_HNO3_calc_deriv_contrib(ModelDataGPU *model_data, TimeDerivativeGPU time_deriv, int *rxn_int_data,
          double *rxn_float_data, double *rxn_env_data, double time_step){
  int *int_data = rxn_int_data;
  double *float_data = rxn_float_data;
  double *state = model_data->grid_cell_state;
  double rate = rxn_env_data[0];
  for (int i_spec=0; i_spec<int_data[0]; i_spec++) rate *= state[int_data[(2 + i_spec)]-1];
  if (rate!=0.) {
    int i_dep_var = 0;
    for (int i_spec=0; i_spec<int_data[0]; i_spec++, i_dep_var++) {
      if (int_data[(2 + int_data[0] + int_data[1] + i_dep_var)] < 0) continue;
      time_derivative_add_value_gpu(time_deriv, int_data[(2 + int_data[0] + int_data[1] + i_dep_var)], -rate);
    }
    for (int i_spec=0; i_spec<int_data[1]; i_spec++, i_dep_var++) {
      if (int_data[(2 + int_data[0] + int_data[1] + i_dep_var)] < 0) continue;
      if (-rate*float_data[(11 + i_spec)]*time_step <= state[int_data[(2 + int_data[0] + i_spec)]-1]) {
        time_derivative_add_value_gpu(time_deriv, int_data[(2 + int_data[0] + int_data[1] + i_dep_var)],rate*float_data[(11 + i_spec)]);
      }
    }
  }
}

__device__
void rxn_gpu_CMAQ_OH_HNO3_calc_jac_contrib(ModelDataGPU *model_data, JacobianGPU jac, int *rxn_int_data,
          double *rxn_float_data, double *rxn_env_data, double time_step){
  int *int_data = rxn_int_data;
  double *float_data = rxn_float_data;
  double *state = model_data->grid_cell_state;
  int i_elem = 0;
  for (int i_ind = 0; i_ind < int_data[0]; i_ind++) {
    double rate = rxn_env_data[0];
    for (int i_spec = 0; i_spec < int_data[0]; i_spec++)
      if (i_ind != i_spec) rate *= state[int_data[(2 + i_spec)]-1];
    for (int i_dep = 0; i_dep < int_data[0]; i_dep++, i_elem++) {
      if (int_data[(2 + 2*(int_data[0]+int_data[1]) + i_elem)] < 0) continue;
      jacobian_add_value_gpu(jac, (unsigned int)int_data[(2 + 2*(int_data[0]+int_data[1]) + i_elem)], JACOBIAN_LOSS,
                         rate);
    }
    for (int i_dep = 0; i_dep < int_data[1]; i_dep++, i_elem++) {
      if (int_data[(2 + 2*(int_data[0]+int_data[1]) + i_elem)] < 0) continue;
      if (-rate * state[int_data[(2 + i_ind)]-1] * float_data[(11 + i_dep)] * time_step <=
          state[int_data[(2 + int_data[0] + i_dep)]-1]) {
        int elem_id=int_data[(2 + 2*(int_data[0]+int_data[1]) + i_elem)];
        //atomicAdd_block(&(jac.production_partials[elem_id]), float_data[(11 + i_dep)] * rate);
      jacobian_add_value_gpu(jac, (unsigned int)int_data[(2 + 2*(int_data[0]+int_data[1]) + i_elem)],JACOBIAN_PRODUCTION, float_data[(11 + i_dep)] * rate);
      }
    }
  }
}

__device__
void rxn_gpu_arrhenius_calc_deriv_contrib(ModelDataGPU *model_data, TimeDerivativeGPU time_deriv,
                                int *rxn_int_data, double *rxn_float_data,
                                double *rxn_env_data, double time_step){
  int *int_data = rxn_int_data;
  double *float_data = rxn_float_data;
  double *state = model_data->grid_cell_state;
  double rate = rxn_env_data[0];
  for (int i_spec=0; i_spec<int_data[0]; i_spec++)
    rate *= state[int_data[(2 + i_spec)]-1];
  if (rate!=0.) {
    int i_dep_var = 0;
    for (int i_spec=0; i_spec<int_data[0]; i_spec++, i_dep_var++) {
      if (int_data[2 + int_data[0] + int_data[1] + i_dep_var] < 0) continue;
      time_derivative_add_value_gpu(time_deriv, int_data[2 + int_data[0] + int_data[1] + i_dep_var], -rate);
    }
    for (int i_spec=0; i_spec<int_data[1]; i_spec++, i_dep_var++) {
      if (int_data[2 + int_data[0] + int_data[1] + i_dep_var] < 0) continue;
      // Negative yields are allowed, but prevented from causing negative
      // concentrations that lead to solver failures
      if (-rate*float_data[6+i_spec]*time_step <= state[int_data[(2 + int_data[0] + i_spec)]-1]) {
        time_derivative_add_value_gpu(time_deriv, int_data[2 + int_data[0] + int_data[1] + i_dep_var],rate*float_data[6+i_spec]);
      }
    }
  }
}

__device__
void rxn_gpu_arrhenius_calc_jac_contrib(ModelDataGPU *model_data, JacobianGPU jac, int *rxn_int_data,
          double *rxn_float_data, double *rxn_env_data, double time_step){
  int *int_data = rxn_int_data;
  double *float_data = rxn_float_data;
  double *state = model_data->grid_cell_state;
  int i_elem = 0;
  for (int i_ind = 0; i_ind < int_data[0]; i_ind++) {
    double rate = rxn_env_data[0];
    for (int i_spec = 0; i_spec < int_data[0]; i_spec++)
      if (i_spec != i_ind) rate *= state[int_data[(2 + i_spec)]-1];
    for (int i_dep = 0; i_dep < int_data[0]; i_dep++, i_elem++) {
      if (int_data[2 + 2*(int_data[0]+int_data[1]) + i_elem] < 0) continue;
      jacobian_add_value_gpu(jac, (unsigned int)int_data[2 + 2*(int_data[0]+int_data[1]) + i_elem],
        JACOBIAN_LOSS, rate);
    }
    for (int i_dep = 0; i_dep < int_data[1]; i_dep++, i_elem++) {
      if (int_data[2 + 2*(int_data[0]+int_data[1]) + i_elem] < 0) continue;
      // Negative yields are allowed, but prevented from causing negative
      // concentrations that lead to solver failures
      if (-rate * state[int_data[(2 + i_ind)]-1] * float_data[6+i_dep] * time_step <=
        state[int_data[(2 + int_data[0] + i_dep)]-1]) {
        int elem_id=int_data[2 + 2*(int_data[0]+int_data[1]) + i_elem];
        //atomicAdd_block(&(jac.production_partials[elem_id]), float_data[6+i_dep] * rate);
        jacobian_add_value_gpu(jac, (unsigned int)int_data[2 + 2*(int_data[0]+int_data[1]) + i_elem],JACOBIAN_PRODUCTION, float_data[6+i_dep] * rate);
      }
    }
  }
}

}

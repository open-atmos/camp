/* Copyright (C) 2021 Barcelona Supercomputing Center and University of
 * Illinois at Urbana-Champaign
 * SPDX-License-Identifier: MIT
 */

extern "C" {
#include "time_derivative_gpu.h"
#include <math.h>
#include <stdio.h>

#ifdef __CUDA_ARCH__
__device__
#endif
void time_derivative_reset_gpu(TimeDerivativeGPU time_deriv) {

#ifdef __CUDA_ARCH__
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i<time_deriv.num_spec){
    time_deriv.production_rates[i] = 0.0;
    time_deriv.loss_rates[i] = 0.0;
    //time_deriv.production_rates[i] = 0.00001;
    //time_deriv.loss_rates[i] = 0.00001;
  }
#else
  for (unsigned int i_spec = 0; i_spec < time_deriv.num_spec; ++i_spec) {
    time_deriv.production_rates[i_spec] = 0.0;
    time_deriv.loss_rates[i_spec] = 0.0;
  }
#endif

}

__device__
void time_derivative_output_gpu(TimeDerivativeGPU time_deriv, double *dest_array,
                            double *deriv_est, unsigned int output_precision) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i<time_deriv.num_spec){
    double *r_p = time_deriv.production_rates;
    double *r_l = time_deriv.loss_rates;
    if (r_p[i] + r_l[i] != 0.0) {
      if (deriv_est) {
        double scale_fact;
        scale_fact =
            1.0 / (r_p[i] + r_l[i]) /
            (1.0 / (r_p[i] + r_l[i]) + MAX_PRECISION_LOSS / fabs(r_p[i]- r_l[i]));
          dest_array[i] = scale_fact * (r_p[i] - r_l[i]) + (1.0 - scale_fact) * (deriv_est[i]);
      } else {
        dest_array[i] = r_p[i] - r_l[i];
      }
    } else {
      dest_array[i] = 0.0;
    }
  }
}

/*
  // Threshhold for precisition loss in rate calculations
  #define MAX_PRECISION_LOSS 1.0e-14
  if(i<deriv_data.num_spec){
    double *r_p = deriv_data.production_rates;
    double *r_l = deriv_data.loss_rates;
    if (r_p[i] + r_l[i] != 0.0) {
      if (md->J_tmp) {
        double scale_fact;
        scale_fact = 1.0 / (r_p[i] + r_l[i]) /
            (1.0 / (r_p[i] + r_l[i]) + MAX_PRECISION_LOSS / fabs(r_p[i]- r_l[i]));
        yout[i] =
        scale_fact * (r_p[i] - r_l[i]) + (1.0 - scale_fact) * (md->J_tmp[i]);
      } else {
        yout[i] = r_p[i] - r_l[i];
      }
    } else {
      yout[i] = 0.0;
    }
  }
  */

#ifdef __CUDA_ARCH__
__device__
#endif
void time_derivative_add_value_gpu(TimeDerivativeGPU time_deriv, unsigned int spec_id,
                               double rate_contribution) {
#ifdef __CUDA_ARCH__
  if (rate_contribution > 0.0) {
    atomicAdd_block(&(time_deriv.production_rates[spec_id]),rate_contribution);
  } else {
    atomicAdd_block(&(time_deriv.loss_rates[spec_id]),-rate_contribution);
  }
#else
  if (rate_contribution > 0.0) {
    time_deriv.production_rates[spec_id] += rate_contribution;
  } else {
    time_deriv.loss_rates[spec_id] += -rate_contribution;
  }
#endif
}

}
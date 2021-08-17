/* Copyright (C) 2019 Matthew Dawson
 * Licensed under the GNU General Public License version 2 or (at your
 * option) any later version. See the file COPYING for details.
 *
 * Functions of the time derivative structure
 *
 */
/** \file
 * \brief Functions of the time derivative structure
 */

extern "C" {
#include "time_derivative_gpu.h"
#include <math.h>
#include <stdio.h>

int time_derivative_initialize_gpu(SolverData *sd) {

  ModelData *md = &(sd->model_data);

  int num_spec = md->n_per_cell_dep_var*md->n_cells;
  //Pre-allocation related arrays (internal struct arrays are mandatory)

  ModelDataGPU *mGPU = &sd->mGPU;

  cudaMalloc((void **) &(mGPU->production_rates),num_spec*sizeof(mGPU->production_rates));
  cudaMalloc((void **) &(mGPU->loss_rates),num_spec*sizeof(mGPU->loss_rates));

  return 1;
}

#ifdef __CUDA_ARCH__
__host__ __device__
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

#ifdef __CUDA_ARCH__
__host__ __device__
#endif
void time_derivative_output_gpu(TimeDerivativeGPU time_deriv, double *dest_array,
                            double *deriv_est, unsigned int output_precision) {

#ifdef PMC_DEBUG
  time_deriv.last_max_loss_precision = 1.0;
#endif

#ifdef __CUDA_ARCH__

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i<time_deriv.num_spec){

    double *r_p = time_deriv.production_rates;
    double *r_l = time_deriv.loss_rates;

    //dest_array[i] = 0.1;
    //dest_array[i] = r_p[i];
    if (r_p[i] + r_l[i] != 0.0) {
      if (deriv_est) {
        double scale_fact;
        scale_fact =
            1.0 / (r_p[i] + r_l[i]) /
            (1.0 / (r_p[i] + r_l[i]) + MAX_PRECISION_LOSS / fabs(r_p[i]- r_l[i]));
          dest_array[i] =
          scale_fact * (r_p[i] - r_l[i]) + (1.0 - scale_fact) * (deriv_est[i]);
        //dest_array[i] = 0.1;
      } else {
        dest_array[i] = r_p[i] - r_l[i];
        //dest_array[i] = 0.2;
      }
    } else {
      dest_array[i] = 0.0;
      //dest_array[i] = 0.000000003;
      //dest_array[i] = r_l[i];
      //dest_array[i] = r_p[i];
    }
    //dest_array[i] = r_p[i];
    //dest_array[i] = r_l[i];
  }

#else

  double *r_p = time_deriv.production_rates;
  double *r_l = time_deriv.loss_rates;

  for (unsigned int i_spec = 0; i_spec < time_deriv.num_spec; ++i_spec) {
    double prec_loss = 1.0;
    if (*r_p + *r_l != 0.0) {
      if (deriv_est) {
        double scale_fact;
        scale_fact =
            1.0 / (*r_p + *r_l) /
            (1.0 / (*r_p + *r_l) + MAX_PRECISION_LOSS / fabsl(*r_p - *r_l));
        *dest_array =
            scale_fact * (*r_p - *r_l) + (1.0 - scale_fact) * (*deriv_est);
      } else {
        *dest_array = *r_p - *r_l;
      }
#ifdef PMC_DEBUG
      if (*r_p != 0.0 && *r_l != 0.0) {
        prec_loss = *r_p > *r_l ? 1.0 - *r_l / *r_p : 1.0 - *r_p / *r_l;
        if (prec_loss < time_deriv.last_max_loss_precision)
          time_deriv.last_max_loss_precision = prec_loss;
      }
#endif
    } else {
      *dest_array = 0.0;
    }
    ++r_p;
    ++r_l;
    ++dest_array;
    if (deriv_est) ++deriv_est;
#ifdef PMC_DEBUG
    if (output_precision == 1) {
      printf("\nspec %d prec_loss %le", i_spec, -log(prec_loss) / log(2.0));
    }
#endif
  }

#endif

}

#ifdef __CUDA_ARCH__
__host__ __device__
#endif
void time_derivative_add_value_gpu(TimeDerivativeGPU time_deriv, unsigned int spec_id,
                               double rate_contribution) {
#ifdef __CUDA_ARCH__
  if (rate_contribution > 0.0) {
    atomicAdd_block(&(time_deriv.production_rates[spec_id]),rate_contribution);
    //atomicAdd(&(time_deriv.production_rates[spec_id]),rate_contribution);
    //atomicAdd(&(time_deriv.production_rates[spec_id]),0.2); //debug
  } else {
    atomicAdd_block(&(time_deriv.loss_rates[spec_id]),-rate_contribution);
    //atomicAdd(&(time_deriv.loss_rates[spec_id]),-rate_contribution);
    //atomicAdd(&(time_deriv.loss_rates[spec_id]),-0.1); //debug
  }
#else
  if (rate_contribution > 0.0) {
    time_deriv.production_rates[spec_id] += rate_contribution;
  } else {
    time_deriv.loss_rates[spec_id] += -rate_contribution;
  }
#endif
}

#ifdef PMC_DEBUG
double time_derivative_max_loss_precision(TimeDerivativeGPU time_deriv) {
  return -log(time_deriv.last_max_loss_precision) / log(2.0);
}
#endif

void time_derivative_free_gpu(TimeDerivativeGPU time_deriv) {
#ifdef __CUDA_ARCH__
  cudaFree(time_deriv.production_rates);
  cudaFree(time_deriv.loss_rates);
#else
  free(time_deriv.production_rates);
  free(time_deriv.loss_rates);
#endif
}

}

//todo merge with default time_derivative "class" using .cpp file
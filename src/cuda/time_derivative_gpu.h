/* Copyright (C) 2021 Barcelona Supercomputing Center and University of
 * Illinois at Urbana-Champaign
 * SPDX-License-Identifier: MIT
 */

#ifndef TIME_DERIVATIVE_H_GPU
#define TIME_DERIVATIVE_H_GPU

#include <math.h>
#include <stdlib.h>
#include "camp_gpu_solver.h"
#include <cuda.h>

// Threshhold for precisition loss in rate calculations
#define MAX_PRECISION_LOSS 1.0e-14

/** \brief Reset the derivative
 *
 * \param time_deriv TimeDerivativeGPU object
 */
#ifdef __CUDA_ARCH__
__device__
#endif
void time_derivative_reset_gpu(TimeDerivativeGPU time_deriv);

/** \brief Output the current derivative array
 *
 * \param time_deriv TimeDerivativeGPU object
 * \param dest_array Pointer to the destination array
 * \param deriv_est Pointer to an estimate of the derivative array (optional)
 * \param output_precision Output the estimated loss of precision for each
 * species if output_precision == 1
 */
#ifdef __CUDA_ARCH__
__device__
#endif
void time_derivative_output_gpu(TimeDerivativeGPU deriv_data, double *yout,
                            double *J_tmp, unsigned int output_precision);

/** \brief Add a contribution to the time derivative
 *
 * \param time_deriv TimeDerivativeGPU object
 * \param spec_id Index of the species to update rates for
 * \param rate_contribution Value to add to the time derivative for speces
 * spec_id
 */
#ifdef __CUDA_ARCH__
__device__
#endif
void time_derivative_add_value_gpu(TimeDerivativeGPU time_deriv, unsigned int spec_id,
                               double rate_contribution);

#endif

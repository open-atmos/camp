/* Copyright (C) 2019 Matthew Dawson
 * Licensed under the GNU General Public License version 2 or (at your
 * option) any later version. See the file COPYING for details.
 *
 * Header for the time derivative structure and related functions
 *
 */
/** \file
 * \brief Header for the time derivative structure and related functions
 */
#ifndef TIME_DERIVATIVE_H_GPU
#define TIME_DERIVATIVE_H_GPU

#include <math.h>
#include <stdlib.h>
#include "camp_gpu_solver.h"

// Threshhold for precisition loss in rate calculations
#define MAX_PRECISION_LOSS 1.0e-14

/** \brief Initialize the derivative
 *
 * \param time_deriv Pointer to the TimeDerivativeGPU object
 * \param num_spec Number of species to include in the derivative
 * \return Flag indicating whether the derivative was sucessfully initialized
 *         (0 = false; 1 = true)
 */
int time_derivative_initialize_gpu(SolverData *sd, unsigned int num_spec);

/** \brief Reset the derivative
 *
 * \param time_deriv TimeDerivativeGPU object
 */
#ifdef __CUDA_ARCH__
__host__ __device__
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
__host__ __device__
#endif
void time_derivative_output_gpu(TimeDerivativeGPU time_deriv, double *dest_array,
                            double *deriv_est, unsigned int output_precision);

/** \brief Add a contribution to the time derivative
 *
 * \param time_deriv TimeDerivativeGPU object
 * \param spec_id Index of the species to update rates for
 * \param rate_contribution Value to add to the time derivative for speces
 * spec_id
 */
#ifdef __CUDA_ARCH__
__host__ __device__
#endif
void time_derivative_add_value_gpu(TimeDerivativeGPU time_deriv, unsigned int spec_id,
                               double rate_contribution);

#ifdef PMC_DEBUG
/** \brief Maximum loss of precision at the last output of the derivative
 *         in bits
 *
 * \param time_deriv TimeDerivativeGPU object
 * \return maximum loss of precision
 */
double time_derivative_max_loss_precision(TimeDerivativeGPU time_deriv);
#endif

/** \brief Free memory associated with a TimeDerivativeGPU
 *
 * \param time_deriv TimeDerivativeGPU object
 */
void time_derivative_free_gpu(TimeDerivativeGPU time_deriv);

#endif

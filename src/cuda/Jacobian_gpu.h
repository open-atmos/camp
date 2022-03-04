/* Copyright (C) 2021 Barcelona Supercomputing Center and University of
 * Illinois at Urbana-Champaign
 * SPDX-License-Identifier: MIT
 */

#ifndef JACOBIAN_H__GPU
#define JACOBIAN_H__GPU

#include <math.h>
#include <stdlib.h>
#include "camp_gpu_solver.h"

// Flags for specifying production or loss elements
#define JACOBIAN_PRODUCTION 0
#define JACOBIAN_LOSS 1


/** \brief Initialize the JacobianGPU
 *
 * \param jac Pointer to the JacobianGPU object
 * \param num_spec Number of species
 * \param jac_struct Dense matrix of flags indicating whether an element is
 *                   (1) potentially non-zero or (0) not.
 * \return Flag indicating whether the derivative was successfully initialized
 *         (0 = false; 1 = true)
 */
int jacobian_initialize_gpu(SolverData *sd);

/** \brief Reset the JacobianGPU
 *
 * \param jac JacobianGPU matrix
 */
#ifdef __CUDA_ARCH__
__host__ __device__
#endif
void jacobian_reset_gpu(JacobianGPU jac);

/** \brief Output the JacobianGPU
 *
 * \param jac JacobianGPU object
 * \param dest_array Pointer to the array to save JacobianGPU data to
 */
#ifdef __CUDA_ARCH__
__host__ __device__
#endif
void jacobian_output_gpu(JacobianGPU jac, double *dest_array);

/** \brief Add a contribution to the JacobianGPU
 *
 * \param jac JacobianGPU object
 * \param elem_id Index of the element to update in the data array
 * \param prod_or_loss Flag indicating whether to update the (0) production or
 *                          (1) loss elements
 * \param jac_contribution Value to add to the JacobianGPU element
 *                         (contributions to loss elements should be positive if
 *                         the contribution increases the loss)
 */
#ifdef __CUDA_ARCH__
__host__ __device__
#endif
void jacobian_add_value_gpu(JacobianGPU jac, unsigned int elem_id,
                        unsigned int prod_or_loss,
                        double jac_contribution);


#endif

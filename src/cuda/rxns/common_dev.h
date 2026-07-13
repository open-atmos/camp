/* Copyright (C) 2021 Barcelona Supercomputing Center and University of
 * Illinois at Urbana-Champaign
 * SPDX-License-Identifier: MIT
 */

/** \file
 * \brief Functions common to the aerosol and gas phase solver functions
 */
#ifndef COMMON_DEV_H_
#define COMMON_DEV_H_

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>

#include "../../camp_common.h"


// Debug mode: remove atomic operations, useful to see that atomicAdd generate
// random results, slightly varying the accuracy of the results
#ifdef IS_DEBUG_MODE_removeAtomic

__device__ void time_derivative_add_value_gpu(TimeDerivativeGPU time_deriv,
                                              unsigned int spec_id,
                                              double rate_contribution) {
  if (rate_contribution > 0.0) {
    time_deriv.production_rates[spec_id] += rate_contribution;
  } else {
    time_deriv.loss_rates[spec_id] += -rate_contribution;
  }
}

__device__ void jacobian_add_value_gpu(JacobianGPU jac, unsigned int elem_id,
                                       int prod_or_loss,
                                       double jac_contribution) {
  if (prod_or_loss == JACOBIAN_PRODUCTION) {
    jac.production_partials[elem_id] += jac_contribution;
  } else {
    jac.loss_partials[elem_id] += jac_contribution;
  }
}

#else

/**
 * @brief Adds a value to the time derivative of a specific species on the GPU.
 *
 * This function adds a rate contribution to the time derivative of a specific
 * species on the GPU. It uses atomic operations to ensure correct
 * synchronization when multiple threads are accessing the same memory location.
 *
 * @param time_deriv The time derivative structure containing the production and
 * loss rates.
 * @param spec_id The ID of the species.
 * @param rate_contribution The rate contribution to be added.
 */
__device__ void time_derivative_add_value_gpu(TimeDerivativeGPU time_deriv,
                                              unsigned int spec_id,
                                              double rate_contribution) {
  // WARNING: Atomicadd is not desirable, because it leads to small deviations
  // in the results, even when scaling the number of data computed in the GPU It
  // would be desirable to remove it
  if (rate_contribution > 0.0) {
    atomicAdd_block(&(time_deriv.production_rates[spec_id]), rate_contribution);
  } else {
    atomicAdd_block(&(time_deriv.loss_rates[spec_id]), -rate_contribution);
  }
}

/**
 * @brief Adds a value to the Jacobian matrix on the GPU.
 *
 * This function adds a contribution to the Jacobian matrix on the GPU for a
 * specific element. The contribution can be either a production or a loss,
 * determined by the `prod_or_loss` parameter. The value of the contribution is
 * specified by the `jac_contribution` parameter.
 *
 * @param jac The Jacobian matrix on the GPU.
 * @param elem_id The ID of the element in the Jacobian matrix.
 * @param prod_or_loss Specifies whether the contribution is a production or a
 * loss. Use the `JACOBIAN_PRODUCTION` constant for production and the
 * `JACOBIAN_LOSS` constant for loss.
 * @param jac_contribution The value of the contribution to be added to the
 * Jacobian matrix.
 */
__device__ void jacobian_add_value_gpu(JacobianGPU jac, unsigned int elem_id,
                                       int prod_or_loss,
                                       double jac_contribution) {
  if (prod_or_loss == JACOBIAN_PRODUCTION) {
    atomicAdd_block(&(jac.production_partials[elem_id]), jac_contribution);
  } else {
    atomicAdd_block(&(jac.loss_partials[elem_id]), jac_contribution);
  }
}

#endif

#endif // COMMON_DEV_H_
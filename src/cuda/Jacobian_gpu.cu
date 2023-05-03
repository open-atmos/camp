/* Copyright (C) 2021 Barcelona Supercomputing Center and University of
 * Illinois at Urbana-Champaign
 * SPDX-License-Identifier: MIT
 */

extern "C" {
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "Jacobian_gpu.h"

#define BUFFER_SIZE 10
#define SMALL_NUMBER 1e-90

#ifdef __CUDA_ARCH__
__device__
#endif
void jacobian_add_value_gpu(JacobianGPU jac, unsigned int elem_id,
                            unsigned int prod_or_loss,
                            double jac_contribution) {
#ifdef __CUDA_ARCH__
  if (prod_or_loss == JACOBIAN_PRODUCTION)
    atomicAdd_block(&(jac.production_partials[elem_id]),jac_contribution);
  if (prod_or_loss == JACOBIAN_LOSS)
    atomicAdd_block(&(jac.loss_partials[elem_id]),jac_contribution);
#else
  if (prod_or_loss == JACOBIAN_PRODUCTION)
    jac.production_partials[elem_id] += jac_contribution;
  if (prod_or_loss == JACOBIAN_LOSS)
    jac.loss_partials[elem_id] += jac_contribution;
#endif
  //check_isnanld(&jac_contribution,1,"post jacobian_add_value jac_contribution");
}

}

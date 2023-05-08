/* Copyright (C) 2021 Barcelona Supercomputing Center and University of
 * Illinois at Urbana-Champaign
 * SPDX-License-Identifier: MIT
 */

extern "C" {
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "Jacobian_gpu.h"

void jacobian_add_value_gpu(JacobianGPU jac, unsigned int elem_id,
                            unsigned int prod_or_loss,
                            double jac_contribution) {
  if (prod_or_loss == JACOBIAN_PRODUCTION)
    atomicAdd_block(&(jac.production_partials[elem_id]),jac_contribution);
  if (prod_or_loss == JACOBIAN_LOSS)
    atomicAdd_block(&(jac.loss_partials[elem_id]),jac_contribution);
}

}

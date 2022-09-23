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

#ifdef __CUDA_ARCH__
__host__ __device__
#endif
void jacobian_add_value_gpu(JacobianGPU jac, unsigned int elem_id,
                        unsigned int prod_or_loss,
                        double jac_contribution);


#endif

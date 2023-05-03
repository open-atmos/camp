/* Copyright (C) 2021 Barcelona Supercomputing Center and University of
 * Illinois at Urbana-Champaign
 * SPDX-License-Identifier: MIT
 */

#ifndef SUB_MODEL_SOLVER_H
#define SUB_MODEL_SOLVER_H
#include "../camp_gpu_solver.h"

#ifdef __CUDA_ARCH__
__device__
#endif
double sub_model_gpu_get_parameter_value(ModelDataGPU *model_data, int parameter_id);

#endif

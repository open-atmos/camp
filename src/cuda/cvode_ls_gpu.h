/* Copyright (C) 2021 Barcelona Supercomputing Center and University of
 * Illinois at Urbana-Champaign
 * SPDX-License-Identifier: MIT
 */

#ifndef CVODE_gpu_ls_SOLVER_H_
#define CVODE_gpu_ls_SOLVER_H_

#include <cuda.h>
#include "../camp_common.h"

int cvInitialSetup_gpu(CVodeMem cv_mem);
int cvHin_gpu(CVodeMem cv_mem, realtype tout);
int cvRcheck1_gpu(CVodeMem cv_mem);
int cvRcheck2_gpu(CVodeMem cv_mem);
int cvRcheck3_gpu(CVodeMem cv_mem);
int cvHandleFailure_gpu(CVodeMem cv_mem, int flag);

#endif

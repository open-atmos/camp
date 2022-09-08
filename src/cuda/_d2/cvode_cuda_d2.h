/* Copyright (C) 2021 Barcelona Supercomputing Center and University of
* Illinois at Urbana-Champaign
* SPDX-License-Identifier: MIT
*/

#ifndef CVODE_CUDA_d2_H_
#define CVODE_CUDA_d2_H_

#include<iostream>
#include<cuda.h>
#include<cuda_runtime.h>
#include<cuda_runtime_api.h>

__device__ void solveBcgCuda_d2_cvode_cuda(
       double *dA, int *djA, int *diA, double *dx, double *dtempv //Input data
       ,int nrows, int blocks, int n_shr_empty, int maxIt
       ,int n_cells, double tolmax, double *ddiag //Init variables
       ,double *dr0, double *dr0h, double *dn0, double *dp0
       ,double *dt, double *ds, double *dAx2, double *dy, double *dz// Auxiliary vectors
);

#endif
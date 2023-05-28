/* Copyright (C) 2021 Barcelona Supercomputing Center and University of
 * Illinois at Urbana-Champaign
 * SPDX-License-Identifier: MIT
 */

#ifndef ITSOLVERGPU_H
#define ITSOLVERGPU_H

#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<iostream>
#include"libsolv.h"
#include<cuda.h>
#include<cuda_runtime.h>
#include<cuda_runtime_api.h>

extern "C" {
#include "../camp_solver.h"
}

#define ONECELL 1
#define MULTICELLS 2
#define  BLOCKCELLSN 3
#define BLOCKCELLS1 4
#define BLOCKCELLSNHALF 5
#define BCG_MAXIT 1000
#define BCG_TOLMAX 1.0E-30

void read_options_bcg(ModelDataGPU *mGPU);
void createLinearSolver(SolverData *sd);
__device__
void cudaDeviceswapCSC_CSR1ThreadBlock(int n_row, int n_col, int* Ap, int* Aj, double* Ax, int* BpGlobal, int* Bi, double* Bx);
__device__
void cudaDeviceswapCSC_CSR(int n_row, int n_col, int* Ap, int* Aj, double* Ax, int* BpGlobal, int* Bi, double* Bx);
__global__
void cudaGlobalswapCSC_CSR(int n_row, int n_col, int* Ap, int* Aj, double* Ax, int* Cp, int* Ci, double* Cx);
__device__ void cudaCVODESwapCSC_CSRBCG(ModelDataGPU *md, ModelDataVariable *dmdv, double* dA);
void swapCSC_CSR(int n_row, int n_col, int* Ap, int* Aj, double* Ax, int* Bp, int* Bi, double* Bx);
void swapCSC_CSR_BCG(SolverData *sd);
void swapCSC_CSR_Indices(SolverData *sd);
void solveBCG(SolverData *sd, double *dA, int *djA, int *diA, double *dx, double *dtempv);
void solveBCGBlocks(SolverData *sd, double *dA, int *djA, int *diA, double *dx, double *dtempv);

#endif
/* Copyright (C) 2021 Barcelona Supercomputing Center and University of
 * Illinois at Urbana-Champaign
 * SPDX-License-Identifier: MIT
 */

#ifndef ITSOLVERGPU_H
#define ITSOLVERGPU_H

#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include"libsolv.h"

extern "C" {
#include "../camp_solver.h"
}

#define ONECELL 1
#define MULTICELLS 2
#define  BLOCKCELLSN 3
#define BLOCKCELLS1 4
#define BLOCKCELLSNHALF 5

void createLinearSolver(SolverData *sd);
__device__
void cudaDeviceswapCSC_CSR1ThreadBlock(int n_row, int n_col, int* Ap, int* Aj, double* Ax, int* BpGlobal, int* Bi, double* Bx);
__device__
void cudaDeviceswapCSC_CSR(int n_row, int n_col, int* Ap, int* Aj, double* Ax, int* BpGlobal, int* Bi, double* Bx);
__global__
void cudaGlobalswapCSC_CSR(int n_row, int n_col, int* Ap, int* Aj, double* Ax, int* Cp, int* Ci, double* Cx);
void swapCSC_CSR_BCG(SolverData *sd);
void swapCSC_CSR_ODE(SolverData *sd);
void solveBCG(SolverData *sd, double *dA, int *djA, int *diA, double *dx, double *dtempv);
void solveBCGBlocks(SolverData *sd, double *dA, int *djA, int *diA, double *dx, double *dtempv);
__device__ void solveBcgCudaDevice(
        double *dA, int *djA, int *diA, double *dx, double *dtempv //Input data
        ,int nrows, int blocks, int n_shr_empty, int maxIt, int mattype
        ,int n_cells, double tolmax, double *ddiag //Init variables
        ,double *dr0, double *dr0h, double *dn0, double *dp0
        ,double *dt, double *ds, double *dAx2, double *dy, double *dz// Auxiliary vectors
#ifdef CAMP_DEBUG_GPU
        ,int *it_pointer
#endif
);

void free_itsolver(SolverData *sd);


#endif
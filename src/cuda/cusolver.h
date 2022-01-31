

#ifndef PARTMC_CUSOLVER_H
#define PARTMC_CUSOLVER_H

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include<cuda.h>
#include<cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cusolverSp.h>
#include <cusparse.h>
#include<math.h>
#include<iostream>

#include"libsolv.h"

#include "itsolver_gpu.h"

//extern "C" {
//#include "../camp_solver.h"
//}

//Time derivative for solver species
typedef struct {
    int test;
    cusolverSpHandle_t handle;
    cusparseMatDescr_t descrA;
    csrqrInfo_t info;
    void *buffer_qr;

} CuSolver;

void createCuSolver(SolverData *sd);
void solveCuSolver(SolverData *sd);


#endif //PARTMC_CUSOLVER_H

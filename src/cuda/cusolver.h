

#ifndef PARTMC_CUSOLVER_H
#define PARTMC_CUSOLVER_H

#include <cusolverSp.h>
#include <cuda_runtime_api.h>
#include <cusparse.h>

#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<iostream>
#include"libsolv.h"
#include<cuda.h>
#include<cuda_runtime.h>
#include<cuda_runtime_api.h>
#include "itsolver_gpu.h"

//extern "C" {
//#include "../camp_solver.h"
//}

//Time derivative for solver species
typedef struct {
    cusolverSpHandle_t handle;
    int test;

} CuSolver;

void createCuSolver(SolverData *sd);



#endif //PARTMC_CUSOLVER_H

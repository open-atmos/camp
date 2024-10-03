/* Copyright (C) 2021 Barcelona Supercomputing Center and University of
 * Illinois at Urbana-Champaign
 * SPDX-License-Identifier: MIT
 */

/*
Interface between GPU solver and CAMP .c interface

Solver extracted from CVODE 3.4 version (BDF method) with the BiConjugate
Gradient algorithm as the linear solver.
*/

#ifndef CVODE_gpu_SOLVER_H_
#define CVODE_gpu_SOLVER_H_

#include <cuda.h>
#include "../camp_common.h"

void init_solve_gpu(SolverData *sd);  // Initialize GPU solver
int cudaCVode(void *cvode_mem, double t_final, N_Vector yout, SolverData *sd,
              double t_initial);                 // Solve
void solver_get_statistics_gpu(SolverData *sd);  // Get statistics
void free_gpu_cu(SolverData *sd);                // Deallocate

#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))
static void HandleError(cudaError_t err, const char *file, int line) {
  if (err != cudaSuccess) {
    printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
    exit(EXIT_FAILURE);
  }
}

#endif

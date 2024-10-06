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

/*
 * cvHandleFailure
 *
 * This routine prints error messages for all cases of failure by
 * cvHin and cvStep.
 * It returns to CVode the value that CVode is to return to the user.
 */
static void cudacvHandleFailure(int flag, int cell) {
  switch (flag) {
    case CV_ERR_FAILURE:
      printf("CVODE ERROR: %s at cell %d\n", MSGCV_ERR_FAILS, cell);
      break;
    case CV_CONV_FAILURE:
      printf("CVODE ERROR: %s at cell %d\n", MSGCV_CONV_FAILS, cell);
      break;
    case CV_LSETUP_FAIL:
      printf("CVODE ERROR: %s at cell %d\n", MSGCV_SETUP_FAILED, cell);
      break;
    case CV_LSOLVE_FAIL:
      printf("CVODE ERROR: %s at cell %d\n", MSGCV_SOLVE_FAILED, cell);
      break;
    case CV_RHSFUNC_FAIL:
      printf("CVODE ERROR: %s at cell %d\n", MSGCV_RHSFUNC_FAILED, cell);
      break;
    case CV_UNREC_RHSFUNC_ERR:
      printf("CVODE ERROR: %s at cell %d\n", MSGCV_RHSFUNC_UNREC, cell);
      break;
    case CV_REPTD_RHSFUNC_ERR:
      printf("CVODE ERROR: %s at cell %d\n", MSGCV_RHSFUNC_REPTD, cell);
      break;
    case CV_RTFUNC_FAIL:
      printf("CVODE ERROR: %s at cell %d\n", MSGCV_RTFUNC_FAILED, cell);
      break;
    case CV_TOO_CLOSE:
      printf("CVODE ERROR: %s at cell %d\n", MSGCV_TOO_CLOSE, cell);
      break;
    default:
      printf("CVODE ERROR: Unknown at cell %d\n", cell);
  }
}

#endif

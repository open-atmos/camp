/* Copyright (C) 2021 Barcelona Supercomputing Center and University of
 * Illinois at Urbana-Champaign
 * SPDX-License-Identifier: MIT
 */

/** \file
 * \brief All of the host functions interfacing between the CPU and GPU
 * as well as the main (__global__) kernel
 *
 * Solver extracted from CVODE 3.4 version (BDF method) with the BiConjugate
 * Gradient algorithm as the linear solver.
 */

#ifndef CVODE_gpu_SOLVER_H_
#define CVODE_gpu_SOLVER_H_

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#ifdef CAMP_USE_MPI
#include <mpi.h>
#endif

#ifdef __CUDACC__
extern "C" {
#endif
#include "../camp_common.h"
#include "../camp_solver.h"
#ifdef __CUDACC__
}
#endif

#define LOAD_BALANCE

#ifdef TRACE_CPUGPU
// Used for profiling traces with Nsight Systems. It allows to add a tag to the
// CPU code.
#include "nvToolsExt.h"
#endif

#include <unistd.h>
#include <math.h>

/**
 * @brief Calculates the next power of two for a given integer.
 *
 * @param v The input integer.
 * @return The next power of two.
 */
static inline int nextPowerOfTwo(int v) {
  v--;
  v |= v >> 1;
  v |= v >> 2;
  v |= v >> 4;
  v |= v >> 8;
  v |= v >> 16;
  v++;
  return v;
}

/**
 * @brief Initialize GPU solver
 * 
 * @param sd 
 */
#ifdef __CUDACC__
extern "C" {
#endif
void init_solve_gpu(SolverData *sd, int max_steps, int max_conv_fails);
#ifdef __CUDACC__
}
#endif

/**
 * @brief Runs the CVode solver on the GPU.
 *
 * @param t_initial The initial time.
 * @param mGPU The pointer to the ModelDataGPU structure.
 * @param blocks The number of blocks for the kernel launch.
 * @param threads_block The number of threads per block for the kernel launch.
 * @param stream The CUDA stream to associate with the kernel launch.
 */
#ifdef __CUDACC__
extern "C" {
#endif
void cvodeRun(double t_initial, ModelDataGPU *mGPU, int blocks,
              int threads_block, cudaStream_t stream);  // launch CVode kernel
#ifdef __CUDACC__
}
#endif

/**
 * @brief Executes the CVode solver on GPU and CPU.
 *
 * This function performs the CVode solver on the GPU and CPU. It transfers
 * the necessary data from the host to the device, solves the system of
 * equations, and transfers the results back to the host. Simultaneously, it
 * solves a part of the cells in the CPU, managed through the variable
 * "load_gpu". It also handles load balancing between the CPU and GPU if the
 * load_balance flag is set. Details explained on C. Guzman PhD Thesis - Chapter
 * 6
 *
 * @param cvode_mem A pointer to the CVode memory structure.
 * @param t_final The final time for the solver.
 * @param yout The output vector where the solution will be stored.
 * @param sd A pointer to the SolverData structure.
 * @param t_initial The initial time for the solver.
 * @return An integer representing the status of the solver.
 */
#ifdef __CUDACC__
extern "C" {
#endif
int cudaCVode(double t_final, SolverData *sd, double t_initial,
              int is_get_solver_stats, int *status_code, int *solver_flag,
              int *num_steps);  // Solve
#ifdef __CUDACC__
}
#endif

/**
 * @brief Get statistics
 * 
 * @param sd 
 */
#ifdef __CUDACC__
extern "C" {
#endif
void solver_get_profile_gpu(SolverData *sd);
#ifdef __CUDACC__
}
#endif

/**
 * @brief Deallocate GPU memory
 * 
 * @param sd 
 */
#ifdef __CUDACC__
extern "C" {
#endif
void free_gpu_cu(SolverData *sd);
#ifdef __CUDACC__
}
#endif

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

#endif  // CVODE_gpu_SOLVER_H_

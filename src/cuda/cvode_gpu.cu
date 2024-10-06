/* Copyright (C) 2021 Barcelona Supercomputing Center and University of
 * Illinois at Urbana-Champaign
 * SPDX-License-Identifier: MIT
 */
#include "cvode_cuda.h"

#define LOAD_BALANCE

extern "C" {
#include "cvode_gpu.h"
}
#ifdef TRACE_CPUGPU
#include "nvToolsExt.h"
#endif

#ifdef CAMP_USE_MPI
#include <mpi.h>
#endif

#include <unistd.h>

int cudaCVode(void *cvode_mem, double t_final, N_Vector yout, SolverData *sd,
              double t_initial) {
  ModelDataGPU *mGPU = sd->mGPU;
  ModelData *md = &(sd->model_data);
  int n_cells = md->n_cells_gpu;
  cudaStream_t stream;  // Variable for asynchronous execution of the GPU
  cudaStreamCreate(&stream);
#ifdef LOAD_BALANCE
  if (sd->load_balance == 1) {
    cudaEventRecord(sd->startGPU, stream);  // Start GPU timer
  }
#endif
  // Transfer data to GPU
  cudaMemcpyAsync(mGPU->rxn_env_data, md->rxn_env_data,
                  md->n_rxn_env_data * n_cells * sizeof(double),
                  cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(mGPU->state, md->total_state,
                  md->n_per_cell_state_var * n_cells * sizeof(double),
                  cudaMemcpyHostToDevice, stream);
  mGPU->init_time_step = sd->init_time_step;
  mGPU->tout = t_final;
  // Solve
  cvodeRun(t_initial, mGPU, n_cells, md->n_per_cell_dep_var,
           stream);  // Asynchronous
#ifdef LOAD_BALANCE
  if (sd->load_balance == 1) {
    cudaEventRecord(sd->stopGPU, stream);  // End GPU timer
  }
#endif
  // CPU solver, equivalent to the CPU solver for the option CPU-Only
#ifdef TRACE_CPUGPU
  nvtxRangePushA("CPU Code");  // Start of profiling trace
#endif
#ifdef LOAD_BALANCE
  double startTime;
  if (sd->load_balance == 1) {
    startTime = MPI_Wtime();
  }
#endif
  // Set data
  n_cells = md->n_cells;
  int flag = CV_SUCCESS;
  int n_state_var = md->n_per_cell_state_var;
  // Get pointers to the first value of the arrays
  double *state = md->total_state;
  double *env = md->total_env;
  double *rxn_env_data = md->rxn_env_data;
  // Set pointers of arrays to the next cell after the GPU cells
  md->total_state += n_state_var * md->n_cells_gpu;
  md->total_env += CAMP_NUM_ENV_PARAM_ * md->n_cells_gpu;
  md->rxn_env_data += md->n_rxn_env_data * md->n_cells_gpu;
  for (int i_cell = md->n_cells_gpu; i_cell < n_cells; i_cell++) {
    int i_dep_var = 0;
    // Update input
    for (int i_spec = 0; i_spec < n_state_var; i_spec++) {
      if (sd->model_data.var_type[i_spec] == CHEM_SPEC_VARIABLE) {
        NV_Ith_S(sd->y, i_dep_var++) = md->total_state[i_spec] > TINY
                                           ? (realtype)md->total_state[i_spec]
                                           : TINY;
      }
    }
    // Reset Jacobian
    if (sd->is_reset_jac == 1) {
      N_VConst(0.0, md->J_state);
      N_VConst(0.0, md->J_deriv);
      SM_NNZ_S(md->J_solver) = SM_NNZ_S(md->J_init);
      for (int i = 0; i < SM_NNZ_S(md->J_solver); i++) {
        (SM_DATA_S(md->J_solver))[i] = 0.0;
      }
    }
    // Reset solver
    flag = CVodeReInit(sd->cvode_mem, t_initial, sd->y);
    flag = SUNKLUReInit(sd->ls, sd->J, SM_NNZ_S(sd->J), SUNKLU_REINIT_PARTIAL);
    flag = CVodeSetInitStep(sd->cvode_mem, sd->init_time_step);
    realtype t_rt = (realtype)t_initial;
    // Solve
    flag = CVode(sd->cvode_mem, t_final, sd->y, &t_rt, CV_NORMAL);
    if (flag < 0) {
      flag = CAMP_SOLVER_FAIL;
      break;
    }
    // Get output
    i_dep_var = 0;
    for (int i_spec = 0; i_spec < n_state_var; i_spec++) {
      if (md->var_type[i_spec] == CHEM_SPEC_VARIABLE) {
        md->total_state[i_spec] = (double)(NV_Ith_S(sd->y, i_dep_var) > 0.0
                                               ? NV_Ith_S(sd->y, i_dep_var)
                                               : 0.0);
        i_dep_var++;
      }
    }
    // Update pointers for next iteration
    md->total_state += n_state_var;
    md->total_env += CAMP_NUM_ENV_PARAM_;
    md->rxn_env_data += md->n_rxn_env_data;
  }
  // Reset pointers
  md->total_state = state;
  md->total_env = env;
  md->rxn_env_data = rxn_env_data;
#ifdef LOAD_BALANCE
  double timeCPU;
  if (sd->load_balance == 1) {
    timeCPU = (MPI_Wtime() - startTime);
  }
#endif
#ifdef TRACE_CPUGPU
  nvtxRangePop();  // End of profiling trace
#endif
#ifdef LOAD_BALANCE
  if (sd->load_balance == 1) {
    // Start synchronization timer between CPU and GPU
    cudaEventRecord(sd->startGPUSync, stream);
  }
#endif
  // Transfer data back to CPU. This is located after the CPU solver and not
  // before because it blocks CPU execution until finish the GPU kernel
  cudaMemcpyAsync(md->total_state, mGPU->state,
                  md->n_per_cell_state_var * md->n_cells_gpu * sizeof(double),
                  cudaMemcpyDeviceToHost, stream);
#ifdef DEBUG_SOLVER_FAILURES
  cudaMemcpyAsync(sd->flags, mGPU->flags, md->n_cells_gpu * sizeof(int),
                  cudaMemcpyDeviceToHost, stream);
#endif
  // Ensure synchronization
  cudaStreamSynchronize(stream);
  cudaDeviceSynchronize();
#ifdef LOAD_BALANCE
  // Balance load between CPU and GPU, changing the number of cells solved on
  // both architectures. Method explained on C. Guzman PhD Thesis - Chapter 6
  if (sd->load_balance == 1) {
    cudaEventRecord(sd->stopGPUSync, stream);  // End synchronization timer
    cudaEventSynchronize(sd->stopGPUSync);
    cudaEventSynchronize(sd->stopGPU);
    float msDevice = 0.0;
    cudaEventElapsedTime(&msDevice, sd->startGPU, sd->stopGPU);
    double timeGPU = msDevice / 1000;
    cudaEventElapsedTime(&msDevice, sd->startGPUSync, sd->stopGPUSync);
    timeGPU += msDevice / 1000;
    double load_balance = 100;
    double min = fmin(timeGPU, timeCPU);
    double max = fmax(timeGPU, timeCPU);
    load_balance = 100 * min / max;
    int short_gpu = 0;
    if (timeGPU < timeCPU) short_gpu = 1;
    double increase_in_load_gpu = sd->load_gpu - sd->last_load_gpu;
    double last_short_gpu = sd->last_short_gpu;
    double diff_load_balance = load_balance - sd->last_load_balance;
    if (short_gpu != last_short_gpu) {
      diff_load_balance = 100 - sd->last_load_balance + 100 - load_balance;
      increase_in_load_gpu *= -1;
    }
    double remaining_load_balance = 100 - load_balance;
    if (remaining_load_balance > diff_load_balance)
      increase_in_load_gpu *= 1.5;
    else
      increase_in_load_gpu /= 2;
    sd->last_short_gpu = short_gpu;
    sd->last_load_balance = load_balance;
    sd->last_load_gpu = sd->load_gpu;
    if (load_balance != 100) sd->load_gpu += increase_in_load_gpu;
    if (sd->load_gpu > 99) sd->load_gpu = 99;
    if (sd->load_gpu < 1) sd->load_gpu = 1;
    sd->acc_load_balance += load_balance;
    sd->iters_load_balance++;
    md->n_cells_gpu = md->n_cells * sd->load_gpu / 100;  // Automatic load
                                                         // balance
#ifdef DEBUG_LOAD_BALANCE
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0)
      printf(
          "load_gpu: %.2lf%% Load balance: %.2lf%% short_gpu %d \
          increase_in_load_gpu %.2lf\n",
          sd->last_load_gpu, load_balance, sd->last_short_gpu,
          increase_in_load_gpu);

#endif
  }
#ifdef CAMP_PROFILE_DEVICE_FUNCTIONS
  cudaMemcpyAsync(&mCPU->mdvCPU, mGPU->mdvo, sizeof(ModelDataVariable),
                  cudaMemcpyDeviceToHost, stream);
#endif
#endif
  cudaStreamDestroy(stream);  // reset stream for next iteration
#ifdef DEBUG_SOLVER_FAILURES
  for (int i = 0; i < md->n_cells_gpu; i++) {
    if (sd->flags[i] != CV_SUCCESS) {        // Check if there was a failure
      cudacvHandleFailure(sd->flags[i], i);  // Print error message
      flag = CAMP_SOLVER_FAIL;               // Update return flag
    }
    sd->flags[i] = CV_SUCCESS;  // Reset flags for next iteration
  }
#endif
  return (flag);
}

void solver_get_statistics_gpu(SolverData *sd) {
#ifdef CAMP_PROFILE_DEVICE_FUNCTIONS
  ModelDataGPU *mGPU = sd->mGPU;
  ModelDataCPU *mCPU = &(sd->mCPU);
  cudaMemcpy(&mCPU->mdvCPU, mGPU->mdvo, sizeof(ModelDataVariable),
             cudaMemcpyDeviceToHost);
#endif
}
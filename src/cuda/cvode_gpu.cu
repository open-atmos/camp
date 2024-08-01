/* Copyright (C) 2021 Barcelona Supercomputing Center and University of
 * Illinois at Urbana-Champaign
 * SPDX-License-Identifier: MIT
 */
#include "cvode_cuda.h"

#define PROFILE_GPU_SOLVING

extern "C" {
#include "cvode_gpu.h"
}
#ifdef TRACE_CPUGPU
#include "nvToolsExt.h"
#endif

#ifdef CAMP_USE_MPI
#include <mpi.h>
#endif

int cudaCVode(void *cvode_mem, double t_final, N_Vector yout,
               SolverData *sd, double t_initial){
  ModelDataGPU *mGPU = sd->mGPU;
  ModelData *md = &(sd->model_data);
  cudaStream_t stream;
  cudaStreamCreate(&stream);
#ifdef PROFILE_GPU_SOLVING
  cudaEventRecord(sd->startGPU,stream);
#endif
  int n_cells=md->n_cells_gpu;
  cudaMemcpyAsync(mGPU->rxn_env_data,md->rxn_env_data,md->n_rxn_env_data * n_cells * sizeof(double),cudaMemcpyHostToDevice,stream);
  cudaMemcpyAsync(mGPU->state,md->total_state,md->n_per_cell_state_var*n_cells*sizeof(double),cudaMemcpyHostToDevice,stream);
  mGPU->init_time_step = sd->init_time_step;
  mGPU->tout = t_final;
  cvodeRun(t_initial, mGPU, n_cells, md->n_per_cell_dep_var, stream); //Asynchronous
  cudaMemcpyAsync(md->total_state, mGPU->state, md->n_per_cell_state_var*n_cells*sizeof(double), cudaMemcpyDeviceToHost, stream);
#ifdef PROFILE_GPU_SOLVING
  cudaEventRecord(sd->stopGPU,stream);
#endif
  //CPU
#ifdef TRACE_CPUGPU
  nvtxRangePushA("CPU Code");
#endif
#ifdef PROFILE_GPU_SOLVING
  double startTime = MPI_Wtime();
#endif
  n_cells=md->n_cells;
  int flag=CV_SUCCESS;
  int n_state_var = md->n_per_cell_state_var;
  double *state = md->total_state;
  double *env = md->total_env;
  double *rxn_env_data = md->rxn_env_data;
  md->total_state += n_state_var*md->n_cells_gpu;
  md->total_env += CAMP_NUM_ENV_PARAM_*md->n_cells_gpu;
  md->rxn_env_data += md->n_rxn_env_data*md->n_cells_gpu;
  for (int i_cell = md->n_cells_gpu; i_cell < n_cells; i_cell++) {
    int i_dep_var = 0;
    for (int i_spec = 0; i_spec < n_state_var; i_spec++) {
      if (sd->model_data.var_type[i_spec] == CHEM_SPEC_VARIABLE) {
        NV_Ith_S(sd->y, i_dep_var++) =
            md->total_state[i_spec] > TINY
           ? (realtype)md->total_state[i_spec] : TINY;
      }
    }
    if (sd->is_reset_jac == 1) {
      N_VConst(0.0, md->J_state);
      N_VConst(0.0, md->J_deriv);
      SM_NNZ_S(md->J_solver) = SM_NNZ_S(md->J_init);
      for (int i = 0; i < SM_NNZ_S(md->J_solver); i++) {
        (SM_DATA_S(md->J_solver))[i] = 0.0;
      }
    }
    flag = CVodeReInit(sd->cvode_mem, t_initial, sd->y);
    flag = SUNKLUReInit(sd->ls, sd->J, SM_NNZ_S(sd->J), SUNKLU_REINIT_PARTIAL);
    flag = CVodeSetInitStep(sd->cvode_mem, sd->init_time_step);
    realtype t_rt = (realtype)t_initial;
    flag=0;
    flag = CVode(sd->cvode_mem, t_final, sd->y, &t_rt, CV_NORMAL);
    if (flag < 0){
      flag = CAMP_SOLVER_FAIL;
      break;
    }
    i_dep_var = 0;
    for (int i_spec = 0; i_spec < n_state_var; i_spec++) {
      if (md->var_type[i_spec] == CHEM_SPEC_VARIABLE) {
        md->total_state[i_spec] =
          (double)(NV_Ith_S(sd->y, i_dep_var) > 0.0
           ? NV_Ith_S(sd->y, i_dep_var)
           : 0.0);
        i_dep_var++;
      }
    }
    md->total_state += n_state_var;
    md->total_env += CAMP_NUM_ENV_PARAM_;
    md->rxn_env_data += md->n_rxn_env_data;
  }
  md->total_state  = state;
  md->total_env = env;
  md->rxn_env_data = rxn_env_data;
#ifdef PROFILE_GPU_SOLVING
  double timeCPU = (MPI_Wtime() - startTime);
#endif
#ifdef TRACE_CPUGPU
  nvtxRangePop();
#endif
#ifdef PROFILE_GPU_SOLVING
  cudaEventRecord(sd->startGPUSync,stream);
#endif
  cudaStreamSynchronize(stream);
  cudaDeviceSynchronize();
#ifdef PROFILE_GPU_SOLVING
  cudaEventRecord(sd->stopGPUSync,stream);
  cudaEventSynchronize(sd->stopGPUSync);
  cudaEventSynchronize(sd->stopGPU);
  float msDevice = 0.0;
  cudaEventElapsedTime(&msDevice, sd->startGPU, sd->stopGPU);
  double timeGPU=msDevice/1000;
  cudaEventElapsedTime(&msDevice, sd->startGPUSync, sd->stopGPUSync);
  timeGPU+=msDevice/1000;
  MPI_Barrier(MPI_COMM_WORLD);
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  double load_balance=100;
  double min=fmin(timeGPU,timeCPU);
  double max=fmax(timeGPU,timeCPU);
  load_balance=100*min/max;
  int short_gpu=0;
  if(timeGPU<timeCPU) short_gpu=1;
  double increase_in_load_gpu=sd->load_gpu-sd->last_load_gpu;
  //if(rank==0)printf("load_gpu: %.2lf%% Load balance: %.2lf%%  short_gpu %d\n",sd->load_gpu,load_balance,short_gpu);
  double last_short_gpu=sd->short_gpu;
  double diff_load_balance=load_balance-sd->last_load_balance; //e.g. 80-20=60; 20-80=-60;
  if(short_gpu != last_short_gpu){
    diff_load_balance=100-sd->last_load_balance+100-load_balance; //e.g. 100-20+100-60=140;
    increase_in_load_gpu*=-1;
  }
  double remaining_load_balance=100-load_balance;
  if(remaining_load_balance > diff_load_balance) increase_in_load_gpu*=1.5;
  else increase_in_load_gpu/=2;
  sd->last_load_balance=load_balance;
  sd->last_load_gpu=sd->load_gpu;
  if(load_balance!=100) sd->load_gpu+=increase_in_load_gpu;
  sd->short_gpu=short_gpu;
  md->n_cells_gpu=md->n_cells*sd->load_gpu/100;
  //if(rank==0)printf("remaining_load_balance %.2lf diff_load_balance %.2lf "
  //"increase_in_load_gpu %.2lf\n",remaining_load_balance,diff_load_balance,increase_in_load_gpu);
  if(rank==0)printf("Load balance: %.2lf%% load_gpu %.2lf%%\n",load_balance,sd->load_gpu);

#ifdef CAMP_PROFILE_DEVICE_FUNCTIONS
  printf("DEBUG: CAMP_PROFILE_DEVICE_FUNCTIONS\n");
  cudaMemcpyAsync(&mCPU->mdvCPU, mGPU->mdvo, sizeof(ModelDataVariable), cudaMemcpyDeviceToHost, stream);
#endif
#endif
  cudaStreamDestroy(stream);
  return(CV_SUCCESS);
}

void solver_get_statistics_gpu(SolverData *sd){
#ifdef CAMP_PROFILE_DEVICE_FUNCTIONS
  ModelDataGPU *mGPU = sd->mGPU;
  ModelDataCPU *mCPU = &(sd->mCPU);
  cudaMemcpy(&mCPU->mdvCPU,mGPU->mdvo,sizeof(ModelDataVariable),cudaMemcpyDeviceToHost);
#endif
}
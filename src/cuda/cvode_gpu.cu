/* Copyright (C) 2021 Barcelona Supercomputing Center and University of
 * Illinois at Urbana-Champaign
 * SPDX-License-Identifier: MIT
 */
#include "cvode_cuda.h"

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
  CVodeMem cv_mem = (CVodeMem) cvode_mem;
  ModelDataGPU *mGPU = sd->mGPU;
  ModelData *md = &(sd->model_data);
#ifdef CAMP_PROFILE_SOLVING
  cudaEventRecord(sd->startcvStep);
#endif
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  int n_cells=md->n_cells_gpu;
  cudaMemcpyAsync(mGPU->rxn_env_data,md->rxn_env_data,md->n_rxn_env_data * n_cells * sizeof(double),cudaMemcpyHostToDevice,stream);
  cudaMemcpyAsync(mGPU->state,md->total_state,md->n_per_cell_state_var*n_cells*sizeof(double),cudaMemcpyHostToDevice,stream);
  mGPU->init_time_step = sd->init_time_step;
  mGPU->tout = t_final;
  cvodeRun(t_initial, mGPU, n_cells, md->n_per_cell_dep_var, stream); //Asynchronous
  //CPU
#ifdef TRACE_CPUGPU
  nvtxRangePushA("CPU Code");
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
  cudaMemcpyAsync(md->total_state, mGPU->state, md->n_per_cell_state_var*md->n_cells_gpu * sizeof(double), cudaMemcpyDeviceToHost, stream);
#ifdef TRACE_CPUGPU
  nvtxRangePop();
#endif
  cudaDeviceSynchronize();
#ifdef CAMP_PROFILE_SOLVING
    cudaEventRecord(sd->stopcvStep);
    cudaEventSynchronize(sd->stopcvStep);
    float mscvStep = 0.0;
    cudaEventElapsedTime(&mscvStep, sd->startcvStep, sd->stopcvStep);
    cv_mem->timecvStep+= mscvStep/1000;
#ifdef CAMP_PROFILE_DEVICE_FUNCTIONS
    cudaMemcpy(&mCPU->mdvCPU, mGPU->mdvo, sizeof(ModelDataVariable), cudaMemcpyDeviceToHost);
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
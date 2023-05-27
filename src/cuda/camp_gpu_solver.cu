/* Copyright (C) 2021 Barcelona Supercomputing Center and University of
 * Illinois at Urbana-Champaign
 * SPDX-License-Identifier: MIT
 */

extern "C" {
#include "camp_gpu_solver.h"
}

void set_jac_data_gpu(SolverData *sd, double *J){
  ModelData *md = &(sd->model_data);
  ModelDataCPU *mCPU = &(sd->mCPU);
  ModelDataGPU *mGPU;
  mGPU = sd->mGPU;
  double *J_ptr = J;
  double *J_solver = SM_DATA_S(md->J_solver);
  double *J_state = N_VGetArrayPointer(md->J_state);
  double *J_deriv = N_VGetArrayPointer(md->J_deriv);
  cudaMemcpy(mGPU->dA, J_ptr, mCPU->jac_size, cudaMemcpyHostToDevice);
  cudaMemcpy(mGPU->J_solver, J_solver, mCPU->jac_size, cudaMemcpyHostToDevice);
  cudaMemcpy(mGPU->J_state, J_state, mCPU->deriv_size, cudaMemcpyHostToDevice);
  cudaMemcpy(mGPU->J_deriv, J_deriv, mCPU->deriv_size, cudaMemcpyHostToDevice);
  cudaMemcpy(mGPU->djA, mCPU->jA, mGPU->nnz/mGPU->n_cells * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(mGPU->diA, mCPU->iA, (mGPU->nrows/mGPU->n_cells + 1) * sizeof(int), cudaMemcpyHostToDevice);
}

void camp_solver_update_model_state_gpu(N_Vector solver_state, SolverData *sd){
  ModelData *md = &(sd->model_data);
  ModelDataGPU *mGPU;
  ModelDataCPU *mCPU = &(sd->mCPU);
  double *total_state = md->total_state;
  mGPU = sd->mGPU;
  cudaMemcpy(mGPU->state, total_state, mCPU->state_size, cudaMemcpyHostToDevice);
}

int rxn_calc_deriv_gpu(SolverData *sd, N_Vector y, N_Vector deriv, double time_step) {
  ModelData *md = &(sd->model_data);
  ModelDataGPU *mGPU;
  ModelDataCPU *mCPU = &(sd->mCPU);
  double *total_state = md->total_state;
  double *deriv_data = N_VGetArrayPointer(deriv);
  if(sd->use_gpu_cvode==0){
    mGPU = sd->mGPU;
    cudaMemcpy(mGPU->deriv_data, deriv_data, mCPU->deriv_size, cudaMemcpyHostToDevice);
    cudaMemcpy(mGPU->state, total_state, mCPU->state_size, cudaMemcpyHostToDevice);
  }
  return 0;
}

void free_gpu_cu(SolverData *sd) {
  ModelDataGPU *mGPU = sd->mGPU;
  ModelDataCPU *mCPU = &(sd->mCPU);
  //printf("free_gpu_cu start\n");
  free(sd->flagCells);
  mGPU = sd->mGPU;
  //ModelDataGPU Start
  cudaFree(mGPU->map_state_deriv);
  cudaFree(mGPU->deriv_data);
  cudaFree(mGPU->J_solver);
  cudaFree(mGPU->J_state);
  cudaFree(mGPU->J_deriv);
  cudaFree(mGPU->J_tmp);
  cudaFree(mGPU->J_tmp2);
  cudaFree(mGPU->indexvals);
  cudaFree(mGPU->indexptrs);
  cudaFree(mGPU->rxn_int);
  cudaFree(mGPU->rxn_double);
  cudaFree(mGPU->state);
  cudaFree(mGPU->env);
  cudaFree(mGPU->rxn_env_data);
  cudaFree(mGPU->rxn_env_data_idx);
  cudaFree(mGPU->production_rates);
  cudaFree(mGPU->loss_rates);
  cudaFree(mGPU->rxn_int_indices);
  cudaFree(mGPU->rxn_float_indices);
#ifdef DEV_AERO_REACTIONS
  cudaFree(mGPU->aero_rep_int_indices);
  cudaFree(mGPU->aero_rep_float_indices);
  cudaFree(mGPU->aero_rep_env_idx);
  cudaFree(mGPU->aero_rep_int_data);
  cudaFree(mGPU->aero_rep_float_data);
  cudaFree(mGPU->aero_rep_env_data);
#endif
  cudaFree(mGPU->n_mapped_values);
  cudaFree(mGPU->jac_map);
  cudaFree(mGPU->yout);
  cudaFree(mGPU->cv_Vabstol);
  cudaFree(mGPU->grid_cell_state);
  cudaFree(mGPU->grid_cell_env);
  cudaFree(mGPU->grid_cell_aero_rep_env_data);
  cudaFree(mGPU->cv_l);
  cudaFree(mGPU->cv_tau);
  cudaFree(mGPU->cv_tq);
  cudaFree(mGPU->cv_last_yn);
  cudaFree(mGPU->cv_acor_init);
  cudaFree(mGPU->dA);
  cudaFree(mGPU->djA);
  cudaFree(mGPU->diA);
  cudaFree(mGPU->dx);
  cudaFree(mGPU->dtempv);
  cudaFree(mGPU->ddiag);
  cudaFree(mGPU->dr0);
  cudaFree(mGPU->dr0h);
  cudaFree(mGPU->dn0);
  cudaFree(mGPU->dp0);
  cudaFree(mGPU->dt);
  cudaFree(mGPU->ds);
  cudaFree(mGPU->dAx2);
  cudaFree(mGPU->dy);
  cudaFree(mGPU->dz);
  cudaFree(mGPU->dftemp);
  cudaFree(mGPU->dcv_y);
  cudaFree(mGPU->dtempv1);
  cudaFree(mGPU->dtempv2);
  cudaFree(mGPU->flag);
  cudaFree(mGPU->flagCells);
  cudaFree(mGPU->cv_acor);
  cudaFree(mGPU->dzn);
  cudaFree(mGPU->dewt);
  cudaFree(mGPU->dsavedJ);
  cudaFree(mCPU->map_state_derivCPU);
  cudaFree(mGPU->mdv);
  cudaFree(mGPU->mdvo);
  cudaFree(mGPU);
}

void print_gpu_specs() {
  printf("GPU specifications \n");
  int nDevicesMax;
  cudaGetDeviceCount(&nDevicesMax);
  for (int i = 0; i < nDevicesMax; i++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    printf("Device Number: %d\n", i);
    printf("  Device name: %s\n", prop.name);
    printf("  Memory Clock Rate (KHz): %d\n",
           prop.memoryClockRate);
    printf("  Peak Memory Bandwidth (GB/s): %f\n",
           2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
    printf("  maxGridSize: %d\n", prop.maxGridSize[1]);
    printf("  maxThreadsPerBlock: %d\n", prop.maxThreadsPerBlock);
    printf("  maxThreadsDim: %d\n", prop.maxThreadsDim[1]);
    printf("  totalGlobalMem: %zu\n", prop.totalGlobalMem);
    printf("  sharedMemPerBlock: %zu\n", prop.sharedMemPerBlock); //bytes
    printf("  multiProcessorCount: %d\n", prop.multiProcessorCount);
  }
}
/* Copyright (C) 2021 Barcelona Supercomputing Center and University of
 * Illinois at Urbana-Champaign
 * SPDX-License-Identifier: MIT
 */

#include "itsolver_gpu.h"

extern "C" {
#include "camp_gpu_solver.h"
#include "rxns_gpu.h"
#ifdef DEV_AERO_REACTIONS
#include "aeros/aero_rep_gpu_solver.h"
#endif
#include "time_derivative_gpu.h"
#include "Jacobian_gpu.h"
}


// Reaction types (Must match parameters defined in camp_rxn_factory)
#define RXN_ARRHENIUS 1
#define RXN_TROE 2
#define RXN_CMAQ_H2O2 3
#define RXN_CMAQ_OH_HNO3 4
#define RXN_PHOTOLYSIS 5
#define RXN_HL_PHASE_TRANSFER 6
#define RXN_AQUEOUS_EQUILIBRIUM 7
#define RXN_SIMPOL_PHASE_TRANSFER 10
#define RXN_CONDENSED_PHASE_ARRHENIUS 11
#define RXN_FIRST_ORDER_LOSS 12
#define RXN_EMISSION 13
#define RXN_WET_DEPOSITION 14

// Status codes for calls to camp_solver functions
#define CAMP_SOLVER_SUCCESS 0
#define CAMP_SOLVER_FAIL 1

//GPU async stream related variables to ensure robustness
//int n_solver_objects=0; //Number of solver_new_gpu calls
//cudaStream_t *stream_gpu; //GPU streams to async computation/data movement
//int n_streams = 16;

static void HandleError(cudaError_t err,
                        const char *file,
                        int line) {
  if (err != cudaSuccess) {
    printf("%s in %s at line %d\n", cudaGetErrorString(err),
           file, line);
    exit(EXIT_FAILURE);
  }
}

void set_jac_data_gpu(SolverData *sd, double *J){

  ModelData *md = &(sd->model_data);
  ModelDataCPU *mCPU = &(sd->mCPU);
  ModelDataGPU *mGPU;

  int offset_nnz_J_solver = 0;
  int offset_nrows = 0;
  for (int iDevice = sd->startDevice; iDevice < sd->endDevice; iDevice++) {
    cudaSetDevice(iDevice);
    sd->mGPU = &(sd->mGPUs[iDevice]);
    mGPU = sd->mGPU;

    double *J_ptr = J+offset_nnz_J_solver;
    double *J_solver = SM_DATA_S(md->J_solver)+offset_nnz_J_solver;
    double *J_state = N_VGetArrayPointer(md->J_state)+offset_nrows;
    double *J_deriv = N_VGetArrayPointer(md->J_deriv)+offset_nrows;
    HANDLE_ERROR(cudaMemcpy(mGPU->dA, J_ptr, mCPU->jac_size, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(mGPU->J_solver, J_solver, mCPU->jac_size, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(mGPU->J_state, J_state, mCPU->deriv_size, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(mGPU->J_deriv, J_deriv, mCPU->deriv_size, cudaMemcpyHostToDevice));

    offset_nnz_J_solver += mCPU->nnz_J_solver;
    offset_nrows += md->n_per_cell_dep_var* mGPU->n_cells;
    cudaMemcpy(mGPU->djA, mCPU->jA, mGPU->nnz/mGPU->n_cells * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(mGPU->diA, mCPU->iA, (mGPU->nrows/mGPU->n_cells + 1) * sizeof(int), cudaMemcpyHostToDevice);
  }
}

void rxn_update_env_state_gpu(SolverData *sd) {

  ModelData *md = &(sd->model_data);
  ModelDataGPU *mGPU;
  ModelDataCPU *mCPU = &(sd->mCPU);
  double *rxn_env_data = md->rxn_env_data;
  double *env = md->total_env;
  double *total_state = md->total_state;

  for (int iDevice = sd->startDevice; iDevice < sd->endDevice; iDevice++) {
    cudaSetDevice(iDevice);
    sd->mGPU = &(sd->mGPUs[iDevice]);
    mGPU = sd->mGPU;

    HANDLE_ERROR(cudaMemcpy(mGPU->rxn_env_data, rxn_env_data, mCPU->rxn_env_data_size, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(mGPU->env, env, mCPU->env_size, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(mGPU->state, total_state, mCPU->state_size, cudaMemcpyHostToDevice));

    rxn_env_data += mGPU->n_rxn_env_data * mGPU->n_cells;
    env += CAMP_NUM_ENV_PARAM_ * mGPU->n_cells;
    total_state += mGPU->state_size_cell * mGPU->n_cells;

  }

}

void camp_solver_update_model_state_gpu(N_Vector solver_state, SolverData *sd,
                                       double threshhold, double replacement_value)
{
  ModelData *md = &(sd->model_data);
  ModelDataGPU *mGPU;
  ModelDataCPU *mCPU = &(sd->mCPU);
  double *total_state = md->total_state;
  for (int iDevice = sd->startDevice; iDevice < sd->endDevice; iDevice++) {
    cudaSetDevice(iDevice);
    sd->mGPU = &(sd->mGPUs[iDevice]);
    mGPU = sd->mGPU;
    HANDLE_ERROR(cudaMemcpy(mGPU->state, total_state, mCPU->state_size, cudaMemcpyHostToDevice));
    total_state += mGPU->state_size_cell * mGPU->n_cells;
  }
}


int rxn_calc_deriv_gpu(SolverData *sd, N_Vector y, N_Vector deriv, double time_step) {

  ModelData *md = &(sd->model_data);
  ModelDataGPU *mGPU;
  ModelDataCPU *mCPU = &(sd->mCPU);

  double *total_state = md->total_state;
  double *deriv_data = N_VGetArrayPointer(deriv);
  if(sd->use_gpu_cvode==0){
    for (int iDevice = sd->startDevice; iDevice < sd->endDevice; iDevice++) {
      cudaSetDevice(iDevice);
      sd->mGPU = &(sd->mGPUs[iDevice]);
      mGPU = sd->mGPU;

      HANDLE_ERROR(cudaMemcpy(mGPU->deriv_data, deriv_data, mCPU->deriv_size, cudaMemcpyHostToDevice));
      HANDLE_ERROR(cudaMemcpy(mGPU->state, total_state, mCPU->state_size, cudaMemcpyHostToDevice));

      total_state += mGPU->state_size_cell * mGPU->n_cells;
      deriv_data += mGPU->nrows;
    }
  }
  return 0;
}

void free_gpu_cu(SolverData *sd) {
  ModelDataGPU *mGPU = sd->mGPU;
  ModelDataCPU *mCPU = &(sd->mCPU);
  //printf("free_gpu_cu start\n");
  free(sd->flagCells);
  for (int iDevice = sd->startDevice; iDevice < sd->endDevice; iDevice++) {
    cudaSetDevice(iDevice);
    sd->mGPU = &(sd->mGPUs[iDevice]);
    mGPU = sd->mGPU;
    //cudaStreamDestroy(mCPU->streams[iDevice]);
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


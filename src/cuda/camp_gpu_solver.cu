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
    HANDLE_ERROR(cudaMemcpy(mGPU->dA, J_ptr, mGPU->jac_size, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(mGPU->J_solver, J_solver, mGPU->jac_size, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(mGPU->J_state, J_state, mGPU->deriv_size, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(mGPU->J_deriv, J_deriv, mGPU->deriv_size, cudaMemcpyHostToDevice));

    offset_nnz_J_solver += mGPU->nnz_J_solver;
    offset_nrows += md->n_per_cell_dep_var* mGPU->n_cells;
    cudaMemcpy(mGPU->djA, mGPU->jA, mGPU->nnz/mGPU->n_cells * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(mGPU->diA, mGPU->iA, (mGPU->nrows/mGPU->n_cells + 1) * sizeof(int), cudaMemcpyHostToDevice);
  }
}

void rxn_update_env_state_gpu(SolverData *sd) {

  ModelData *md = &(sd->model_data);
  ModelDataGPU *mGPU;
  double *rxn_env_data = md->rxn_env_data;
  double *env = md->total_env;
  double *total_state = md->total_state;

  for (int iDevice = sd->startDevice; iDevice < sd->endDevice; iDevice++) {
    cudaSetDevice(iDevice);
    sd->mGPU = &(sd->mGPUs[iDevice]);
    mGPU = sd->mGPU;

    HANDLE_ERROR(cudaMemcpy(mGPU->rxn_env_data, rxn_env_data, mGPU->rxn_env_data_size, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(mGPU->env, env, mGPU->env_size, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(mGPU->state, total_state, mGPU->state_size, cudaMemcpyHostToDevice));

    rxn_env_data += mGPU->n_rxn_env_data * mGPU->n_cells;
    env += CAMP_NUM_ENV_PARAM_ * mGPU->n_cells;
    total_state += mGPU->state_size_cell * mGPU->n_cells;

  }

}

__device__
void cudaDevicecamp_solver_check_model_state0(double *state, double *y,
                                        int *map_state_deriv, double threshhold, double replacement_value, int *flag,
                                        int deriv_length_cell, int n_cells)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int active_threads = n_cells*deriv_length_cell;
  //extern __shared__ int flag_shr[];
  __shared__ int flag_shr[1];
  flag_shr[0] = 0;

  if(tid<active_threads) {

    if (y[tid] < threshhold) {

      //*flag = CAMP_SOLVER_FAIL;
      flag_shr[0] = CAMP_SOLVER_FAIL;
#ifdef FAILURE_DETAIL
      printf("\nFailed model state update gpu (Negative value on 'y'):[spec %d] = %le",tid,y[tid]);
#endif

    } else {
      state[map_state_deriv[tid]] =
              y[tid] <= threshhold ?
              replacement_value : y[tid];

      //printf("tid %d map_state_deriv %d\n", tid, map_state_deriv[tid]);

    }

    /*
    if (y[tid] > -SMALL) {
      state_init[map_state_deriv[tid]] =
      y[tid] > threshhold ?
      y[tid] : replacement_value;

      //state_init[map_state_deriv[tid]] = 0.1;
      //printf("tid %d map_state_deriv %d\n", tid, map_state_deriv[tid]);
    } else {
      *status = CAMP_SOLVER_FAIL;
#ifdef FAILURE_DETAIL
      printf("\nFailed model state update gpu (Negative value on 'y'):[spec %d] = %le",tid,y[tid]);
#endif
    }
     */
  }

  __syncthreads();
  *flag = flag_shr[0];
  return;

}

__global__
void camp_solver_check_model_state_cuda(double *state_init, double *y,
        int *map_state_deriv, double threshhold, double replacement_value, int *flag,
        int deriv_length_cell, int n_cells)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int active_threads = n_cells*deriv_length_cell;

  if(tid<active_threads) {

    if (y[tid] < threshhold) {

      *flag = CAMP_SOLVER_FAIL;
#ifdef FAILURE_DETAIL
      printf("\nFailed model state update gpu (Negative value on 'y'):[spec %d] = %le",tid,y[tid]);
#endif

    } else {
      state_init[map_state_deriv[tid]] =
              y[tid] <= threshhold ?
              replacement_value : y[tid];

      //state_init[map_state_deriv[tid]] = 0.1;
      //printf("tid %d map_state_deriv %d\n", tid, map_state_deriv[tid]);

    }

    /*
    if (y[tid] > -SMALL) {
      state_init[map_state_deriv[tid]] =
      y[tid] > threshhold ?
      y[tid] : replacement_value;

      //state_init[map_state_deriv[tid]] = 0.1;
      //printf("tid %d map_state_deriv %d\n", tid, map_state_deriv[tid]);
    } else {
      *status = CAMP_SOLVER_FAIL;
#ifdef FAILURE_DETAIL
      printf("\nFailed model state update gpu (Negative value on 'y'):[spec %d] = %le",tid,y[tid]);
#endif
    }
     */
  }

}

int camp_solver_check_model_state_gpu(N_Vector solver_state, SolverData *sd,
                                      double threshhold0, double replacement_value0)
{
  ModelData *md = &(sd->model_data);
  int flag = CAMP_SOLVER_SUCCESS; //0
  ModelDataGPU *mGPU;

  double replacement_value = TINY;
  double threshhold = -SMALL;

  for (int iDevice = sd->startDevice; iDevice < sd->endDevice; iDevice++) {
    cudaSetDevice(iDevice);
    sd->mGPU = &(sd->mGPUs[iDevice]);
    mGPU = sd->mGPU;

    int n_cells = mGPU->n_cells;
    int n_state_var = md->n_per_cell_state_var;
    int n_dep_var = md->n_per_cell_dep_var;
    int n_threads = n_dep_var*n_cells;
    int n_blocks = ((n_threads + mGPU->max_n_gpu_thread - 1) / mGPU->max_n_gpu_thread);

    camp_solver_check_model_state_cuda << < n_blocks, mGPU->max_n_gpu_thread >> >
    (mGPU->state, mGPU->dcv_y, mGPU->map_state_deriv,
            threshhold, replacement_value, &flag, n_dep_var, n_cells);

    HANDLE_ERROR(cudaMemcpy(md->total_state, mGPU->state, mGPU->state_size, cudaMemcpyDeviceToHost));

  }

#ifdef DEBUG_CHECK_MODEL_STATE_CUDA
  for (int i_cell = 0; i_cell < n_cells; i_cell++) {
   for (int i_dep_var = 0; i_dep_var < n_dep_var; i_dep_var++) {

     printf("(%d) %-le \n", i_dep_var+1,
            md->total_state[mGPU->map_state_derivCPU[i_dep_var]]);
   }
}
#endif


  //printf("camp_solver_check_model_state_gpu flag %d\n",flag);

  return flag;
}

void camp_solver_update_model_state_gpu(N_Vector solver_state, SolverData *sd,
                                       double threshhold, double replacement_value)
{
  ModelData *md = &(sd->model_data);
  ModelDataGPU *mGPU;
  double *total_state = md->total_state;
  for (int iDevice = sd->startDevice; iDevice < sd->endDevice; iDevice++) {
    cudaSetDevice(iDevice);
    sd->mGPU = &(sd->mGPUs[iDevice]);
    mGPU = sd->mGPU;
    HANDLE_ERROR(cudaMemcpy(mGPU->state, total_state, mGPU->state_size, cudaMemcpyHostToDevice));
    total_state += mGPU->state_size_cell * mGPU->n_cells;
  }
}

__device__ void solveRXN0(
        TimeDerivativeGPU deriv_data,
       double time_step,
       ModelDataGPU *md
)
{

#ifdef REVERSE_INT_FLOAT_MATRIX

  double *rxn_float_data = &( md->rxn_double[md->i_rxn]);
  int *int_data = &(md->rxn_int[md->i_rxn]);
  int rxn_type = int_data[0];
  int *rxn_int_data = (int *) &(int_data[1*md->n_rxn]);

#else

  double *rxn_float_data = &( md->rxn_double[md->rxn_float_indices[md->i_rxn]]);
  int *int_data = &(md->rxn_int[md->rxn_int_indices[md->i_rxn]]);

  //double *rxn_float_data = &( md->rxn_double[md->i_rxn]);
  //int *int_data = &(md->rxn_int[md->i_rxn]);


  int rxn_type = int_data[0];
  int *rxn_int_data = (int *) &(int_data[1]);

#endif

  //Get indices for rates
  double *rxn_env_data = &(md->rxn_env_data
  [md->n_rxn_env_data*md->i_cell+md->rxn_env_data_idx[md->i_rxn]]);

#ifdef DEBUG_DERIV_GPU
  if(tid==0){
    printf("[DEBUG] GPU solveRXN tid %d, \n", tid);
  }
#endif

  switch (rxn_type) {
    //case RXN_AQUEOUS_EQUILIBRIUM :
    //fix run-time error
    //rxn_gpu_aqueous_equilibrium_calc_deriv_contrib(md, deriv_data, rxn_int_data,
    //                                               rxn_float_data, rxn_env_data,time_step);
    //break;
    case RXN_ARRHENIUS :
      rxn_gpu_arrhenius_calc_deriv_contrib(md, deriv_data, rxn_int_data,
                                           rxn_float_data, rxn_env_data,time_step);
      break;
    case RXN_CMAQ_H2O2 :
      rxn_gpu_CMAQ_H2O2_calc_deriv_contrib(md, deriv_data, rxn_int_data,
                                          rxn_float_data, rxn_env_data,time_step);
      break;
    case RXN_CMAQ_OH_HNO3 :
      rxn_gpu_CMAQ_OH_HNO3_calc_deriv_contrib(md, deriv_data, rxn_int_data,
                                           rxn_float_data, rxn_env_data,time_step);
      break;
    case RXN_CONDENSED_PHASE_ARRHENIUS :
      //rxn_gpu_condensed_phase_arrhenius_calc_deriv_contrib(md, deriv_data, rxn_int_data,
      //                                     rxn_float_data, rxn_env_data,time_step);
      break;
    case RXN_EMISSION :
      //rxn_gpu_emission_calc_deriv_contrib(md, deriv_data, rxn_int_data,
      //                                     rxn_float_data, rxn_env_data,time_step);
      break;
    case RXN_FIRST_ORDER_LOSS :
      //rxn_gpu_first_order_loss_calc_deriv_contrib(md, deriv_data, rxn_int_data,
      //                                     rxn_float_data, rxn_env_data,time_step);
      break;
    case RXN_HL_PHASE_TRANSFER :
      //rxn_gpu_HL_phase_transfer_calc_deriv_contrib(md, deriv_data, rxn_int_data,
      //                                             rxn_float_data, rxn_env_data,time_stepn);
      break;
    case RXN_PHOTOLYSIS :
      rxn_gpu_photolysis_calc_deriv_contrib(md, deriv_data, rxn_int_data,
                                           rxn_float_data, rxn_env_data,time_step);
      break;
    case RXN_SIMPOL_PHASE_TRANSFER :
      //rxn_gpu_SIMPOL_phase_transfer_calc_deriv_contrib(md, deriv_data,
      //        rxn_int_data, rxn_float_data, rxn_env_data, time_step);
      break;
    case RXN_TROE :
      rxn_gpu_troe_calc_deriv_contrib(md, deriv_data, rxn_int_data,
                                      rxn_float_data, rxn_env_data,time_step);
      break;
    case RXN_WET_DEPOSITION :
      //rxn_gpu_wet_deposition_calc_deriv_contrib(md, deriv_data, rxn_int_data,
      //                                     rxn_float_data, rxn_env_data,time_step);
      break;
  }
/*
*/

}

__device__ void cudaDevicecalc_deriv0(
#ifdef CAMP_DEBUG_GPU
        int counterDeriv2,
#endif
        //check_model_state
        //double threshhold, double replacement_value, int *flag,
        //f_cuda
        double time_step, int deriv_length_cell, int state_size_cell,
        int n_cells,
        int i_kernel, int threads_block, int n_shr_empty, double *y,
        ModelDataGPU md_object
) //Interface CPU/GPU
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int tid_cell=tid%deriv_length_cell;
  int active_threads = n_cells*deriv_length_cell;
  ModelDataGPU *md = &md_object;

#ifdef DEBUG_DERIV_GPU
  if(tid==0){
    printf("[DEBUG] GPU solveDerivative tid %d, \n", tid);
  }__syncthreads();
#endif

  if(tid<active_threads){

#ifdef DEBUG_solveDerivative_J_DERIV_IN_CPU
#else

    //N_VLinearSum(1.0, y, -1.0, md->J_state, md->J_tmp);
  cudaDevicezaxpby(1.0, y, -1.0, md->J_state, md->J_tmp, active_threads);
  //SUNMatMatvec(md->J_solver, md->J_tmp, md->J_tmp2);
  cudaDeviceSpmvCSC_block(md->J_tmp2, md->J_tmp, md->J_solver, md->jA, md->iA);
  //N_VLinearSum(1.0, md->J_deriv, 1.0, md->J_tmp2, md->J_tmp);
  cudaDevicezaxpby(1.0, md->J_deriv, 1.0, md->J_tmp2, md->J_tmp, active_threads);
  cudaDevicesetconst(md->J_tmp2, 0.0, active_threads); //Reset for next iter


#endif

    //Debug
    /*
    if(counterDeriv2<=1){
      printf("(%d) y %-le J_state %-le J_solver %-le J_tmp %-le J_tmp2 %-le J_deriv %-le\n",tid+1,
             y[tid], md->J_state[tid], md->J_solver[tid], md->J_tmp[tid], md->J_tmp2[tid], md->J_deriv[tid]);
      //printf("gpu threads %d\n", active_threads);
    }
*/
    TimeDerivativeGPU deriv_data;
    deriv_data.num_spec = deriv_length_cell*n_cells;

#ifdef AEROS_CPU
#else
    deriv_data.production_rates = md->production_rates;
    deriv_data.loss_rates = md->loss_rates;
    time_derivative_reset_gpu(deriv_data);
    __syncthreads();
#endif

    int i_cell = tid/deriv_length_cell;
    md->i_cell = i_cell;
    deriv_data.production_rates = &( md->production_rates[deriv_length_cell*i_cell]);
    deriv_data.loss_rates = &( md->loss_rates[deriv_length_cell*i_cell]);

    md->grid_cell_state = &( md->state[state_size_cell*i_cell]);
    md->grid_cell_env = &( md->env[CAMP_NUM_ENV_PARAM_*i_cell]);

    //Filter threads for n_rxn
    int n_rxn = md->n_rxn;
    if( tid_cell < n_rxn) {
      int n_iters = n_rxn / deriv_length_cell;
      //Repeat if there are more reactions than species
      for (int i = 0; i < n_iters; i++) {
        md->i_rxn = tid_cell + i*deriv_length_cell;

        solveRXN0(deriv_data, time_step, md);
      }

      //Limit tid to pending rxns to compute
      int residual=n_rxn-(deriv_length_cell*n_iters);
      if(tid_cell < residual){
        md->i_rxn = tid_cell + deriv_length_cell*n_iters;

        solveRXN0(deriv_data, time_step, md);
      }
    }
    __syncthreads();

    /*if(tid==0){
      printf("tid %d time_deriv.production_rates %-le time_deriv.loss_rates %-le\n",
              tid, deriv_data.production_rates[tid],
             deriv_data.loss_rates[tid]);
    }*/

    deriv_data.production_rates = md->production_rates;
    deriv_data.loss_rates = md->loss_rates;
    __syncthreads();
    time_derivative_output_gpu(deriv_data, md->deriv_data, md->J_tmp,0);

  }

}

__device__
void cudaDevicef0(
#ifdef CAMP_DEBUG_GPU
        int counterDeriv2,
#endif
        //check_model_state
        double threshhold, double replacement_value, int *flag,
        //f_cuda
        double time_step, int deriv_length_cell, int state_size_cell,
        int n_cells,
        int i_kernel, int threads_block, int n_shr_empty, double *y,
        ModelDataGPU md_object
) //Interface CPU/GPU
{

  ModelDataGPU *md = &md_object;

  cudaDevicecamp_solver_check_model_state0(md->state, y,
                                          md->map_state_deriv, threshhold, replacement_value,
                                          flag, deriv_length_cell, n_cells);

  //__syncthreads();
  //study flag block effect: flag is global for all threads or for only the block?
  if(*flag==CAMP_SOLVER_FAIL)
    return;

  cudaDevicecalc_deriv0(
#ifdef CAMP_DEBUG_GPU
           counterDeriv2,
#endif
          //f_cuda
        time_step, deriv_length_cell, state_size_cell,
           n_cells, i_kernel, threads_block, n_shr_empty, y,
           md_object
          );
}

__global__
void cudaGlobalf(
#ifdef CAMP_DEBUG_GPU
        int counterDeriv2,
#endif
        //check_model_state
        double threshhold, double replacement_value, int *flag,
        //f_cuda
        double time_step, int deriv_length_cell, int state_size_cell,
        int n_cells,
        int i_kernel, int threads_block, int n_shr_empty, double *y,
        ModelDataGPU md_object
) //Interface CPU/GPU
{

  cudaDevicef0(
#ifdef CAMP_DEBUG_GPU
          counterDeriv2,
#endif
          //check_model_state
                threshhold, replacement_value, flag,
                //f_cuda
          time_step, deriv_length_cell, state_size_cell,
          n_cells, i_kernel, threads_block, n_shr_empty, y,
          md_object
  );
}



/** Old routine
 */
__global__ void solveDerivative(
#ifdef CAMP_DEBUG_GPU
        int counterDeriv2,
#endif
        double time_step, int deriv_length_cell, int state_size_cell,
        int n_cells,
        int i_kernel, int threads_block, int n_shr_empty, double *y,
        double threshhold, double replacement_value, ModelDataGPU md_object
) //Interface CPU/GPU
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int tid_cell=tid%deriv_length_cell;
  int active_threads = n_cells*deriv_length_cell;
  ModelDataGPU *md = &md_object;

#ifdef DEBUG_DERIV_GPU
  if(tid==0){
    printf("[DEBUG] GPU solveDerivative tid %d, \n", tid);
  }__syncthreads();
#endif

  if(tid<active_threads){

#ifdef DEBUG_solveDerivative_J_DERIV_IN_CPU
#else

    //N_VLinearSum(1.0, y, -1.0, md->J_state, md->J_tmp);
    cudaDevicezaxpby(1.0, y, -1.0, md->J_state, md->J_tmp, active_threads);
    //SUNMatMatvec(md->J_solver, md->J_tmp, md->J_tmp2);
    cudaDeviceSpmvCSC_block(md->J_tmp2, md->J_tmp, md->J_solver, md->jA, md->iA);
    //N_VLinearSum(1.0, md->J_deriv, 1.0, md->J_tmp2, md->J_tmp);
    cudaDevicezaxpby(1.0, md->J_deriv, 1.0, md->J_tmp2, md->J_tmp, active_threads);
    cudaDevicesetconst(md->J_tmp2, 0.0, active_threads); //Reset for next iter


#endif

    //Debug
    //printf("HOLA\n");
    /*
    if(counterDeriv2<=1){
      printf("(%d) y %-le J_state %-le J_solver %-le J_tmp %-le J_tmp2 %-le J_deriv %-le\n",tid+1,
             y[tid], md->J_state[tid], md->J_solver[tid], md->J_tmp[tid], md->J_tmp2[tid], md->J_deriv[tid]);
      //printf("gpu threads %d\n", active_threads);
    }
*/

    TimeDerivativeGPU deriv_data;
    deriv_data.num_spec = deriv_length_cell*n_cells;

#ifdef AEROS_CPU
#else
    deriv_data.production_rates = md->production_rates;
    deriv_data.loss_rates = md->loss_rates;
    time_derivative_reset_gpu(deriv_data);
    __syncthreads();
#endif

    int i_cell = tid/deriv_length_cell;
    md->i_cell = i_cell;
    deriv_data.production_rates = &( md->production_rates[deriv_length_cell*i_cell]);
    deriv_data.loss_rates = &( md->loss_rates[deriv_length_cell*i_cell]);

    md->grid_cell_state = &( md->state[state_size_cell*i_cell]);
    md->grid_cell_env = &( md->env[CAMP_NUM_ENV_PARAM_*i_cell]);

    //Filter threads for n_rxn
    int n_rxn = md->n_rxn;
    if( tid_cell < n_rxn) {
      int n_iters = n_rxn / deriv_length_cell;
      //Repeat if there are more reactions than species
      for (int i = 0; i < n_iters; i++) {
        md->i_rxn = tid_cell + i*deriv_length_cell;

        solveRXN0(deriv_data, time_step, md);
      }

      //Limit tid to pending rxns to compute
      int residual=n_rxn-(deriv_length_cell*n_iters);
      if(tid_cell < residual){
        md->i_rxn = tid_cell + deriv_length_cell*n_iters;

        solveRXN0(deriv_data, time_step, md);
      }
    }
    __syncthreads();

    /*if(tid==0){
      printf("tid %d time_deriv.production_rates %-le time_deriv.loss_rates %-le\n",
              tid, deriv_data.production_rates[tid],
             deriv_data.loss_rates[tid]);
    }*/

    deriv_data.production_rates = md->production_rates;
    deriv_data.loss_rates = md->loss_rates;
    __syncthreads();
    time_derivative_output_gpu(deriv_data, md->deriv_data, md->J_tmp,0);

    /*
    if(tid<deriv_data.num_spec && tid>1022){
      //if(tid<1){
      //deriv_init[tid] = deriv_data.production_rates[tid];
      //deriv_init[tid] = deriv_data.loss_rates[tid];
      printf("tid %d time_deriv.production_rates %-le time_deriv.loss_rates %-le"
             "deriv_init %-le\n",
             tid, deriv_data.production_rates[tid],
             deriv_data.loss_rates[tid],
             //deriv_data.loss_rates[tid]);
             deriv_init[tid]);
    }*/

  }

}

int rxn_calc_deriv_gpu(SolverData *sd, N_Vector y, N_Vector deriv, double time_step) {

  ModelData *md = &(sd->model_data);
  ModelDataGPU *mGPU;

  double *total_state = md->total_state;
  double *deriv_data = N_VGetArrayPointer(deriv);
  if(sd->use_gpu_cvode==0){

    for (int iDevice = sd->startDevice; iDevice < sd->endDevice; iDevice++) {
      cudaSetDevice(iDevice);
      sd->mGPU = &(sd->mGPUs[iDevice]);
      mGPU = sd->mGPU;

      HANDLE_ERROR(cudaMemcpy(mGPU->deriv_data, deriv_data, mGPU->deriv_size, cudaMemcpyHostToDevice));
      HANDLE_ERROR(cudaMemcpy(mGPU->state, total_state, mGPU->state_size, cudaMemcpyHostToDevice));

      total_state += mGPU->state_size_cell * mGPU->n_cells;
      deriv_data += mGPU->nrows;
    }

  }else{

    ModelDataGPU *mGPU = sd->mGPU;
    int n_cells = md->n_cells;
    int n_kernels = 1; // Divide load into multiple kernel calls
    int total_threads = mGPU->nrows/n_kernels;
    int n_shr_empty = mGPU->max_n_gpu_thread%mGPU->nrows;
    int threads_block = mGPU->max_n_gpu_thread - n_shr_empty; //last multiple of size_cell before max_threads
    int n_blocks = ((total_threads + threads_block - 1) / threads_block);
    double *J_tmp = N_VGetArrayPointer(md->J_tmp);

    //Update state
    double replacement_value = TINY;
    double threshhold = -SMALL;
    int flag = CAMP_SOLVER_SUCCESS; //0
#ifdef DEBUG_rxn_calc_deriv_gpu
    printf("rxn_calc_deriv_gpu start\n");
#endif

    if (camp_solver_check_model_state_gpu(y, sd, -SMALL, TINY) != CAMP_SOLVER_SUCCESS)
      return 1;

   //debug
   /*
    if(sd->counterDerivGPU<=0){
      printf("f_cuda start total_state [(id),conc], n_state_var %d, n_cells %d\n", md->n_per_cell_state_var, n_cells);
      printf("n_deriv %d\n", md->n_per_cell_dep_var);
      for (int i = 0; i < md->n_per_cell_state_var*n_cells; i++) {
        printf("(%d) %-le \n",i+1, md->total_state[i]);
      }
    }
    */

#ifdef CAMP_DEBUG_GPU
  //timeDerivSend += (clock() - t1);
  //clock_t t2 = clock();

    cudaEventRecord(md->startDerivKernel);

#endif

#ifdef AEROS_CPU

    update_aero_contrib_gpu(sd);

#endif

#ifdef DEBUG_solveDerivative_J_DERIV_IN_CPU

    HANDLE_ERROR(cudaMemcpy(mGPU->J_tmp, J_tmp, mGPU->deriv_size, cudaMemcpyHostToDevice));

#endif

  //Loop to test multiple kernel executions
    for (int i_kernel=0; i_kernel<n_kernels; i_kernel++){
      //cudaDeviceSynchronize();
      //solveDerivative << < (n_blocks), threads_block >> >(
      cudaGlobalf << < (n_blocks), threads_block >> >(
#ifdef CAMP_DEBUG_GPU
     sd->counterDerivGPU,
#endif
      //update_state
      threshhold, replacement_value, &flag,
       //f_cuda
      time_step, md->n_per_cell_dep_var,
       md->n_per_cell_state_var,n_cells,
       i_kernel, threads_block,n_shr_empty, mGPU->dcv_y,
       *sd->mGPU
       );
    }

    if(flag==CAMP_SOLVER_FAIL)
      return flag;

#ifdef CAMP_DEBUG_GPU
    cudaEventRecord(md->stopDerivKernel);
    cudaEventSynchronize(md->stopDerivKernel);
    float msDerivKernel = 0.0;
    cudaEventElapsedTime(&msDerivKernel, md->startDerivKernel, md->stopDerivKernel);
   md->timeDerivKernel+= msDerivKernel;
#endif
      HANDLE_ERROR(cudaMemcpy(deriv_data, mGPU->deriv_data, mGPU->deriv_size, cudaMemcpyDeviceToHost));
  }
  return 0;
}

void free_gpu_cu(SolverData *sd) {

  ModelDataGPU *mGPU = sd->mGPU;

  //printf("free_gpu_cu start\n");

  free(sd->flagCells);

  for (int iDevice = sd->startDevice; iDevice < sd->endDevice; iDevice++) {
    cudaSetDevice(iDevice);
    sd->mGPU = &(sd->mGPUs[iDevice]);
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
    cudaFree(mGPU->jac_aux);
    cudaFree(mGPU->indexvals_gpu);
    cudaFree(mGPU->indexptrs_gpu);
    cudaFree(mGPU->map_state_derivCPU);
    cudaFree(mGPU->mdv);
    cudaFree(mGPU->mdvo);
    cudaFree(mGPU);
  }
}
/* Auxiliar functions */
void bubble_sort_gpu(unsigned int *n_zeros, unsigned int *rxn_position, int n_rxn){

  int tmp,s=1,i_rxn=n_rxn;

  while(s){
    s=0;
    for (int i = 1; i < i_rxn; i++) {
      //Few zeros go first
      if (n_zeros[i] < n_zeros[i - 1]) {
        //Swap positions
        tmp = rxn_position[i];
        rxn_position[i] = rxn_position[i - 1];
        rxn_position[i - 1] = tmp;

        tmp = n_zeros[i];
        n_zeros[i] = n_zeros[i - 1];
        n_zeros[i - 1] = tmp;
        s=1;
      }
    }
    i_rxn--;
  }

}



/* Prints */

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


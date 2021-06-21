/* Copyright (C) 2019 Christian Guzman
 * Licensed under the GNU General Public License version 1 or (at your
 * option) any later version. See the file COPYING for details.
 *
 * Interface Host-Device (CPU-GPU) to compute reaction-specific functions on GPU
 *
 */

#include "f_jac.h"
/*
#include "itsolver_gpu.h"

extern "C" {
#include "f_jac.h"
//#include "rxns_gpu.h"
#include "aeros/aero_rep_gpu_solver.h"
#include "time_derivative_gpu.h"
}



// Reaction types (Must match parameters defined in pmc_rxn_factory)
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

#define STREAM_RXN_ENV_GPU 0
#define STREAM_ENV_GPU 1
#define STREAM_DERIV_GPU 2

// Status codes for calls to camp_solver functions
#define CAMP_SOLVER_SUCCESS 0
#define CAMP_SOLVER_FAIL 1

/*

__device__ void solveRXN(
#ifdef BASIC_CALC_DERIV
        double *deriv_data,
#else
        TimeDerivativeGPU deriv_data,
#endif
       double time_step,
       ModelDataGPU *md
)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

#ifdef REVERSE_INT_FLOAT_MATRIX

  double *rxn_float_data = &( md->rxn_double[md->i_rxn]);
  int *int_data = &(md->rxn_int[md->i_rxn]);
  int rxn_type = int_data[0];
  int *rxn_int_data = (int *) &(int_data[1*md->n_rxn]);

#else

  double *rxn_float_data = (double *)&( md->rxn_double[md->rxn_float_indices[md->i_rxn]]);
  int *int_data = (int *)&(md->rxn_int[md->rxn_int_indices[md->i_rxn]]);

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
      printf("RXN_EMISSION");
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
#ifdef BASIC_CALC_DERIV
#else
      rxn_gpu_troe_calc_deriv_contrib(md, deriv_data, rxn_int_data,
                                      rxn_float_data, rxn_env_data,time_step);
#endif
      break;
    case RXN_WET_DEPOSITION :
      printf("RXN_WET_DEPOSITION");
      //rxn_gpu_wet_deposition_calc_deriv_contrib(md, deriv_data, rxn_int_data,
      //                                     rxn_float_data, rxn_env_data,time_step);
      break;
  }


}

__device__ void cudaDevicecalc_deriv(
#ifdef PMC_DEBUG_GPU
        int counterDeriv2,
#endif
        //check_model_state
        //double threshhold, double replacement_value, int *flag,
        //f_gpu
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
  cudaDeviceSpmvCSC_block(md->J_tmp2, md->J_tmp, active_threads, md->J_solver, md->jJ_solver, md->iJ_solver, 0);
  //N_VLinearSum(1.0, md->J_deriv, 1.0, md->J_tmp2, md->J_tmp);
  cudaDevicezaxpby(1.0, md->J_deriv, 1.0, md->J_tmp2, md->J_tmp, active_threads);
  cudaDevicesetconst(md->J_tmp2, 0.0, active_threads); //Reset for next iter


#endif

#ifdef BASIC_CALC_DERIV
    md->i_rxn=tid%n_rxn;
    double *deriv_init = md->deriv_data;
    md->deriv_data = &( md->deriv_init[deriv_length_cell*md->i_cell]);
    if(tid < n_rxn*n_cells){
        solveRXN(deriv_data, time_step, md);
    }
#else
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
    md->grid_cell_env = &( md->env[PMC_NUM_ENV_PARAM_*i_cell]);

    //Filter threads for n_rxn
    int n_rxn = md->n_rxn;
    if( tid_cell < n_rxn) {
      int n_iters = n_rxn / deriv_length_cell;
      //Repeat if there are more reactions than species
      for (int i = 0; i < n_iters; i++) {
        md->i_rxn = tid_cell + i*deriv_length_cell;

        solveRXN(deriv_data, time_step, md);
      }

      //Limit tid to pending rxns to compute
      int residual=n_rxn-(deriv_length_cell*n_iters);
      if(tid_cell < residual){
        md->i_rxn = tid_cell + deriv_length_cell*n_iters;

        solveRXN(deriv_data, time_step, md);
      }
    }
    __syncthreads();

    deriv_data.production_rates = md->production_rates;
    deriv_data.loss_rates = md->loss_rates;
    __syncthreads();
    time_derivative_output_gpu(deriv_data, md->deriv_data, md->J_tmp,0);
#endif



  }

}

__device__
void cudaDevicef(
#ifdef PMC_DEBUG_GPU
        int counterDeriv2,
#endif
        //check_model_state
        double threshhold, double replacement_value, int *flag,
        //f_gpu
        double time_step, int deriv_length_cell, int state_size_cell,
        int n_cells,
        int i_kernel, int threads_block, int n_shr_empty, double *y,
        ModelDataGPU md_object
) //Interface CPU/GPU
{

  ModelDataGPU *md = &md_object;

  cudaDevicecamp_solver_check_model_state(md->state, y,
                                          md->map_state_deriv, threshhold, replacement_value,
                                          flag, deriv_length_cell, n_cells);

  __syncthreads;
  //study flag block effect: flag is global for all threads or for only the block?
  if(*flag==CAMP_SOLVER_FAIL)
    return;

  cudaDevicecalc_deriv(
#ifdef PMC_DEBUG_GPU
           counterDeriv2,
#endif
        //check_model_state          md->map_state_deriv, threshhold, replacement_value, flag,
          //f_gpu
        time_step, deriv_length_cell, state_size_cell,
           n_cells, i_kernel, threads_block, n_shr_empty, y,
           md_object
          );
}
//*/

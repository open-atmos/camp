/* Copyright (C) 2020 Christian Guzman
 * Licensed under the GNU General Public License version 1 or (at your
 * option) any later version. See the file COPYING for details.
 *
 * ODE GPU solver
 *
 */

#include "itsolver_gpu.h"

extern "C" {
#include "cvode_gpu.h"
//#include "cuda_structs.h"
#include "rxns_gpu.h"
#include "aeros/aero_rep_gpu_solver.h"
#include "time_derivative_gpu.h"
#include "Jacobian_gpu.h"

}

#include "cusolver.h"

#ifdef CAMP_USE_MPI
#include <mpi.h>
#endif


#define CV_SUCCESS               0

#define DO_ERROR_TEST    +2
#define PREDICT_AGAIN    +3
#define CONV_FAIL        +4
#define TRY_AGAIN        +5
#define FIRST_CALL       +6
#define PREV_CONV_FAIL   +7
#define PREV_ERR_FAIL    +8
#define RHSFUNC_RECVR    +9

#define NUM_TESTS    5     /* number of error test quantities     */

/*=================================================================*/
/*             CVODE Private Constants                             */
/*=================================================================*/

//#define ZERO    RCONST(0.0)     /* real 0.0     */
//#define TINY    RCONST(1.0e-10) /* small number */
#define PT1     RCONST(0.1)     /* real 0.1     */
#define POINT2  RCONST(0.2)     /* real 0.2     */
#define FOURTH  RCONST(0.25)    /* real 0.25    */
//#define HALF    RCONST(0.5)     /* real 0.5     */
//#define ONE     RCONST(1.0)     /* real 1.0     */
#define TWO     RCONST(2.0)     /* real 2.0     */
#define THREE   RCONST(3.0)     /* real 3.0     */
#define FOUR    RCONST(4.0)     /* real 4.0     */
#define FIVE    RCONST(5.0)     /* real 5.0     */
#define TWELVE  RCONST(12.0)    /* real 12.0    */
#define HUNDRED RCONST(100.0)   /* real 100.0   */
#define CAMP_TINY RCONST(1.0e-30) /* small number for CAMP */

/*=================================================================*/
/*             CVODE Routine-Specific Constants                    */
/*=================================================================*/

/*
 * Control constants for lower-level functions used by cvStep
 * ----------------------------------------------------------
 *
 * cvHin return values:
 *    CV_SUCCESS
 *    CV_RHSFUNC_FAIL
 *    CV_TOO_CLOSE
 *
 * cvStep control constants:
 *    DO_ERROR_TEST
 *    PREDICT_AGAIN
 *
 * cvStep return values:
 *    CV_SUCCESS,
 *    CV_LSETUP_FAIL,  CV_LSOLVE_FAIL,
 *    CV_RHSFUNC_FAIL, CV_RTFUNC_FAIL
 *    CV_CONV_FAILURE, CV_ERR_FAILURE,
 *    CV_FIRST_RHSFUNC_ERR
 *
 * cvNls input nflag values:
 *    FIRST_CALL
 *    PREV_CONV_FAIL
 *    PREV_ERR_FAIL
 *
 * cvNls return values:
 *    CV_SUCCESS,
 *    CV_LSETUP_FAIL, CV_LSOLVE_FAIL, CV_RHSFUNC_FAIL,
 *    CONV_FAIL, RHSFUNC_RECVR
 *
 * cvNewtonIteration return values:
 *    CV_SUCCESS,
 *    CV_LSOLVE_FAIL, CV_RHSFUNC_FAIL
 *    CONV_FAIL, RHSFUNC_RECVR,
 *    TRY_AGAIN
 *
 */

#define DO_ERROR_TEST    +2
#define PREDICT_AGAIN    +3

#define CONV_FAIL        +4
#define TRY_AGAIN        +5

#define FIRST_CALL       +6
#define PREV_CONV_FAIL   +7
#define PREV_ERR_FAIL    +8

#define RHSFUNC_RECVR    +9

/*
 * Control constants for lower-level rootfinding functions
 * -------------------------------------------------------
 *
 * cvRcheck1 return values:
 *    CV_SUCCESS,
 *    CV_RTFUNC_FAIL,
 * cvRcheck2 return values:
 *    CV_SUCCESS
 *    CV_RTFUNC_FAIL,
 *    CLOSERT
 *    RTFOUND
 * cvRcheck3 return values:
 *    CV_SUCCESS
 *    CV_RTFUNC_FAIL,
 *    RTFOUND
 * cvRootfind return values:
 *    CV_SUCCESS
 *    CV_RTFUNC_FAIL,
 *    RTFOUND
 */

#define RTFOUND          +1
#define CLOSERT          +3

/*
 * Control constants for tolerances
 * --------------------------------
 */

#define CV_NN  0
#define CV_SS  1
#define CV_SV  2
#define CV_WF  3

/*
 * Algorithmic constants
 * ---------------------
 *
 * CVodeGetDky and cvStep
 *
 *    FUZZ_FACTOR
 *
 * cvHin
 *
 *    HLB_FACTOR
 *    HUB_FACTOR
 *    H_BIAS
 *    MAX_ITERS
 *
 * CVodeCreate
 *
 *   CORTES
 *
 * cvStep
 *
 *    THRESH
 *    ETAMX1
 *    ETAMX2
 *    ETAMX3
 *    ETAMXF
 *    ETAMIN
 *    ETACF
 *    ADDON
 *    BIAS1
 *    BIAS2
 *    BIAS3
 *    ONEPSM
 *
 *    SMALL_NST   nst > SMALL_NST => use ETAMX3
 *    MXNCF       max no. of convergence failures during one step try
 *    MXNEF       max no. of error test failures during one step try
 *    MXNEF1      max no. of error test failures before forcing a reduction of order
 *    SMALL_NEF   if an error failure occurs and SMALL_NEF <= nef <= MXNEF1, then
 *                reset eta =  SUNMIN(eta, ETAMXF)
 *    LONG_WAIT   number of steps to wait before considering an order change when
 *                q==1 and MXNEF1 error test failures have occurred
 *
 * cvNls
 *
 *    NLS_MAXCOR  maximum no. of corrector iterations for the nonlinear solver
 *    CRDOWN      constant used in the estimation of the convergence rate (crate)
 *                of the iterates for the nonlinear equation
 *    DGMAX       iter == CV_NEWTON, |gamma/gammap-1| > DGMAX => call lsetup
 *    RDIV        declare divergence if ratio del/delp > RDIV
 *    MSBP        max no. of steps between lsetup calls
 *
 */

#define FUZZ_FACTOR RCONST(100.0)

#define HLB_FACTOR RCONST(100.0)
#define HUB_FACTOR RCONST(0.1)
#define H_BIAS     HALF
#define MAX_ITERS  4000

#define CORTES RCONST(0.1)

#define THRESH RCONST(1.5)
#define ETAMX1 RCONST(10000.0)
#define ETAMX2 RCONST(10.0)
#define ETAMX3 RCONST(10.0)
#define ETAMXF RCONST(0.2)
#define ETAMIN RCONST(0.1)
#define ETACF  RCONST(0.25)
#define ADDON  RCONST(0.000001)
#define BIAS1  RCONST(6.0)
#define BIAS2  RCONST(6.0)
#define BIAS3  RCONST(10.0)
#define ONEPSM RCONST(1.000001)

#define SMALL_NST    10
#define MXNCF        10
#define MXNEF         7
#define MXNEF1        3
#define SMALL_NEF     2
#define LONG_WAIT    10

#define NLS_MAXCOR 3
#define CRDOWN RCONST(0.3)
#define DGMAX  RCONST(0.3)

#define RDIV      TWO
#define MSBP       20



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

#define STREAM_RXN_ENV_GPU 0
#define STREAM_ENV_GPU 1
#define STREAM_DERIV_GPU 2

// Status codes for calls to camp_solver functions
#define CAMP_SOLVER_SUCCESS 0
#define CAMP_SOLVER_FAIL 1

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

static void HandleError(cudaError_t err,
                        const char *file,
                        int line) {
  if (err != cudaSuccess) {
    printf("%s in %s at line %d\n", cudaGetErrorString(err),
           file, line);
    exit(EXIT_FAILURE);
  }
}

/*
void check_isnand(double *x, int len, int var_id){

  int n_zeros=0;
  for (int i=0; i<len; i++){
    if(isnan(x[i]))
      printf("NAN %d[%d]",var_id,i);
  }

}*/

void check_isnand(double *x, int len, const char *s){

  for (int i=0; i<len; i++){
    if(isnan(x[i])){
      printf("NAN %s[%d]",s,i);
      exit(0);
    }
  }

}

__global__
void check_isnand_global(double *x, int len, int var_id)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  //if(i<2)
  if(i<len)
  {
    if(isnan(x[i]))
      printf("NAN %d[%d]",var_id,i);
    //printf("%d[%d]=%-le\n",var_id,i,x[i]);
  }
}

__global__
void check_isnand_global0(double *x, int len, int var_id)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i==0)
    for (int i=0; i<len; i++){
      if(isnan(x[i]))
        printf("NAN %d[%d]",var_id,i);
    }
}

int compare_doubles(double *x, double *y, int len, const char *s) {

  int flag = 1;
  double tol = 0.01;
  //float tol=0.0001;
  double rel_error;
  int n_fails = 0;
  for (int i = 0; i < len; i++) {
    if (x[i] == 0)
      rel_error = 0.;
    else
      rel_error = abs((x[i] - y[i]) / x[i]);
    //rel_error=(x[i]-y[i]/(x[i]+1.0E-60));
    if (rel_error > tol) {
      printf("compare_doubles %s rel_error %le for tol %le at [%d]: %le vs %le\n",
             s, rel_error, tol, i, x[i], y[i]);
      flag = 0;
      n_fails++;
      if (n_fails == 4)
        return flag;
    }
  }
  return flag;
}

int compare_ints(int *x, int *y, int len, const char *s) {

  int flag = 1;
  double tol = 0.01;
  //float tol=0.0001;
  double rel_error;
  int n_fails = 0;
  for (int i = 0; i < len; i++) {
    if (x[i] == 0)
      rel_error = 0.;
    else
      rel_error = abs((x[i] - y[i]) / x[i]);
    //rel_error=(x[i]-y[i]/(x[i]+1.0E-60));
    if (rel_error > tol) {
      printf("compare_ints %s rel_error %le for tol %le at [%d]: %d vs %d\n",
             s, rel_error, tol, i, x[i], y[i]);
      flag = 0;
      n_fails++;
      if (n_fails == 4)
        return flag;
    }
  }
  return flag;
}


int nextPowerOfTwoCVODE(int v){

  v--;
  v |= v >> 1;
  v |= v >> 2;
  v |= v >> 4;
  v |= v >> 8;
  v |= v >> 16;
  v++;

  return v;
}

__device__
void printmin(ModelDataGPU *md,double* y, const char *s) {

  __syncthreads();
  extern __shared__ double flag_shr2[];
  int i= threadIdx.x + blockDim.x*blockIdx.x;
  __syncthreads();

  double min;
  cudaDevicemin(&min, y[i], flag_shr2, md->n_shr_empty);
  __syncthreads();
  if(i==0)printf("%s min %le\n",s,min);
  __syncthreads();

}

//Copy A to B
__device__
void cudaDeviceJacCopy(int n_row, int* Ap, int* Aj, double* Ax, int* Bp, int* Bi, double* Bx) {

  int i= threadIdx.x + blockDim.x*blockIdx.x;
  __syncthreads();

  for(int j=Ap[i]; j<Ap[i+1]; j++){
    Bi[j]=Aj[j];
    Bx[j]=Ax[j];
  }

  Bp[i]=Ap[i];
  if(i==gridDim.x*blockDim.x-1){
    Bp[i+1]=Ap[i+1];
  }

  __syncthreads();

}

__device__
int cudaDevicecamp_solver_check_model_state(ModelDataGPU *md, ModelDataVariable *dmdv, double *y, int *flag)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  __syncthreads();
  extern __shared__ int flag_shr[];
  flag_shr[0] = 0;

  //printmin(md,md->state,"cudaDevicecamp_solver_check_model_state start state");

  __syncthreads();
  if (y[i] < md->threshhold) {
    flag_shr[0] = CAMP_SOLVER_FAIL;

#ifdef DEBUG_cudaDevicecamp_solver_check_model_state
    printf("Failed model state update gpu:[spec %d] = %le flag_shr %d\n",i,y[i],flag_shr[0]);
#endif

  } else {
    md->state[md->map_state_deriv[i]] =
            y[i] <= md->threshhold ?
            md->replacement_value : y[i];
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

  __syncthreads();
  *flag = (int)flag_shr[0];
  __syncthreads();
#ifdef DEV_printmin
  printmin(md,md->state,"cudaDevicecamp_solver_check_model_state end state");
#endif

  //printmin(md,y,"cudaDevicecamp_solver_check_model_state end y");
  //printmin(md,md->state,"cudaDevicecamp_solver_check_model_state end state");

#ifdef DEBUG_cudaDevicecamp_solver_check_model_state
  __syncthreads();if(i==0)printf("flag %d flag_shr %d\n",*flag,flag_shr2[0]);
#endif

  return *flag;
}


__device__ void solveRXN(
#ifdef BASIC_CALC_DERIV
        double *deriv_data,
#else
        TimeDerivativeGPU deriv_data,
#endif
        double time_step,
        ModelDataGPU *md, ModelDataVariable *dmdv
)
{

#ifdef REVERSE_INT_FLOAT_MATRIX

  double *rxn_float_data = &( md->rxn_double[dmdv->i_rxn]);
  int *int_data = &(md->rxn_int[dmdv->i_rxn]);
  int rxn_type = int_data[0];
  int *rxn_int_data = (int *) &(int_data[1*md->n_rxn]);

#else

  double *rxn_float_data = (double *)&( md->rxn_double[md->rxn_float_indices[dmdv->i_rxn]]);
  int *int_data = (int *)&(md->rxn_int[md->rxn_int_indices[dmdv->i_rxn]]);

  //double *rxn_float_data = &( md->rxn_double[dmdv->i_rxn]);
  //int *int_data = &(md->rxn_int[dmdv->i_rxn]);


  int rxn_type = int_data[0];
  int *rxn_int_data = (int *) &(int_data[1]);

#endif

  //Get indices for rates
  double *rxn_env_data = &(md->rxn_env_data
  [md->n_rxn_env_data*dmdv->i_cell+md->rxn_env_data_idx[dmdv->i_rxn]]);

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
        double time_step, int deriv_length_cell, int state_size_cell,
        int n_cells,
        int i_kernel, int threads_block, int n_shr_empty, double *y,
        double *yout, ModelDataGPU *md, ModelDataVariable *dmdv
) //Interface CPU/GPU
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int tid_cell=tid%deriv_length_cell;
  //int active_threads = md->n_cells*md->deriv_length_cell;
  int active_threads = md->nrows;

#ifdef DEBUG_DERIV_GPU
  if(tid==0){
    printf("[DEBUG] GPU solveDerivative tid %d, \n", tid);
  }__syncthreads();
#endif

#ifdef DEV_printmin

  //__syncthreads();//no effect, but printmin yes
  printmin(md,yout,"cudaDevicecalc_deriv start end yout");
  printmin(md,md->J_tmp,"cudaDevicecalc_deriv start end J_tmp");
  printmin(md,md->J_state,"cudaDevicecalc_deriv start end J_state");
#endif
    //N_VLinearSum(1.0, y, -1.0, md->J_state, md->J_tmp);
    cudaDevicezaxpby(1.0, y, -1.0, md->J_state, md->J_tmp, active_threads);
    //SUNMatMatvec(md->J_solver, md->J_tmp, md->J_tmp2);
    cudaDeviceSpmvCSC_block(md->J_tmp2, md->J_tmp, active_threads, md->J_solver, md->jJ_solver, md->iJ_solver, 0);
    //N_VLinearSum(1.0, md->J_deriv, 1.0, md->J_tmp2, md->J_tmp);
    cudaDevicezaxpby(1.0, md->J_deriv, 1.0, md->J_tmp2, md->J_tmp, active_threads);
    cudaDevicesetconst(md->J_tmp2, 0.0, active_threads); //Reset for next iter
#ifdef DEV_printmin
    printmin(md,md->J_tmp,"cudaDevicecalc_deriv start end J_tmp");
    printmin(md,md->J_state,"cudaDevicecalc_deriv start end J_state");
#endif
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
    dmdv->i_cell = i_cell;
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
        dmdv->i_rxn = tid_cell + i*deriv_length_cell;

        solveRXN(deriv_data, time_step, md, dmdv);
      }

      //Limit tid to pending rxns to compute
      int residual=n_rxn-(deriv_length_cell*n_iters);
      if(tid_cell < residual){
        dmdv->i_rxn = tid_cell + deriv_length_cell*n_iters;

        solveRXN(deriv_data, time_step, md, dmdv);
      }
    }
    __syncthreads();

    deriv_data.production_rates = md->production_rates;
    deriv_data.loss_rates = md->loss_rates;
#ifdef DEV_printmin
    printmin(md,yout,"cudaDevicecalc_deriv start end yout");
#endif
    __syncthreads();
    time_derivative_output_gpu(deriv_data, yout, md->J_tmp,0);
#ifdef DEV_printmin
    printmin(md,yout,"cudaDevicecalc_deriv start end yout");
#endif

  __syncthreads();

}

__device__
int cudaDevicef(
        double time_step, int deriv_length_cell, int state_size_cell,
        int n_cells,
        int i_kernel, int threads_block, int n_shr_empty, double *y,
        double *yout, ModelDataGPU *md, ModelDataVariable *dmdv, int *flag
) //Interface CPU/GPU
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;

#ifdef CAMP_DEBUG_GPU

  int clock_khz=md->clock_khz;
  clock_t start;
  start = clock();

#endif
#ifdef DEV_printmin
  printmin(md,y,"cudaDevicef Start y");
#endif
  //double time_step = dmdv->cv_next_h; //CVodeGetCurrentStep(sd->cvode_mem, &time_step);
  // On the first call to f(), the time step hasn't been set yet, so use the
  // default value
  time_step = time_step > 0. ? time_step : dmdv->init_time_step;
#ifdef DEV_printmin
  printmin(md,md->state,"cudaDevicef start state");
#endif

  int checkflag=cudaDevicecamp_solver_check_model_state(md, dmdv, y, flag);

  __syncthreads();
  if(checkflag==CAMP_SOLVER_FAIL){
    *flag=CAMP_SOLVER_FAIL;

#ifdef DEV_printmin
    printmin(md,y,"cudaDevicef End y");
#endif

#ifdef CAMP_DEBUG_GPU
#ifdef cudaGlobalCVode_timers_max_blocks
    md->dtf[i] += ((double)(int)(clock() - start))/(clock_khz*1000);
#else
    if(i==0) *md->dtf += ((double)(int)(clock() - start))/(clock_khz*1000);
#endif
#endif

#ifdef DEBUG_cudaDevicef
    if(i==0)printf("cudaDevicef CAMP_SOLVER_FAIL %d\n",i);
#endif
    return CAMP_SOLVER_FAIL;
  }
#ifdef DEV_printmin
  printmin(md,yout,"cudaDevicef End yout");
#endif
  cudaDevicecalc_deriv(
          //f_gpu
          time_step, deriv_length_cell, state_size_cell,
          n_cells, i_kernel, threads_block, n_shr_empty, y,
          yout, md, dmdv
  );

  //printmin(md,yout,"cudaDevicef End yout");
  //printmin(md,y,"cudaDevicef End y");


#ifdef CAMP_DEBUG_GPU
#ifdef cudaGlobalCVode_timers_max_blocks

    md->dtf[i] += ((double)(int)(clock() - start))/(clock_khz*1000);

#else

    if(i==0) *md->dtf += ((double)(int)(clock() - start))/(clock_khz*1000);

#endif
#endif

  __syncthreads();
  *flag=0;
  __syncthreads();

  return 0;

}

__device__
int CudaDeviceguess_helper(double t_n, double h_n, double* y_n,
                           double* y_n1, double* hf, double* tmp1,
                           double* corr, double cv_reltol, int nrows,
        //check_model_state
                           double threshhold, double replacement_value, int *flag,
        //f_gpu
                           double time_step, int deriv_length_cell, int state_size_cell,
                           int n_cells,
                           int i_kernel, int threads_block, int n_shr_empty, //double *y,
                           ModelDataGPU *md, ModelDataVariable *dmdv
) {

  extern __shared__ double flag_shr2[];
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int tid = threadIdx.x;

#ifdef DEBUG_CudaDeviceguess_helper
  if(i==0)printf("CudaDeviceguess_helper start gpu\n");
#endif

  // Only try improvements when negative concentrations are predicted
  //if (N_VMin(y_n) > -SMALL) return 0;
  __syncthreads();

  double min;
  cudaDevicemin(&min, y_n[i], flag_shr2, n_shr_empty);

#ifdef DEBUG_CudaDeviceguess_helper
  if(i==0)printf("min %le -SMALL %le\n",min, -SMALL);
#endif

  if(min>-SMALL){

#ifdef DEBUG_CudaDeviceguess_helper
    if(i==0)printf("Return 0 %le\n",y_n[i]);
#endif

    return 0;
  }

#ifdef CAMP_DEBUG_GPU

  int clock_khz=md->clock_khz;
  clock_t start;
  start = clock();

#endif

  // Copy \f$y(t_{n-1})\f$ to working array
  //N_VScale(ONE, y_n1, tmp1);
  tmp1[i]=y_n1[i];
  __syncthreads();
  // Get  \f$f(t_{n-1})\f$
  /*if (h_n > ZERO) {
    N_VScale(ONE / h_n, hf, corr);
  } else {
    N_VScale(ONE, hf, corr);
  }*/

  if (h_n > 0.) {
    corr[i]=(1./h_n)*hf[i];
  } else {
    corr[i]=hf[i];
  }

  // Advance state interatively
  double t_0 = h_n > 0. ? t_n - h_n : t_n - 1.;
  double t_j = 0.;
  int GUESS_MAX_ITER = 5; //5 //reduce this to improve perf
  __syncthreads();
  for (int iter = 0; iter < GUESS_MAX_ITER && t_0 + t_j < t_n; iter++) {
    // Calculate \f$h_j\f$
    //double h_j = t_n - (t_0 + t_j);
    //int i_fast = -1;
    __syncthreads();

    double h_j = t_n - (t_0 + t_j);
    /*
    for (int i = 0; i < n_elem; i++) {
     realtype t_star = -atmp1[i] / acorr[i];
      if ((t_star > ZERO || (t_star == ZERO && acorr[i] < ZERO)) &&
          t_star < h_j) {
        h_j = t_star;
        i_fast = i;
      }
    }
     */
    __syncthreads();

    double t_star;
    double h_j_init=h_j;

    //if(i==0)printf("*md->h_jPtrInit %le\n",*md->h_jPtr);

    if(corr[i]==0){
      t_star=h_j;
    }else{
      t_star = -tmp1[i] / corr[i];
    }

    if( !(t_star > 0. || (t_star == 0. && corr[i] < 0.)) ){//&&corr[i]==0.)
      t_star=h_j;
    }

    __syncthreads();
    //(blockIdx.x==0 && iter<=0)printf("i %d t_star %le atmp1 %le acorr %le\n",i,t_star,tmp1[i],corr[i]);

    flag_shr2[tid]=h_j_init;
    cudaDevicemin(&h_j, t_star, flag_shr2, n_shr_empty);
    flag_shr2[0]=1;
    __syncthreads();

#ifdef DEBUG_CudaDeviceguess_helper
    //if(tid==0 && iter<=5) printf("CudaDeviceguess_helper h_j %le h_j_init %le t_star %le block %d iter %d\n",h_j,h_j_init,t_star,blockIdx.x,iter);
#endif

    // Scale incomplete jumps
    /*
    if (i_fast >= 0 && h_n > ZERO)
      h_j *= 0.95 + 0.1 ;
    //h_j *= 0.95 + 0.1 * rand() / (double)RAND_MAX;
    h_j = t_n < t_0 + t_j + h_j ? t_n - (t_0 + t_j) : h_j;
     */

    //if (i_fast >= 0 && h_n > 0.)
    if (h_n > 0.)
      h_j *= 0.95 + 0.1; //todo use the same than cpu version( pull request 8 github)
    //h_j *= 0.95 + 0.1 * rand() / (double)RAND_MAX;
    h_j = t_n < t_0 + t_j + h_j ? t_n - (t_0 + t_j) : h_j;


    __syncthreads();
    // Only make small changes to adjustment vectors used in Newton iteration
    if (h_n == 0. &&
        t_n - (h_j + t_j + t_0) > cv_reltol) {

#ifdef DEBUG_CudaDeviceguess_helper
      if(i==0)printf("CudaDeviceguess_helper small changes \n");
#endif


#ifdef CAMP_DEBUG_GPU
#ifdef cudaGlobalCVode_timers_max_blocks
    md->dtguess_helper[i] += ((double)(clock() - start))/(clock_khz*1000);
#else
    if(i==0) *md->dtguess_helper += ((double)(clock() - start))/(clock_khz*1000);
#endif
#endif

    return -1;
    }

    // Advance the state
    //N_VLinearSum(ONE, tmp1, h_j, corr, tmp1);
    cudaDevicezaxpby(1., tmp1, h_j, corr, tmp1, md->nrows);

    __syncthreads();
    // Advance t_j
    t_j += h_j;

#ifdef DEBUG_CudaDeviceguess_helper
    //  printf("dcorr[%d] %le dhf %le dt_star %le dh_j %le dh_n %le\n",
    //         i,corr[i],hf[i],t_star,h_j,h_n);

    //if(i==0)
    //  for(int j=0;j<nrows;j++)
    //    printf("dcorr[%d] %le dtmp1 %le dhf %le dt_star %le dh_j %le dh_n %le\n",
    //           j,corr[j],tmp1[j],hf[j],t_star,h_j,h_n);

#endif

    // Recalculate the time derivative \f$f(t_j)\f$
    /*
    if (f(t_0 + t_j, tmp1, corr, solver_data) != 0) {
      N_VConst(ZERO, corr);
      return -1;
    }*/


//#ifndef DEV_CudaDeviceguess_helper
//#else

#ifdef DEV_printmin
    printmin(md,md->state,"cudaDevicef start state");
#endif

    int aux_flag=0;

    int fflag=cudaDevicef(
            t_0 + t_j, md->deriv_length_cell, md->state_size_cell,
            md->n_cells, md->i_kernel, threads_block,
            md->n_shr_empty, tmp1, corr,md,dmdv,&aux_flag
    );
#ifdef DEV_printmin
    printmin(md,tmp1,"cudaDevicef end tmp1");
#endif
    __syncthreads();

    if (fflag == CAMP_SOLVER_FAIL) {
      //N_VConst(ZERO, corr);
      corr[i] = 0.;

#ifdef DEBUG_CudaDeviceguess_helper
      if(i==0)printf("CudaDeviceguess_helper df(t)\n");
#endif

#ifdef CAMP_DEBUG_GPU
#ifdef cudaGlobalCVode_timers_max_blocks
    md->dtguess_helper[i] += ((double)(clock() - start))/(clock_khz*1000);
#else
    if(i==0) *md->dtguess_helper += ((double)(clock() - start))/(clock_khz*1000);
#endif
#endif

     return -1;
    }

//#endif
//DEV_CudaDeviceguess_helper

    //((CVodeMem)sd->cvode_mem)->cv_nfe++;

    if (iter == GUESS_MAX_ITER - 1 && t_0 + t_j < t_n) {
      if (h_n == 0.){


#ifdef CAMP_DEBUG_GPU
#ifdef cudaGlobalCVode_timers_max_blocks
    md->dtguess_helper[i] += ((double)(clock() - start))/(clock_khz*1000);
#else
    if(i==0) *md->dtguess_helper += ((double)(clock() - start))/(clock_khz*1000);
#endif
#endif

        return -1;
      }
    }
    __syncthreads();
  }

  __syncthreads();
#ifdef DEBUG_CudaDeviceguess_helper
   if(i==0)printf("CudaDeviceguess_helper return 1\n");
#endif

  // Set the correction vector
  //N_VLinearSum(ONE, tmp1, -ONE, y_n, corr);
  cudaDevicezaxpby(1., tmp1, -1., y_n, corr, md->nrows);


  // Scale the initial corrections
  //if (h_n > 0.) N_VScale(0.999, corr, corr);
  if (h_n > 0.) corr[i]=corr[i]*0.999;

  // Update the hf vector
  //N_VLinearSum(ONE, tmp1, -ONE, y_n1, hf);
  cudaDevicezaxpby(1., tmp1, -1., y_n1, hf, md->nrows);


#ifdef CAMP_DEBUG_GPU
#ifdef cudaGlobalCVode_timers_max_blocks
    md->dtguess_helper[i] += ((double)(clock() - start))/(clock_khz*1000);
#else
    if(i==0) *md->dtguess_helper += ((double)(clock() - start))/(clock_khz*1000);
#endif
#endif


  __syncthreads();
  return 1;
}


__device__ void solveRXNJac(
        JacobianGPU jac,
        double time_step,
        ModelDataGPU *md, ModelDataVariable *dmdv
)
{

#ifdef CAMP_DEBUG_GPU

  int clock_khz=md->clock_khz;
  clock_t start;
  start = clock();

#endif

#ifdef REVERSE_INT_FLOAT_MATRIX

  double *rxn_float_data = &( md->rxn_double[dmdv->i_rxn]);
  int *int_data = &(md->rxn_int[dmdv->i_rxn]);
  int rxn_type = int_data[0];
  int *rxn_int_data = (int *) &(int_data[1*md->n_rxn]);

#else

  double *rxn_float_data = (double *)&( md->rxn_double[md->rxn_float_indices[dmdv->i_rxn]]);
  int *int_data = (int *)&(md->rxn_int[md->rxn_int_indices[dmdv->i_rxn]]);

  //double *rxn_float_data = &( md->rxn_double[dmdv->i_rxn]);
  //int *int_data = &(md->rxn_int[dmdv->i_rxn]);

  int rxn_type = int_data[0];
  int *rxn_int_data = (int *) &(int_data[1]);

#endif

  //Get indices for rates
  double *rxn_env_data = &(md->rxn_env_data
  [md->n_rxn_env_data*dmdv->i_cell+md->rxn_env_data_idx[dmdv->i_rxn]]);

#ifdef DEBUG_solveRXNJac
  if(tid==0){
    printf("[DEBUG] GPU solveRXN tid %d, \n", tid);
  }
#endif

  switch (rxn_type) {
    //case RXN_AQUEOUS_EQUILIBRIUM :
    //fix run-time error
    //rxn_gpu_aqueous_equilibrium_calc_jac_contrib(md, jac, rxn_int_data,
    //                                               rxn_float_data, rxn_env_data,time_step);
    //break;
    case RXN_ARRHENIUS :
      rxn_gpu_arrhenius_calc_jac_contrib(md, jac, rxn_int_data,
                                         rxn_float_data, rxn_env_data,time_step);
      break;
    case RXN_CMAQ_H2O2 :
      rxn_gpu_CMAQ_H2O2_calc_jac_contrib(md, jac, rxn_int_data,
                                         rxn_float_data, rxn_env_data,time_step);
      break;
    case RXN_CMAQ_OH_HNO3 :
      rxn_gpu_CMAQ_OH_HNO3_calc_jac_contrib(md, jac, rxn_int_data,
                                            rxn_float_data, rxn_env_data,time_step);
      break;
    case RXN_CONDENSED_PHASE_ARRHENIUS :
      //rxn_gpu_condensed_phase_arrhenius_calc_jac_contrib(md, jac, rxn_int_data,
      //                                     rxn_float_data, rxn_env_data,time_step);
      break;
    case RXN_EMISSION :
      //printf("RXN_EMISSION");
      //rxn_gpu_emission_calc_jac_contrib(md, jac, rxn_int_data,
      //                                     rxn_float_data, rxn_env_data,time_step);
      break;
    case RXN_FIRST_ORDER_LOSS :
      //rxn_gpu_first_order_loss_calc_jac_contrib(md, jac, rxn_int_data,
      //                                     rxn_float_data, rxn_env_data,time_step);
      break;
    case RXN_HL_PHASE_TRANSFER :
      //rxn_gpu_HL_phase_transfer_calc_jac_contrib(md, jac, rxn_int_data,
      //                                             rxn_float_data, rxn_env_data,time_stepn);
      break;
    case RXN_PHOTOLYSIS :
      rxn_gpu_photolysis_calc_jac_contrib(md, jac, rxn_int_data,
                                          rxn_float_data, rxn_env_data,time_step);
      break;
    case RXN_SIMPOL_PHASE_TRANSFER :
      //rxn_gpu_SIMPOL_phase_transfer_calc_jac_contrib(md, jac,
      //        rxn_int_data, rxn_float_data, rxn_env_data, time_step);
      break;
    case RXN_TROE :
      rxn_gpu_troe_calc_jac_contrib(md, jac, rxn_int_data,
                                    rxn_float_data, rxn_env_data,time_step);
      break;
    case RXN_WET_DEPOSITION :
      //printf("RXN_WET_DEPOSITION");
      //rxn_gpu_wet_deposition_calc_jac_contrib(md, jac, rxn_int_data,
      //                                     rxn_float_data, rxn_env_data,time_step);
      break;
  }
/*
*/

#ifdef CAMP_DEBUG_GPU
  int i = blockIdx.x * blockDim.x + threadIdx.x;
#ifdef cudaGlobalCVode_timers_max_blocks

  md->dtRXNJac[i] += ((double)(clock() - start))/(clock_khz*1000);

#else

  if(i==0) *md->dtRXNJac += ((double)(clock() - start))/(clock_khz*1000);

#endif
#endif

}

__device__ void cudaDevicecalc_Jac(
        //check_model_state
        //double threshhold, double replacement_value, int *flag,
        //f_gpu
        double time_step, int deriv_length_cell, int state_size_cell,
        int n_cells, int i_kernel,
        int threads_block, int n_shr_empty, double *y,
        ModelDataGPU *md, ModelDataVariable *dmdv
) //Interface CPU/GPU
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int tid_cell=tid%deriv_length_cell;
  int active_threads = md->n_cells*md->deriv_length_cell;

#ifdef CAMP_DEBUG_GPU

  int clock_khz=md->clock_khz;
  clock_t start;
  start = clock();

#endif

#ifdef DEBUG_cudaDeviceJac
  if(tid==0){
    printf("[DEBUG] GPU solveDerivative tid %d, \n", tid);
  }__syncthreads();
#endif

  if(tid<active_threads){

    //Debug
    /*
    if(i==0){
      printf("(%d) y %-le J_state %-le J_solver %-le J_tmp %-le J_tmp2 %-le J_deriv %-le\n",tid+1,
             y[tid], md->J_state[tid], md->J_solver[tid], md->J_tmp[tid], md->J_tmp2[tid], md->J_deriv[tid]);
      //printf("gpu threads %d\n", active_threads);
    }
*/

    JacobianGPU *jac = &md->jac;
    JacobianGPU jacBlock;

#ifdef DEV_JACOBIANGPUNUMSPEC
    jac->num_spec = state_size_cell;
    jacBlock.num_spec = state_size_cell;
#endif

#ifdef DEV_MULTICELLSGPU

    jacBlock.num_elem[0] = jac->num_elem[0]*(blockDim.x/deriv_length_cell);

#else

    jacBlock.num_elem = jac->num_elem;

#endif

    //if(threadIdx.x==0) printf("*jac->num_elem %d\n",jac->num_elem[0]);
    //if(threadIdx.x==0) printf("deriv_length_cell %d\n",deriv_length_cell);
    //if(threadIdx.x==0) printf("state_size_cell %d\n",state_size_cell);

#ifdef AEROS_CPU
#else
#endif

    int i_cell = tid/deriv_length_cell;
    dmdv->i_cell = i_cell;
    //jacBlock.production_partials = &( jac->production_partials[jac.num_elem[0]*i_cell]);
    //jacBlock.loss_partials = &( jac->loss_partials[jac.num_elem[0]*i_cell]);
    jacBlock.production_partials = &( jac->production_partials[jacBlock.num_elem[0]*blockIdx.x]);
    jacBlock.loss_partials = &( jac->loss_partials[jacBlock.num_elem[0]*blockIdx.x]);
    //jacBlock.col_ptrs = &( md->col_ptrs[algo*i_cell]);

    jacobian_reset_gpu(jacBlock);

    md->grid_cell_state = &( md->state[state_size_cell*i_cell]);
    md->grid_cell_env = &( md->env[CAMP_NUM_ENV_PARAM_*i_cell]);

    /*
    md->grid_cell_aero_rep_env_data =
    &(md->aero_rep_env_data[md->i_cell*md->n_aero_rep_env_data]);

    //Filter threads for n_aero_rep
    int n_aero_rep = md->n_aero_rep;
    if( tid_cell < n_aero_rep) {
      int n_iters = n_aero_rep / deriv_length_cell;
      //Repeat if there are more reactions than species
      for (int i = 0; i < n_iters; i++) {
        dmdv->i_aero_rep = tid_cell + i*deriv_length_cell;

        aero_rep_gpu_update_state(md);
      }

      //Limit tid to pending rxns to compute
      int residual=n_aero_rep-(deriv_length_cell*n_iters);
      if(tid_cell < residual){
        dmdv->i_aero_rep = tid_cell + deriv_length_cell*n_iters;

        aero_rep_gpu_update_state(md);
      }
    }
     */

#ifdef DEBUG_cudaDevicecalc_Jac

    if(tid==0)printf("cudaDevicecalc_Jac01\n");

    //if(threadIdx.x==0) {
    //  printf("jac.num_elem %d\n",jacBlock.num_elem);
    //  printf("*md->n_mapped_values %d\n",*md->n_mapped_values);
      //for (int i=0; i<*md->n_mapped_values; i++){
      //  printf("cudaDevicecalc_Jac0 jacBlock [%d]=%le\n",i,jacBlock.production_partials[i]);
      //}
    //}

#endif

    __syncthreads();
    //Filter threads for n_rxn
    int n_rxn = md->n_rxn;
    if( tid_cell < n_rxn) {
      int n_iters = n_rxn / deriv_length_cell;
      //Repeat if there are more reactions than species
      for (int i = 0; i < n_iters; i++) {
        dmdv->i_rxn = tid_cell + i*deriv_length_cell;

        solveRXNJac(jacBlock, time_step, md, dmdv);
      }

      //Limit tid to pending rxns to compute
      int residual=n_rxn-(deriv_length_cell*n_iters);
      if(tid_cell < residual){
        dmdv->i_rxn = tid_cell + deriv_length_cell*n_iters;

        solveRXNJac(jacBlock, time_step, md, dmdv);
      }
    }
    __syncthreads();


    jacobian_output_gpu(jacBlock, &(md->J_rxn[jacBlock.num_elem[0]*blockIdx.x]) );


#ifdef DEBUG_cudaDevicecalc_Jac

    if(threadIdx.x==0) {
        printf("jac.num_elem %d\n",jacBlock.num_elem[0]);
        printf("*md->n_mapped_values %d\n",*md->n_mapped_values);
        for (int i=0; i<10; i++){//*md->n_mapped_values
          printf("cudaDevicecalc_Jac0 J_rxn [%d]=%le\n",i,md->J_rxn[i]);
        }
      }

#endif


#ifdef DEV_REMOVE_threadIdx0

    //try using diA and djA pointers to better memory access
    JacMap *jac_map = md->jac_map;

    int nnz = md->n_mapped_values[0];

    for (int n = 0; n < nnz; n++) {
      md->J[jac_map[n].solver_id + nnz * blockIdx.x] = 0.0;
      md->J[jac_map[n].solver_id + nnz * blockIdx.x] =
              md->J_rxn[jac_map[n].rxn_id + jacBlock.num_elem[0] * blockIdx.x];

    }

    /*
    int nnz_tid = nnz/active_threads;

    for (int n = i*nnz_tid; n < i*(nnz_tid+1); ++n) {

      md->J[jac_map[n].solver_id+nnz*blockIdx.x] = //+=0.;
              md->J_rxn[jac_map[n].rxn_id+jacBlock.num_elem[0]*blockIdx.x];

    }

    int nnz_left = nnz-(nnz_tid*active_threads);
    if(i<nnz_left)
      for (int n = nnz_tid; n < nnz_tid; ++n) {

        md->J[jac_map[n].solver_id+nnz*blockIdx.x] = //+=0.;
                md->J_rxn[jac_map[n].rxn_id+jacBlock.num_elem[0]*blockIdx.x];
    }
*/

#else

    __syncthreads();
    if(threadIdx.x==0) {

      JacMap *jac_map = md->jac_map;
      int nnz = md->n_mapped_values[0];
      //int nnz = jac->num_elem;
      //SM_DATA_S(md->J_params)[0] = 1.0;  // dummy value for non-sub model calcs
      //for (int i_map = 0; i_map < md->n_mapped_values; ++i_map) {
      //for (int n = (nnz/gridDim.x)*blockIdx.x; n < (nnz/gridDim.x)*(blockIdx.x+1); n++) {

#ifdef DEV_MULTICELLSGPU

      //*(blockDim.x/deriv_length_cell)
      for (int j = 0; j < (blockDim.x/deriv_length_cell); j++) {
        for (int n = 0; n < nnz; n++) {
          //md->J[n] = 0.0;
          //md->J[n] = 0.0;

          //o = (j-1)*(I_E) + (i-1)
          //z = (k-1)*(I_E*I_N) + o
          int z = blockIdx.x*blockDim.x + j*nnz;
          int z_rxn = blockIdx.x*blockDim.x + j*jacBlock.num_elem[0];

          md->J[jac_map[n].solver_id + z] = //+=0.;
                  md->J_rxn[jac_map[n].rxn_id + jacBlock.num_elem[0] * blockIdx.x];

        }
      }

#else

      for (int n = 0; n < nnz; n++) {
        md->J[jac_map[n].solver_id + nnz * blockIdx.x] = 0.0;
        md->J[jac_map[n].solver_id + nnz * blockIdx.x] =
                md->J_rxn[jac_map[n].rxn_id + jacBlock.num_elem[0] * blockIdx.x];

        //SM_DATA_S(J)
        //[i_cell * md->n_per_cell_solver_jac_elem + jac_map[i_map].solver_id] +=
        //        SM_DATA_S(md->J_rxn)[jac_map[i_map].rxn_id] *
        //        SM_DATA_S(md->J_params)[jac_map[i_map].param_id];

      }

#endif

    }__syncthreads();

#endif

#ifdef CAMP_DEBUG_GPU
#ifdef cudaGlobalCVode_timers_max_blocks

    md->dtcalc_Jac[i] += ((double)(clock() - start))/(clock_khz*1000);

#else

    if(i==0) *md->dtcalc_Jac += ((double)(clock() - start))/(clock_khz*1000);

#endif
#endif

  }
}

__device__
int cudaDeviceJac(
        //check_model_state
        double threshhold, double replacement_value, int *flag,
        //f_gpu
        double time_step, int deriv_length_cell, int state_size_cell,
        int n_cells, int i_kernel,
        int threads_block, int n_shr_empty, double *y,
        ModelDataGPU *md, ModelDataVariable *dmdv, double *dftemp
) //Interface CPU/GPU
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int active_threads = md->n_cells*md->deriv_length_cell;

#ifdef CAMP_DEBUG_GPU

  int clock_khz=md->clock_khz;
  clock_t start;
  start = clock();

#endif

#ifdef DEV_printmin
  printmin(md,md->state,"cudaDeviceJac start state");
#endif
  int aux_flag=0;

  //int guessflag=
  int retval=cudaDevicef(
          time_step, deriv_length_cell, state_size_cell,
          n_cells, i_kernel, threads_block, n_shr_empty, y,
          dftemp,md,dmdv,&aux_flag
  );__syncthreads();
#ifdef DEBUG_cudaDevicef
  printmin(md,dftemp,"cudaDeviceJac dftemp");
#endif

  if(retval==CAMP_SOLVER_FAIL)
    return CAMP_SOLVER_FAIL;

#ifdef DEV_printmin
  printmin(md,y,"cudaDeviceJac y");
  printmin(md,md->state,"cudaDeviceJac start state");
#endif

  //helps with debugging
/*
  int checkflag=cudaDevicecamp_solver_check_model_state(md, dmdv, y, flag);
  __syncthreads();
  if(checkflag==CAMP_SOLVER_FAIL){
    *flag=CAMP_SOLVER_FAIL;
    //printf("cudaDeviceJac cudaDevicecamp_solver_check_model_state *flag==CAMP_SOLVER_FAIL\n");
    //printmin(md,y,"cudaDeviceJac end y");//Never enters?
    return CAMP_SOLVER_FAIL;
  }
*/

#ifdef DEV_printmin
  printmin(md,y,"cudaDeviceJac end y");
#endif

  //printmin(md,dftemp,"cudaDeviceJac end dftemp");

  cudaDevicecalc_Jac(
          //check_model_state          md->map_state_deriv, threshhold, replacement_value, flag,
          //f_gpu
          time_step, deriv_length_cell, state_size_cell,
          n_cells, i_kernel, threads_block, n_shr_empty, y,
          md, dmdv
  );
  __syncthreads();
#ifdef DEV_printmin
 printmin(md,dftemp,"cudaDevicecalc_Jac end dftemp");
#endif

#ifdef DEV_REMOVE_threadIdx0
#else
  if(i<active_threads){

    __syncthreads();
    if(threadIdx.x==0) {
      int nnz = md->n_mapped_values[0];
      for (int n = 0; n < nnz; n++) {
        md->J_solver[n]=md->J[n];
      }
    }__syncthreads();

    md->J_state[i]=y[i];
    md->J_deriv[i]=dftemp[i];
  }

#endif

#ifdef CAMP_DEBUG_GPU
#ifdef cudaGlobalCVode_timers_max_blocks

    md->dtJac[i] += ((double)(clock() - start))/(clock_khz*1000);

#else

    if(i==0) *md->dtJac += ((double)(clock() - start))/(clock_khz*1000);

#endif
#endif

  __syncthreads();
  *flag = 0;
  __syncthreads();
  return 0;

}


__device__
int cudaDevicelinsolsetup(
        //LS
        double *dA, int *djA, int *diA, double *dtempv, //Input data
        int nrows, int blocks, int n_shr_empty, int maxIt, int mattype,
        int n_cells, double tolmax, double *ddiag, //Init variables
        double *dr0, double *dr0h, double *dn0, double *dp0,
        double *dt, double *ds, double *dAx2, double *dy, double *dz,// Auxiliary vectors
        //swapCSC_CSR_BCG
        int *diB, int *djB, double *dB,
        //Guess_helper
        double t_n, double h_n, double* dftemp,
        double* dcv_y, double* tmp1,
        double* corr, double cv_reltol,
        //update_state
        double threshhold, double replacement_value, int *flag,
        //f_gpu
        double time_step, int deriv_length_cell, int state_size_cell,
        int i_kernel, int threads_block, ModelDataGPU *md, ModelDataVariable *dmdv,
        //cudacvNewtonIteration
        double *cv_acor,double *dewt,int convfail,
        int *disavedJ, int *djsavedJ, double *dsavedJ
) {

  extern __shared__ int flag_shr[];

  double dgamma;
  int jbad, jok;
#ifdef DEV_printmin
  printmin(md,dcv_y,"cudaDevicelinsolsetup Start dcv_y");
#endif
  dgamma = fabs((dmdv->cv_gamma / dmdv->cv_gammap) - 1.);//SUNRabs

  jbad = (dmdv->cv_nst == 0) ||
         (dmdv->cv_nst > dmdv->nstlj + CVD_MSBJ) ||
         ((convfail == CV_FAIL_BAD_J) && (dgamma < CVD_DGMAX)) ||
         (convfail == CV_FAIL_OTHER);
  jok = !jbad;


  //if(i==0)printf("cudaGlobalinsolsetupjok %d",jok);

  //if(i==0) printf("cudaGlobalinsolsetupcv_nst %d dmdv->nstlj %d cv_jcur %d jok %d gamma %le\n",
  //               *dmdv->cv_nst,dmdv->nstlj,*cv_jcur,jok,dmdv->cv_gamma);__syncthreads();


  if (jok==1) {
  //  if (0) {

    __syncthreads();

    dmdv->cv_jcur = 0; //all blocks update this variable
    //flag_shr[1] = 0;


    //todo Optimization: JacCopy of dA only (pointers iA and jA remains unchanged during GPU ODE solver in theory)
    cudaDeviceJacCopy(nrows, disavedJ, djsavedJ, dsavedJ, diA, djA, dA);

    __syncthreads();

    //cv_mem->cv_jcur = SUNFALSE;
    //retval = SUNMatCopy(cvdls_mem->savedJ, cvdls_mem->A);

    // If jok = SUNFALSE, reset J, call jac routine for new J value and save a copy
  } else {

  __syncthreads();

    dmdv->nje++;
    dmdv->nstlj = dmdv->cv_nst;
    dmdv->cv_jcur = 1;
    //flag_shr[1] = 1; //if used, assign to 1 if retval fails

  __syncthreads();

    int aux_flag=0;

    int guess_flag=cudaDeviceJac(
            //check_model_state
            threshhold, replacement_value, &aux_flag,
            //f_gpu
            time_step, deriv_length_cell, state_size_cell,
            n_cells, i_kernel, threads_block, n_shr_empty, dcv_y,
            md,dmdv, dftemp
    );
    __syncthreads();
    //if(i==0)printf("cudaGlobalinsolsetupflag_shr[1] %d\n",flag_shr[1]);

    if (guess_flag < 0) {
      //last_flag = CVDLS_JACFUNC_UNRECVR;
      return -1;
    }
    if (guess_flag > 0) {
      //last_flag = CVDLS_JACFUNC_RECVR;
      return 1;
    }

   cudaDeviceJacCopy(nrows, diA, djA, dA, disavedJ, djsavedJ, dsavedJ);

  }

  __syncthreads();


  cudaDevicematScaleAddI(nrows, dA, djA, diA, -dmdv->cv_gamma);
  cudaDevicediagprecond(nrows, dA, djA, diA, ddiag); //Setup linear solver


  *flag=0;
  __syncthreads();

  //cudaMemcpy(bicg->A,mGPU->dA,mGPU->nnz*sizeof(double),cudaMemcpyDeviceToHost);


  //__syncthreads();
  //*cv_jcur = flag_shr[1]; //BUG HERE 1000 CELLS
  //if(i==0)*cv_jcur = flag_shr[1]; //BUG HERE 1000 CELLS
  //__syncthreads();

  return 0;
}


//Algorithm: Biconjugate gradient
__device__
void solveBcgCudaDeviceCVODE(
        double *dA, int *djA, int *diA, double *dx, double *dtempv //Input data
        ,int nrows, int blocks, int n_shr_empty, int maxIt, int mattype
        ,int n_cells, double tolmax, double *ddiag //Init variables
        ,double *dr0, double *dr0h, double *dn0, double *dp0
        ,double *dt, double *ds, double *dAx2, double *dy, double *dz// Auxiliary vectors
        ,ModelDataGPU *md, ModelDataVariable *dmdv
)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int active_threads = nrows;
#ifdef DEV_printmin
  printmin(md,dtempv,"solveBcgCudaDeviceCVODEStart dtempv");
#endif
  //if(i<1){
  if(i<active_threads){

    double alpha,rho0,omega0,beta,rho1,temp1,temp2;
    alpha=rho0=omega0=beta=rho1=temp1=temp2=1.0;

    /*alpha  = 1.0;
    rho0   = 1.0;
    omega0 = 1.0;*/

    //gpu_yequalsconst(dn0,0.0,nrows,blocks,threads);  //n0=0.0 //memset???
    //gpu_yequalsconst(dp0,0.0,nrows,blocks,threads);  //p0=0.0
    cudaDevicesetconst(dn0, 0.0, nrows);
    cudaDevicesetconst(dp0, 0.0, nrows);

    //Not needed
    /*
    cudaDevicesetconst(dr0h, 0.0, nrows);
    cudaDevicesetconst(dt, 0.0, nrows);
    cudaDevicesetconst(ds, 0.0, nrows);
    cudaDevicesetconst(dAx2, 0.0, nrows);
    cudaDevicesetconst(dy, 0.0, nrows);
    cudaDevicesetconst(dz, 0.0, nrows);
     */

    __syncthreads();
    cudaDeviceSpmv(dr0,dx,nrows,dA,djA,diA,n_shr_empty); //y=A*x
    __syncthreads();

#ifdef DEBUG_SOLVEBCGCUDA_DEEP
    __syncthreads();
    if(i==0)printf("%d dr0 %-le\n",i,dr0[i]);
    __syncthreads();
#endif

    //gpu_axpby(dr0,dtempv,1.0,-1.0,nrows,blocks,threads); // r0=1.0*rhs+-1.0r0 //y=ax+by
    cudaDeviceaxpby(dr0,dtempv,1.0,-1.0,nrows);

    __syncthreads();
    //gpu_yequalsx(dr0h,dr0,nrows,blocks,threads);  //r0h=r0
    cudaDeviceyequalsx(dr0h,dr0,nrows);

    int it=0;

#ifdef DEBUG_SOLVEBCGCUDA_DEEP
    __syncthreads();
    if(i==0){
      printf("%d dr0 %-le dr0 %-le\n",it,dr0[i],dr0h[i]);
    }
    if(i==0)printf("solveBcgCudaDeviceCVODEStart dx %le ddiag %le diA %d"
                   "djA %d dA %le it %d block %d\n",
                   dx[(blockDim.x-1)*0],ddiag[(blockDim.x-1)*0],diA[(blockDim.x-1)*0],
                   djA[(blockDim.x-1)*0],dA[(blockDim.x-1)*0],it,blockIdx.x);

    __syncthreads();
#endif


    do
    {
      //rho1=gpu_dotxy(dr0, dr0h, aux, daux, nrows,(blocks + 1) / 2, threads);
      __syncthreads();

      cudaDevicedotxy(dr0, dr0h, &rho1, nrows, n_shr_empty);

#ifdef DEBUG_SOLVEBCGCUDA_DEEP
      __syncthreads();
    if(i==0 && it<2){
      //printf("%d dr0[%d] %-le\n",it,i,dr0[i]);
      printf("%d %d rho1 rho0 %-le %-le\n",it,i,rho1,rho0);
      printf("%d dr0 %-le dr0 %-le\n",it,dr0[i],dr0h[i]);
    }
      __syncthreads();
#endif

      __syncthreads();
      beta = (rho1 / rho0) * (alpha / omega0);

      __syncthreads();
      //gpu_zaxpbypc(dp0,dr0,dn0,beta,-1.0*omega0*beta,nrows,blocks,threads);   //z = ax + by + c
      cudaDevicezaxpbypc(dp0, dr0, dn0, beta, -1.0 * omega0 * beta, nrows);   //z = ax + by + c

      __syncthreads();
      //gpu_multxy(dy,ddiag,dp0,nrows,blocks,threads);  // precond y= p0*diag
      cudaDevicemultxy(dy, ddiag, dp0, nrows);

      __syncthreads();
      cudaDevicesetconst(dn0, 0.0, nrows);
      //gpu_spmv(dn0,dy,nrows,dA,djA,diA,mattype,blocks,threads);  // n0= A*y
      __syncthreads();
      cudaDeviceSpmv(dn0, dy, nrows, dA, djA, diA,n_shr_empty);

#ifdef DEBUG_SOLVEBCGCUDA_DEEP

      if(i==0 && it<2){
        printf("%d %d dy dn0 ddiag %-le %-le %le\n",it,i,dy[i],dn0[i],ddiag[i]);
        //printf("%d %d dn0 %-le\n",it,i,dn0[i]);
        //printf("%d %d &temp1 %p\n",it,i,&temp1);
        //printf("%d %d &test %p\n",it,i,&test);
        //printf("%d %d &i %p\n",it,i,&i);
      }

#endif

      __syncthreads();
      //temp1=gpu_dotxy(dr0h, dn0, aux, daux, nrows,(blocks + 1) / 2, threads);
      cudaDevicedotxy(dr0h, dn0, &temp1, nrows, n_shr_empty);

      __syncthreads();
      alpha = rho1 / temp1;

      //gpu_zaxpby(1.0,dr0,-1.0*alpha,dn0,ds,nrows,blocks,threads); // a*x + b*y = z
      cudaDevicezaxpby(1.0, dr0, -1.0 * alpha, dn0, ds, nrows);

      __syncthreads();
      //gpu_multxy(dz,ddiag,ds,nrows,blocks,threads); // precond z=diag*s
      cudaDevicemultxy(dz, ddiag, ds, nrows); // precond z=diag*s

      __syncthreads();
      //gpu_spmv(dt,dz,nrows,dA,djA,diA,mattype,blocks,threads);
      cudaDeviceSpmv(dt, dz, nrows, dA, djA, diA,n_shr_empty);

      __syncthreads();
      //gpu_multxy(dAx2,ddiag,dt,nrows,blocks,threads);
      cudaDevicemultxy(dAx2, ddiag, dt, nrows);

      __syncthreads();
      //temp1=gpu_dotxy(dz, dAx2, aux, daux, nrows,(blocks + 1) / 2, threads);
      cudaDevicedotxy(dz, dAx2, &temp1, nrows, n_shr_empty);

      __syncthreads();
      //temp2=gpu_dotxy(dAx2, dAx2, aux, daux, nrows,(blocks + 1) / 2, threads);
      cudaDevicedotxy(dAx2, dAx2, &temp2, nrows, n_shr_empty);

      __syncthreads();
      omega0 = temp1 / temp2;
      //gpu_axpy(dx,dy,alpha,nrows,blocks,threads); // x=alpha*y +x
      cudaDeviceaxpy(dx, dy, alpha, nrows); // x=alpha*y +x

      __syncthreads();
      //gpu_axpy(dx,dz,omega0,nrows,blocks,threads);
      cudaDeviceaxpy(dx, dz, omega0, nrows);

      __syncthreads();
      //gpu_zaxpby(1.0,ds,-1.0*omega0,dt,dr0,nrows,blocks,threads);
      cudaDevicezaxpby(1.0, ds, -1.0 * omega0, dt, dr0, nrows);
      __syncthreads();
      cudaDevicesetconst(dt, 0.0, nrows);

      __syncthreads();
      //temp1=gpu_dotxy(dr0, dr0, aux, daux, nrows,(blocks + 1) / 2, threads);
      cudaDevicedotxy(dr0, dr0, &temp1, nrows, n_shr_empty);
      __syncthreads();

      //temp1 = sqrt(temp1);
      temp1 = sqrtf(temp1);

      rho0 = rho1;
      /**/
      __syncthreads();
      /**/

      //if (tid==0) it++;
      it++;
    } while(it<maxIt && temp1>tolmax);//while(it<maxIt && temp1>tolmax);//while(0);
#ifdef DEV_printmin
    printmin(md,dx,"solveBcgCudaDeviceCVODEEnd dx");
#endif

#ifdef DEBUG_SOLVEBCGCUDA_DEEP
    __syncthreads();
    if(i==0)printf("solveBcgCudaDeviceCVODEEnd dx %le ddiag %le diA %d"
                   "djA d dA %le it %d block %d\n",
                   dx[(blockDim.x-1)*0],ddiag[(blockDim.x-1)*0],diA[(blockDim.x-1)*0],
                   djA[(blockDim.x-1)*0],dA[(blockDim.x-1)*0],it,blockIdx.x);
    __syncthreads();
#endif

    //if(it>=maxIt-1)
    //  dvcheck_input_gpud(dr0,nrows,999);

    //dvcheck_input_gpud(dr0,nrows,k++);


#ifdef CAMP_DEBUG_GPU

  dmdv->counterBCGInternal += it;
  dmdv->counterBCG++;

#endif

  }

}

void constructor_cvode_gpu(CVodeMem cv_mem, SolverData *sd)
{
  itsolver *bicg = &(sd->bicg);
  ModelData *md = &(sd->model_data);
  ModelDataGPU *mGPU = &sd->mGPU;
  CVDlsMem cvdls_mem = (CVDlsMem) cv_mem->cv_lmem;
  SUNMatrix J = cvdls_mem->A;
  int device=0;//Selected GPU
  double *ewt = N_VGetArrayPointer(cv_mem->cv_ewt);
  double *tempv = N_VGetArrayPointer(cv_mem->cv_tempv);
  double *cv_last_yn = N_VGetArrayPointer(cv_mem->cv_last_yn);
  double *cv_acor_init = N_VGetArrayPointer(cv_mem->cv_acor_init);
  int flag = 999; //CAMP_SOLVER_SUCCESS
  cudaDeviceProp prop;

  mGPU->n_cells=md->n_cells;
  mGPU->dftemp=mGPU->deriv_data; //deriv is gpu pointer
  //Init GPU ODE solver variables
  //Linking Matrix data, later this data must be allocated in GPU
  mGPU->nnz=SM_NNZ_S(J);
  mGPU->nrows=SM_NP_S(J);
  bicg->A=(double*)SM_DATA_S(J);

  //Using int per default as sundindextype give wrong results in CPU, so translate from int64 to int
  bicg->jA=(int*)malloc(sizeof(int)*SM_NNZ_S(J));
  bicg->iA=(int*)malloc(sizeof(int)*(SM_NP_S(J)+1));
  for(int i=0;i<SM_NNZ_S(J);i++)
    bicg->jA[i]=SM_INDEXVALS_S(J)[i];
  for(int i=0;i<=SM_NP_S(J);i++)
    bicg->iA[i]=SM_INDEXPTRS_S(J)[i];
  //bicg->jA=(int*)SM_INDEXVALS_S(J);
  //bicg->iA=(int*)SM_INDEXPTRS_S(J);

  cudaGetDeviceProperties(&prop, device);
  mGPU->threads=prop.maxThreadsPerBlock;//set at max gpu
  //mGPU->threads=1024;
  //printf("mGPU->threads %d \n", mGPU->threads);
  mGPU->blocks=(mGPU->nrows+mGPU->threads-1)/mGPU->threads;

  //ODE aux variables
  mGPU->dA=mGPU->J;//set itsolver gpu pointer to jac pointer initialized at camp

  sd->flagCells=(int *)malloc((mGPU->n_cells)*sizeof(int));

  cudaMalloc((void**)&mGPU->mdv,sizeof(ModelDataVariable));
  cudaMalloc((void**)&mGPU->mdvo,sizeof(ModelDataVariable));
  cudaMalloc((void**)&mGPU->flag,1*sizeof(int));

  cudaMalloc((void**)&mGPU->flagCells,mGPU->n_cells*sizeof(int));
  cudaMalloc((void**)&mGPU->djA,mGPU->nnz*sizeof(int));
  cudaMalloc((void**)&mGPU->diA,(mGPU->nrows+1)*sizeof(int));
  cudaMalloc((void**)&mGPU->dB,mGPU->nnz*sizeof(double));
  cudaMalloc((void**)&mGPU->djB,mGPU->nnz*sizeof(int));
  cudaMalloc((void**)&mGPU->diB,(mGPU->nrows+1)*sizeof(int));
  cudaMalloc((void**)&mGPU->dsavedJ,mGPU->nnz*sizeof(double));
  cudaMalloc((void**)&mGPU->djsavedJ,mGPU->nnz*sizeof(int));
  cudaMalloc((void**)&mGPU->disavedJ,(mGPU->nrows+1)*sizeof(int));
  cudaMalloc((void**)&mGPU->dewt,mGPU->nrows*sizeof(double));
  cudaMalloc((void**)&mGPU->cv_acor,mGPU->nrows*sizeof(double));
  cudaMalloc((void**)&mGPU->dtempv,mGPU->nrows*sizeof(double));
  cudaMalloc((void**)&mGPU->dtempv1,mGPU->nrows*sizeof(double));
  cudaMalloc((void**)&mGPU->dtempv2,mGPU->nrows*sizeof(double));
  cudaMalloc((void**)&mGPU->dzn,mGPU->nrows*(cv_mem->cv_qmax+1)*sizeof(double));//L_MAX 6
  cudaMalloc((void**)&mGPU->dcv_y,mGPU->nrows*sizeof(double));
  cudaMalloc((void**)&mGPU->dx,mGPU->nrows*sizeof(double));
  cudaMalloc((void**)&mGPU->cv_last_yn,mGPU->nrows*sizeof(double));
  cudaMalloc((void**)&mGPU->cv_acor_init,mGPU->nrows*sizeof(double));
  cudaMalloc((void**)&mGPU->cv_acor,mGPU->nrows*sizeof(double));
  cudaMalloc((void**)&mGPU->yout,mGPU->nrows*sizeof(double));
  cudaMalloc((void**)&mGPU->cv_l,L_MAX*mGPU->n_cells*sizeof(double));
  cudaMalloc((void**)&mGPU->cv_tau,(L_MAX+1)*mGPU->n_cells*sizeof(double));
  cudaMalloc((void**)&mGPU->cv_tq,(NUM_TESTS+1)*mGPU->n_cells*sizeof(double));
  cudaMalloc((void**)&mGPU->cv_Vabstol,mGPU->nrows*sizeof(double));

  HANDLE_ERROR(cudaMemset(mGPU->flagCells, CV_SUCCESS, mGPU->n_cells*sizeof(int)));
  cudaMemcpy(mGPU->djA,bicg->jA,mGPU->nnz*sizeof(int),cudaMemcpyHostToDevice);
  cudaMemcpy(mGPU->diA,bicg->iA,(mGPU->nrows+1)*sizeof(int),cudaMemcpyHostToDevice);
  cudaMemcpy(mGPU->dsavedJ,bicg->A,mGPU->nnz*sizeof(double),cudaMemcpyHostToDevice);
  cudaMemcpy(mGPU->djsavedJ,bicg->jA,mGPU->nnz*sizeof(int),cudaMemcpyHostToDevice);
  cudaMemcpy(mGPU->disavedJ,bicg->iA,(mGPU->nrows+1)*sizeof(int),cudaMemcpyHostToDevice);
  cudaMemcpy(mGPU->dewt,ewt,mGPU->nrows*sizeof(double),cudaMemcpyHostToDevice);
  cudaMemcpy(mGPU->cv_acor,ewt,mGPU->nrows*sizeof(double),cudaMemcpyHostToDevice);
  cudaMemcpy(mGPU->dftemp,ewt,mGPU->nrows*sizeof(double),cudaMemcpyHostToDevice);
  cudaMemcpy(mGPU->dx,tempv,mGPU->nrows*sizeof(double),cudaMemcpyHostToDevice);
  cudaMemcpy(mGPU->cv_last_yn,cv_last_yn,mGPU->nrows*sizeof(double),cudaMemcpyHostToDevice);
  cudaMemcpy(mGPU->cv_acor_init,cv_acor_init,mGPU->nrows*sizeof(double),cudaMemcpyHostToDevice);


  mGPU->n_cells=md->n_cells;
  mGPU->dftemp=mGPU->deriv_data; //deriv is gpu pointer
  mGPU->nnz=SM_NNZ_S(J);
  cudaGetDeviceProperties(&prop, device);
  mGPU->threads=prop.maxThreadsPerBlock;//set at max gpu (1024)
  mGPU->blocks=(mGPU->nrows+mGPU->threads-1)/mGPU->threads;
  mGPU->replacement_value=TINY;
  mGPU->threshhold=-SMALL;
  mGPU->deriv_length_cell=mGPU->nrows/mGPU->n_cells;
  mGPU->state_size_cell=md->n_per_cell_state_var;

  cudaMemcpy(mGPU->flag,&flag,1*sizeof(int),cudaMemcpyHostToDevice);

  // Allocating matrix data to the GPU
  mGPU->dA=mGPU->J;

  //Init Linear Solver variables
  createSolver(sd);
  createCuSolver(sd);

  sd->mdv.cv_reltol=((CVodeMem) sd->cvode_mem)->cv_reltol;
  sd->mdv.cv_nfe=0;
  sd->mdv.cv_nsetups=0;
  sd->mdv.nje=0;
  sd->mdv.nstlj=0;

  //Check if everything is correct
#ifdef FAILURE_DETAIL
  if(md->n_per_cell_dep_var > prop.maxThreadsPerBlock/2)
    printf("ERROR: The GPU can't handle so much species"
           " [NOT ENOUGH THREADS/BLOCK FOR ALL THE SPECIES]\n");
#endif

#ifdef CAMP_DEBUG_GPU

  sd->mdv.countercvStep=0;
  sd->mdv.counterDerivGPU=0;
  sd->mdv.counterBCGInternal=0;
  sd->mdv.counterBCG=0;
  sd->mdv.dtBCG=0.;
  sd->mdv.dtcudaDeviceCVode=0.;
  sd->mdv.dtPostBCG=0.;

  bicg->counterNewtonIt=0;
  bicg->counterLinSolSetup=0;
  bicg->counterLinSolSolve=0;
  //bicg->countercvStep=0;
  bicg->counterDerivNewton=0;
  bicg->counterBiConjGrad=0;
  bicg->counterDerivSolve=0;
  bicg->counterJac=0;
  bicg->countersolveCVODEGPU=0;

#ifdef cudaGlobalCVode_timers_max_blocks
  int lendt=blocks;

#else
  sd->mdv.dtBCG=0.;
  sd->mdv.dtcudaDeviceCVode=0.;
  sd->mdv.dtPostBCG=0.;
  int lendt=1;

#endif

  cudaDeviceGetAttribute(&mGPU->clock_khz, cudaDevAttrClockRate, 0);
  cudaMalloc((void**)&mGPU->tguessNewton,lendt*sizeof(double));
  cudaMemset(mGPU->tguessNewton, 0., lendt*sizeof(double));
  cudaMalloc((void**)&mGPU->dtNewtonIteration,lendt*sizeof(double));
  cudaMemset(mGPU->dtNewtonIteration, 0., lendt*sizeof(double));
  cudaMalloc((void**)&mGPU->dtJac,lendt*sizeof(double));
  cudaMemset(mGPU->dtJac, 0., lendt*sizeof(double));
  cudaMalloc((void**)&mGPU->dtlinsolsetup,lendt*sizeof(double));
  cudaMemset(mGPU->dtlinsolsetup, 0., lendt*sizeof(double));
  cudaMalloc((void**)&mGPU->dtcalc_Jac,lendt*sizeof(double));
  cudaMemset(mGPU->dtcalc_Jac, 0., lendt*sizeof(double));
  cudaMalloc((void**)&mGPU->dtRXNJac,lendt*sizeof(double));
  cudaMemset(mGPU->dtRXNJac, 0., lendt*sizeof(double));
  cudaMalloc((void**)&mGPU->dtf,lendt*sizeof(double));
  cudaMemset(mGPU->dtf, 0., lendt*sizeof(double));
  cudaMalloc((void**)&mGPU->dtguess_helper,lendt*sizeof(double));
  cudaMemset(mGPU->dtguess_helper, 0., lendt*sizeof(double));

  bicg->timeNewtonIt=CAMP_TINY;
  bicg->timeLinSolSetup=CAMP_TINY;
  bicg->timeLinSolSolve=CAMP_TINY;
  bicg->timecvStep=CAMP_TINY;
  bicg->timeDerivNewton=CAMP_TINY;
  bicg->timeBiConjGrad=CAMP_TINY;
  bicg->timeBiConjGradMemcpy=CAMP_TINY;
  bicg->timeDerivSolve=CAMP_TINY;
  bicg->timeJac=CAMP_TINY;
  bicg->timesolveCVODEGPU=CAMP_TINY;

  cudaEventCreate(&bicg->startDerivNewton);
  cudaEventCreate(&bicg->startDerivSolve);
  cudaEventCreate(&bicg->startLinSolSetup);
  cudaEventCreate(&bicg->startLinSolSolve);
  cudaEventCreate(&bicg->startNewtonIt);
  cudaEventCreate(&bicg->startcvStep);
  cudaEventCreate(&bicg->startBCG);
  cudaEventCreate(&bicg->startBCGMemcpy);
  cudaEventCreate(&bicg->startJac);
  cudaEventCreate(&bicg->startsolveCVODEGPU);

  cudaEventCreate(&bicg->stopDerivNewton);
  cudaEventCreate(&bicg->stopDerivSolve);
  cudaEventCreate(&bicg->stopLinSolSetup);
  cudaEventCreate(&bicg->stopLinSolSolve);
  cudaEventCreate(&bicg->stopNewtonIt);
  cudaEventCreate(&bicg->stopcvStep);
  cudaEventCreate(&bicg->stopBCG);
  cudaEventCreate(&bicg->stopBCGMemcpy);
  cudaEventCreate(&bicg->stopJac);
  cudaEventCreate(&bicg->stopsolveCVODEGPU);

#endif

  cudaMemcpy(mGPU->mdv,&sd->mdv,sizeof(ModelDataVariable),cudaMemcpyHostToDevice);
  cudaMemcpy(mGPU->mdvo,&sd->mdv,sizeof(ModelDataVariable),cudaMemcpyHostToDevice); //todo check this is fine for cudacvode

  if(cv_mem->cv_sldeton){
    printf("ERROR: cudaDevicecvBDFStab is pending to implement "
                   "(disabled by default on CAMP)\n");
    exit(0);
  }

#ifdef CHECK_GPU_LINSOLVE
  sd->max_error_linsolver = 0.0;
  sd->max_error_linsolver_i = 0;
  sd->n_linsolver_i = 0;
#endif

}

int check_jac_status_error_gpu2(SUNMatrix A)
{
  sunindextype j, i, newvals, M, N;
  booleantype newmat, found;
  sunindextype *Ap, *Ai;
  //realtype *Ax;
  int flag;

  /* store shortcuts to matrix dimensions (M is inner dimension, N is outer) */
  if (SM_SPARSETYPE_S(A) == CSC_MAT) {
    M = SM_ROWS_S(A);
    N = SM_COLUMNS_S(A);
  } else {
    M = SM_COLUMNS_S(A);
    N = SM_ROWS_S(A);
  }

  /* access data arrays from A (return if failure) */
  Ap = Ai = NULL;
  //Ax = NULL;
  if (SM_INDEXPTRS_S(A)) Ap = SM_INDEXPTRS_S(A);
  else return (-1);
  if (SM_INDEXVALS_S(A)) Ai = SM_INDEXVALS_S(A);
  else return (-1);
  //if (SM_DATA_S(A)) Ax = SM_DATA_S(A);
  //else return (-1);


  /* determine if A: contains values on the diagonal (so I can just be added in);
     if not, then increment counter for extra storage that should be required. */
  newvals = 0;
  for (j = 0; j < SUNMIN(M, N); j++) {
    /* scan column (row if CSR) of A, searching for diagonal value */
    found = SUNFALSE;
    for (i = Ap[j]; i < Ap[j + 1]; i++) {
      if (Ai[i] == j) {
        found = SUNTRUE;
        break;
      }
    }
    /* if no diagonal found, increment necessary storage counter */
    if (!found) newvals += 1;
  }

  /* If extra nonzeros required, check whether matrix has sufficient storage space
     for new nonzero entries  (so I can be inserted into existing storage) */
  newmat = SUNFALSE;   /* no reallocation needed */
  if (newvals > (SM_NNZ_S(A) - Ap[N]))
    newmat = SUNTRUE;

  //case 1: A already contains a diagonal
  if (newvals == 0) {

    flag = 0;
    //printf("jac_indices had or need change to fill the diagonal");

    //   case 2: A has sufficient storage, but does not already contain a diagonal
  } else if (!newmat) {

    printf("Jacobian does not contain a diagonal, jac_indices had/need to change");
    flag = 1;
    //case 3: A must be reallocated with sufficient storage */
  } else {

    printf("Jacobian must be reallocated with sufficient storage");
    flag = 1;
  }

  return flag;
}

/*
 * cvHandleFailure
 *
 * This routine prints error messages for all cases of failure by
 * cvHin and cvStep.
 * It returns to CVode the value that CVode is to return to the user.
 */

int cvHandleFailure_gpu2(CVodeMem cv_mem, int flag)
{

  /* Set vector of  absolute weighted local errors */
  /*
  N_VProd(acor, ewt, tempv);
  N_VAbs(tempv, tempv);
  */

  /* Depending on flag, print error message and return error flag */
  switch (flag) {
    case CV_ERR_FAILURE:
      cvProcessError(cv_mem, CV_ERR_FAILURE, "CVODE", "CVode", MSGCV_ERR_FAILS,
                     cv_mem->cv_tn, cv_mem->cv_h);
      break;
    case CV_CONV_FAILURE:
      cvProcessError(cv_mem, CV_CONV_FAILURE, "CVODE", "CVode", MSGCV_CONV_FAILS,
                     cv_mem->cv_tn, cv_mem->cv_h);
      break;
    case CV_LSETUP_FAIL:
      cvProcessError(cv_mem, CV_LSETUP_FAIL, "CVODE", "CVode", MSGCV_SETUP_FAILED,
                     cv_mem->cv_tn);
      break;
    case CV_LSOLVE_FAIL:
      cvProcessError(cv_mem, CV_LSOLVE_FAIL, "CVODE", "CVode", MSGCV_SOLVE_FAILED,
                     cv_mem->cv_tn);
      break;
    case CV_RHSFUNC_FAIL:
      cvProcessError(cv_mem, CV_RHSFUNC_FAIL, "CVODE", "CVode", MSGCV_RHSFUNC_FAILED,
                     cv_mem->cv_tn);
      break;
    case CV_UNREC_RHSFUNC_ERR:
      cvProcessError(cv_mem, CV_UNREC_RHSFUNC_ERR, "CVODE", "CVode", MSGCV_RHSFUNC_UNREC,
                     cv_mem->cv_tn);
      break;
    case CV_REPTD_RHSFUNC_ERR:
      cvProcessError(cv_mem, CV_REPTD_RHSFUNC_ERR, "CVODE", "CVode", MSGCV_RHSFUNC_REPTD,
                     cv_mem->cv_tn);
      break;
    case CV_RTFUNC_FAIL:
      cvProcessError(cv_mem, CV_RTFUNC_FAIL, "CVODE", "CVode", MSGCV_RTFUNC_FAILED,
                     cv_mem->cv_tn);
      break;
    case CV_TOO_CLOSE:
      cvProcessError(cv_mem, CV_TOO_CLOSE, "CVODE", "CVode", MSGCV_TOO_CLOSE);
      break;
    default:
      return(CV_SUCCESS);
  }

  return(flag);
}

__device__
int cudaDevicecvNewtonIteration(
        //LS
        double *dA, int *djA, int *diA, double *dx, double *dtempv, //Input data
        int nrows, int blocks, int n_shr_empty, int maxIt, int mattype,
        int n_cells, double tolmax, double *ddiag, //Init variables
        double *dr0, double *dr0h, double *dn0, double *dp0,
        double *dt, double *ds, double *dAx2, double *dy, double *dz,// Auxiliary vectors
        //swapCSC_CSR_BCG
        int *diB, int *djB, double *dB,
        //Guess_helper
        double t_n, double h_n, double* dftemp,
        double* dcv_y, double* tmp1,
        double* corr, double cv_reltol,
        //update_state
        double threshhold, double replacement_value, int *flag,
        //f_gpu
        double time_step, int deriv_length_cell, int state_size_cell,
        int i_kernel, int threads_block, ModelDataGPU *md, ModelDataVariable *dmdv,
        //cudacvNewtonIteration
        double *cv_acor, double *dzn,
        double *dewt
)
{
  extern __shared__ double flag_shr2[];
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int aux_flag=0;

  double del, delp, dcon, m;
  del = delp = 0.0;
  dmdv->cv_mnewt = m = 0;

#ifdef CAMP_DEBUG_GPU

  int clock_khz=md->clock_khz;
  clock_t start;

#endif

  //if(i==0)printf("cudaDevicecvNewtonIterationStart dzn[(blockDim.x*(blockIdx.x+1)-1)*0] %le counterNewton %d block %d\n",dzn[(blockDim.x*(blockIdx.x+1)-1)*0],counterNewton,blockIdx.x);
#ifdef DEV_printmin
  printmin(md,dtempv,"cudaDevicecvNewtonIterationStart dtempv");
#endif

  for(;;) {

#ifdef DEV_printmin
    printmin(md,dftemp,"cudaDevicecvNewtonIteration dftemp");
#endif

#ifdef CAMP_DEBUG_GPU

    //Useful for %: https://stackoverflow.com/questions/19527038/how-to-measure-the-time-of-the-device-functions-when-they-are-called-in-kernel-f
    start = clock();

#endif


    cudaDevicezaxpby(dmdv->cv_rl1, (dzn + 1 * nrows), 1.0, cv_acor, dtempv, nrows);
    cudaDevicezaxpby(dmdv->cv_gamma, dftemp, -1.0, dtempv, dtempv, nrows);
    //}//GOOD

    //if(i==0)printf("cudaDevicecvNewtonIteration dftemp %le dtempv %le dcv_y %le it %d block %d\n",
    //               dftemp[(blockDim.x-1)*0],dtempv[(blockDim.x-1)*0],dcv_y[(blockDim.x-1)*0],it,blockIdx.x);

    solveBcgCudaDeviceCVODE(
            dA, djA, diA, dx, dtempv, nrows, blocks,
            n_shr_empty,maxIt, mattype, n_cells,
            tolmax, ddiag, dr0, dr0h, dn0, dp0, dt,
            ds, dAx2, dy, dz, md, dmdv
    );

#ifdef CAMP_DEBUG_GPU

#ifdef cudaGlobalCVode_timers_max_blocks

    dmdv->dtBCG[i] += ((double)(int)(clock() - startBCG))/(clock_khz*1000);

#else

    //if(threadIdx.x==0)dmdv->dtBCG += ((double)(int)(clock() - start))/(clock_khz*1000);//wrong
    dmdv->dtBCG += ((double)(int)(clock() - start))/(clock_khz*1000);


#endif

    start = clock(); //almost accurate :https://stackoverflow.com/questions/19527038/how-to-measure-the-time-of-the-device-functions-when-they-are-called-in-kernel-f

#endif

    __syncthreads();
    dtempv[i] = dx[i];
    __syncthreads();
#ifdef DEV_printmin
    printmin(md,dcv_y,"cudaDevicecvNewtonIteration dcv_y");
    printmin(md,dtempv,"cudaDevicecvNewtonIteration dtempv");
#endif
    //if (cv_mem->cv_ghfun){//Function is always defined in CAMP
    //N_VLinearSum(ONE, cv_mem->cv_y, ONE, b, cv_mem->cv_ftemp);
    cudaDevicezaxpby(1.0, dcv_y, 1.0, dtempv, dftemp, nrows);
#ifdef DEBUG_cudaDevicecvNewtonIteration
    //if(i==0)printf("cudaDevicecvNewtonIteration dftemp %le dtempv %le dcv_y %le it %d block %d\n",
    //               dftemp[(blockDim.x-1)*0],dtempv[(blockDim.x-1)*0],dcv_y[(blockDim.x-1)*0],it,blockIdx.x);
#endif
#ifdef DEV_printmin
    printmin(md,dftemp,"cudaDevicecvNewtonIteration dftemp");
#endif

    __syncthreads();
    int guessflag=CudaDeviceguess_helper(t_n, 0., dftemp,
                           dcv_y, dtempv, tmp1,
                           corr, cv_reltol, nrows,
            //check_model_state
                           threshhold, replacement_value, &aux_flag,
            //f_gpu
                           time_step, deriv_length_cell, state_size_cell,
                           n_cells, i_kernel, threads_block, n_shr_empty,
                           md, dmdv
    );
    __syncthreads();
    //if(i==0)printf("cudaDevicecvNewtonIteration guessflag %d block %d\n",guessflag,blockIdx.x);

    if (guessflag < 0) {
      if (!(dmdv->cv_jcur)) { //Bool set up during linsolsetup just before Jacobian
        //&& (cv_lsetup)) { //cv_mem->cv_lsetup// Setup routine, always exists for BCG
        return TRY_AGAIN;
      } else {
        return RHSFUNC_RECVR;
      }
    }

    // Check for negative concentrations (CAMP addition)
    cudaDevicezaxpby(1., dcv_y, 1., dtempv, dftemp, nrows);

    //    if (N_VMin(cv_mem->cv_ftemp) < -CAMP_TINY) {
    //      return(CONV_FAIL);
    //    }
    double min;
    cudaDevicemin(&min, dftemp[i], flag_shr2, md->n_shr_empty);

    if (min < -CAMP_TINY) {
      //if (dftemp[i] < -CAMP_TINY) {
      return CONV_FAIL;
    }
    __syncthreads();

    //cv_acor[i]+=dx[i];
    cudaDevicezaxpby(1., cv_acor, 1., dx, cv_acor, nrows);
    cudaDevicezaxpby(1., dzn, 1., cv_acor, dcv_y, nrows);

    cudaDeviceVWRMS_Norm(dx, dewt, &del, nrows, n_shr_empty);

// Test for convergence.  If m > 0, an estimate of the convergence
    // rate constant is stored in crate, and used in the test.
//#define SUNMAX(A, B) ((A) > (B) ? (A) : (B))
    if (m > 0) {
      dmdv->cv_crate = SUNMAX(0.3 * dmdv->cv_crate, del / delp);
    }

    dcon = del * SUNMIN(1.0, dmdv->cv_crate) / md->cv_tq[4+blockIdx.x*(NUM_TESTS + 1)];

    flag_shr2[0]=0;//needed?
    __syncthreads();
    if (dcon <= 1.0) {
      cudaDeviceVWRMS_Norm(cv_acor, dewt, &dmdv->cv_acnrm, nrows, n_shr_empty);
      //cv_mem->cv_acnrm = gpu_VWRMS_Norm(mGPU->nrows, mGPU->cv_acor, mGPU->dewt, bicg->aux,
      //  //                                    mGPU->daux, (mGPU->blocks + 1) / 2, mGPU->threads);

      __syncthreads();
      dmdv->cv_jcur = 0;
      __syncthreads();

      return CV_SUCCESS;
    }

    dmdv->cv_mnewt = ++m;

    // Stop at maxcor iterations or if iter. seems to be diverging.
    //     If still not converged and Jacobian data is not current,
    //     signal to try the solution again
    if ((m == dmdv->cv_maxcor) || ((m >= 2) && (del > RDIV * delp))) {
      if (!(dmdv->cv_jcur)) {
        return TRY_AGAIN;
      } else {
        return RHSFUNC_RECVR;
      }
    }

    // Save norm of correction, evaluate f, and loop again
    delp = del;

    __syncthreads();

#ifdef DEV_printmin
    printmin(md,md->state,"cudaDevicef start state");
#endif

    int retval=cudaDevicef(
            time_step, deriv_length_cell, state_size_cell,
            n_cells, i_kernel, threads_block, n_shr_empty, dcv_y,
            dftemp, md, dmdv, &aux_flag
    );
    //retval = f_gpu(cv_mem->cv_tn, cv_mem->cv_y, cv_mem->cv_ftemp, cv_mem->cv_user_data);

    __syncthreads();

    // a*x + b*y = z
    cudaDevicezaxpby(1., dcv_y, 1., dzn, cv_acor, nrows);
    //gpu_zaxpby(1.0, mGPU->dcv_y, -1.0, mGPU->dzn, mGPU->cv_acor, mGPU->nrows, mGPU->blocks, mGPU->threads);

    if (retval < 0) {
      return CV_RHSFUNC_FAIL;
    }
    if (retval > 0) {
      if (!(dmdv->cv_jcur)) {
        return TRY_AGAIN;
      } else {
        return RHSFUNC_RECVR;
      }
    }

    dmdv->cv_nfe=dmdv->cv_nfe+1;


#ifdef CAMP_DEBUG_GPU
#ifdef cudaGlobalCVode_timers_max_blocks

    __syncthreads();
    dmdv->dtPostBCG[i] += ((double)(clock() - start))/(clock_khz*1000);

#else

    if(i==0) dmdv->dtPostBCG += ((double)(clock() - start))/(clock_khz*1000);

#endif

#endif

#ifdef DEBUG_cudaDevicecvNewtonIteration
    if(i==0)printf("cudaDevicecvNewtonIteration dzn[(blockDim.x*(blockIdx.x+1)-1)*0] %le it %d block %d\n",dzn[(blockDim.x*(blockIdx.x+1)-1)*0],it,blockIdx.x);
#endif

  }

}

__device__
int cudaDevicecvNlsNewton(
        //LS
        double *dA, int *djA, int *diA, double *dx, double *dtempv, //Input data
        int nrows, int blocks, int n_shr_empty, int maxIt, int mattype,
        int n_cells, double tolmax, double *ddiag, //Init variables
        double *dr0, double *dr0h, double *dn0, double *dp0,
        double *dt, double *ds, double *dAx2, double *dy, double *dz,// Auxiliary vectors
        //swapCSC_CSR_BCG
        int *diB, int *djB, double *dB,
        //Guess_helper
        double t_n, double h_n, double* dftemp,
        double* dcv_y, double* tmp1,
        double* corr, double cv_reltol,
        //update_state
        double threshhold, double replacement_value, int *flag,
        //f_gpu
        double time_step, int deriv_length_cell, int state_size_cell,
        int i_kernel, int threads_block, ModelDataGPU *md, ModelDataVariable *dmdv,
        //cudacvNewtonIteration
        double *cv_acor, double *dzn,
        double *dewt,
        int *disavedJ, int *djsavedJ, double *dsavedJ
) {
  extern __shared__ int flag_shr[];
  int flagDevice = 0;//99
  __syncthreads();*flag = flag_shr[0];__syncthreads();
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  //if(threadIdx.x==0)printf("cudaDevicecvNlsNewton start %d\n",blockIdx.x);
#ifdef DEV_printmin
  printmin(md,dtempv,"cudaDevicecvNlsNewtonStart dtempv");
#endif
#ifdef CAMP_DEBUG_GPU

  int clock_khz=md->clock_khz;
  clock_t start;

#endif

  int convfail = ((dmdv->nflag == FIRST_CALL) || (dmdv->nflag == PREV_ERR_FAIL)) ?
                 CV_NO_FAILURES : CV_FAIL_OTHER;

  //Decide whether or not to call setup routine (if one exists)
  //if (cv_mem->cv_lsetup) { // Setup routine, always exists for BCG
  int dgamrat=fabs(dmdv->cv_gamrat - 1.);
  int callSetup = (dmdv->nflag == PREV_CONV_FAIL) || (dmdv->nflag == PREV_ERR_FAIL) ||
                  (dmdv->cv_nst == 0) ||
                  (dmdv->cv_nst >= dmdv->cv_nstlp + MSBP) ||
                  (dgamrat > DGMAX);



  //cudaDevicezaxpby(1.0, dzn, -1.0, md->cv_last_yn, dftemp, nrows);
  dftemp[i]=dzn[i]+(-md->cv_last_yn[i]);

  __syncthreads();
  int guessflag=CudaDeviceguess_helper(t_n, h_n, dzn,
                                       md->cv_last_yn, dftemp, dtempv,
                                       md->cv_acor_init, cv_reltol, nrows,
          //check_model_state
                         threshhold, replacement_value, &flagDevice,
          //f_gpu
                         time_step, deriv_length_cell, state_size_cell,
                         n_cells, i_kernel, threads_block, n_shr_empty,
                         md, dmdv
  );
  __syncthreads();

#ifdef DEV_printmin
  printmin(md,dtempv,"cudaDevicecvSet after guess_helper dtempv");
#endif

  //if(i==0)printf("cudaDevicecvNlsNewton guessflag %d block %d\n",guessflag,blockIdx.x);

  if(guessflag<0){
  //if(*flag<0){
    *flag=RHSFUNC_RECVR;
    //if(threadIdx.x==0)printf("CudaDeviceguess_helper guessflag RHSFUNC_RECVR block %d\n", blockIdx.x);
    return RHSFUNC_RECVR;
  }

/*
    if (cv_mem->cv_ghfun) {
    //N_VScale(cv_mem->cv_rl1, cv_mem->cv_zn[1], cv_mem->cv_ftemp);

    N_VLinearSum(ONE, cv_mem->cv_zn[0], -ONE, cv_mem->cv_last_yn, cv_mem->cv_ftemp);
    retval = cv_mem->cv_ghfun(cv_mem->cv_tn, cv_mem->cv_h, cv_mem->cv_zn[0],
                              cv_mem->cv_last_yn, cv_mem->cv_ftemp, cv_mem->cv_user_data,
                              cv_mem->cv_tempv, cv_mem->cv_acor_init);
    if (retval < 0){
      return (RHSFUNC_RECVR);
    }
  }
*/

  for(;;) {

    __syncthreads();
    dcv_y[i] = dzn[i];

#ifdef DEV_printmin
    //printmin(md,md->state,"cudaDevicef start state");
#endif

    int aux_flag=0;

    int retval=cudaDevicef(
            //f_gpu
            time_step, deriv_length_cell, state_size_cell,
            n_cells, i_kernel, threads_block, n_shr_empty, dcv_y,
            dftemp,md,dmdv,&aux_flag
    );

    if (retval < 0) {
      return CV_RHSFUNC_FAIL;
    }
    if (retval> 0) {
      return RHSFUNC_RECVR;
    }

    __syncthreads();
    //if (i == 0)
    dmdv->cv_nfe++;
    __syncthreads();

    if (callSetup==1) {

#ifdef CAMP_DEBUG_GPU

      start = clock();

#endif

      __syncthreads();
      int linflag=cudaDevicelinsolsetup(
              dA, djA, diA, dtempv, //Input data
              nrows, blocks, n_shr_empty, maxIt, mattype,
              n_cells, tolmax, ddiag, //Init variables
              dr0, dr0h, dn0, dp0,
              dt, ds, dAx2, dy, dz,// Auxiliary vectors
              //swapCSC_CSR_BCG
              diB, djB, dB,
              //Guess_helper
              t_n, h_n, dftemp,
              dcv_y, tmp1,
              corr, cv_reltol,
              //update_state
              threshhold, replacement_value, flag,
              //f_gpu
              time_step, deriv_length_cell, state_size_cell,
              i_kernel, threads_block, md, dmdv,
              //cudacvNewtonIteration
              cv_acor,dewt,convfail,
              disavedJ, djsavedJ, dsavedJ
      );
      __syncthreads();

#ifdef CAMP_DEBUG_GPU
#ifdef cudaGlobalCVode_timers_max_blocks

      md->dtlinsolsetup[i] += ((double)(clock() - start))/(clock_khz*1000);

#else

      if(i==0) *md->dtlinsolsetup += ((double)(clock() - start))/(clock_khz*1000);

#endif
#endif

      //if (i == 0)
      dmdv->cv_nsetups++; //needed?
      callSetup = 0;
      dmdv->cv_gamrat = dmdv->cv_crate = 1.0;
      dmdv->cv_gammap = dmdv->cv_gamma;
      dmdv->cv_nstlp = dmdv->cv_nst;
      // Return if lsetup failed

      if (linflag < 0) {
        flag_shr[0] = CV_LSETUP_FAIL;
        break;
      }
      if (linflag > 0) {
        flag_shr[0] = CONV_FAIL;
        break;
      }

    }

    __syncthreads();
    cv_acor[i] = 0.0;


#ifdef CAMP_DEBUG_GPU

    start = clock();

#endif

    __syncthreads();
    int nItflag=cudaDevicecvNewtonIteration(
            dA, djA, diA, dx, dtempv, //Input data
            nrows, blocks, n_shr_empty, maxIt, mattype,
            n_cells, tolmax, ddiag, //Init variables
            dr0, dr0h, dn0, dp0,
            dt, ds, dAx2, dy, dz,// Auxiliary vectors
            //swapCSC_CSR_BCG
            diB, djB, dB,
            //Guess_helper
            t_n, h_n, dftemp,
            dcv_y, tmp1,
            corr, cv_reltol,
            //update_state
            threshhold, replacement_value, flag,
            //f_gpu
            time_step, deriv_length_cell, state_size_cell,
            i_kernel, threads_block, md, dmdv,
            //cudacvNewtonIteration
            cv_acor, dzn,
            dewt
    );
    __syncthreads();

#ifdef CAMP_DEBUG_GPU
#ifdef cudaGlobalCVode_timers_max_blocks

    md->dtNewtonIteration[i] += ((double)(clock() - start))/(clock_khz*1000);

#else

    if(i==0) *md->dtNewtonIteration += ((double)(clock() - start))/(clock_khz*1000);

#endif
#endif

    if (nItflag != TRY_AGAIN) {
      return nItflag;
    }

    __syncthreads();
    callSetup = 1;
    __syncthreads();
    convfail = CV_FAIL_BAD_J;

    __syncthreads();


  }//for(;;)

  //if(threadIdx.x==0)printf("cudaDevicecvNlsNewton2 flag_shr[0] %d block %d\n",flag_shr[0], blockIdx.x);

  __syncthreads();
  return *flag;

}

__device__
void cudaDevicecvRescale(ModelDataGPU *md, ModelDataVariable *dmdv) {

  extern __shared__ double dzn[];

  int j;
  double factor;

  //if(i==0)printf("cudaDevicecvRescale2 start\n");

  __syncthreads();

  factor = dmdv->cv_eta;
  for (j=1; j <= dmdv->cv_q; j++) {
    //N_VScale(factor, md->dzn[j], md->dzn[j]);

    cudaDevicescaley(&md->dzn[md->nrows*(j)],factor,md->nrows);

    __syncthreads();
    //if(i==0)printf("cudaDevicecvRescale2 factor %le j %d\n",factor,j);
    factor *= dmdv->cv_eta;
    __syncthreads();
  }

  dmdv->cv_h = dmdv->cv_hscale * dmdv->cv_eta;
  dmdv->cv_next_h = dmdv->cv_h;
  dmdv->cv_hscale = dmdv->cv_h;
  dmdv->cv_nscon = 0;

  __syncthreads();

}

__device__
void cudaDevicecvRestore(ModelDataGPU *md, ModelDataVariable *dmdv, double saved_t) {

  extern __shared__ double dzn[];
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  int j, k;

#ifdef DEV_cudaDevicecvRestore
  //if(i==0)printf("DEV_cudaDevicecvRestore start\n");
  //if(i==0)printf("DEV_cudaDevicecvRestore dmdv->cv_tn %le saved_t %le\n",dmdv->cv_tn,saved_t);

#endif

  __syncthreads();
  dmdv->cv_tn=saved_t;

  for (k = 1; k <= dmdv->cv_q; k++){
    for (j = dmdv->cv_q; j >= k; j--) {
      //N_VLinearSum(ONE, cv_mem->cv_zn[j-1], -ONE,
      //             cv_mem->cv_zn[j], cv_mem->cv_zn[j-1]);

    cudaDevicezaxpby(1., &md->dzn[md->nrows*(j-1)], -1.,
            &md->dzn[md->nrows*(j)], &md->dzn[md->nrows*(j-1)], md->nrows);

    }
  }

  //N_VScale(ONE, cv_mem->cv_last_yn, cv_mem->cv_zn[0]);
  md->dzn[i]=md->cv_last_yn[i];

  __syncthreads();

#ifdef DEV_cudaDevicecvRestore
  //if(i==0)printf("DEV_cudaDevicecvRestore end\n");
#endif

}

__device__
int cudaDevicecvHandleNFlag(ModelDataGPU *md, ModelDataVariable *dmdv, int *nflagPtr, double saved_t,
                             int *ncfPtr) {

  extern __shared__ int flag_shr[];

  //if(i==0)printf("cudaDevicecvHandleNFlag *md->flag %d \n",*md->flag);

  if (*nflagPtr == CV_SUCCESS){
    return(DO_ERROR_TEST);
  }

  // The nonlinear soln. failed; increment ncfn and restore zn
  //if(i==0)
    dmdv->cv_ncfn++;

  cudaDevicecvRestore(md, dmdv, saved_t);
  //__syncthreads();

  if (*nflagPtr == CV_LSETUP_FAIL)  return(CV_LSETUP_FAIL);
  if (*nflagPtr == CV_LSOLVE_FAIL)  return(CV_LSOLVE_FAIL);
  if (*nflagPtr == CV_RHSFUNC_FAIL) return(CV_RHSFUNC_FAIL);


  (*ncfPtr)++;
  dmdv->cv_etamax = 1.;

  // If we had maxncf failures or |h| = hmin,
  //   return CV_CONV_FAILURE or CV_REPTD_RHSFUNC_ERR.

  __syncthreads();

  if ((fabs(dmdv->cv_h) <= dmdv->cv_hmin*ONEPSM) ||
      (*ncfPtr == dmdv->cv_maxncf)) {
    if (*nflagPtr == CONV_FAIL)     return(CV_CONV_FAILURE);
    if (*nflagPtr == RHSFUNC_RECVR) return(CV_REPTD_RHSFUNC_ERR);
  }

  // Reduce step size; return to reattempt the step
  __syncthreads();
  dmdv->cv_eta = SUNMAX(ETACF,
          dmdv->cv_hmin / fabs(dmdv->cv_h));
  __syncthreads();
  *nflagPtr = PREV_CONV_FAIL;
  cudaDevicecvRescale(md, dmdv);
  __syncthreads();

  return (PREDICT_AGAIN);

}

__device__
void cudaDevicecvSetTqBDFt(ModelDataGPU *md, ModelDataVariable *dmdv,
                           double hsum, double alpha0,
                           double alpha0_hat, double xi_inv, double xistar_inv) {

  extern __shared__ int flag_shr[];

  double A1, A2, A3, A4, A5, A6;
  double C, Cpinv, Cppinv;

  __syncthreads();

  A1 = 1. - alpha0_hat + alpha0;
  A2 = 1. + dmdv->cv_q * A1;

  md->cv_tq[2+blockIdx.x*(NUM_TESTS + 1)] = fabs(A1 / (alpha0 * A2));

  md->cv_tq[5+blockIdx.x*(NUM_TESTS + 1)] = fabs(A2 * xistar_inv / (md->cv_l[dmdv->cv_q+blockIdx.x*L_MAX] * xi_inv));
  if (dmdv->cv_qwait == 1) {
    if (dmdv->cv_q > 1) {
      C = xistar_inv / md->cv_l[dmdv->cv_q+blockIdx.x*L_MAX];
      A3 = alpha0 + 1. / dmdv->cv_q;
      A4 = alpha0_hat + xi_inv;
      Cpinv = (1. - A4 + A3) / A3;
      md->cv_tq[1+blockIdx.x*(NUM_TESTS + 1)] = fabs(C * Cpinv);
    }
    else md->cv_tq[1+blockIdx.x*(NUM_TESTS + 1)] = 1.;

    __syncthreads();

    hsum += md->cv_tau[dmdv->cv_q+blockIdx.x*(L_MAX + 1)];
    xi_inv = dmdv->cv_h / hsum;
    A5 = alpha0 - (1. / (dmdv->cv_q+1));
    A6 = alpha0_hat - xi_inv;
    Cppinv = (1. - A6 + A5) / A2;
    md->cv_tq[3+blockIdx.x*(NUM_TESTS + 1)] = fabs(Cppinv / (xi_inv * (dmdv->cv_q+2) * A5));
    __syncthreads();
  }

  md->cv_tq[4+blockIdx.x*(NUM_TESTS + 1)] = dmdv->cv_nlscoef / md->cv_tq[2+blockIdx.x*(NUM_TESTS + 1)];

}

__device__
void cudaDevicecvSetBDF(ModelDataGPU *md, ModelDataVariable *dmdv) {

  extern __shared__ int flag_shr[];

  double alpha0, alpha0_hat, xi_inv, xistar_inv, hsum;
  int z,j;

  __syncthreads();

  md->cv_l[0+blockIdx.x*L_MAX] = md->cv_l[1+blockIdx.x*L_MAX] = xi_inv = xistar_inv = 1.;
  for (z=2; z <= dmdv->cv_q; z++) md->cv_l[z+blockIdx.x*L_MAX] = 0.;
  alpha0 = alpha0_hat = -1.;
  hsum = dmdv->cv_h;
  __syncthreads();
  if (dmdv->cv_q > 1) {
    for (j=2; j < dmdv->cv_q; j++) {
      hsum += md->cv_tau[j-1+blockIdx.x*(L_MAX + 1)];
      xi_inv = dmdv->cv_h / hsum;
      alpha0 -= 1. / j;
      for (z=j; z >= 1; z--) md->cv_l[z+blockIdx.x*L_MAX] += md->cv_l[z-1+blockIdx.x*L_MAX]*xi_inv;
      // The l[z] are coefficients of product(1 to j) (1 + x/xi_i)
    }
    __syncthreads();
    // j = q
    alpha0 -= 1. / dmdv->cv_q;
    xistar_inv = -md->cv_l[1+blockIdx.x*L_MAX] - alpha0;
    hsum += md->cv_tau[dmdv->cv_q-1+blockIdx.x*(L_MAX + 1)];
    xi_inv = dmdv->cv_h / hsum;
    alpha0_hat = -md->cv_l[1+blockIdx.x*L_MAX] - xi_inv;
    for (z=dmdv->cv_q; z >= 1; z--)
      md->cv_l[z+blockIdx.x*L_MAX] += md->cv_l[z-1+blockIdx.x*L_MAX]*xistar_inv;
  }
  __syncthreads();
  cudaDevicecvSetTqBDFt(md, dmdv, hsum, alpha0, alpha0_hat, xi_inv, xistar_inv);

}

__device__
void cudaDevicecvSet(ModelDataGPU *md, ModelDataVariable *dmdv) {

  extern __shared__ int flag_shr[];
#ifdef DEV_printmin
  printmin(md,md->dtempv,"cudaDevicecvSet Start dtempv");
#endif
  __syncthreads();
  cudaDevicecvSetBDF(md,dmdv);
  __syncthreads();

  dmdv->cv_rl1 = 1.0 / md->cv_l[1+blockIdx.x*L_MAX];
  dmdv->cv_gamma = dmdv->cv_h * dmdv->cv_rl1;
  __syncthreads();
  if (dmdv->cv_nst == 0){
    //if(threadIdx.x == 0)
      //printf("dmdv->cv_nst == 0\n");
    dmdv->cv_gammap = dmdv->cv_gamma;

  }
  //if(threadIdx.x == 0)printf("cudaDevicecvSet3 dmdv->cv_nst %d dmdv->cv_gammap %le block %d\n", dmdv->cv_nst, dmdv->cv_gammap, blockIdx.x);
  __syncthreads();
  dmdv->cv_gamrat = (dmdv->cv_nst > 0) ?
                    dmdv->cv_gamma / dmdv->cv_gammap : 1.;  // protect x / x != 1.0
  __syncthreads();
}

__device__
void cudaDevicecvPredict(ModelDataGPU *md, ModelDataVariable *dmdv) {

  extern __shared__ double dzn[];
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  int j, k;
#ifdef DEV_printmin
  printmin(md,md->dtempv,"cudaDevicecvPredict start dtempv");
#endif
  __syncthreads();
  dmdv->cv_tn += dmdv->cv_h;
  __syncthreads();
  if (dmdv->cv_tstopset) {
    if ((dmdv->cv_tn - dmdv->cv_tstop)*dmdv->cv_h > 0.)
      dmdv->cv_tn = dmdv->cv_tstop;
  }

  //N_VScale(ONE, cv_mem->cv_zn[0], cv_mem->cv_last_yn);
  md->cv_last_yn[i]=md->dzn[i];

  for (k = 1; k <= dmdv->cv_q; k++){
    __syncthreads();
    for (j = dmdv->cv_q; j >= k; j--){
      __syncthreads();
      //N_VLinearSum(ONE, cv_mem->cv_zn[j-1], ONE,
      //             cv_mem->cv_zn[j], cv_mem->cv_zn[j-1]);
      cudaDevicezaxpby(1., &md->dzn[md->nrows*(j-1)], 1.,
                       &md->dzn[md->nrows*(j)], &md->dzn[md->nrows*(j-1)], md->nrows);

    }
    __syncthreads();
  }
  __syncthreads();
}

__device__
void cudaDevicecvDecreaseBDF(ModelDataGPU *md, ModelDataVariable *dmdv) {

  extern __shared__ double dzn[];

  double hsum, xi;
  int z, j;

  for (z=0; z <= dmdv->cv_qmax; z++) md->cv_l[z+blockIdx.x*L_MAX] = 0.;
  md->cv_l[2+blockIdx.x*L_MAX] = 1.;

  hsum = 0.;
  for (j=1; j <= dmdv->cv_q-2; j++) {
    hsum += md->cv_tau[j+blockIdx.x*(L_MAX + 1)];
    xi = hsum /dmdv->cv_hscale;
    for (z=j+2; z >= 2; z--)
      md->cv_l[z+blockIdx.x*L_MAX] = md->cv_l[z+blockIdx.x*L_MAX]*xi + md->cv_l[z-1+blockIdx.x*L_MAX];
  }

  for (j=2; j < dmdv->cv_q; j++){


    //N_VLinearSum(-cv_mem->cv_l[j], cv_mem->cv_zn[cv_mem->cv_q],
    //             ONE, cv_mem->cv_zn[j], cv_mem->cv_zn[j]);

    cudaDevicezaxpby(-md->cv_l[j+blockIdx.x*L_MAX],
                     &md->dzn[md->nrows*(dmdv->cv_q)],
                     1., &md->dzn[md->nrows*(j)],
                     &md->dzn[md->nrows*(j)], md->nrows);

    }

}

__device__
int cudaDevicecvDoErrorTest(ModelDataGPU *md, ModelDataVariable *dmdv,
                             int *nflagPtr,
                             double saved_t, int *nefPtr, double *dsmPtr) {

  //extern __shared__ int flag_shr[];
  //extern __shared__ double flag_shr2[];
  extern __shared__ double dzn[];
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  double dsm;
  double min_val;
  int retval;

  // Find the minimum concentration and if it's small and negative, make it
  // positive
  //N_VLinearSum(cv_mem->cv_l[0], cv_mem->cv_acor, ONE, cv_mem->cv_zn[0],
  //             cv_mem->cv_ftemp);

  cudaDevicezaxpby(md->cv_l[0+blockIdx.x*L_MAX],
                   md->cv_acor, 1., md->dzn, md->dftemp, md->nrows);

  //min_val = N_VMin(cv_mem->cv_ftemp);
  cudaDevicemin(&min_val, md->dftemp[i], dzn, md->n_shr_empty);

  if (min_val < 0. && min_val > -CAMP_TINY) {
    //N_VAbs(cv_mem->cv_ftemp, cv_mem->cv_ftemp);
    md->dftemp[i]=fabs(md->dftemp[i]);

    //N_VLinearSum(-cv_mem->cv_l[0], cv_mem->cv_acor, ONE, cv_mem->cv_ftemp,
    //             cv_mem->cv_zn[0]);
    cudaDevicezaxpby(-md->cv_l[0+blockIdx.x*L_MAX],
                     md->cv_acor, 1., md->dftemp, md->dzn, md->nrows);

    min_val = 0.;
  }

  dsm = dmdv->cv_acnrm * md->cv_tq[2+blockIdx.x*(NUM_TESTS + 1)];

  // If est. local error norm dsm passes test and there are no negative values,
  // return CV_SUCCESS
  *dsmPtr = dsm;
  if (dsm <= 1. && min_val >= 0.) return(CV_SUCCESS);

  // Test failed; increment counters, set nflag, and restore zn array
  (*nefPtr)++;
  dmdv->cv_netf++;
  *nflagPtr = PREV_ERR_FAIL;
  cudaDevicecvRestore(md, dmdv, saved_t);

  __syncthreads();

  // At maxnef failures or |h| = hmin, return CV_ERR_FAILURE
  if ((fabs(dmdv->cv_h) <= dmdv->cv_hmin*ONEPSM) ||
      (*nefPtr == dmdv->cv_maxnef)) return(CV_ERR_FAILURE);

  // Set etamax = 1 to prevent step size increase at end of this step
  dmdv->cv_etamax = 1.;

  __syncthreads();

  // Set h ratio eta from dsm, rescale, and return for retry of step
  if (*nefPtr <= MXNEF1) {
    //dmdv->cv_eta = 1. / (SUNRpowerR(BIAS2*dsm,ONE/cv_mem->cv_L) + ADDON);
    dmdv->cv_eta = 1. / (pow(BIAS2*dsm,1./dmdv->cv_L) + ADDON);
    __syncthreads();
    dmdv->cv_eta = SUNMAX(ETAMIN, SUNMAX(dmdv->cv_eta,
                           dmdv->cv_hmin / fabs(dmdv->cv_h)));
    __syncthreads();
    if (*nefPtr >= SMALL_NEF)
      dmdv->cv_eta = SUNMIN(dmdv->cv_eta, ETAMXF);
    __syncthreads();

    cudaDevicecvRescale(md, dmdv);
    return(TRY_AGAIN);
  }

  __syncthreads();

  // After MXNEF1 failures, force an order reduction and retry step
  if (dmdv->cv_q > 1) {
    dmdv->cv_eta = SUNMAX(ETAMIN,
    dmdv->cv_hmin / fabs(dmdv->cv_h));
    //never enters?
    //if(i==0)printf("dmdv->cv_q > 1\n");
    //cvAdjustOrder_gpu2(cv_mem,-1);
    cudaDevicecvDecreaseBDF(md, dmdv);

    dmdv->cv_L = dmdv->cv_q;
    dmdv->cv_q--;
    dmdv->cv_qwait = dmdv->cv_L;
    cudaDevicecvRescale(md, dmdv);
    __syncthreads();
    return(TRY_AGAIN);
  }

  // If already at order 1, restart: reload zn from scratch

  __syncthreads();

  dmdv->cv_eta = SUNMAX(ETAMIN, dmdv->cv_hmin / fabs(dmdv->cv_h));
  __syncthreads();
  dmdv->cv_h *= dmdv->cv_eta;
  dmdv->cv_next_h = dmdv->cv_h;
  dmdv->cv_hscale = dmdv->cv_h;
  dmdv->cv_qwait = 10;
  dmdv->cv_nscon = 0;


  //retval = cv_mem->cv_f(cv_mem->cv_tn, cv_mem->cv_zn[0],
  //                      cv_mem->cv_tempv, cv_mem->cv_user_data);

  int aux_flag=0;

#ifdef DEV_printmin
  printmin(md,md->state,"cudaDevicef start state");
#endif

  retval=cudaDevicef(
          dmdv->cv_tn, md->deriv_length_cell, md->state_size_cell,
          md->n_cells, md->i_kernel, md->threads_block, md->n_shr_empty,
          md->dzn, md->dtempv,md,dmdv, &aux_flag
  );


  dmdv->cv_nfe++;
  if (retval < 0)  return(CV_RHSFUNC_FAIL);
  if (retval > 0)  return(CV_UNREC_RHSFUNC_ERR);

  //N_VScale(cv_mem->cv_h, cv_mem->cv_tempv, cv_mem->cv_zn[1]);
    md->dzn[1*md->nrows+i]=dmdv->cv_h*md->dtempv[i];

  return(TRY_AGAIN);

}

__device__
void cudaDevicecvCompleteStep(ModelDataGPU *md, ModelDataVariable *dmdv) {

  extern __shared__ double dzn[];
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  int z, j;
#ifdef DEV_printmin
  printmin(md,md->dtempv,"cudaDevicecvCompleteStep start dtempv");
#endif
  dmdv->cv_nst++;
  dmdv->cv_nscon++;
  dmdv->cv_hu = dmdv->cv_h;
  dmdv->cv_qu = dmdv->cv_q;

  for (z=dmdv->cv_q; z >= 2; z--)  md->cv_tau[z+blockIdx.x*(L_MAX + 1)] = md->cv_tau[z-1+blockIdx.x*(L_MAX + 1)];
  if ((dmdv->cv_q==1) && (dmdv->cv_nst > 1))
    md->cv_tau[2+blockIdx.x*(L_MAX + 1)] = md->cv_tau[1+blockIdx.x*(L_MAX + 1)];
  md->cv_tau[1+blockIdx.x*(L_MAX + 1)] = dmdv->cv_h;

  __syncthreads();


  // Apply correction to column j of zn: l_j * Delta_n
  for (j=0; j <= dmdv->cv_q; j++){

    //N_VLinearSum(md->cv_l[j], md->cv_acor, ONE,
    //            md->cv_zn[j], md->cv_zn[j]);

    cudaDevicezaxpby(md->cv_l[j+blockIdx.x*L_MAX],
                     md->cv_acor,
                     1., &md->dzn[md->nrows*(j)],
                     &md->dzn[md->nrows*(j)], md->nrows);

  }
  dmdv->cv_qwait--;
  if ((dmdv->cv_qwait == 1) && (dmdv->cv_q != dmdv->cv_qmax)) {

    //N_VScale(ONE, md->cv_acor, md->cv_zn[dmdv->cv_qmax]);
    md->dzn[md->nrows*(dmdv->cv_qmax)+i]=md->cv_acor[i];

    dmdv->cv_saved_tq5 = md->cv_tq[5+blockIdx.x*(NUM_TESTS + 1)];
    dmdv->cv_indx_acor = dmdv->cv_qmax;
  }

}

__device__
void cudaDevicecvChooseEta(ModelDataGPU *md, ModelDataVariable *dmdv) {

  extern __shared__ double dzn[];
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  double etam;

  etam = SUNMAX(dmdv->cv_etaqm1, SUNMAX(dmdv->cv_etaq, dmdv->cv_etaqp1));

  __syncthreads();

  if (etam < THRESH) {
    dmdv->cv_eta = 1.;
    dmdv->cv_qprime = dmdv->cv_q;
    return;
  }

  __syncthreads();

  if (etam == dmdv->cv_etaq) {

    dmdv->cv_eta = dmdv->cv_etaq;
    dmdv->cv_qprime = dmdv->cv_q;

  } else if (etam == dmdv->cv_etaqm1) {

    dmdv->cv_eta = dmdv->cv_etaqm1;
    dmdv->cv_qprime = dmdv->cv_q - 1;

  } else {

    dmdv->cv_eta = dmdv->cv_etaqp1;
    dmdv->cv_qprime = dmdv->cv_q + 1;

    __syncthreads();

    if (dmdv->cv_lmm == CV_BDF) {
      //
       // Store Delta_n in zn[qmax] to be used in order increase
       //
       // This happens at the last step of order q before an increase
       // to order q+1, so it represents Delta_n in the ELTE at q+1
       //

      //N_VScale(ONE, dmdv->cv_acor, dmdv->cv_zn[dmdv->cv_qmax]);
      md->dzn[md->nrows*(dmdv->cv_qmax)+i]=md->cv_acor[i];

    }
  }

  __syncthreads();

}

__device__
void cudaDevicecvSetEta(ModelDataGPU *md, ModelDataVariable *dmdv) {

  extern __shared__ int flag_shr[];

  __syncthreads();

  // If eta below the threshhold THRESH, reject a change of step size
  if (dmdv->cv_eta < THRESH) {
    dmdv->cv_eta = 1.;
    //Never enters (ensures it works anyway
    dmdv->cv_hprime = dmdv->cv_h;
  } else {
    // Limit eta by etamax and hmax, then set hprime
    __syncthreads();
    dmdv->cv_eta = SUNMIN(dmdv->cv_eta, dmdv->cv_etamax);
    __syncthreads();
    dmdv->cv_eta /= SUNMAX(ONE,
            fabs(dmdv->cv_h)*dmdv->cv_hmax_inv*dmdv->cv_eta);
    __syncthreads();
    dmdv->cv_hprime = dmdv->cv_h * dmdv->cv_eta;
    //printf("dmdv->cv_eta NOT < THRESH %le dmdv->cv_hprime block %d\n", dmdv->cv_hprime, blockIdx.x);

    __syncthreads();
    if (dmdv->cv_qprime < dmdv->cv_q) dmdv->cv_nscon = 0;
  }

  __syncthreads();

}

__device__
int cudaDevicecvPrepareNextStep(ModelDataGPU *md, ModelDataVariable *dmdv, double dsm) {

  extern __shared__ double sdata[];
  __syncthreads();
#ifdef DEV_printmin
  printmin(md,md->dtempv,"cudaDevicecvPrepareNextStep start dtempv");
#endif
  // If etamax = 1, defer step size or order changes
  if (dmdv->cv_etamax == 1.) {
    dmdv->cv_qwait = SUNMAX(dmdv->cv_qwait, 2);
    dmdv->cv_qprime = dmdv->cv_q;
    dmdv->cv_hprime = dmdv->cv_h;
    dmdv->cv_eta = 1.;
    return 0;
  }

  __syncthreads();

  // etaq is the ratio of new to old h at the current order
  //dmdv->cv_etaq = 1. /(SUNRpowerR(BIAS2*dsm,1./dmdv->cv_L) + ADDON);
  dmdv->cv_etaq = 1. /(pow(BIAS2*dsm,1./dmdv->cv_L) + ADDON);

  __syncthreads();

  // If no order change, adjust eta and acor in cvSetEta and return
  if (dmdv->cv_qwait != 0) {
    dmdv->cv_eta = dmdv->cv_etaq;
    dmdv->cv_qprime = dmdv->cv_q;
    cudaDevicecvSetEta(md, dmdv);
    return 0;
  }

  __syncthreads();

  // If qwait = 0, consider an order change.   etaqm1 and etaqp1 are
  //  the ratios of new to old h at orders q-1 and q+1, respectively.
  //  cvChooseEta selects the largest; cvSetEta adjusts eta and acor
  dmdv->cv_qwait = 2;

  //compute cv_etaqm1
  double ddn;
  dmdv->cv_etaqm1 = 0.;
  __syncthreads();
  if (dmdv->cv_q > 1) {
    cudaDeviceVWRMS_Norm(&md->dzn[md->nrows*(dmdv->cv_q)],
                         md->dewt, &ddn, md->nrows, md->n_shr_empty);
    __syncthreads();
    ddn *= md->cv_tq[1+blockIdx.x*(NUM_TESTS + 1)];
    __syncthreads();
    dmdv->cv_etaqm1 = 1./(pow(BIAS1*ddn, 1./dmdv->cv_q) + ADDON);
  }

  //compute cv_etaqp1
  double dup, cquot;
  dmdv->cv_etaqp1 = 0.;
  __syncthreads();
  if (dmdv->cv_q != dmdv->cv_qmax && dmdv->cv_saved_tq5 != 0.) {
    //cquot = (dmdv->cv_tq[5] / dmdv->cv_saved_tq5) *
    //        SUNRpowerI(dmdv->cv_h/md->cv_tau[2], dmdv->cv_L); //maybe need custom function?
    cquot = (md->cv_tq[5+blockIdx.x*(NUM_TESTS + 1)] / dmdv->cv_saved_tq5) *
            pow(double(dmdv->cv_h/md->cv_tau[2+blockIdx.x*(L_MAX + 1)]), double(dmdv->cv_L));

    //N_VLinearSum(-cquot, dmdv->cv_zn[dmdv->cv_qmax], ONE,
    //             dmdv->cv_acor, dmdv->cv_tempv);

    cudaDevicezaxpby(-cquot,
    &md->dzn[md->nrows*(dmdv->cv_qmax)],
    1., md->cv_acor,
    md->dtempv, md->nrows);

    //dup = N_VWrmsNorm(md->dtempv, cv_mem->cv_ewt) * cv_mem->cv_tq[3];
    cudaDeviceVWRMS_Norm(md->dtempv, md->dewt, &dup, md->nrows, md->n_shr_empty);

    __syncthreads();
    dup *= md->cv_tq[3+blockIdx.x*(NUM_TESTS + 1)];
    __syncthreads();
    dmdv->cv_etaqp1 = 1. / (pow(BIAS3*dup, 1./(dmdv->cv_L+1)) + ADDON);
  }

  __syncthreads();
  cudaDevicecvChooseEta(md, dmdv);
  __syncthreads();
  cudaDevicecvSetEta(md, dmdv);
  __syncthreads();

  return CV_SUCCESS;

}

__device__
void cudaDevicecvIncreaseBDF(ModelDataGPU *md, ModelDataVariable *dmdv) {

  extern __shared__ double dzn[];
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int tid = threadIdx.x;

  double alpha0, alpha1, prod, xi, xiold, hsum, A1;
  int z, j;

  for (z=0; z <= dmdv->cv_qmax; z++) md->cv_l[z+blockIdx.x*L_MAX] = 0.;
  md->cv_l[2+blockIdx.x*L_MAX] = alpha1 = prod = xiold = 1.;

  alpha0 = -1.;
  hsum = dmdv->cv_hscale;
  if (dmdv->cv_q > 1) {
    for (j=1; j < dmdv->cv_q; j++) {
      hsum += md->cv_tau[j+1+blockIdx.x*(L_MAX + 1)];
      xi = hsum / dmdv->cv_hscale;
      prod *= xi;
      alpha0 -= 1. / (j+1);
      alpha1 += 1. / xi;
      for (z=j+2; z >= 2; z--)
        md->cv_l[z+blockIdx.x*L_MAX] = md->cv_l[z+blockIdx.x*L_MAX]*xiold + md->cv_l[z-1+blockIdx.x*L_MAX];
      xiold = xi;
    }
  }

  A1 = (-alpha0 - alpha1) / prod;
  //N_VScale(A1, md->cv_zn[dmdv->cv_indx_acor],
  //         md->cv_zn[dmdv->cv_L]);

  //__syncthreads();
  dzn[tid]=md->dzn[md->nrows*(dmdv->cv_L)+i];

  dzn[tid]=A1*md->dzn[md->nrows*(dmdv->cv_indx_acor)+i];

  md->dzn[md->nrows*(dmdv->cv_L)+i]=dzn[tid];
  //__syncthreads();

  for (j=2; j <= dmdv->cv_q; j++){
    //N_VLinearSum(md->cv_l[j], md->cv_zn[dmdv->cv_L], ONE,
    //             md->cv_zn[j], md->cv_zn[j]);

    cudaDevicezaxpby(md->cv_l[j+blockIdx.x*L_MAX],
    &md->dzn[md->nrows*(dmdv->cv_L)],
    1., &md->dzn[md->nrows*(j)],
    &md->dzn[md->nrows*(j)], md->nrows);

  }

}

__device__
void cudaDevicecvAdjustParams(ModelDataGPU *md, ModelDataVariable *dmdv) {

  if (dmdv->cv_qprime != dmdv->cv_q) {

    int deltaq = dmdv->cv_qprime-dmdv->cv_q;
    switch(deltaq) {
      case 1:
        cudaDevicecvIncreaseBDF(md, dmdv);
        break;
      case -1:
        cudaDevicecvDecreaseBDF(md, dmdv);
        break;
    }

    dmdv->cv_q = dmdv->cv_qprime;
    dmdv->cv_L = dmdv->cv_q+1;
    dmdv->cv_qwait = dmdv->cv_L;
  }
  cudaDevicecvRescale(md, dmdv);
}

__device__
int cudaDevicecvStep(ModelDataGPU *md, ModelDataVariable *dmdv) {

  extern __shared__ double sdata[];
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  double saved_t = dmdv->cv_tn;
  int ncf = 0;
  int nef = 0;
  dmdv->nflag = FIRST_CALL;
  int nflag=FIRST_CALL;
  double dsm;

  __syncthreads();

  if ((dmdv->cv_nst > 0) && (dmdv->cv_hprime != dmdv->cv_h)){
    cudaDevicecvAdjustParams(md, dmdv);
  }

  __syncthreads();

  for (;;) {
    __syncthreads();
    cudaDevicecvPredict(md, dmdv);
    __syncthreads();
    cudaDevicecvSet(md, dmdv);
    __syncthreads();

    nflag = cudaDevicecvNlsNewton(
            md->dA, md->djA, md->diA, md->dx, md->dtempv, //Input data
            md->nrows, md->blocks, md->n_shr_empty, md->maxIt, md->mattype,
            md->n_cells, md->tolmax, md->ddiag, //Init variables
            md->dr0, md->dr0h, md->dn0, md->dp0,
            md->dt, md->ds, md->dAx2, md->dy, md->dz,// Auxiliary vectors
            //swapCSC_CSR_BCG
            md->diB, md->djB, md->dB,
            //Guess_helper //*md->cv_tn
            dmdv->cv_tn, dmdv->cv_h, md->dftemp,
            md->dcv_y, md->dtempv1,
            md->dtempv2, dmdv->cv_reltol,
            //update_state
            md->threshhold, md->replacement_value, &nflag,
            //f_gpu
            dmdv->cv_next_h, md->deriv_length_cell, md->state_size_cell,
            md->i_kernel, md->threads_block, md, dmdv,
            //cudacvNewtonIteration
            md->cv_acor, md->dzn, md->dewt,
            md->disavedJ, md->djsavedJ, md->dsavedJ
    );

    __syncthreads();
    dmdv->nflag = nflag;
    __syncthreads();
#ifdef DEBUG_cudaDevicecvStep
    if(threadIdx.x==0)printf("DEV_CUDACVSTEP nflag %d dmdv->nflag %d block %d\n",dmdv->nflag, dmdv->nflag, blockIdx.x);
#endif
    int kflag = cudaDevicecvHandleNFlag(md, dmdv, &nflag, saved_t, &ncf);
    //dmdv->kflag = cudaDevicecvHandleNFlag(md, dmdv, &nflag, saved_t, &ncf);
    //dmdv->kflag = cudaDevicecvHandleNFlag(md, dmdv, &dmdv->nflag, saved_t, &ncf);

    __syncthreads();
    dmdv->nflag = nflag;//needed?
    dmdv->kflag = kflag;
    __syncthreads();
#ifdef DEBUG_cudaDevicecvStep
    if(threadIdx.x==0)printf("DEV_CUDACVSTEP kflag %d block %d\n",dmdv->kflag, blockIdx.x);
#endif
    // Go back in loop if we need to predict again (nflag=PREV_CONV_FAIL)

    if (dmdv->kflag == PREDICT_AGAIN) {
      //if (threadIdx.x == 0)printf("DEV_CUDACVSTEP kflag PREDICT_AGAIN block %d\n", blockIdx.x);
      continue;
    }

    // Return if nonlinear solve failed and recovery not possible.
    if (dmdv->kflag != DO_ERROR_TEST) {
      //if(threadIdx.x==0)printf("DEV_CUDACVSTEP kflag!=DO_ERROR_TEST block %d\n", blockIdx.x);
      return (dmdv->kflag);
    }

    __syncthreads();
    int eflag=cudaDevicecvDoErrorTest(md,dmdv,&nflag,saved_t,&nef,&dsm);
    //dmdv->eflag=cudaDevicecvDoErrorTest(md,dmdv,&dmdv->nflag,saved_t,&nef,&dsm);
    __syncthreads();
    dmdv->nflag = nflag;
    dmdv->eflag = eflag;
    __syncthreads();
#ifdef DEBUG_cudaDevicecvStep
    if(threadIdx.x==0)printf("DEV_CUDACVSTEP nflag %d eflag %d block %d\n",dmdv->nflag, dmdv->eflag, blockIdx.x);    //if(i==0)printf("eflag %d\n", eflag);
#endif
    // Go back in loop if we need to predict again (nflag=PREV_ERR_FAIL)
    if (dmdv->eflag == TRY_AGAIN){
      //if (threadIdx.x == 0)printf("DEV_CUDACVSTEP eflag TRY_AGAIN block %d\n", blockIdx.x);
      continue;
    }

    // Return if error test failed and recovery not possible.
    if (dmdv->eflag != CV_SUCCESS){
      //if (threadIdx.x == 0)printf("DEV_CUDACVSTEP eflag!=CV_SUCCESS block %d\n", blockIdx.x);
      return (dmdv->eflag);
    }

    // Error test passed (eflag=CV_SUCCESS), break from loop
    break;

  }

  // Nonlinear system solve and error test were both successful.
  // Update data, and consider change of step and/or order.
  __syncthreads();
  cudaDevicecvCompleteStep(md, dmdv);
  __syncthreads();
  cudaDevicecvPrepareNextStep(md, dmdv, dsm);
  __syncthreads();
  // If Stablilty Limit Detection is turned on, call stability limit
  // detection routine for possible order reduction.

 //  dmdv->cv_etamax = (dmdv->cv_nst <= SMALL_NST) ? ETAMX2 : ETAMX3;
  //dmdv->cv_etamax = (dmdv->cv_nst <= 10) ? 10. : 10.;
  dmdv->cv_etamax=10.;

  //if(i==0)printf("DEV_cudaDevicecvStep dmdv->cv._etamax %le dmdv->cv_nst %d\n",
  //               dmdv->cv_etamax,dmdv->cv_nst);

  //  Finally, we rescale the acor array to be the
  //  estimated local error vector.
  //N_VScale(cv_mem->cv_tq[2], cv_mem->cv_acor, cv_mem->cv_acor);
  md->cv_acor[i]*=md->cv_tq[2+blockIdx.x*(NUM_TESTS + 1)];

  //if(i==0)printf("DEV_cudaDevicecvStep end mdvo->cv_tn %le\n",mdvo->cv_tn);
  __syncthreads();
  //if(threadIdx.x==0)printf("DEV_CUDACVSTEP return CV_SUCCESS block %d\n",blockIdx.x);    //if(i==0)printf("eflag %d\n", eflag);

  return(CV_SUCCESS);

  }

__device__
int cudaDeviceCVodeGetDky(ModelDataGPU *md, ModelDataVariable *dmdv,
                           double t, int k, double *dky) {

  //extern __shared__ int flag_shr[];
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  double s, c, r;
  double tfuzz, tp, tn1;
  int z, j;

  __syncthreads();
   // Allow for some slack
   tfuzz = FUZZ_FACTOR * dmdv->cv_uround * (fabs(dmdv->cv_tn) + fabs(dmdv->cv_hu));
   if (dmdv->cv_hu < 0.) tfuzz = -tfuzz;
   tp = dmdv->cv_tn - dmdv->cv_hu - tfuzz;
   tn1 = dmdv->cv_tn + tfuzz;
   if ((t-tp)*(t-tn1) > 0.) {
     //cvProcessError(dmdv, CV_BAD_T, "CVODE", "CVodeGetDky", MSGCV_BAD_T,
     //               t, dmdv->cv_tn-dmdv->cv_hu, dmdv->cv_tn);
     return(CV_BAD_T);
   }

  __syncthreads();
   // Sum the differentiated interpolating polynomial

   s = (t - dmdv->cv_tn) / dmdv->cv_h;
   for (j=dmdv->cv_q; j >= k; j--) {
     c = 1.;
     for (z=j; z >= j-k+1; z--) c *= z;
     //if(i==0){ printf("cudaDeviceCVodeGetDky c %le s %le j %d dmdv->cv_q %d\n",
     //        c, s, j), dmdv->cv_q;
      //for(int n=0;n<md->nrows)
      //printf("")
     //}

     if (j == dmdv->cv_q) {
       //N_VScale(c, md->dzn[dmdv->cv_q], dky);
      //dky[i]=c*md->dzn[md->nrows*(dmdv->cv_q)+i];
       dky[i]=c*md->dzn[md->nrows*(j)+i];

     } else {
       //N_VLinearSum(c, md->cv_zn[j], s, dky, dky);
       cudaDevicezaxpby(c,
        &md->dzn[md->nrows*(j)],
        s, dky,
        dky, md->nrows);

     }
   }
  __syncthreads();
   if (k == 0) return(CV_SUCCESS); //always?
  __syncthreads();
   //r = SUNRpowerI(dmdv->cv_h,-k);
   r = pow(double(dmdv->cv_h),double(-k));
   //N_VScale(r, dky, dky);
  __syncthreads();

   dky[i]=dky[i]*r;

   return(CV_SUCCESS);



}

__device__
int cudaDevicecvEwtSetSV(ModelDataGPU *md, ModelDataVariable *dmdv,
                         double *dzn, double *weight) {

  extern __shared__ double flag_shr2[];
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  //N_VAbs(ycur, md->dtempv);
  //N_VAbs(cv_mem->cv_ftemp, cv_mem->cv_ftemp);
  //md->dftemp[i]=fabs(md->dftemp[i]);
  md->dtempv[i]=fabs(dzn[i]);

  //N_VLinearSum(dmdv->cv_reltol, md->dtempv, ONE,
  //             md->cv_Vabstol, md->dtempv);
 cudaDevicezaxpby(dmdv->cv_reltol, md->dtempv, 1.,
        md->cv_Vabstol, md->dtempv, md->nrows);

  double min;
  cudaDevicemin(&min, md->dtempv[i], flag_shr2, md->n_shr_empty);
  __syncthreads();
  if (min <= 0.) return(-1);

  //N_VInv(md->dtempv, weight);
  //zd[i] = ONE/xd[i];
  weight[i]= 1./md->dtempv[i];

  return(0);
}

__device__
int cudaDeviceCVode(ModelDataGPU *md, ModelDataVariable *dmdv) {

  extern __shared__ int flag_shr[];
  int i = blockIdx.x * blockDim.x + threadIdx.x;

#ifdef DEV_printmin
  printmin(md,md->state,"cudaDeviceCVode start state");
#endif

#ifndef DEV_CUDACVODE

  for(;;) {

#endif

#ifdef DEV_ATOMICMAX
  atomicMin(md->flag,md->jJ_solver[i]);
#else
    //Very small diff 100 cells (~0.01 error) sometimes (1/3 exec)
#endif

  //if(tid==0)printf("md->flagCells[blockIdx.x] %d\n",md->flagCells[blockIdx.x]);

#ifdef CAMP_DEBUG_GPU
    dmdv->countercvStep++;
#endif

    flag_shr[0] = 0;
    dmdv->flag = 0;

    __syncthreads();

    dmdv->cv_next_h = dmdv->cv_h;
    dmdv->cv_next_q = dmdv->cv_q;

    int ewtsetOK = 0;
    if (dmdv->cv_nst > 0) {

      //ewtsetOK = cvEwtSetSV(cv_mem, cv_mem->cv_zn[0], cv_mem->cv_ewt);
      ewtsetOK = cudaDevicecvEwtSetSV(md, dmdv, md->dzn, md->dewt);

      if (ewtsetOK != 0) {

        //if (cv_mem->cv_itol == CV_WF)
        //  cvProcessError(cv_mem, CV_ILL_INPUT, "CVODE", "CVode",
        //                 MSGCV_EWT_NOW_FAIL, cv_mem->cv_tn);
        //else
        //  cvProcessError(cv_mem, CV_ILL_INPUT, "CVODE", "CVode",
        //                 MSGCV_EWT_NOW_BAD, cv_mem->cv_tn);
        dmdv->cv_tretlast = dmdv->tret = dmdv->cv_tn;
        //N_VScale(ONE, cv_mem->cv_zn[0], yout);
        md->yout[i] = md->dzn[i];

        if(i==0) printf("ERROR: ewtsetOK istate %d\n",dmdv->istate);
        return CV_ILL_INPUT;
      }
    }

    /* Check for too many steps */
    if ((dmdv->cv_mxstep > 0) && (dmdv->nstloc >= dmdv->cv_mxstep)) {
      //cvProcessError(cv_mem, CV_TOO_MUCH_WORK, "CVODE", "CVode",
      //               MSGCV_MAX_STEPS, cv_mem->cv_tn);

      dmdv->cv_tretlast = dmdv->tret = dmdv->cv_tn;
      //N_VScale(ONE, md->dzn, yout);
      md->yout[i] = md->dzn[i];

      if(i==0) printf("ERROR: cv_mxstep istate %d\n",dmdv->istate);
      return CV_TOO_MUCH_WORK;
    }

    /* Check for too much accuracy requested */
    //double nrm = N_VWrmsNorm(dmdv->cv_zn[0], dmdv->cv_ewt);
    double nrm;
    cudaDeviceVWRMS_Norm(md->dzn,
                         md->dewt, &nrm, md->nrows, md->n_shr_empty);

    dmdv->cv_tolsf = dmdv->cv_uround * nrm;
    if (dmdv->cv_tolsf > 1.) {
      //cvProcessError(cv_mem, CV_TOO_MUCH_ACC, "CVODE", "CVode",
      //               MSGCV_TOO_MUCH_ACC, cv_mem->cv_tn);
      dmdv->cv_tretlast = dmdv->tret = dmdv->cv_tn;
      //N_VScale(1., md->dzn[0], md->yout);
      md->yout[i] = md->dzn[i];

      dmdv->cv_tolsf *= 2.;

      if(i==0) printf("ERROR: cv_tolsf istate %d\n",dmdv->istate);
      __syncthreads();
      return CV_TOO_MUCH_ACC;
    } else {
      dmdv->cv_tolsf = 1.;
    }

#ifdef ODE_WARNING
    // Check for h below roundoff level in tn
    if (dmdv->cv_tn + dmdv->cv_h == dmdv->cv_tn) {
      dmdv->cv_nhnil++;
      //if (dmdv->cv_nhnil <= dmdv->cv_mxhnil)
      //  cvProcessError(dmdv, CV_WARNING, "CVODE", "CVode",
      //                 MSGCV_HNIL, dmdv->cv_tn, dmdv->cv_h);
      //if (dmdv->cv_nhnil == dmdv->cv_mxhnil)
      //  cvProcessError(dmdv, CV_WARNING, "CVODE", "CVode", MSGCV_HNIL_DONE);
      if ((dmdv->cv_nhnil <= dmdv->cv_mxhnil) ||
              (dmdv->cv_nhnil == dmdv->cv_mxhnil))
        if(i==0)printf("WARNING: h below roundoff level in tn");
    }
#endif

    int kflag2 = cudaDevicecvStep(md, dmdv);

    __syncthreads();
    dmdv->kflag2=kflag2;
    __syncthreads();

#ifdef DEV_ATOMICMAX
  atomicMin(md->flag,md->jJ_solver[i]);
#else
    //Very small diff 100 cells (~0.01 error) sometimes (1/3 exec)
#endif

  __syncthreads();

#ifdef DEBUG_cudaDeviceCVode
    if(i==0){
      printf("DEV_cudaDeviceCVode countercvStep %d thread %d\n",dmdv->countercvStep, i);
      printf("dmdv->cv_tn %le dmdv->tout %le dmdv->cv_h %le dmdv->cv_hprime %le\n",
             dmdv->cv_tn,dmdv->tout,dmdv->cv_h,dmdv->cv_hprime);
    }
#endif

    //if(i==0)printf("cudaDeviceCVode dmdv->kflag %d\n",dmdv->kflag);

    if (dmdv->kflag2 != CV_SUCCESS) {

      //istate = cvHandleFailure_gpu2(cv_mem, kflag);

      //cv_mem->cv_tretlast = *tret = cv_mem->cv_tn;
      dmdv->cv_tretlast = dmdv->tret = dmdv->cv_tn;

      //N_VScale(ONE, md->dzn[0], yout);
      md->yout[i] = md->dzn[i];

      if(i==0) printf("ERROR: dmdv->kflag != CV_SUCCESS istate %d\n",dmdv->istate);

      //if(i==0)printf("cudaDeviceCVode2 dmdv->kflag %d\n",dmdv->kflag);

      return dmdv->kflag2;
    }

    dmdv->nstloc++;

    //check if tout reached
    if ((dmdv->cv_tn - dmdv->tout) * dmdv->cv_h >= 0.) {

      dmdv->istate = CV_SUCCESS;
      dmdv->cv_tretlast = dmdv->tret = dmdv->tout;
      //(void) CVodeGetDky(cv_mem, dmdv->tout, 0, md->yout);

      cudaDeviceCVodeGetDky(md, dmdv, dmdv->tout, 0, md->yout);

      //if(i==0) printf("SUCCESS: dmdv->cv_tn - dmdv->tout) istate %d\n",dmdv->istate);

      //istate = CV_SUCCESS;
      return CV_SUCCESS;
    }

    if (dmdv->cv_tstopset) {//needed?
      double troundoff = FUZZ_FACTOR * dmdv->cv_uround * (fabs(dmdv->cv_tn) + fabs(dmdv->cv_h));
      if (fabs(dmdv->cv_tn - dmdv->cv_tstop) <= troundoff) {
        //(void) CVodeGetDky(dmdv, dmdv->cv_tstop, 0, md->yout);
        cudaDeviceCVodeGetDky(md, dmdv, dmdv->cv_tstop, 0, md->yout);
        dmdv->cv_tretlast = dmdv->tret = dmdv->cv_tstop;
        dmdv->cv_tstopset = SUNFALSE;
        dmdv->istate = CV_TSTOP_RETURN;
        if(i==0) printf("ERROR: cv_tstopset istate %d\n",dmdv->istate);
        __syncthreads();
        return CV_TSTOP_RETURN;
      }
      if ((dmdv->cv_tn + dmdv->cv_hprime - dmdv->cv_tstop) * dmdv->cv_h > 0.) {
        dmdv->cv_hprime = (dmdv->cv_tstop - dmdv->cv_tn) * (1.0 - 4.0 * dmdv->cv_uround);
        if(i==0) printf("ERROR: dmdv->cv_tn + dmdv->cv_hprime - dmdv->cv_tstop istate %d\n",dmdv->istate);
        dmdv->cv_eta = dmdv->cv_hprime / dmdv->cv_h;
      }

    }

#ifndef DEV_CUDACVODE
  }
  //return CV_SUCCESS;//needed?, it should always return instead of break and never reach here
#else
  return dmdv->kflag2;
#endif

}

__global__
void cudaGlobalCVode(ModelDataGPU md_object) {

  ModelDataGPU *md = &md_object;
  ModelDataVariable *mdvo = md->mdvo;
  extern __shared__ int flag_shr[];
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int tid = threadIdx.x;
  int active_threads = md->nrows;
  int istate;
  ModelDataVariable dmdv_object = *md_object.mdv;
  ModelDataVariable *dmdv = &dmdv_object;

  __syncthreads();
  if(i<active_threads){

#ifdef CAMP_DEBUG_GPU
    int clock_khz=md->clock_khz;
    clock_t start;
    start = clock();
    __syncthreads();
#endif
    istate=cudaDeviceCVode(md,dmdv);

#ifdef CAMP_DEBUG_GPU
    __syncthreads();
     dmdv->dtcudaDeviceCVode += ((double)(int)(clock() - start))/(clock_khz*1000);
#endif

  }
  __syncthreads();
  dmdv->istate=istate;
  __syncthreads();

  //if(i==0)printf("countercvStep %d\n",dmdv->countercvStep);

  if(tid==0)md->flagCells[blockIdx.x]=dmdv->istate;
  *mdvo = *dmdv;
}

void solveCVODEGPU_thr(int blocks, int threads_block, int n_shr_memory, int n_shr_empty, int offset_cells,
                       SolverData *sd, CVodeMem cv_mem)
{

  itsolver *bicg = &(sd->bicg);
  ModelDataGPU *mGPU = &sd->mGPU;

  int len_cell=mGPU->deriv_length_cell;
  //f_gpu
  mGPU->i_kernel=0;

  /*

  int threads_block = len_cell;
  int blocks = mGPU->n_cells;
  int n_shr = nextPowerOfTwo2(len_cell);
  int n_shr_empty = n_shr-threads_block;

   */

#ifndef DEBUG_SOLVEBCGCUDA
  if(bicg->counterBiConjGrad==0) {
    printf("solveCVODEGPU_thr n_cells %d len_cell %d nrows %d nnz %d max_threads_block %d blocks %d threads_block %d n_shr_empty %d offset_cells %d\n",
           mGPU->n_cells,len_cell,mGPU->nrows,mGPU->nnz,n_shr_memory,blocks,threads_block,n_shr_empty,offset_cells);

    //print_double(bicg->A,nnz,"A");
    //print_int(bicg->jA,nnz,"jA");
    //print_int(bicg->iA,nrows+1,"iA");

  }
#endif

  cudaGlobalCVode <<<blocks,threads_block,n_shr_memory*sizeof(double)>>>
                                             (sd->mGPU);

}

void solveCVODEGPU(SolverData *sd, CVodeMem cv_mem)
{

  itsolver *bicg = &(sd->bicg);
  ModelDataGPU *mGPU = &sd->mGPU;

#ifndef DEBUG_SOLVEBCGCUDA
  if(bicg->counterBiConjGrad==0) {
    printf("solveCVODEGPU\n");
  }
#endif

#ifdef CAMP_DEBUG_GPU
  cudaEventRecord(bicg->startsolveCVODEGPU);
#endif

  int len_cell = mGPU->nrows/mGPU->n_cells;
  int max_threads_block=mGPU->threads;

  int offset_cells=0;

  int threads_block = len_cell;
  //int blocks = mGPU->blocks = mGPU->n_cells;
  int blocks = mGPU->n_cells;
  int n_shr = max_threads_block = nextPowerOfTwoCVODE(len_cell);
  int n_shr_empty = mGPU->n_shr_empty= n_shr-threads_block;

  solveCVODEGPU_thr(blocks, threads_block, max_threads_block, n_shr_empty, offset_cells,
                    sd,cv_mem);

#ifdef DEBUG_SOLVEBCGCUDA
  if(bicg->counterBiConjGrad<2) {
    printf("solveGPUBlock end\n");
  }
#endif

#ifdef CAMP_DEBUG_GPU
    cudaEventRecord(bicg->stopsolveCVODEGPU);
    cudaEventSynchronize(bicg->stopsolveCVODEGPU);
    float mssolveCVODEGPU = 0.0;
    cudaEventElapsedTime(&mssolveCVODEGPU, bicg->startsolveCVODEGPU, bicg->stopsolveCVODEGPU);

    bicg->timesolveCVODEGPU+= mssolveCVODEGPU/1000;

    bicg->countersolveCVODEGPU++;
#endif

#ifdef DEBUG_SOLVEBCGCUDA
  if(bicg->counterBiConjGrad<2) {
    printf("solveCVODEGPU end\n");
  }
#endif

}

static int cvEwtSetSV(CVodeMem cv_mem, N_Vector ycur, N_Vector weight)
{
  N_VAbs(ycur, cv_mem->cv_tempv);
  N_VLinearSum(cv_mem->cv_reltol, cv_mem->cv_tempv, ONE,
               cv_mem->cv_Vabstol, cv_mem->cv_tempv);
  if (N_VMin(cv_mem->cv_tempv) <= ZERO) return(-1);
  N_VInv(cv_mem->cv_tempv, weight);
  return(0);
}


int cvEwtSet(N_Vector ycur, N_Vector weight, void *data)
{
  CVodeMem cv_mem;
  int flag = 0;

  /* data points to cv_mem here */

  cv_mem = (CVodeMem) data;

  flag = cvEwtSetSV(cv_mem, ycur, weight);

  return(flag);
}

int CVode_gpu2(void *cvode_mem, realtype tout, N_Vector yout,
          realtype *tret, int itask, SolverData *sd)
{
  CVodeMem cv_mem;
  long int nstloc;
  int retval, hflag, kflag, istate, ir, ier, irfndp;
  int ewtsetOK;
  realtype troundoff, tout_hin, rh, nrm;
  booleantype inactive_roots;

  itsolver *bicg = &(sd->bicg);

  /*
   * -------------------------------------
   * 1. Check and process inputs
   * -------------------------------------
   */

  /* Check if cvode_mem exists */
  if (cvode_mem == NULL) {
    cvProcessError(NULL, CV_MEM_NULL, "CVODE", "CVode", MSGCV_NO_MEM);
    return(CV_MEM_NULL);
  }
  cv_mem = (CVodeMem) cvode_mem;

  /* Check if cvode_mem was allocated */
  if (cv_mem->cv_MallocDone == SUNFALSE) {
    cvProcessError(cv_mem, CV_NO_MALLOC, "CVODE", "CVode", MSGCV_NO_MALLOC);
    return(CV_NO_MALLOC);
  }

  /* Check for yout != NULL */
  if ((cv_mem->cv_y = yout) == NULL) {
    cvProcessError(cv_mem, CV_ILL_INPUT, "CVODE", "CVode", MSGCV_YOUT_NULL);
    return(CV_ILL_INPUT);
  }

  /* Check for tret != NULL */
  if (tret == NULL) {
    cvProcessError(cv_mem, CV_ILL_INPUT, "CVODE", "CVode", MSGCV_TRET_NULL);
    return(CV_ILL_INPUT);
  }

  /* Check for valid itask */
  if ( (itask != CV_NORMAL) && (itask != CV_ONE_STEP) ) {
    cvProcessError(cv_mem, CV_ILL_INPUT, "CVODE", "CVode", MSGCV_BAD_ITASK);
    return(CV_ILL_INPUT);
  }

  if (itask == CV_NORMAL) cv_mem->cv_toutc = tout;
  cv_mem->cv_taskc = itask;

  /*
   * ----------------------------------------
   * 2. Initializations performed only at
   *    the first step (nst=0):
   *    - initial setup
   *    - initialize Nordsieck history array
   *    - compute initial step size
   *    - check for approach to tstop
   *    - check for approach to a root
   * ----------------------------------------
   */

  // GPU initializations
  //set_data_gpu2(cv_mem, sd);

  if (cv_mem->cv_nst == 0) {

    cv_mem->cv_tretlast = *tret = cv_mem->cv_tn;

    ier = cvInitialSetup_gpu2(cv_mem);
    if (ier!= CV_SUCCESS) return(ier);

    /* Call f at (t0,y0), set zn[1] = y'(t0),
       set initial h (from H0 or cvHin), and scale zn[1] by h.
       Also check for zeros of root function g at and near t0.    */
    //retval = cv_mem->cv_f(cv_mem->cv_tn, cv_mem->cv_zn[0],
    //                      cv_mem->cv_zn[1], cv_mem->cv_user_data);
    retval = f(cv_mem->cv_tn, cv_mem->cv_zn[0], cv_mem->cv_zn[1], cv_mem->cv_user_data);

    N_VScale(ONE, cv_mem->cv_zn[0], yout);

    cv_mem->cv_nfe++;
    if (retval < 0) {
      cvProcessError(cv_mem, CV_RHSFUNC_FAIL, "CVODE", "CVode",
                     MSGCV_RHSFUNC_FAILED, cv_mem->cv_tn);
      return(CV_RHSFUNC_FAIL);
    }
    if (retval > 0) {
      cvProcessError(cv_mem, CV_FIRST_RHSFUNC_ERR, "CVODE", "CVode",
                     MSGCV_RHSFUNC_FIRST);
      return(CV_FIRST_RHSFUNC_ERR);
    }



    /* Test input tstop for legality. */

    if (cv_mem->cv_tstopset) {
      if ( (cv_mem->cv_tstop - cv_mem->cv_tn)*(tout - cv_mem->cv_tn) <= ZERO ) {
        cvProcessError(cv_mem, CV_ILL_INPUT, "CVODE", "CVode",
                       MSGCV_BAD_TSTOP, cv_mem->cv_tstop, cv_mem->cv_tn);
        return(CV_ILL_INPUT);
      }
    }

    /* Set initial h (from H0 or cvHin). */

    cv_mem->cv_h = cv_mem->cv_hin;
    if ( (cv_mem->cv_h != ZERO) && ((tout-cv_mem->cv_tn)*cv_mem->cv_h < ZERO) ) {
      cvProcessError(cv_mem, CV_ILL_INPUT, "CVODE", "CVode", MSGCV_BAD_H0);
      return(CV_ILL_INPUT);
    }
    if (cv_mem->cv_h == ZERO) {
      tout_hin = tout;
      if ( cv_mem->cv_tstopset && (tout-cv_mem->cv_tn)*(tout-cv_mem->cv_tstop) > ZERO )
        tout_hin = cv_mem->cv_tstop;
      hflag = cvHin_gpu2(cv_mem, tout_hin); //set cv_y
      if (hflag != CV_SUCCESS) {
        istate = cvHandleFailure_gpu2(cv_mem, hflag);
        return(istate);
      }
    }
    rh = SUNRabs(cv_mem->cv_h)*cv_mem->cv_hmax_inv;
    if (rh > ONE) cv_mem->cv_h /= rh;
    if (SUNRabs(cv_mem->cv_h) < cv_mem->cv_hmin)
      cv_mem->cv_h *= cv_mem->cv_hmin/SUNRabs(cv_mem->cv_h);

    /* Check for approach to tstop */

    if (cv_mem->cv_tstopset) {
      if ( (cv_mem->cv_tn + cv_mem->cv_h - cv_mem->cv_tstop)*cv_mem->cv_h > ZERO )
        cv_mem->cv_h = (cv_mem->cv_tstop - cv_mem->cv_tn)*(ONE-FOUR*cv_mem->cv_uround);
    }

    /* Scale zn[1] by h.*/

    cv_mem->cv_hscale = cv_mem->cv_h;
    cv_mem->cv_h0u    = cv_mem->cv_h;
    cv_mem->cv_hprime = cv_mem->cv_h;

    N_VScale(cv_mem->cv_h, cv_mem->cv_zn[1], cv_mem->cv_zn[1]);
    /* Try to improve initial guess of zn[1] */
    if (cv_mem->cv_ghfun) {

      N_VLinearSum(ONE, cv_mem->cv_zn[0], ONE, cv_mem->cv_zn[1], cv_mem->cv_tempv1);
      cv_mem->cv_ghfun(cv_mem->cv_tn + cv_mem->cv_h, cv_mem->cv_h, cv_mem->cv_tempv1,
                       cv_mem->cv_zn[0], cv_mem->cv_zn[1], cv_mem->cv_user_data,
                       cv_mem->cv_tempv2, cv_mem->cv_acor_init);
    }
    /* Check for zeros of root function g at and near t0. */

    if (cv_mem->cv_nrtfn > 0) {

      retval = cvRcheck1_gpu2(cv_mem);

      if (retval == CV_RTFUNC_FAIL) {
        cvProcessError(cv_mem, CV_RTFUNC_FAIL, "CVODE", "cvRcheck1",
                       MSGCV_RTFUNC_FAILED, cv_mem->cv_tn);
        return(CV_RTFUNC_FAIL);
      }

    }

  } /* end of first call block */

  /*
   * ------------------------------------------------------
   * 3. At following steps, perform stop tests:
   *    - check for root in last step
   *    - check if we passed tstop
   *    - check if we passed tout (NORMAL mode)
   *    - check if current tn was returned (ONE_STEP mode)
   *    - check if we are close to tstop
   *      (adjust step size if needed)
   * -------------------------------------------------------
   */

  if (cv_mem->cv_nst > 0) {

    /* Estimate an infinitesimal time interval to be used as
       a roundoff for time quantities (based on current time
       and step size) */
    troundoff = FUZZ_FACTOR*cv_mem->cv_uround*(SUNRabs(cv_mem->cv_tn) + SUNRabs(cv_mem->cv_h));

    /* First, check for a root in the last step taken, other than the
       last root found, if any.  If itask = CV_ONE_STEP and y(tn) was not
       returned because of an intervening root, return y(tn) now.     */
    if (cv_mem->cv_nrtfn > 0) {

      irfndp = cv_mem->cv_irfnd;

      retval = cvRcheck2_gpu2(cv_mem);

      if (retval == CLOSERT) {
        cvProcessError(cv_mem, CV_ILL_INPUT, "CVODE", "cvRcheck2",
                       MSGCV_CLOSE_ROOTS, cv_mem->cv_tlo);
        return(CV_ILL_INPUT);
      } else if (retval == CV_RTFUNC_FAIL) {
        cvProcessError(cv_mem, CV_RTFUNC_FAIL, "CVODE", "cvRcheck2",
                       MSGCV_RTFUNC_FAILED, cv_mem->cv_tlo);
        return(CV_RTFUNC_FAIL);
      } else if (retval == RTFOUND) {
        cv_mem->cv_tretlast = *tret = cv_mem->cv_tlo;
        return(CV_ROOT_RETURN);
      }

      /* If tn is distinct from tretlast (within roundoff),
         check remaining interval for roots */
      if ( SUNRabs(cv_mem->cv_tn - cv_mem->cv_tretlast) > troundoff ) {

        retval = cvRcheck3_gpu2(cv_mem);

        if (retval == CV_SUCCESS) {     /* no root found */
          cv_mem->cv_irfnd = 0;
          if ((irfndp == 1) && (itask == CV_ONE_STEP)) {
            cv_mem->cv_tretlast = *tret = cv_mem->cv_tn;
            N_VScale(ONE, cv_mem->cv_zn[0], yout);
            return(CV_SUCCESS);
          }
        } else if (retval == RTFOUND) {  /* a new root was found */
          cv_mem->cv_irfnd = 1;
          cv_mem->cv_tretlast = *tret = cv_mem->cv_tlo;
          return(CV_ROOT_RETURN);
        } else if (retval == CV_RTFUNC_FAIL) {  /* g failed */
          cvProcessError(cv_mem, CV_RTFUNC_FAIL, "CVODE", "cvRcheck3",
                         MSGCV_RTFUNC_FAILED, cv_mem->cv_tlo);
          return(CV_RTFUNC_FAIL);
        }

      }

    } /* end of root stop check */

    /* In CV_NORMAL mode, test if tout was reached */
    if ( (itask == CV_NORMAL) && ((cv_mem->cv_tn-tout)*cv_mem->cv_h >= ZERO) ) {
      cv_mem->cv_tretlast = *tret = tout;
      ier =  CVodeGetDky(cv_mem, tout, 0, yout);
      if (ier != CV_SUCCESS) {
        cvProcessError(cv_mem, CV_ILL_INPUT, "CVODE", "CVode",
                       MSGCV_BAD_TOUT, tout);
        return(CV_ILL_INPUT);
      }
      return(CV_SUCCESS);
    }

    /* In CV_ONE_STEP mode, test if tn was returned */
    if ( itask == CV_ONE_STEP &&
         SUNRabs(cv_mem->cv_tn - cv_mem->cv_tretlast) > troundoff ) {
      cv_mem->cv_tretlast = *tret = cv_mem->cv_tn;
      N_VScale(ONE, cv_mem->cv_zn[0], yout);
      return(CV_SUCCESS);
    }

    /* Test for tn at tstop or near tstop */
    if ( cv_mem->cv_tstopset ) {

      if ( SUNRabs(cv_mem->cv_tn - cv_mem->cv_tstop) <= troundoff) {
        ier =  CVodeGetDky(cv_mem, cv_mem->cv_tstop, 0, yout);
        if (ier != CV_SUCCESS) {
          cvProcessError(cv_mem, CV_ILL_INPUT, "CVODE", "CVode",
                         MSGCV_BAD_TSTOP, cv_mem->cv_tstop, cv_mem->cv_tn);
          return(CV_ILL_INPUT);
        }
        cv_mem->cv_tretlast = *tret = cv_mem->cv_tstop;
        cv_mem->cv_tstopset = SUNFALSE;
        return(CV_TSTOP_RETURN);
      }

      /* If next step would overtake tstop, adjust stepsize */
      if ( (cv_mem->cv_tn + cv_mem->cv_hprime - cv_mem->cv_tstop)*cv_mem->cv_h > ZERO ) {
        cv_mem->cv_hprime = (cv_mem->cv_tstop - cv_mem->cv_tn)*(ONE-FOUR*cv_mem->cv_uround);
        cv_mem->cv_eta = cv_mem->cv_hprime/cv_mem->cv_h;
      }

    }

  } /* end stopping tests block */

  /*
   * --------------------------------------------------
   * 4. Looping point for internal steps
   *
   *    4.1. check for errors (too many steps, too much
   *         accuracy requested, step size too small)
   *    4.2. take a new step (call cvStep)
   *    4.3. stop on error
   *    4.4. perform stop tests:
   *         - check for root in last step
   *         - check if tout was passed
   *         - check if close to tstop
   *         - check if in ONE_STEP mode (must return)
   * --------------------------------------------------
   */

#ifdef CAMP_DEBUG_GPU
  cudaEventRecord(bicg->startcvStep);
#endif

  nstloc = 0;
  for(;;) {

#ifdef CAMP_DEBUG_GPU
    sd->mdv.countercvStep++;
#endif

    cv_mem->cv_next_h = cv_mem->cv_h;
    cv_mem->cv_next_q = cv_mem->cv_q;

    /* Reset and check ewt */
    if (cv_mem->cv_nst > 0) {

      ewtsetOK = cv_mem->cv_efun(cv_mem->cv_zn[0], cv_mem->cv_ewt, cv_mem->cv_e_data);
      //set here copy of ewt to gpu

      if (ewtsetOK != 0) {

        if (cv_mem->cv_itol == CV_WF)
          cvProcessError(cv_mem, CV_ILL_INPUT, "CVODE", "CVode",
                         MSGCV_EWT_NOW_FAIL, cv_mem->cv_tn);
        else
          cvProcessError(cv_mem, CV_ILL_INPUT, "CVODE", "CVode",
                         MSGCV_EWT_NOW_BAD, cv_mem->cv_tn);

        istate = CV_ILL_INPUT;
        cv_mem->cv_tretlast = *tret = cv_mem->cv_tn;
        N_VScale(ONE, cv_mem->cv_zn[0], yout);
        break;

      }
    }

    /* Check for too many steps */
    if ( (cv_mem->cv_mxstep>0) && (nstloc >= cv_mem->cv_mxstep) ) {
      cvProcessError(cv_mem, CV_TOO_MUCH_WORK, "CVODE", "CVode",
                     MSGCV_MAX_STEPS, cv_mem->cv_tn);
      istate = CV_TOO_MUCH_WORK;
      cv_mem->cv_tretlast = *tret = cv_mem->cv_tn;
      N_VScale(ONE, cv_mem->cv_zn[0], yout);
      break;
    }

    /* Check for too much accuracy requested */
    nrm = N_VWrmsNorm(cv_mem->cv_zn[0], cv_mem->cv_ewt);
    cv_mem->cv_tolsf = cv_mem->cv_uround * nrm;
    if (cv_mem->cv_tolsf > ONE) {
      cvProcessError(cv_mem, CV_TOO_MUCH_ACC, "CVODE", "CVode",
                     MSGCV_TOO_MUCH_ACC, cv_mem->cv_tn);
      istate = CV_TOO_MUCH_ACC;
      cv_mem->cv_tretlast = *tret = cv_mem->cv_tn;
      N_VScale(ONE, cv_mem->cv_zn[0], yout);
      cv_mem->cv_tolsf *= TWO;
      break;
    } else {
      cv_mem->cv_tolsf = ONE;
    }

    /* Check for h below roundoff level in tn */
    if (cv_mem->cv_tn + cv_mem->cv_h == cv_mem->cv_tn) {
      cv_mem->cv_nhnil++;
      if (cv_mem->cv_nhnil <= cv_mem->cv_mxhnil)
        cvProcessError(cv_mem, CV_WARNING, "CVODE", "CVode",
                       MSGCV_HNIL, cv_mem->cv_tn, cv_mem->cv_h);
      if (cv_mem->cv_nhnil == cv_mem->cv_mxhnil)
        cvProcessError(cv_mem, CV_WARNING, "CVODE", "CVode", MSGCV_HNIL_DONE);
    }

    /* Call cvStep to take a step */
    //kflag = cvStep(cv_mem);
    kflag = cvStep_gpu2(sd, cv_mem);
    //kflag = cudacvStep(sd, cv_mem);

    /* Process failed step cases, and exit loop */
    if (kflag != CV_SUCCESS) {
      istate = cvHandleFailure_gpu2(cv_mem, kflag);
      cv_mem->cv_tretlast = *tret = cv_mem->cv_tn;
      N_VScale(ONE, cv_mem->cv_zn[0], yout);
      break;
    }

    nstloc++;

    /* Check for root in last step taken. */
    if (cv_mem->cv_nrtfn > 0) {

      retval = cvRcheck3_gpu2(cv_mem);

      if (retval == RTFOUND) {  /* A new root was found */
        cv_mem->cv_irfnd = 1;
        istate = CV_ROOT_RETURN;
        cv_mem->cv_tretlast = *tret = cv_mem->cv_tlo;
        break;
      } else if (retval == CV_RTFUNC_FAIL) { /* g failed */
        cvProcessError(cv_mem, CV_RTFUNC_FAIL, "CVODE", "cvRcheck3",
                       MSGCV_RTFUNC_FAILED, cv_mem->cv_tlo);
        istate = CV_RTFUNC_FAIL;
        break;
      }

      /* If we are at the end of the first step and we still have
       * some event functions that are inactive, issue a warning
       * as this may indicate a user error in the implementation
       * of the root function. */

      if (cv_mem->cv_nst==1) {
        inactive_roots = SUNFALSE;
        for (ir=0; ir<cv_mem->cv_nrtfn; ir++) {
          if (!cv_mem->cv_gactive[ir]) {
            inactive_roots = SUNTRUE;
            break;
          }
        }
        if ((cv_mem->cv_mxgnull > 0) && inactive_roots) {
          cvProcessError(cv_mem, CV_WARNING, "CVODES", "CVode",
                         MSGCV_INACTIVE_ROOTS);
        }
      }

    }

    /* In NORMAL mode, check if tout reached */
    if ( (itask == CV_NORMAL) &&  (cv_mem->cv_tn-tout)*cv_mem->cv_h >= ZERO ) {
      istate = CV_SUCCESS;
      cv_mem->cv_tretlast = *tret = tout;
      (void) CVodeGetDky(cv_mem, tout, 0, yout);
      cv_mem->cv_next_q = cv_mem->cv_qprime;
      cv_mem->cv_next_h = cv_mem->cv_hprime;
      break;
    }

    /* Check if tn is at tstop or near tstop */
    if ( cv_mem->cv_tstopset ) {

      troundoff = FUZZ_FACTOR*cv_mem->cv_uround*(SUNRabs(cv_mem->cv_tn) + SUNRabs(cv_mem->cv_h));
      if ( SUNRabs(cv_mem->cv_tn - cv_mem->cv_tstop) <= troundoff) {
        (void) CVodeGetDky(cv_mem, cv_mem->cv_tstop, 0, yout);
        cv_mem->cv_tretlast = *tret = cv_mem->cv_tstop;
        cv_mem->cv_tstopset = SUNFALSE;
        istate = CV_TSTOP_RETURN;
        break;
      }

      if ( (cv_mem->cv_tn + cv_mem->cv_hprime - cv_mem->cv_tstop)*cv_mem->cv_h > ZERO ) {
        cv_mem->cv_hprime = (cv_mem->cv_tstop - cv_mem->cv_tn)*(ONE-FOUR*cv_mem->cv_uround);
        cv_mem->cv_eta = cv_mem->cv_hprime/cv_mem->cv_h;
      }

    }

    /* In ONE_STEP mode, copy y and exit loop */
    if (itask == CV_ONE_STEP) {
      istate = CV_SUCCESS;
      cv_mem->cv_tretlast = *tret = cv_mem->cv_tn;
      N_VScale(ONE, cv_mem->cv_zn[0], yout);
      cv_mem->cv_next_q = cv_mem->cv_qprime;
      cv_mem->cv_next_h = cv_mem->cv_hprime;
      break;
    }

  } /* end looping for internal steps */

#ifdef CAMP_DEBUG_GPU

  ModelDataGPU *mGPU = &sd->mGPU;
  cudaMemcpy(&sd->mdv, mGPU->mdvo, sizeof(ModelDataVariable), cudaMemcpyDeviceToHost);

  cudaEventRecord(bicg->stopcvStep);
  cudaEventSynchronize(bicg->stopcvStep);
  float mscvStep = 0.0;
  cudaEventElapsedTime(&mscvStep, bicg->startcvStep, bicg->stopcvStep);
  bicg->timecvStep+= mscvStep/1000;
#endif

  return(istate);
}

int cudaCVode(void *cvode_mem, realtype tout, N_Vector yout,
               realtype *tret, int itask, SolverData *sd)
{
  CVodeMem cv_mem;
  long int nstloc;
  int retval, hflag, kflag, istate, ier, irfndp;
  realtype troundoff, tout_hin, rh;

  itsolver *bicg = &(sd->bicg);
  ModelDataGPU *mGPU = &sd->mGPU;
  ModelData *md = &(sd->model_data);
  //double *youtArray = N_VGetArrayPointer(yout);

  /*
   * -------------------------------------
   * 1. Check and process inputs
   * -------------------------------------
   */

  /* Check if cvode_mem exists */
  if (cvode_mem == NULL) {
    cvProcessError(NULL, CV_MEM_NULL, "CVODE", "CVode", MSGCV_NO_MEM);
    return(CV_MEM_NULL);
  }
  cv_mem = (CVodeMem) cvode_mem;

  /* Check if cvode_mem was allocated */
  if (cv_mem->cv_MallocDone == SUNFALSE) {
    cvProcessError(cv_mem, CV_NO_MALLOC, "CVODE", "CVode", MSGCV_NO_MALLOC);
    return(CV_NO_MALLOC);
  }

  /* Check for yout != NULL */
  if ((cv_mem->cv_y = yout) == NULL) {
    cvProcessError(cv_mem, CV_ILL_INPUT, "CVODE", "CVode", MSGCV_YOUT_NULL);
    return(CV_ILL_INPUT);
  }

  /* Check for tret != NULL */
  if (tret == NULL) {
    cvProcessError(cv_mem, CV_ILL_INPUT, "CVODE", "CVode", MSGCV_TRET_NULL);
    return(CV_ILL_INPUT);
  }

  /* Check for valid itask */
  if ( (itask != CV_NORMAL) && (itask != CV_ONE_STEP) ) {
    cvProcessError(cv_mem, CV_ILL_INPUT, "CVODE", "CVode", MSGCV_BAD_ITASK);
    return(CV_ILL_INPUT);
  }

  if (itask == CV_NORMAL) cv_mem->cv_toutc = tout;
  cv_mem->cv_taskc = itask;

  /*
   * ----------------------------------------
   * 2. Initializations performed only at
   *    the first step (nst=0):
   *    - initial setup
   *    - initialize Nordsieck history array
   *    - compute initial step size
   *    - check for approach to tstop
   *    - check for approach to a root
   * ----------------------------------------
   */

  // GPU initializations
  //set_data_gpu2(cv_mem, sd);

  if (cv_mem->cv_nst == 0) {

    cv_mem->cv_tretlast = *tret = cv_mem->cv_tn;

    ier = cvInitialSetup_gpu2(cv_mem);
    if (ier!= CV_SUCCESS) return(ier);

    /* Call f at (t0,y0), set zn[1] = y'(t0),
       set initial h (from H0 or cvHin), and scale zn[1] by h.
       Also check for zeros of root function g at and near t0.    */
    //retval = cv_mem->cv_f(cv_mem->cv_tn, cv_mem->cv_zn[0],
    //                      cv_mem->cv_zn[1], cv_mem->cv_user_data);
    retval = f(cv_mem->cv_tn, cv_mem->cv_zn[0], cv_mem->cv_zn[1], cv_mem->cv_user_data);

    N_VScale(ONE, cv_mem->cv_zn[0], yout);

    cv_mem->cv_nfe++;
    if (retval < 0) {
      cvProcessError(cv_mem, CV_RHSFUNC_FAIL, "CVODE", "CVode",
                     MSGCV_RHSFUNC_FAILED, cv_mem->cv_tn);
      return(CV_RHSFUNC_FAIL);
    }
    if (retval > 0) {
      cvProcessError(cv_mem, CV_FIRST_RHSFUNC_ERR, "CVODE", "CVode",
                     MSGCV_RHSFUNC_FIRST);
      return(CV_FIRST_RHSFUNC_ERR);
    }



    /* Test input tstop for legality. */

    if (cv_mem->cv_tstopset) {
      if ( (cv_mem->cv_tstop - cv_mem->cv_tn)*(tout - cv_mem->cv_tn) <= ZERO ) {
        cvProcessError(cv_mem, CV_ILL_INPUT, "CVODE", "CVode",
                       MSGCV_BAD_TSTOP, cv_mem->cv_tstop, cv_mem->cv_tn);
        return(CV_ILL_INPUT);
      }
    }

    /* Set initial h (from H0 or cvHin). */

    cv_mem->cv_h = cv_mem->cv_hin;
    if ( (cv_mem->cv_h != ZERO) && ((tout-cv_mem->cv_tn)*cv_mem->cv_h < ZERO) ) {
      cvProcessError(cv_mem, CV_ILL_INPUT, "CVODE", "CVode", MSGCV_BAD_H0);
      return(CV_ILL_INPUT);
    }
    if (cv_mem->cv_h == ZERO) {
      tout_hin = tout;
      if ( cv_mem->cv_tstopset && (tout-cv_mem->cv_tn)*(tout-cv_mem->cv_tstop) > ZERO )
        tout_hin = cv_mem->cv_tstop;
      hflag = cvHin_gpu2(cv_mem, tout_hin); //set cv_y
      if (hflag != CV_SUCCESS) {
        istate = cvHandleFailure_gpu2(cv_mem, hflag);
        return(istate);
      }
    }
    rh = SUNRabs(cv_mem->cv_h)*cv_mem->cv_hmax_inv;
    if (rh > ONE) cv_mem->cv_h /= rh;
    if (SUNRabs(cv_mem->cv_h) < cv_mem->cv_hmin)
      cv_mem->cv_h *= cv_mem->cv_hmin/SUNRabs(cv_mem->cv_h);

    /* Check for approach to tstop */

    if (cv_mem->cv_tstopset) {
      if ( (cv_mem->cv_tn + cv_mem->cv_h - cv_mem->cv_tstop)*cv_mem->cv_h > ZERO )
        cv_mem->cv_h = (cv_mem->cv_tstop - cv_mem->cv_tn)*(ONE-FOUR*cv_mem->cv_uround);
    }

    /* Scale zn[1] by h.*/

    cv_mem->cv_hscale = cv_mem->cv_h;
    cv_mem->cv_h0u    = cv_mem->cv_h;
    cv_mem->cv_hprime = cv_mem->cv_h;

    N_VScale(cv_mem->cv_h, cv_mem->cv_zn[1], cv_mem->cv_zn[1]);
    /* Try to improve initial guess of zn[1] */
    if (cv_mem->cv_ghfun) {

      N_VLinearSum(ONE, cv_mem->cv_zn[0], ONE, cv_mem->cv_zn[1], cv_mem->cv_tempv1);
      cv_mem->cv_ghfun(cv_mem->cv_tn + cv_mem->cv_h, cv_mem->cv_h, cv_mem->cv_tempv1,
                       cv_mem->cv_zn[0], cv_mem->cv_zn[1], cv_mem->cv_user_data,
                       cv_mem->cv_tempv2, cv_mem->cv_acor_init);
    }
    /* Check for zeros of root function g at and near t0. */

    if (cv_mem->cv_nrtfn > 0) {

      retval = cvRcheck1_gpu2(cv_mem);

      if (retval == CV_RTFUNC_FAIL) {
        cvProcessError(cv_mem, CV_RTFUNC_FAIL, "CVODE", "cvRcheck1",
                       MSGCV_RTFUNC_FAILED, cv_mem->cv_tn);
        return(CV_RTFUNC_FAIL);
      }

    }

  } /* end of first call block */

  /*
   * ------------------------------------------------------
   * 3. At following steps, perform stop tests:
   *    - check for root in last step
   *    - check if we passed tstop
   *    - check if we passed tout (NORMAL mode)
   *    - check if current tn was returned (ONE_STEP mode)
   *    - check if we are close to tstop
   *      (adjust step size if needed)
   * -------------------------------------------------------
   */

  if (cv_mem->cv_nst > 0) {

    /* Estimate an infinitesimal time interval to be used as
       a roundoff for time quantities (based on current time
       and step size) */
    troundoff = FUZZ_FACTOR*cv_mem->cv_uround*(SUNRabs(cv_mem->cv_tn) + SUNRabs(cv_mem->cv_h));

    /* First, check for a root in the last step taken, other than the
       last root found, if any.  If itask = CV_ONE_STEP and y(tn) was not
       returned because of an intervening root, return y(tn) now.     */
    if (cv_mem->cv_nrtfn > 0) {

      irfndp = cv_mem->cv_irfnd;

      retval = cvRcheck2_gpu2(cv_mem);

      if (retval == CLOSERT) {
        cvProcessError(cv_mem, CV_ILL_INPUT, "CVODE", "cvRcheck2",
                       MSGCV_CLOSE_ROOTS, cv_mem->cv_tlo);
        return(CV_ILL_INPUT);
      } else if (retval == CV_RTFUNC_FAIL) {
        cvProcessError(cv_mem, CV_RTFUNC_FAIL, "CVODE", "cvRcheck2",
                       MSGCV_RTFUNC_FAILED, cv_mem->cv_tlo);
        return(CV_RTFUNC_FAIL);
      } else if (retval == RTFOUND) {
        cv_mem->cv_tretlast = *tret = cv_mem->cv_tlo;
        return(CV_ROOT_RETURN);
      }

      /* If tn is distinct from tretlast (within roundoff),
         check remaining interval for roots */
      if ( SUNRabs(cv_mem->cv_tn - cv_mem->cv_tretlast) > troundoff ) {

        retval = cvRcheck3_gpu2(cv_mem);

        if (retval == CV_SUCCESS) {     /* no root found */
          cv_mem->cv_irfnd = 0;
          if ((irfndp == 1) && (itask == CV_ONE_STEP)) {
            cv_mem->cv_tretlast = *tret = cv_mem->cv_tn;
            N_VScale(ONE, cv_mem->cv_zn[0], yout);
            return(CV_SUCCESS);
          }
        } else if (retval == RTFOUND) {  /* a new root was found */
          cv_mem->cv_irfnd = 1;
          cv_mem->cv_tretlast = *tret = cv_mem->cv_tlo;
          return(CV_ROOT_RETURN);
        } else if (retval == CV_RTFUNC_FAIL) {  /* g failed */
          cvProcessError(cv_mem, CV_RTFUNC_FAIL, "CVODE", "cvRcheck3",
                         MSGCV_RTFUNC_FAILED, cv_mem->cv_tlo);
          return(CV_RTFUNC_FAIL);
        }

      }

    } /* end of root stop check */

    /* In CV_NORMAL mode, test if tout was reached */
    if ( (itask == CV_NORMAL) && ((cv_mem->cv_tn-tout)*cv_mem->cv_h >= ZERO) ) {
      cv_mem->cv_tretlast = *tret = tout;
      ier =  CVodeGetDky(cv_mem, tout, 0, yout);
      if (ier != CV_SUCCESS) {
        cvProcessError(cv_mem, CV_ILL_INPUT, "CVODE", "CVode",
                       MSGCV_BAD_TOUT, tout);
        return(CV_ILL_INPUT);
      }
      return(CV_SUCCESS);
    }

    /* In CV_ONE_STEP mode, test if tn was returned */
    if ( itask == CV_ONE_STEP &&
         SUNRabs(cv_mem->cv_tn - cv_mem->cv_tretlast) > troundoff ) {
      cv_mem->cv_tretlast = *tret = cv_mem->cv_tn;
      N_VScale(ONE, cv_mem->cv_zn[0], yout);
      return(CV_SUCCESS);
    }

    /* Test for tn at tstop or near tstop */
    if ( cv_mem->cv_tstopset ) {

      if ( SUNRabs(cv_mem->cv_tn - cv_mem->cv_tstop) <= troundoff) {
        ier =  CVodeGetDky(cv_mem, cv_mem->cv_tstop, 0, yout);
        if (ier != CV_SUCCESS) {
          cvProcessError(cv_mem, CV_ILL_INPUT, "CVODE", "CVode",
                         MSGCV_BAD_TSTOP, cv_mem->cv_tstop, cv_mem->cv_tn);
          return(CV_ILL_INPUT);
        }
        cv_mem->cv_tretlast = *tret = cv_mem->cv_tstop;
        cv_mem->cv_tstopset = SUNFALSE;
        return(CV_TSTOP_RETURN);
      }

      /* If next step would overtake tstop, adjust stepsize */
      if ( (cv_mem->cv_tn + cv_mem->cv_hprime - cv_mem->cv_tstop)*cv_mem->cv_h > ZERO ) {
        cv_mem->cv_hprime = (cv_mem->cv_tstop - cv_mem->cv_tn)*(ONE-FOUR*cv_mem->cv_uround);
        cv_mem->cv_eta = cv_mem->cv_hprime/cv_mem->cv_h;
      }

    }

  } /* end stopping tests block */

  /*
   * --------------------------------------------------
   * 4. Looping point for internal steps
   *
   *    4.1. check for errors (too many steps, too much
   *         accuracy requested, step size too small)
   *    4.2. take a new step (call cvStep)
   *    4.3. stop on error
   *    4.4. perform stop tests:
   *         - check for root in last step
   *         - check if tout was passed
   *         - check if close to tstop
   *         - check if in ONE_STEP mode (must return)
   * --------------------------------------------------
   */

  if (cv_mem->cv_y == NULL) {
    cvProcessError(cv_mem, CV_BAD_DKY, "CVODE", "CVodeGetDky", MSGCV_NULL_DKY);
    return(CV_BAD_DKY);
  }

#ifdef CAMP_DEBUG_GPU
  cudaEventRecord(bicg->startcvStep);
#endif

  istate = 99;
  kflag = 99;
  nstloc = 0;
  int flag;

  double *ewt = NV_DATA_S(cv_mem->cv_ewt);
  double *acor = NV_DATA_S(cv_mem->cv_acor);
  double *tempv = NV_DATA_S(cv_mem->cv_tempv);
  double *ftemp = NV_DATA_S(cv_mem->cv_ftemp);
  double *cv_last_yn = N_VGetArrayPointer(cv_mem->cv_last_yn);
  double *cv_acor_init = N_VGetArrayPointer(cv_mem->cv_acor_init);
  double *youtArray = N_VGetArrayPointer(yout);
  double *cv_Vabstol = N_VGetArrayPointer(cv_mem->cv_Vabstol);

  cudaMemcpy(mGPU->dewt,ewt,mGPU->nrows*sizeof(double),cudaMemcpyHostToDevice);
  cudaMemcpy(mGPU->cv_acor, acor, mGPU->nrows * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(mGPU->dtempv, tempv, mGPU->nrows * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(mGPU->dftemp, ftemp, mGPU->nrows * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(mGPU->cv_last_yn, cv_last_yn, mGPU->nrows * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(mGPU->cv_acor_init, cv_acor_init, mGPU->nrows * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(mGPU->yout, youtArray, mGPU->nrows * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(mGPU->cv_Vabstol,cv_Vabstol,mGPU->nrows*sizeof(double),cudaMemcpyHostToDevice);

  for (int i = 0; i < mGPU->n_cells; i++) {
    cudaMemcpy(mGPU->cv_l+i*L_MAX, cv_mem->cv_l, L_MAX * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(mGPU->cv_tau+i*(L_MAX + 1), cv_mem->cv_tau, (L_MAX + 1) * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(mGPU->cv_tq+i*(NUM_TESTS + 1), cv_mem->cv_tq, (NUM_TESTS + 1) * sizeof(double), cudaMemcpyHostToDevice);
  }

  for (int i = 0; i <= cv_mem->cv_qmax; i++) {//cv_qmax+1 (6)?
    double *zn = NV_DATA_S(cv_mem->cv_zn[i]);
    cudaMemcpy((i * mGPU->nrows + mGPU->dzn), zn, mGPU->nrows * sizeof(double), cudaMemcpyHostToDevice);

  }

  for(int i=0;i<mGPU->n_cells;i++)
    sd->flagCells[i]=99;

  cudaMemcpy(mGPU->flagCells, sd->flagCells, mGPU->n_cells*sizeof(int), cudaMemcpyHostToDevice);
  HANDLE_ERROR(cudaMemcpy(mGPU->state, md->total_state, md->state_size, cudaMemcpyHostToDevice));

#ifndef DEV_CUDACVODE

#else
  for(;;) {

#endif

  sd->mdv.init_time_step = sd->init_time_step;
  sd->mdv.cv_mxstep = cv_mem->cv_mxstep;
  sd->mdv.cv_taskc = cv_mem->cv_taskc;
  sd->mdv.cv_uround = cv_mem->cv_uround;
  sd->mdv.cv_nrtfn = cv_mem->cv_nrtfn;
  sd->mdv.cv_tretlast = cv_mem->cv_tretlast;
  sd->mdv.cv_hmax_inv = cv_mem->cv_hmax_inv;
  sd->mdv.cv_lmm = cv_mem->cv_lmm;
  sd->mdv.cv_iter = cv_mem->cv_iter;
  sd->mdv.cv_itol = cv_mem->cv_itol;
  sd->mdv.cv_reltol = cv_mem->cv_reltol;
  sd->mdv.cv_nhnil = cv_mem->cv_nhnil;
  sd->mdv.cv_etaqm1 = cv_mem->cv_etaqm1;
  sd->mdv.cv_etaq = cv_mem->cv_etaq;
  sd->mdv.cv_etaqp1 = cv_mem->cv_etaqp1;
  sd->mdv.cv_lrw1 = cv_mem->cv_lrw1;
  sd->mdv.cv_liw1 = cv_mem->cv_liw1;
  sd->mdv.cv_lrw = (int) cv_mem->cv_lrw;
  sd->mdv.cv_liw = (int) cv_mem->cv_liw;
  sd->mdv.cv_saved_tq5 = cv_mem->cv_saved_tq5;
  sd->mdv.cv_tolsf = cv_mem->cv_tolsf;
  sd->mdv.cv_qmax_alloc = cv_mem->cv_qmax_alloc;
  sd->mdv.cv_indx_acor = cv_mem->cv_indx_acor;
  sd->mdv.cv_qu = cv_mem->cv_qu;
  sd->mdv.cv_h0u = cv_mem->cv_h0u;
  sd->mdv.cv_hu = cv_mem->cv_hu;
  sd->mdv.cv_jcur = cv_mem->cv_jcur;
  sd->mdv.cv_mnewt = cv_mem->cv_mnewt;
  sd->mdv.cv_maxcor = cv_mem->cv_maxcor;
  sd->mdv.cv_nstlp = (int) cv_mem->cv_nstlp;
  sd->mdv.cv_qmax = cv_mem->cv_qmax;
  sd->mdv.cv_L = cv_mem->cv_L;
  sd->mdv.cv_maxnef = cv_mem->cv_maxnef;
  sd->mdv.cv_netf = (int) cv_mem->cv_netf;
  sd->mdv.cv_acnrm = cv_mem->cv_acnrm;
  sd->mdv.cv_tstop = cv_mem->cv_tstop;
  sd->mdv.cv_tstopset = cv_mem->cv_tstopset;
  sd->mdv.cv_nlscoef = cv_mem->cv_nlscoef;
  sd->mdv.cv_qwait = cv_mem->cv_qwait;
  sd->mdv.cv_crate = cv_mem->cv_crate;
  sd->mdv.cv_gamrat = cv_mem->cv_gamrat;
  sd->mdv.cv_gammap = cv_mem->cv_gammap;
  sd->mdv.cv_nst = cv_mem->cv_nst;
  sd->mdv.cv_gamma = cv_mem->cv_gamma;
  sd->mdv.cv_rl1 = cv_mem->cv_rl1;
  sd->mdv.cv_eta = cv_mem->cv_eta;
  sd->mdv.cv_q = cv_mem->cv_q;
  sd->mdv.cv_qprime = cv_mem->cv_qprime;
  sd->mdv.cv_h = cv_mem->cv_h;
  sd->mdv.cv_next_h = cv_mem->cv_next_h;//needed?
  sd->mdv.cv_hscale = cv_mem->cv_hscale;
  sd->mdv.cv_nscon = cv_mem->cv_nscon;
  sd->mdv.cv_hprime = cv_mem->cv_hprime;
  sd->mdv.cv_hmin = cv_mem->cv_hmin;
  sd->mdv.cv_tn = cv_mem->cv_tn;
  sd->mdv.cv_etamax = cv_mem->cv_etamax;
  sd->mdv.cv_maxncf = cv_mem->cv_maxncf;

  sd->mdv.nstloc = (int)nstloc;
  sd->mdv.tout = tout;
  sd->mdv.tret = *tret;
  sd->mdv.istate = istate;
  sd->mdv.kflag = kflag;
  sd->mdv.kflag2 = 99;

  cudaMemcpy(mGPU->mdv, &sd->mdv, sizeof(ModelDataVariable), cudaMemcpyHostToDevice);

#ifdef CAMP_DEBUG_GPU
    //cudaEventRecord(bicg->startBCG);
#endif

  solveCVODEGPU(sd, cv_mem);
  cudaDeviceSynchronize();

#ifdef CAMP_DEBUG_GPU
  //cudaEventRecord(bicg->stopBCG);
  //cudaEventSynchronize(bicg->stopBCG);
  //float msBiConjGrad = 0.0;
  //cudaEventElapsedTime(&msBiConjGrad, bicg->startBCG, bicg->stopBCG);
  //bicg->timeBiConjGrad+= msBiConjGrad;
#endif

  cudaMemcpy(&sd->mdv, mGPU->mdvo, sizeof(ModelDataVariable), cudaMemcpyDeviceToHost);

  cv_mem->cv_next_q = sd->mdv.cv_next_q;
  tout = sd->mdv.tout; //Not output?
  cv_mem->cv_taskc = sd->mdv.cv_taskc;
  cv_mem->cv_uround = sd->mdv.cv_uround;
  cv_mem->cv_nrtfn = sd->mdv.cv_nrtfn;
  nstloc = sd->mdv.nstloc;
  cv_mem->cv_tretlast = sd->mdv.cv_tretlast;
  cv_mem->cv_hmax_inv = sd->mdv.cv_hmax_inv;
  cv_mem->cv_lmm = sd->mdv.cv_lmm;
  cv_mem->cv_iter = sd->mdv.cv_iter;
  cv_mem->cv_itol = sd->mdv.cv_itol;
  cv_mem->cv_reltol = sd->mdv.cv_reltol;
  cv_mem->cv_nhnil = sd->mdv.cv_nhnil;
  cv_mem->cv_etaqm1 = sd->mdv.cv_etaqm1;
  cv_mem->cv_etaq = sd->mdv.cv_etaq;
  cv_mem->cv_etaqp1 = sd->mdv.cv_etaqp1;
  cv_mem->cv_lrw1 = sd->mdv.cv_lrw1;
  cv_mem->cv_liw1 = sd->mdv.cv_liw1;
  cv_mem->cv_lrw = sd->mdv.cv_lrw;
  cv_mem->cv_liw = sd->mdv.cv_liw;
  cv_mem->cv_saved_tq5 = sd->mdv.cv_saved_tq5;
  cv_mem->cv_tolsf = sd->mdv.cv_tolsf;
  cv_mem->cv_qmax_alloc = sd->mdv.cv_qmax_alloc;
  cv_mem->cv_indx_acor = sd->mdv.cv_indx_acor;
  cv_mem->cv_qu = sd->mdv.cv_qu;
  cv_mem->cv_h0u = sd->mdv.cv_h0u;
  cv_mem->cv_hu = sd->mdv.cv_hu;
  cv_mem->cv_jcur = sd->mdv.cv_jcur;//not needed?
  cv_mem->cv_mnewt = sd->mdv.cv_mnewt;
  cv_mem->cv_maxcor = sd->mdv.cv_maxcor;
  cv_mem->cv_nstlp = sd->mdv.cv_nstlp;
  cv_mem->cv_qmax = sd->mdv.cv_qmax;
  cv_mem->cv_L = sd->mdv.cv_L;
  cv_mem->cv_maxnef = sd->mdv.cv_maxnef;
  cv_mem->cv_netf = sd->mdv.cv_netf;
  cv_mem->cv_acnrm = sd->mdv.cv_acnrm;
  cv_mem->cv_tstop = sd->mdv.cv_tstop;
  cv_mem->cv_tstopset = sd->mdv.cv_tstopset;
  cv_mem->cv_nlscoef = sd->mdv.cv_nlscoef;
  cv_mem->cv_qwait = sd->mdv.cv_qwait;
  cv_mem->cv_crate = sd->mdv.cv_crate;
  cv_mem->cv_gamrat = sd->mdv.cv_gamrat;
  cv_mem->cv_gammap = sd->mdv.cv_gammap;
  cv_mem->cv_nst = sd->mdv.cv_nst;
  cv_mem->cv_gamma = sd->mdv.cv_gamma;
  cv_mem->cv_rl1 = sd->mdv.cv_rl1;
  cv_mem->cv_eta = sd->mdv.cv_eta;
  cv_mem->cv_q = sd->mdv.cv_q;
  cv_mem->cv_qprime = sd->mdv.cv_qprime;
  cv_mem->cv_h = sd->mdv.cv_h;
  cv_mem->cv_hscale = sd->mdv.cv_hscale;
  cv_mem->cv_nscon = sd->mdv.cv_nscon;
  cv_mem->cv_hprime = sd->mdv.cv_hprime;
  cv_mem->cv_hmin = sd->mdv.cv_hmin;
  cv_mem->cv_etamax = sd->mdv.cv_etamax;
  cv_mem->cv_maxncf = sd->mdv.cv_maxncf;
  cv_mem->cv_tn = sd->mdv.cv_tn;

  istate = sd->mdv.istate;
  kflag = sd->mdv.kflag;
  *tret = sd->mdv.tret;

  cudaMemcpy(cv_mem->cv_l, mGPU->cv_l, L_MAX * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(cv_mem->cv_tau, mGPU->cv_tau, (L_MAX + 1) * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(cv_mem->cv_tq, mGPU->cv_tq, (NUM_TESTS + 1) * sizeof(double), cudaMemcpyDeviceToHost);

  cudaMemcpy(ewt, mGPU->dewt, mGPU->nrows * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(acor, mGPU->cv_acor, mGPU->nrows * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(tempv, mGPU->dtempv, mGPU->nrows * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(ftemp, mGPU->dftemp, mGPU->nrows * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(cv_last_yn, mGPU->cv_last_yn, mGPU->nrows * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(cv_acor_init, mGPU->cv_acor_init, mGPU->nrows * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(youtArray, mGPU->yout, mGPU->nrows * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(cv_Vabstol, mGPU->cv_Vabstol, mGPU->nrows * sizeof(double), cudaMemcpyDeviceToHost);

  for (int i = 0; i <= cv_mem->cv_qmax; i++) {//cv_qmax+1 (6)?
    double *zn = NV_DATA_S(cv_mem->cv_zn[i]);
    cudaMemcpy(zn, (i * mGPU->nrows + mGPU->dzn), mGPU->nrows * sizeof(double), cudaMemcpyDeviceToHost);

  }

  HANDLE_ERROR(cudaMemcpy(sd->flagCells, mGPU->flagCells, mGPU->n_cells * sizeof(int), cudaMemcpyDeviceToHost));
  flag = CV_SUCCESS;
  for (int i = 0; i < mGPU->n_cells; i++) {
    if (sd->flagCells[i] != flag) {
      flag = sd->flagCells[i];
      break;
    }
  }
#ifndef DEV_CUDACVODE
    istate=flag;
#else
    kflag=flag;
#endif

    //printf("DEV_cudacvStep counterBiConjGrad %d\n",bicg->counterBiConjGrad);
    //printf("cudaCVode flag %d kflag %d\n",flag, sd->mdv.flag);

#ifndef DEV_CUDACVODE

    // In NORMAL mode, check if tout reached
    //if ( (cv_mem->cv_tn-tout)*cv_mem->cv_h >= ZERO ) {
    if ( istate==CV_SUCCESS ) {

      //istate = CV_SUCCESS;

      //printf("istate==CV_SUCCESS\n";

      cv_mem->cv_tretlast = sd->mdv.cv_tretlast;

      //cv_mem->cv_next_q = sd->mdv.cv_next_q;
      //cv_mem->cv_next_h = sd->mdv.cv_next_h;
      cv_mem->cv_next_q = sd->mdv.cv_qprime;//needed?
      cv_mem->cv_next_h = sd->mdv.cv_hprime;

    }else{
#endif

      if (kflag != CV_SUCCESS) {
        printf("cudaCVode2 kflag %d\n",kflag);
        istate = cvHandleFailure_gpu2(cv_mem, kflag);
        //cv_mem->cv_tretlast = *tret = cv_mem->cv_tn;
        //N_VScale(ONE, cv_mem->cv_zn[0], yout);

        cv_mem->cv_next_h = sd->mdv.cv_next_h;

#ifndef DEV_CUDACVODE
#else
        break;
#endif
      }

      if ( (itask == CV_NORMAL) &&  (cv_mem->cv_tn-tout)*cv_mem->cv_h >= ZERO ) {
        istate = CV_SUCCESS;
        cv_mem->cv_tretlast = *tret = tout;
        (void) CVodeGetDky(cv_mem, tout, 0, yout);
        cv_mem->cv_next_q = cv_mem->cv_qprime;
        cv_mem->cv_next_h = cv_mem->cv_hprime;
#ifndef DEV_CUDACVODE
#else
        break;
#endif
      }

      // Reset and check ewt
      if (istate==CV_ILL_INPUT) {
        if (cv_mem->cv_itol == CV_WF)
          cvProcessError(cv_mem, CV_ILL_INPUT, "CVODE", "CVode",
                         MSGCV_EWT_NOW_FAIL, cv_mem->cv_tn);
        else
          cvProcessError(cv_mem, CV_ILL_INPUT, "CVODE", "CVode",
                         MSGCV_EWT_NOW_BAD, cv_mem->cv_tn);
        //Remove break after removing for(;;) in cpu
#ifndef DEV_CUDACVODE
#else
        break;
#endif
      }

      // Check for too many steps
      if ( (cv_mem->cv_mxstep>0) && (nstloc >= cv_mem->cv_mxstep) ) {
        cvProcessError(cv_mem, CV_TOO_MUCH_WORK, "CVODE", "CVode",
                       MSGCV_MAX_STEPS, cv_mem->cv_tn);
        istate = CV_TOO_MUCH_WORK;
        //cv_mem->cv_tretlast = *tret = cv_mem->cv_tn;
        //N_VScale(ONE, cv_mem->cv_zn[0], yout);
#ifndef DEV_CUDACVODE
#else
        break;
#endif
      }

      // Check for too much accuracy requested
      //nrm = N_VWrmsNorm(cv_mem->cv_zn[0], cv_mem->cv_ewt);
      //cv_mem->cv_tolsf = cv_mem->cv_uround * nrm;
      if (cv_mem->cv_tolsf > ONE) {
        cvProcessError(cv_mem, CV_TOO_MUCH_ACC, "CVODE", "CVode",
                       MSGCV_TOO_MUCH_ACC, cv_mem->cv_tn);
        istate = CV_TOO_MUCH_ACC;
        //cv_mem->cv_tretlast = *tret = cv_mem->cv_tn;
        //N_VScale(ONE, cv_mem->cv_zn[0], yout);
        //cv_mem->cv_tolsf *= TWO;
#ifndef DEV_CUDACVODE
#else
        break;
#endif
      }

      //Check for h below roundoff level in tn
      if (cv_mem->cv_tn + cv_mem->cv_h == cv_mem->cv_tn) {
          //cv_mem->cv_nhnil++;
          if (cv_mem->cv_nhnil <= cv_mem->cv_mxhnil)
            cvProcessError(cv_mem, CV_WARNING, "CVODE", "CVode",
                           MSGCV_HNIL, cv_mem->cv_tn, cv_mem->cv_h);
          if (cv_mem->cv_nhnil == cv_mem->cv_mxhnil)
            cvProcessError(cv_mem, CV_WARNING, "CVODE", "CVode", MSGCV_HNIL_DONE);
          }

      if ( cv_mem->cv_tstopset ) {

        troundoff = FUZZ_FACTOR*cv_mem->cv_uround*(SUNRabs(cv_mem->cv_tn) + SUNRabs(cv_mem->cv_h));
        if ( SUNRabs(cv_mem->cv_tn - cv_mem->cv_tstop) <= troundoff) {
          //(void) CVodeGetDky(cv_mem, cv_mem->cv_tstop, 0, yout);
          //cv_mem->cv_tretlast = *tret = cv_mem->cv_tstop;
          //cv_mem->cv_tstopset = SUNFALSE;
          istate = CV_TSTOP_RETURN;

#ifndef DEV_CUDACVODE
#else
          break;
#endif
        }

        if ( (cv_mem->cv_tn + cv_mem->cv_hprime - cv_mem->cv_tstop)*cv_mem->cv_h > ZERO ) {
          cv_mem->cv_hprime = (cv_mem->cv_tstop - cv_mem->cv_tn)*(ONE-FOUR*cv_mem->cv_uround);
          cv_mem->cv_eta = cv_mem->cv_hprime/cv_mem->cv_h;
        }

      }

#ifndef DEV_CUDACVODE
    }
#endif

#ifndef DEV_CUDACVODE
#else

  } /* end looping for internal steps */

#endif

#ifdef CAMP_DEBUG_GPU
  cudaEventRecord(bicg->stopcvStep);
  cudaEventSynchronize(bicg->stopcvStep);
  float mscvStep = 0.0;
  cudaEventElapsedTime(&mscvStep, bicg->startcvStep, bicg->stopcvStep);
  bicg->timecvStep+= mscvStep/1000;

  bicg->timeBiConjGrad=bicg->timesolveCVODEGPU*sd->mdv.dtBCG/sd->mdv.dtcudaDeviceCVode;
  bicg->counterBiConjGrad+= sd->mdv.counterBCG;


#endif

  return(istate);
}



/*-----------------------------------------------------------------*/

/*
 * CVodeGetDky
 *
 * This routine computes the k-th derivative of the interpolating
 * polynomial at the time t and stores the result in the vector dky.
 * The formula is:
 *         q
 *  dky = SUM c(j,k) * (t - tn)^(j-k) * h^(-j) * zn[j] ,
 *        j=k
 * where c(j,k) = j*(j-1)*...*(j-k+1), q is the current order, and
 * zn[j] is the j-th column of the Nordsieck history array.
 *
 * This function is called by CVode with k = 0 and t = tout, but
 * may also be called directly by the user.
 */

int CVodeGetDky_gpu2(void *cvode_mem, realtype t, int k, N_Vector dky)
{
  realtype s, c, r;
  realtype tfuzz, tp, tn1;
  int i, j;
  CVodeMem cv_mem;

  /* Check all inputs for legality */

  if (cvode_mem == NULL) {
    cvProcessError(NULL, CV_MEM_NULL, "CVODE", "CVodeGetDky", MSGCV_NO_MEM);
    return(CV_MEM_NULL);
  }
  cv_mem = (CVodeMem) cvode_mem;

  if (dky == NULL) {
    cvProcessError(cv_mem, CV_BAD_DKY, "CVODE", "CVodeGetDky", MSGCV_NULL_DKY);
    return(CV_BAD_DKY);
  }

  if ((k < 0) || (k > cv_mem->cv_q)) {
    cvProcessError(cv_mem, CV_BAD_K, "CVODE", "CVodeGetDky", MSGCV_BAD_K);
    return(CV_BAD_K);
  }

  /* Allow for some slack */
  tfuzz = FUZZ_FACTOR * cv_mem->cv_uround * (SUNRabs(cv_mem->cv_tn) + SUNRabs(cv_mem->cv_hu));
  if (cv_mem->cv_hu < ZERO) tfuzz = -tfuzz;
  tp = cv_mem->cv_tn - cv_mem->cv_hu - tfuzz;
  tn1 = cv_mem->cv_tn + tfuzz;
  if ((t-tp)*(t-tn1) > ZERO) {
    cvProcessError(cv_mem, CV_BAD_T, "CVODE", "CVodeGetDky", MSGCV_BAD_T,
                   t, cv_mem->cv_tn-cv_mem->cv_hu, cv_mem->cv_tn);
    return(CV_BAD_T);
  }

  /* Sum the differentiated interpolating polynomial */

  s = (t - cv_mem->cv_tn) / cv_mem->cv_h;
  for (j=cv_mem->cv_q; j >= k; j--) {
    c = ONE;
    for (i=j; i >= j-k+1; i--) c *= i;
    if (j == cv_mem->cv_q) {
      N_VScale(c, cv_mem->cv_zn[cv_mem->cv_q], dky);
    } else {
      N_VLinearSum(c, cv_mem->cv_zn[j], s, dky, dky);
    }
  }
  if (k == 0) return(CV_SUCCESS);
  r = SUNRpowerI(cv_mem->cv_h,-k);
  N_VScale(r, dky, dky);
  return(CV_SUCCESS);
}

/*
 * CVodeFree
 *
 * This routine frees the problem memory allocated by CVodeInit.
 * Such memory includes all the vectors allocated by cvAllocVectors,
 * and the memory lmem for the linear solver (deallocated by a call
 * to lfree).
 */

void CVodeFree_gpu2(void **cvode_mem)
{
  CVodeMem cv_mem;

  if (*cvode_mem == NULL) return;

  cv_mem = (CVodeMem) (*cvode_mem);

  cvFreeVectors_gpu2(cv_mem);

  if (cv_mem->cv_lfree != NULL) cv_mem->cv_lfree(cv_mem);

  if (cv_mem->cv_nrtfn > 0) {
    free(cv_mem->cv_glo); cv_mem->cv_glo = NULL;
    free(cv_mem->cv_ghi); cv_mem->cv_ghi = NULL;
    free(cv_mem->cv_grout); cv_mem->cv_grout = NULL;
    free(cv_mem->cv_iroots); cv_mem->cv_iroots = NULL;
    free(cv_mem->cv_rootdir); cv_mem->cv_rootdir = NULL;
    free(cv_mem->cv_gactive); cv_mem->cv_gactive = NULL;
  }

  free(*cvode_mem);
  *cvode_mem = NULL;
}

/*
 * =================================================================
 *  Private Functions Implementation
 * =================================================================
 */

/*
 * cvCheckNvector
 * This routine checks if all required vector operations are present.
 * If any of them is missing it returns SUNFALSE.
 */

booleantype cvCheckNvector_gpu2(N_Vector tmpl)
{
  if((tmpl->ops->nvclone     == NULL) ||
     (tmpl->ops->nvdestroy   == NULL) ||
     (tmpl->ops->nvlinearsum == NULL) ||
     (tmpl->ops->nvconst     == NULL) ||
     (tmpl->ops->nvprod      == NULL) ||
     (tmpl->ops->nvdiv       == NULL) ||
     (tmpl->ops->nvscale     == NULL) ||
     (tmpl->ops->nvabs       == NULL) ||
     (tmpl->ops->nvinv       == NULL) ||
     (tmpl->ops->nvaddconst  == NULL) ||
     (tmpl->ops->nvmaxnorm   == NULL) ||
     (tmpl->ops->nvwrmsnorm  == NULL) ||
     (tmpl->ops->nvmin       == NULL))
    return(SUNFALSE);
  else
    return(SUNTRUE);
}

/*
 * cvAllocVectors
 *
 * This routine allocates the CVODE vectors ewt, acor, tempv, ftemp, and
 * zn[0], ..., zn[maxord].
 * If all memory allocations are successful, cvAllocVectors returns SUNTRUE.
 * Otherwise all allocated memory is freed and cvAllocVectors returns SUNFALSE.
 * This routine also sets the optional outputs lrw and liw, which are
 * (respectively) the lengths of the real and integer work spaces
 * allocated here.
 */

booleantype cvAllocVectors_gpu2(CVodeMem cv_mem, N_Vector tmpl)
{
  int i, j;

  /* Allocate ewt, acor, tempv, ftemp */

  cv_mem->cv_ewt = N_VClone(tmpl);
  if (cv_mem->cv_ewt == NULL) return(SUNFALSE);

  cv_mem->cv_acor = N_VClone(tmpl);
  if (cv_mem->cv_acor == NULL) {
    N_VDestroy(cv_mem->cv_ewt);
    return(SUNFALSE);
  }

  cv_mem->cv_tempv = N_VClone(tmpl);
  if (cv_mem->cv_tempv == NULL) {
    N_VDestroy(cv_mem->cv_ewt);
    N_VDestroy(cv_mem->cv_acor);
    return(SUNFALSE);
  }
  N_VConst(ZERO, cv_mem->cv_tempv);

  cv_mem->cv_tempv1 = N_VClone(tmpl);
  if (cv_mem->cv_tempv1 == NULL) {
    N_VDestroy(cv_mem->cv_tempv);
    N_VDestroy(cv_mem->cv_ewt);
    N_VDestroy(cv_mem->cv_acor);
    return(SUNFALSE);
  }
  N_VConst(ZERO, cv_mem->cv_tempv1);

  cv_mem->cv_tempv2 = N_VClone(tmpl);
  if (cv_mem->cv_tempv2 == NULL) {
    N_VDestroy(cv_mem->cv_tempv);
    N_VDestroy(cv_mem->cv_tempv1);
    N_VDestroy(cv_mem->cv_ewt);
    N_VDestroy(cv_mem->cv_acor);
    return(SUNFALSE);
  }
  N_VConst(ZERO, cv_mem->cv_tempv2);
  cv_mem->cv_acor_init = N_VClone(tmpl);
  if (cv_mem->cv_acor_init == NULL) {
    N_VDestroy(cv_mem->cv_tempv);
    N_VDestroy(cv_mem->cv_tempv1);
    N_VDestroy(cv_mem->cv_tempv2);
    N_VDestroy(cv_mem->cv_ewt);
    N_VDestroy(cv_mem->cv_acor);
    return(SUNFALSE);
  }

  cv_mem->cv_last_yn = N_VClone(tmpl);
  if (cv_mem->cv_last_yn == NULL) {
    N_VDestroy(cv_mem->cv_acor_init);
    N_VDestroy(cv_mem->cv_tempv);
    N_VDestroy(cv_mem->cv_tempv1);
    N_VDestroy(cv_mem->cv_tempv2);
    N_VDestroy(cv_mem->cv_ewt);
    N_VDestroy(cv_mem->cv_acor);
    return(SUNFALSE);
  }

  cv_mem->cv_ftemp = N_VClone(tmpl);
  if (cv_mem->cv_ftemp == NULL) {
    N_VDestroy(cv_mem->cv_last_yn);
    N_VDestroy(cv_mem->cv_acor_init);
    N_VDestroy(cv_mem->cv_tempv);
    N_VDestroy(cv_mem->cv_tempv1);
    N_VDestroy(cv_mem->cv_tempv2);
    N_VDestroy(cv_mem->cv_ewt);
    N_VDestroy(cv_mem->cv_acor);
    return(SUNFALSE);
  }

  /* Allocate zn[0] ... zn[qmax] */

  for (j=0; j <= cv_mem->cv_qmax; j++) {
    cv_mem->cv_zn[j] = N_VClone(tmpl);
    if (cv_mem->cv_zn[j] == NULL) {
      N_VDestroy(cv_mem->cv_ewt);
      N_VDestroy(cv_mem->cv_acor);
      N_VDestroy(cv_mem->cv_acor_init);
      N_VDestroy(cv_mem->cv_last_yn);
      N_VDestroy(cv_mem->cv_tempv);
      N_VDestroy(cv_mem->cv_tempv1);
      N_VDestroy(cv_mem->cv_tempv2);
      N_VDestroy(cv_mem->cv_ftemp);
      for (i=0; i < j; i++) N_VDestroy(cv_mem->cv_zn[i]);
      return(SUNFALSE);
    }
  }

  /* Update solver workspace lengths  */
  cv_mem->cv_lrw += (cv_mem->cv_qmax + 5)*cv_mem->cv_lrw1;
  cv_mem->cv_liw += (cv_mem->cv_qmax + 5)*cv_mem->cv_liw1;

  /* Store the value of qmax used here */
  cv_mem->cv_qmax_alloc = cv_mem->cv_qmax;

  return(SUNTRUE);
}

/*
 * cvFreeVectors
 *
 * This routine frees the CVODE vectors allocated in cvAllocVectors.
 */

void cvFreeVectors_gpu2(CVodeMem cv_mem)
{
  int j, maxord;

  maxord = cv_mem->cv_qmax_alloc;

  N_VDestroy(cv_mem->cv_ewt);
  N_VDestroy(cv_mem->cv_acor);
  N_VDestroy(cv_mem->cv_tempv);
  N_VDestroy(cv_mem->cv_tempv1);
  N_VDestroy(cv_mem->cv_tempv2);
  N_VDestroy(cv_mem->cv_acor_init);
  N_VDestroy(cv_mem->cv_last_yn);
  N_VDestroy(cv_mem->cv_ftemp);
  for (j=0; j <= maxord; j++) N_VDestroy(cv_mem->cv_zn[j]);

  cv_mem->cv_lrw -= (maxord + 5)*cv_mem->cv_lrw1;
  cv_mem->cv_liw -= (maxord + 5)*cv_mem->cv_liw1;

  if (cv_mem->cv_VabstolMallocDone) {
    N_VDestroy(cv_mem->cv_Vabstol);
    cv_mem->cv_lrw -= cv_mem->cv_lrw1;
    cv_mem->cv_liw -= cv_mem->cv_liw1;
  }
}

/*
 * cvInitialSetup
 *
 * This routine performs input consistency checks at the first step.
 * If needed, it also checks the linear solver module and calls the
 * linear solver initialization routine.
 */

int cvInitialSetup_gpu2(CVodeMem cv_mem)
{
  int ier;

  /* Did the user specify tolerances? */
  if (cv_mem->cv_itol == CV_NN) {
    cvProcessError(cv_mem, CV_ILL_INPUT, "CVODE", "cvInitialSetup", MSGCV_NO_TOLS);
    return(CV_ILL_INPUT);
  }

  /* Set data for efun */
  if (cv_mem->cv_user_efun) cv_mem->cv_e_data = cv_mem->cv_user_data;
  else                      cv_mem->cv_e_data = cv_mem;

  /* Load initial error weights */
  ier = cv_mem->cv_efun(cv_mem->cv_zn[0], cv_mem->cv_ewt, cv_mem->cv_e_data);
  if (ier != 0) {
    if (cv_mem->cv_itol == CV_WF)
      cvProcessError(cv_mem, CV_ILL_INPUT, "CVODE", "cvInitialSetup", MSGCV_EWT_FAIL);
    else
      cvProcessError(cv_mem, CV_ILL_INPUT, "CVODE", "cvInitialSetup", MSGCV_BAD_EWT);
    return(CV_ILL_INPUT);
  }

  /* Check if lsolve function exists (if needed) and call linit function (if it exists) */
  if (cv_mem->cv_iter == CV_NEWTON) {
    if (cv_mem->cv_lsolve == NULL) {
      cvProcessError(cv_mem, CV_ILL_INPUT, "CVODE", "cvInitialSetup", MSGCV_LSOLVE_NULL);
      return(CV_ILL_INPUT);
    }
    if (cv_mem->cv_linit != NULL) {
      ier = cv_mem->cv_linit(cv_mem);
      if (ier != 0) {
        cvProcessError(cv_mem, CV_LINIT_FAIL, "CVODE", "cvInitialSetup", MSGCV_LINIT_FAIL);
        return(CV_LINIT_FAIL);
      }
    }
  }

  return(CV_SUCCESS);
}

/*
 * -----------------------------------------------------------------
 * PRIVATE FUNCTIONS FOR CVODE
 * -----------------------------------------------------------------
 */

/*
 * cvHin
 *
 * This routine computes a tentative initial step size h0.
 * If tout is too close to tn (= t0), then cvHin returns CV_TOO_CLOSE
 * and h remains uninitialized. Note that here tout is either the value
 * passed to CVode at the first call or the value of tstop (if tstop is
 * enabled and it is closer to t0=tn than tout).
 * If the RHS function fails unrecoverably, cvHin returns CV_RHSFUNC_FAIL.
 * If the RHS function fails recoverably too many times and recovery is
 * not possible, cvHin returns CV_REPTD_RHSFUNC_ERR.
 * Otherwise, cvHin sets h to the chosen value h0 and returns CV_SUCCESS.
 *
 * The algorithm used seeks to find h0 as a solution of
 *       (WRMS norm of (h0^2 ydd / 2)) = 1,
 * where ydd = estimated second derivative of y.
 *
 * We start with an initial estimate equal to the geometric mean of the
 * lower and upper bounds on the step size.
 *
 * Loop up to MAX_ITERS times to find h0.
 * Stop if new and previous values differ by a factor < 2.
 * Stop if hnew/hg > 2 after one iteration, as this probably means
 * that the ydd value is bad because of cancellation error.
 *
 * For each new proposed hg, we allow MAX_ITERS attempts to
 * resolve a possible recoverable failure from f() by reducing
 * the proposed stepsize by a factor of 0.2. If a legal stepsize
 * still cannot be found, fall back on a previous value if possible,
 * or else return CV_REPTD_RHSFUNC_ERR.
 *
 * Finally, we apply a bias (0.5) and verify that h0 is within bounds.
 */
int cvHin_gpu2(CVodeMem cv_mem, realtype tout)
{
  int retval, sign, count1, count2;
  realtype tdiff, tdist, tround, hlb, hub;
  realtype hg, hgs, hs, hnew, hrat, h0, yddnrm;
  booleantype hgOK, hnewOK;

  /* If tout is too close to tn, give up */

  if ((tdiff = tout-cv_mem->cv_tn) == ZERO) return(CV_TOO_CLOSE);

  sign = (tdiff > ZERO) ? 1 : -1;
  tdist = SUNRabs(tdiff);
  tround = cv_mem->cv_uround * SUNMAX(SUNRabs(cv_mem->cv_tn), SUNRabs(tout));

  if (tdist < TWO*tround) return(CV_TOO_CLOSE);

  /*
     Set lower and upper bounds on h0, and take geometric mean
     as first trial value.
     Exit with this value if the bounds cross each other.
  */

  hlb = HLB_FACTOR * tround;
  hub = cvUpperBoundH0_gpu2(cv_mem, tdist);

  hg  = SUNRsqrt(hlb*hub);

  if (hub < hlb) {
    if (sign == -1) cv_mem->cv_h = -hg;
    else            cv_mem->cv_h =  hg;
    return(CV_SUCCESS);
  }

  /* Outer loop */

  hnewOK = SUNFALSE;
  hs = hg;         /* safeguard against 'uninitialized variable' warning */

  for(count1 = 1; count1 <= MAX_ITERS; count1++) {

    /* Attempts to estimate ydd */

    hgOK = SUNFALSE;

    for (count2 = 1; count2 <= MAX_ITERS; count2++) {
      hgs = hg*sign;
      retval = cvYddNorm_gpu2(cv_mem, hgs, &yddnrm);
      /* If f() failed unrecoverably, give up */
      if (retval < 0) return(CV_RHSFUNC_FAIL);
      /* If successful, we can use ydd */
      if (retval == CV_SUCCESS) {hgOK = SUNTRUE; break;}
      /* f() failed recoverably; cut step size and test it again */
      hg *= POINT2;
    }

    /* If f() failed recoverably MAX_ITERS times */

    if (!hgOK) {
      /* Exit if this is the first or second pass. No recovery possible */
      if (count1 <= 2) return(CV_REPTD_RHSFUNC_ERR);
      /* We have a fall-back option. The value hs is a previous hnew which
         passed through f(). Use it and break */
      hnew = hs;
      break;
    }

    /* The proposed step size is feasible. Save it. */
    hs = hg;

    /* If the stopping criteria was met, or if this is the last pass, stop */
    if ( (hnewOK) || (count1 == MAX_ITERS))  {hnew = hg; break;}

    /* Propose new step size */
    hnew = (yddnrm*hub*hub > TWO) ? SUNRsqrt(TWO/yddnrm) : SUNRsqrt(hg*hub);
    hrat = hnew/hg;

    /* Accept hnew if it does not differ from hg by more than a factor of 2 */
    if ((hrat > HALF) && (hrat < TWO)) {
      hnewOK = SUNTRUE;
    }

    /* After one pass, if ydd seems to be bad, use fall-back value. */
    if ((count1 > 1) && (hrat > TWO)) {
      hnew = hg;
      hnewOK = SUNTRUE;
    }

    /* Send this value back through f() */
    hg = hnew;

  }

  /* Apply bounds, bias factor, and attach sign */

  h0 = H_BIAS*hnew;
  if (h0 < hlb) h0 = hlb;
  if (h0 > hub) h0 = hub;
  if (sign == -1) h0 = -h0;
  cv_mem->cv_h = h0;

  return(CV_SUCCESS);
}

/*
 * cvUpperBoundH0
 *
 * This routine sets an upper bound on abs(h0) based on
 * tdist = tn - t0 and the values of y[i]/y'[i].
 */

realtype cvUpperBoundH0_gpu2(CVodeMem cv_mem, realtype tdist)
{
  realtype hub_inv, hub;
  N_Vector temp1, temp2;

  /*
   * Bound based on |y0|/|y0'| -- allow at most an increase of
   * HUB_FACTOR in y0 (based on a forward Euler step). The weight
   * factor is used as a safeguard against zero components in y0.
   */

  temp1 = cv_mem->cv_tempv;
  temp2 = cv_mem->cv_acor;

  N_VAbs(cv_mem->cv_zn[0], temp2);
  cv_mem->cv_efun(cv_mem->cv_zn[0], temp1, cv_mem->cv_e_data);
  N_VInv(temp1, temp1);
  N_VLinearSum(HUB_FACTOR, temp2, ONE, temp1, temp1);

  N_VAbs(cv_mem->cv_zn[1], temp2);

  N_VDiv(temp2, temp1, temp1);
  hub_inv = N_VMaxNorm(temp1);

  /*
   * bound based on tdist -- allow at most a step of magnitude
   * HUB_FACTOR * tdist
   */

  hub = HUB_FACTOR*tdist;

  /* Use the smaller of the two */

  if (hub*hub_inv > ONE) hub = ONE/hub_inv;

  return(hub);
}

/*
 * cvYddNorm
 *
 * This routine computes an estimate of the second derivative of y
 * using a difference quotient, and returns its WRMS norm.
 */

int cvYddNorm_gpu2(CVodeMem cv_mem, realtype hg, realtype *yddnrm)
{
  int retval;

  N_VLinearSum(hg, cv_mem->cv_zn[1], ONE, cv_mem->cv_zn[0], cv_mem->cv_y);
  //retval = cv_mem->cv_f(cv_mem->cv_tn+hg, cv_mem->cv_y,
  //                      cv_mem->cv_tempv, cv_mem->cv_user_data);
  retval = f(cv_mem->cv_tn+hg, cv_mem->cv_y, cv_mem->cv_tempv, cv_mem->cv_user_data);
  cv_mem->cv_nfe++;
  if (retval < 0) return(CV_RHSFUNC_FAIL);
  if (retval > 0) return(RHSFUNC_RECVR);

  N_VLinearSum(ONE, cv_mem->cv_tempv, -ONE, cv_mem->cv_zn[1], cv_mem->cv_tempv);
  N_VScale(ONE/hg, cv_mem->cv_tempv, cv_mem->cv_tempv);

  *yddnrm = N_VWrmsNorm(cv_mem->cv_tempv, cv_mem->cv_ewt);

  return(CV_SUCCESS);
}

/*
 * -----------------------------------------------------------------
 * Functions for rootfinding
 * -----------------------------------------------------------------
 */

/*
 * cvRcheck1
 *
 * This routine completes the initialization of rootfinding memory
 * information, and checks whether g has a zero both at and very near
 * the initial point of the IVP.
 *
 * This routine returns an int equal to:
 *  CV_RTFUNC_FAIL < 0 if the g function failed, or
 *  CV_SUCCESS     = 0 otherwise.
 */

int cvRcheck1_gpu2(CVodeMem cv_mem)
{
  int i, retval;
  realtype smallh, hratio, tplus;
  booleantype zroot;

  for (i = 0; i < cv_mem->cv_nrtfn; i++) cv_mem->cv_iroots[i] = 0;
  cv_mem->cv_tlo = cv_mem->cv_tn;
  cv_mem->cv_ttol = (SUNRabs(cv_mem->cv_tn) + SUNRabs(cv_mem->cv_h)) *
                    cv_mem->cv_uround*HUNDRED;

  /* Evaluate g at initial t and check for zero values. */
  retval = cv_mem->cv_gfun(cv_mem->cv_tlo, cv_mem->cv_zn[0],
                           cv_mem->cv_glo, cv_mem->cv_user_data);
  cv_mem->cv_nge = 1;
  if (retval != 0) return(CV_RTFUNC_FAIL);

  zroot = SUNFALSE;
  for (i = 0; i < cv_mem->cv_nrtfn; i++) {
    if (SUNRabs(cv_mem->cv_glo[i]) == ZERO) {
      zroot = SUNTRUE;
      cv_mem->cv_gactive[i] = SUNFALSE;
    }
  }
  if (!zroot) return(CV_SUCCESS);

  /* Some g_i is zero at t0; look at g at t0+(small increment). */
  hratio = SUNMAX(cv_mem->cv_ttol/SUNRabs(cv_mem->cv_h), PT1);
  smallh = hratio*cv_mem->cv_h;
  tplus = cv_mem->cv_tlo + smallh;
  N_VLinearSum(ONE, cv_mem->cv_zn[0], hratio,
               cv_mem->cv_zn[1], cv_mem->cv_y);
  retval = cv_mem->cv_gfun(tplus, cv_mem->cv_y,
                           cv_mem->cv_ghi, cv_mem->cv_user_data);
  cv_mem->cv_nge++;
  if (retval != 0) return(CV_RTFUNC_FAIL);

  /* We check now only the components of g which were exactly 0.0 at t0
   * to see if we can 'activate' them. */
  for (i = 0; i < cv_mem->cv_nrtfn; i++) {
    if (!cv_mem->cv_gactive[i] && SUNRabs(cv_mem->cv_ghi[i]) != ZERO) {
      cv_mem->cv_gactive[i] = SUNTRUE;
      cv_mem->cv_glo[i] = cv_mem->cv_ghi[i];
    }
  }
  return(CV_SUCCESS);
}

/*
 * cvRcheck2
 *
 * This routine checks for exact zeros of g at the last root found,
 * if the last return was a root.  It then checks for a close pair of
 * zeros (an error condition), and for a new root at a nearby point.
 * The array glo = g(tlo) at the left endpoint of the search interval
 * is adjusted if necessary to assure that all g_i are nonzero
 * there, before returning to do a root search in the interval.
 *
 * On entry, tlo = tretlast is the last value of tret returned by
 * CVode.  This may be the previous tn, the previous tout value,
 * or the last root location.
 *
 * This routine returns an int equal to:
 *     CV_RTFUNC_FAIL  < 0 if the g function failed, or
 *     CLOSERT         = 3 if a close pair of zeros was found, or
 *     RTFOUND         = 1 if a new zero of g was found near tlo, or
 *     CV_SUCCESS      = 0 otherwise.
 */

int cvRcheck2_gpu2(CVodeMem cv_mem)
{
  int i, retval;
  realtype smallh, hratio, tplus;
  booleantype zroot;

  if (cv_mem->cv_irfnd == 0) return(CV_SUCCESS);

  (void) CVodeGetDky(cv_mem, cv_mem->cv_tlo, 0, cv_mem->cv_y);
  retval = cv_mem->cv_gfun(cv_mem->cv_tlo, cv_mem->cv_y,
                           cv_mem->cv_glo, cv_mem->cv_user_data);
  cv_mem->cv_nge++;
  if (retval != 0) return(CV_RTFUNC_FAIL);

  zroot = SUNFALSE;
  for (i = 0; i < cv_mem->cv_nrtfn; i++) cv_mem->cv_iroots[i] = 0;
  for (i = 0; i < cv_mem->cv_nrtfn; i++) {
    if (!cv_mem->cv_gactive[i]) continue;
    if (SUNRabs(cv_mem->cv_glo[i]) == ZERO) {
      zroot = SUNTRUE;
      cv_mem->cv_iroots[i] = 1;
    }
  }
  if (!zroot) return(CV_SUCCESS);

  /* One or more g_i has a zero at tlo.  Check g at tlo+smallh. */
  cv_mem->cv_ttol = (SUNRabs(cv_mem->cv_tn) + SUNRabs(cv_mem->cv_h)) *
                    cv_mem->cv_uround * HUNDRED;
  smallh = (cv_mem->cv_h > ZERO) ? cv_mem->cv_ttol : -cv_mem->cv_ttol;
  tplus = cv_mem->cv_tlo + smallh;
  if ( (tplus - cv_mem->cv_tn)*cv_mem->cv_h >= ZERO) {
    hratio = smallh/cv_mem->cv_h;
    N_VLinearSum(ONE, cv_mem->cv_y, hratio, cv_mem->cv_zn[1], cv_mem->cv_y);
  } else {
    (void) CVodeGetDky(cv_mem, tplus, 0, cv_mem->cv_y);
  }
  retval = cv_mem->cv_gfun(tplus, cv_mem->cv_y,
                           cv_mem->cv_ghi, cv_mem->cv_user_data);
  cv_mem->cv_nge++;
  if (retval != 0) return(CV_RTFUNC_FAIL);

  /* Check for close roots (error return), for a new zero at tlo+smallh,
  and for a g_i that changed from zero to nonzero. */
  zroot = SUNFALSE;
  for (i = 0; i < cv_mem->cv_nrtfn; i++) {
    if (!cv_mem->cv_gactive[i]) continue;
    if (SUNRabs(cv_mem->cv_ghi[i]) == ZERO) {
      if (cv_mem->cv_iroots[i] == 1) return(CLOSERT);
      zroot = SUNTRUE;
      cv_mem->cv_iroots[i] = 1;
    } else {
      if (cv_mem->cv_iroots[i] == 1)
        cv_mem->cv_glo[i] = cv_mem->cv_ghi[i];
    }
  }
  if (zroot) return(RTFOUND);
  return(CV_SUCCESS);
}

/*
 * cvRcheck3
 *
 * This routine interfaces to cvRootfind to look for a root of g
 * between tlo and either tn or tout, whichever comes first.
 * Only roots beyond tlo in the direction of integration are sought.
 *
 * This routine returns an int equal to:
 *     CV_RTFUNC_FAIL  < 0 if the g function failed, or
 *     RTFOUND         = 1 if a root of g was found, or
 *     CV_SUCCESS      = 0 otherwise.
 */

int cvRcheck3_gpu2(CVodeMem cv_mem)
{
  int i, ier, retval;

  /* Set thi = tn or tout, whichever comes first; set y = y(thi). */
  if (cv_mem->cv_taskc == CV_ONE_STEP) {
    cv_mem->cv_thi = cv_mem->cv_tn;
    N_VScale(ONE, cv_mem->cv_zn[0], cv_mem->cv_y);
  }
  if (cv_mem->cv_taskc == CV_NORMAL) {
    if ( (cv_mem->cv_toutc - cv_mem->cv_tn)*cv_mem->cv_h >= ZERO) {
      cv_mem->cv_thi = cv_mem->cv_tn;
      N_VScale(ONE, cv_mem->cv_zn[0], cv_mem->cv_y);
    } else {
      cv_mem->cv_thi = cv_mem->cv_toutc;
      (void) CVodeGetDky(cv_mem, cv_mem->cv_thi, 0, cv_mem->cv_y);
    }
  }

  /* Set ghi = g(thi) and call cvRootfind to search (tlo,thi) for roots. */
  retval = cv_mem->cv_gfun(cv_mem->cv_thi, cv_mem->cv_y,
                           cv_mem->cv_ghi, cv_mem->cv_user_data);
  cv_mem->cv_nge++;
  if (retval != 0) return(CV_RTFUNC_FAIL);

  cv_mem->cv_ttol = (SUNRabs(cv_mem->cv_tn) + SUNRabs(cv_mem->cv_h)) *
                    cv_mem->cv_uround * HUNDRED;
  ier = cvRootfind_gpu2(cv_mem);
  if (ier == CV_RTFUNC_FAIL) return(CV_RTFUNC_FAIL);
  for(i=0; i<cv_mem->cv_nrtfn; i++) {
    if(!cv_mem->cv_gactive[i] && cv_mem->cv_grout[i] != ZERO)
      cv_mem->cv_gactive[i] = SUNTRUE;
  }
  cv_mem->cv_tlo = cv_mem->cv_trout;
  for (i = 0; i < cv_mem->cv_nrtfn; i++)
    cv_mem->cv_glo[i] = cv_mem->cv_grout[i];

  /* If no root found, return CV_SUCCESS. */
  if (ier == CV_SUCCESS) return(CV_SUCCESS);

  /* If a root was found, interpolate to get y(trout) and return.  */
  (void) CVodeGetDky(cv_mem, cv_mem->cv_trout, 0, cv_mem->cv_y);
  return(RTFOUND);
}

/*
 * cvRootfind
 *
 * This routine solves for a root of g(t) between tlo and thi, if
 * one exists.  Only roots of odd multiplicity (i.e. with a change
 * of sign in one of the g_i), or exact zeros, are found.
 * Here the sign of tlo - thi is arbitrary, but if multiple roots
 * are found, the one closest to tlo is returned.
 *
 * The method used is the Illinois algorithm, a modified secant method.
 * Reference: Kathie L. Hiebert and Lawrence F. Shampine, Implicitly
 * Defined Output Points for Solutions of ODEs, Sandia National
 * Laboratory Report SAND80-0180, February 1980.
 *
 * This routine uses the following parameters for communication:
 *
 * nrtfn    = number of functions g_i, or number of components of
 *            the vector-valued function g(t).  Input only.
 *
 * gfun     = user-defined function for g(t).  Its form is
 *            (void) gfun(t, y, gt, user_data)
 *
 * rootdir  = in array specifying the direction of zero-crossings.
 *            If rootdir[i] > 0, search for roots of g_i only if
 *            g_i is increasing; if rootdir[i] < 0, search for
 *            roots of g_i only if g_i is decreasing; otherwise
 *            always search for roots of g_i.
 *
 * gactive  = array specifying whether a component of g should
 *            or should not be monitored. gactive[i] is initially
 *            set to SUNTRUE for all i=0,...,nrtfn-1, but it may be
 *            reset to SUNFALSE if at the first step g[i] is 0.0
 *            both at the I.C. and at a small perturbation of them.
 *            gactive[i] is then set back on SUNTRUE only after the
 *            corresponding g function moves away from 0.0.
 *
 * nge      = cumulative counter for gfun calls.
 *
 * ttol     = a convergence tolerance for trout.  Input only.
 *            When a root at trout is found, it is located only to
 *            within a tolerance of ttol.  Typically, ttol should
 *            be set to a value on the order of
 *               100 * UROUND * max (SUNRabs(tlo), SUNRabs(thi))
 *            where UROUND is the unit roundoff of the machine.
 *
 * tlo, thi = endpoints of the interval in which roots are sought.
 *            On input, these must be distinct, but tlo - thi may
 *            be of either sign.  The direction of integration is
 *            assumed to be from tlo to thi.  On return, tlo and thi
 *            are the endpoints of the final relevant interval.
 *
 * glo, ghi = arrays of length nrtfn containing the vectors g(tlo)
 *            and g(thi) respectively.  Input and output.  On input,
 *            none of the glo[i] should be zero.
 *
 * trout    = root location, if a root was found, or thi if not.
 *            Output only.  If a root was found other than an exact
 *            zero of g, trout is the endpoint thi of the final
 *            interval bracketing the root, with size at most ttol.
 *
 * grout    = array of length nrtfn containing g(trout) on return.
 *
 * iroots   = int array of length nrtfn with root information.
 *            Output only.  If a root was found, iroots indicates
 *            which components g_i have a root at trout.  For
 *            i = 0, ..., nrtfn-1, iroots[i] = 1 if g_i has a root
 *            and g_i is increasing, iroots[i] = -1 if g_i has a
 *            root and g_i is decreasing, and iroots[i] = 0 if g_i
 *            has no roots or g_i varies in the direction opposite
 *            to that indicated by rootdir[i].
 *
 * This routine returns an int equal to:
 *      CV_RTFUNC_FAIL  < 0 if the g function failed, or
 *      RTFOUND         = 1 if a root of g was found, or
 *      CV_SUCCESS      = 0 otherwise.
 */

int cvRootfind_gpu2(CVodeMem cv_mem)
{
  realtype alph, tmid, gfrac, maxfrac, fracint, fracsub;
  int i, retval, imax, side, sideprev;
  booleantype zroot, sgnchg;

  imax = 0;

  /* First check for change in sign in ghi or for a zero in ghi. */
  maxfrac = ZERO;
  zroot = SUNFALSE;
  sgnchg = SUNFALSE;
  for (i = 0;  i < cv_mem->cv_nrtfn; i++) {
    if(!cv_mem->cv_gactive[i]) continue;
    if (SUNRabs(cv_mem->cv_ghi[i]) == ZERO) {
      if(cv_mem->cv_rootdir[i]*cv_mem->cv_glo[i] <= ZERO) {
        zroot = SUNTRUE;
      }
    } else {
      if ( (cv_mem->cv_glo[i]*cv_mem->cv_ghi[i] < ZERO) &&
           (cv_mem->cv_rootdir[i]*cv_mem->cv_glo[i] <= ZERO) ) {
        gfrac = SUNRabs(cv_mem->cv_ghi[i]/(cv_mem->cv_ghi[i] - cv_mem->cv_glo[i]));
        if (gfrac > maxfrac) {
          sgnchg = SUNTRUE;
          maxfrac = gfrac;
          imax = i;
        }
      }
    }
  }

  /* If no sign change was found, reset trout and grout.  Then return
     CV_SUCCESS if no zero was found, or set iroots and return RTFOUND.  */
  if (!sgnchg) {
    cv_mem->cv_trout = cv_mem->cv_thi;
    for (i = 0; i < cv_mem->cv_nrtfn; i++) cv_mem->cv_grout[i] = cv_mem->cv_ghi[i];
    if (!zroot) return(CV_SUCCESS);
    for (i = 0; i < cv_mem->cv_nrtfn; i++) {
      cv_mem->cv_iroots[i] = 0;
      if(!cv_mem->cv_gactive[i]) continue;
      if ( (SUNRabs(cv_mem->cv_ghi[i]) == ZERO) &&
           (cv_mem->cv_rootdir[i]*cv_mem->cv_glo[i] <= ZERO) )
        cv_mem->cv_iroots[i] = cv_mem->cv_glo[i] > 0 ? -1 : 1;
    }
    return(RTFOUND);
  }

  /* Initialize alph to avoid compiler warning */
  alph = ONE;

  /* A sign change was found.  Loop to locate nearest root. */

  side = 0;  sideprev = -1;
  for(;;) {                                    /* Looping point */

    /* If interval size is already less than tolerance ttol, break. */
    if (SUNRabs(cv_mem->cv_thi - cv_mem->cv_tlo) <= cv_mem->cv_ttol) break;

    /* Set weight alph.
       On the first two passes, set alph = 1.  Thereafter, reset alph
       according to the side (low vs high) of the subinterval in which
       the sign change was found in the previous two passes.
       If the sides were opposite, set alph = 1.
       If the sides were the same, then double alph (if high side),
       or halve alph (if low side).
       The next guess tmid is the secant method value if alph = 1, but
       is closer to tlo if alph < 1, and closer to thi if alph > 1.    */

    if (sideprev == side) {
      alph = (side == 2) ? alph*TWO : alph*HALF;
    } else {
      alph = ONE;
    }

    /* Set next root approximation tmid and get g(tmid).
       If tmid is too close to tlo or thi, adjust it inward,
       by a fractional distance that is between 0.1 and 0.5.  */
    tmid = cv_mem->cv_thi - (cv_mem->cv_thi - cv_mem->cv_tlo) *
                            cv_mem->cv_ghi[imax] / (cv_mem->cv_ghi[imax] - alph*cv_mem->cv_glo[imax]);
    if (SUNRabs(tmid - cv_mem->cv_tlo) < HALF*cv_mem->cv_ttol) {
      fracint = SUNRabs(cv_mem->cv_thi - cv_mem->cv_tlo)/cv_mem->cv_ttol;
      fracsub = (fracint > FIVE) ? PT1 : HALF/fracint;
      tmid = cv_mem->cv_tlo + fracsub*(cv_mem->cv_thi - cv_mem->cv_tlo);
    }
    if (SUNRabs(cv_mem->cv_thi - tmid) < HALF*cv_mem->cv_ttol) {
      fracint = SUNRabs(cv_mem->cv_thi - cv_mem->cv_tlo)/cv_mem->cv_ttol;
      fracsub = (fracint > FIVE) ? PT1 : HALF/fracint;
      tmid = cv_mem->cv_thi - fracsub*(cv_mem->cv_thi - cv_mem->cv_tlo);
    }

    (void) CVodeGetDky(cv_mem, tmid, 0, cv_mem->cv_y);
    retval = cv_mem->cv_gfun(tmid, cv_mem->cv_y, cv_mem->cv_grout,
                             cv_mem->cv_user_data);
    cv_mem->cv_nge++;
    if (retval != 0) return(CV_RTFUNC_FAIL);

    /* Check to see in which subinterval g changes sign, and reset imax.
       Set side = 1 if sign change is on low side, or 2 if on high side.  */
    maxfrac = ZERO;
    zroot = SUNFALSE;
    sgnchg = SUNFALSE;
    sideprev = side;
    for (i = 0;  i < cv_mem->cv_nrtfn; i++) {
      if(!cv_mem->cv_gactive[i]) continue;
      if (SUNRabs(cv_mem->cv_grout[i]) == ZERO) {
        if(cv_mem->cv_rootdir[i]*cv_mem->cv_glo[i] <= ZERO) zroot = SUNTRUE;
      } else {
        if ( (cv_mem->cv_glo[i]*cv_mem->cv_grout[i] < ZERO) &&
             (cv_mem->cv_rootdir[i]*cv_mem->cv_glo[i] <= ZERO) ) {
          gfrac = SUNRabs(cv_mem->cv_grout[i]/(cv_mem->cv_grout[i] - cv_mem->cv_glo[i]));
          if (gfrac > maxfrac) {
            sgnchg = SUNTRUE;
            maxfrac = gfrac;
            imax = i;
          }
        }
      }
    }
    if (sgnchg) {
      /* Sign change found in (tlo,tmid); replace thi with tmid. */
      cv_mem->cv_thi = tmid;
      for (i = 0; i < cv_mem->cv_nrtfn; i++)
        cv_mem->cv_ghi[i] = cv_mem->cv_grout[i];
      side = 1;
      /* Stop at root thi if converged; otherwise loop. */
      if (SUNRabs(cv_mem->cv_thi - cv_mem->cv_tlo) <= cv_mem->cv_ttol) break;
      continue;  /* Return to looping point. */
    }

    if (zroot) {
      /* No sign change in (tlo,tmid), but g = 0 at tmid; return root tmid. */
      cv_mem->cv_thi = tmid;
      for (i = 0; i < cv_mem->cv_nrtfn; i++)
        cv_mem->cv_ghi[i] = cv_mem->cv_grout[i];
      break;
    }

    /* No sign change in (tlo,tmid), and no zero at tmid.
       Sign change must be in (tmid,thi).  Replace tlo with tmid. */
    cv_mem->cv_tlo = tmid;
    for (i = 0; i < cv_mem->cv_nrtfn; i++)
      cv_mem->cv_glo[i] = cv_mem->cv_grout[i];
    side = 2;
    /* Stop at root thi if converged; otherwise loop back. */
    if (SUNRabs(cv_mem->cv_thi - cv_mem->cv_tlo) <= cv_mem->cv_ttol) break;

  } /* End of root-search loop */

  /* Reset trout and grout, set iroots, and return RTFOUND. */
  cv_mem->cv_trout = cv_mem->cv_thi;
  for (i = 0; i < cv_mem->cv_nrtfn; i++) {
    cv_mem->cv_grout[i] = cv_mem->cv_ghi[i];
    cv_mem->cv_iroots[i] = 0;
    if(!cv_mem->cv_gactive[i]) continue;
    if ( (SUNRabs(cv_mem->cv_ghi[i]) == ZERO) &&
         (cv_mem->cv_rootdir[i]*cv_mem->cv_glo[i] <= ZERO) )
      cv_mem->cv_iroots[i] = cv_mem->cv_glo[i] > 0 ? -1 : 1;
    if ( (cv_mem->cv_glo[i]*cv_mem->cv_ghi[i] < ZERO) &&
         (cv_mem->cv_rootdir[i]*cv_mem->cv_glo[i] <= ZERO) )
      cv_mem->cv_iroots[i] = cv_mem->cv_glo[i] > 0 ? -1 : 1;
  }
  return(RTFOUND);
}

/*
 * cvStep
 *
 * This routine performs one internal cvode step, from tn to tn + h.
 * It calls other routines to do all the work.
 *
 * The main operations done here are as follows:
 * - preliminary adjustments if a new step size was chosen;
 * - prediction of the Nordsieck history array zn at tn + h;
 * - setting of multistep method coefficients and test quantities;
 * - solution of the nonlinear system;
 * - testing the local error;
 * - updating zn and other state data if successful;
 * - resetting stepsize and order for the next step.
 * - if SLDET is on, check for stability, reduce order if necessary.
 * On a failure in the nonlinear system solution or error test, the
 * step may be reattempted, depending on the nature of the failure.
 */
int cvStep_gpu2(SolverData *sd, CVodeMem cv_mem)
{
  ModelDataGPU *mGPU = &sd->mGPU;
  realtype saved_t, dsm;
  int ncf, nef;
  int nflag, kflag, eflag;

  double *ewt = NV_DATA_S(cv_mem->cv_ewt);
  cudaMemcpy(mGPU->dewt,ewt,mGPU->nrows*sizeof(double),cudaMemcpyHostToDevice);

  saved_t = cv_mem->cv_tn;
  ncf = nef = 0;
  nflag = FIRST_CALL;

  if ((cv_mem->cv_nst > 0) && (cv_mem->cv_hprime != cv_mem->cv_h))
    cvAdjustParams_gpu2(cv_mem);

  /* Looping point for attempts to take a step */
  for(;;) {

    cvPredict_gpu2(cv_mem);

    cvSet_gpu2(cv_mem);

    //nflag = cvNls(cv_mem, nflag);
    nflag = cvNlsNewton_gpu2(sd, cv_mem, nflag);//f(y)+BCG

    kflag = cvHandleNFlag_gpu2(cv_mem, &nflag, saved_t, &ncf);

    /* Go back in loop if we need to predict again (nflag=PREV_CONV_FAIL)*/
    if (kflag == PREDICT_AGAIN) continue;

    /* Return if nonlinear solve failed and recovery not possible. */
    if (kflag != DO_ERROR_TEST) return(kflag);

    /* Perform error test (nflag=CV_SUCCESS) */
    eflag = cvDoErrorTest_gpu2(cv_mem, &nflag, saved_t, &nef, &dsm);

    /* Go back in loop if we need to predict again (nflag=PREV_ERR_FAIL) */
    if (eflag == TRY_AGAIN)  continue;

    /* Return if error test failed and recovery not possible. */
    if (eflag != CV_SUCCESS) return(eflag);

    /* Error test passed (eflag=CV_SUCCESS), break from loop */
    break;

  }

  /* Nonlinear system solve and error test were both successful.
     Update data, and consider change of step and/or order.       */

  cvCompleteStep_gpu2(cv_mem);

  cvPrepareNextStep_gpu2(cv_mem, dsm);//use tq calculated in cvset and tempv calc in cvnewton

  /* If Stablilty Limit Detection is turned on, call stability limit
     detection routine for possible order reduction. */

  if (cv_mem->cv_sldeton) cvBDFStab_gpu2(cv_mem);

  cv_mem->cv_etamax = (cv_mem->cv_nst <= SMALL_NST) ? ETAMX2 : ETAMX3;

  /*  Finally, we rescale the acor array to be the
      estimated local error vector. */

  N_VScale(cv_mem->cv_tq[2], cv_mem->cv_acor, cv_mem->cv_acor);
  return(CV_SUCCESS);

}

/*
 * cvAdjustParams
 *
 * This routine is called when a change in step size was decided upon,
 * and it handles the required adjustments to the history array zn.
 * If there is to be a change in order, we call cvAdjustOrder and reset
 * q, L = q+1, and qwait.  Then in any case, we call cvRescale, which
 * resets h and rescales the Nordsieck array.
 */

void cvAdjustParams_gpu2(CVodeMem cv_mem)
{
  if (cv_mem->cv_qprime != cv_mem->cv_q) {
    //cvAdjustOrder(cv_mem, cv_mem->cv_qprime-cv_mem->cv_q);

    int deltaq = cv_mem->cv_qprime-cv_mem->cv_q;
    switch(deltaq) {
      case 1:
        cvIncreaseBDF_gpu2(cv_mem);
        break;
      case -1:
        cvDecreaseBDF_gpu2(cv_mem);
        break;
    }

    cv_mem->cv_q = cv_mem->cv_qprime;
    cv_mem->cv_L = cv_mem->cv_q+1;
    cv_mem->cv_qwait = cv_mem->cv_L;
  }
  cvRescale_gpu2(cv_mem);
}

/*
 * cvIncreaseBDF
 *
 * This routine adjusts the history array on an increase in the
 * order q in the case that lmm == CV_BDF.
 * A new column zn[q+1] is set equal to a multiple of the saved
 * vector (= acor) in zn[indx_acor].  Then each zn[j] is adjusted by
 * a multiple of zn[q+1].  The coefficients in the adjustment are the
 * coefficients of the polynomial x*x*(x+xi_1)*...*(x+xi_j),
 * where xi_j = [t_n - t_(n-j)]/h.
 */

void cvIncreaseBDF_gpu2(CVodeMem cv_mem)
{
  realtype alpha0, alpha1, prod, xi, xiold, hsum, A1;
  int i, j;

  for (i=0; i <= cv_mem->cv_qmax; i++) cv_mem->cv_l[i] = ZERO;
  cv_mem->cv_l[2] = alpha1 = prod = xiold = ONE;
  alpha0 = -ONE;
  hsum = cv_mem->cv_hscale;
  if (cv_mem->cv_q > 1) {
    for (j=1; j < cv_mem->cv_q; j++) {
      hsum += cv_mem->cv_tau[j+1];
      xi = hsum / cv_mem->cv_hscale;
      prod *= xi;
      alpha0 -= ONE / (j+1);
      alpha1 += ONE / xi;
      for (i=j+2; i >= 2; i--)
        cv_mem->cv_l[i] = cv_mem->cv_l[i]*xiold + cv_mem->cv_l[i-1];
      xiold = xi;
    }
  }
  A1 = (-alpha0 - alpha1) / prod;
  N_VScale(A1, cv_mem->cv_zn[cv_mem->cv_indx_acor],
           cv_mem->cv_zn[cv_mem->cv_L]);
  for (j=2; j <= cv_mem->cv_q; j++)
    N_VLinearSum(cv_mem->cv_l[j], cv_mem->cv_zn[cv_mem->cv_L], ONE,
                 cv_mem->cv_zn[j], cv_mem->cv_zn[j]);
}

/*
 * cvDecreaseBDF
 *
 * This routine adjusts the history array on a decrease in the
 * order q in the case that lmm == CV_BDF.
 * Each zn[j] is adjusted by a multiple of zn[q].  The coefficients
 * in the adjustment are the coefficients of the polynomial
 *   x*x*(x+xi_1)*...*(x+xi_j), where xi_j = [t_n - t_(n-j)]/h.
 */

void cvDecreaseBDF_gpu2(CVodeMem cv_mem)
{
  realtype hsum, xi;
  int i, j;

  for (i=0; i <= cv_mem->cv_qmax; i++) cv_mem->cv_l[i] = ZERO;
  cv_mem->cv_l[2] = ONE;
  hsum = ZERO;
  for (j=1; j <= cv_mem->cv_q-2; j++) {
    hsum += cv_mem->cv_tau[j];
    xi = hsum /cv_mem->cv_hscale;
    for (i=j+2; i >= 2; i--)
      cv_mem->cv_l[i] = cv_mem->cv_l[i]*xi + cv_mem->cv_l[i-1];
  }

  for (j=2; j < cv_mem->cv_q; j++)
    N_VLinearSum(-cv_mem->cv_l[j], cv_mem->cv_zn[cv_mem->cv_q],
                 ONE, cv_mem->cv_zn[j], cv_mem->cv_zn[j]);
}

/*
 * cvRescale
 *
 * This routine rescales the Nordsieck array by multiplying the
 * jth column zn[j] by eta^j, j = 1, ..., q.  Then the value of
 * h is rescaled by eta, and hscale is reset to h.
 */

void cvRescale_gpu2(CVodeMem cv_mem)
{
  int j;
  realtype factor;

  factor = cv_mem->cv_eta;
  for (j=1; j <= cv_mem->cv_q; j++) {
    N_VScale(factor, cv_mem->cv_zn[j], cv_mem->cv_zn[j]);
    factor *= cv_mem->cv_eta;
  }
  cv_mem->cv_h = cv_mem->cv_hscale * cv_mem->cv_eta;
  cv_mem->cv_next_h = cv_mem->cv_h;
  cv_mem->cv_hscale = cv_mem->cv_h;
  cv_mem->cv_nscon = 0;
}

/*
 * cvPredict
 *
 * This routine advances tn by the tentative step size h, and computes
 * the predicted array z_n(0), which is overwritten on zn.  The
 * prediction of zn is done by repeated additions.
 * If tstop is enabled, it is possible for tn + h to be past tstop by roundoff,
 * and in that case, we reset tn (after incrementing by h) to tstop.
 */

void cvPredict_gpu2(CVodeMem cv_mem)
{
  int j, k;

  cv_mem->cv_tn += cv_mem->cv_h;
  if (cv_mem->cv_tstopset) {
    if ((cv_mem->cv_tn - cv_mem->cv_tstop)*cv_mem->cv_h > ZERO)
      cv_mem->cv_tn = cv_mem->cv_tstop;
  }
  N_VScale(ONE, cv_mem->cv_zn[0], cv_mem->cv_last_yn);
  for (k = 1; k <= cv_mem->cv_q; k++)
    for (j = cv_mem->cv_q; j >= k; j--)
      N_VLinearSum(ONE, cv_mem->cv_zn[j-1], ONE,
                   cv_mem->cv_zn[j], cv_mem->cv_zn[j-1]);

}

/*
 * cvSet
 *
 * This routine is a high level routine which calls cvSetAdams or
 * cvSetBDF to set the polynomial l, the test quantity array tq,
 * and the related variables  rl1, gamma, and gamrat.
 *
 * The array tq is loaded with constants used in the control of estimated
 * local errors and in the nonlinear convergence test.  Specifically, while
 * running at order q, the components of tq are as follows:
 *   tq[1] = a coefficient used to get the est. local error at order q-1
 *   tq[2] = a coefficient used to get the est. local error at order q
 *   tq[3] = a coefficient used to get the est. local error at order q+1
 *   tq[4] = constant used in nonlinear iteration convergence test
 *   tq[5] = coefficient used to get the order q+2 derivative vector used in
 *           the est. local error at order q+1
 */

void cvSet_gpu2(CVodeMem cv_mem)
{

  cvSetBDF_gpu2(cv_mem);
  cv_mem->cv_rl1 = ONE / cv_mem->cv_l[1];
  cv_mem->cv_gamma = cv_mem->cv_h * cv_mem->cv_rl1;
  if (cv_mem->cv_nst == 0) cv_mem->cv_gammap = cv_mem->cv_gamma;
  cv_mem->cv_gamrat = (cv_mem->cv_nst > 0) ?
                      cv_mem->cv_gamma / cv_mem->cv_gammap : ONE;  /* protect x / x != 1.0 */
}

/*
 * cvSetBDF
 *
 * This routine computes the coefficients l and tq in the case
 * lmm == CV_BDF.  cvSetBDF calls cvSetTqBDF to set the test
 * quantity array tq.
 *
 * The components of the array l are the coefficients of a
 * polynomial Lambda(x) = l_0 + l_1 x + ... + l_q x^q, given by
 *                                 q-1
 * Lambda(x) = (1 + x / xi*_q) * PRODUCT (1 + x / xi_i) , where
 *                                 i=1
 *  xi_i = [t_n - t_(n-i)] / h.
 *
 * The array tq is set to test quantities used in the convergence
 * test, the error test, and the selection of h at a new order.
 */

void cvSetBDF_gpu2(CVodeMem cv_mem)
{
  realtype alpha0, alpha0_hat, xi_inv, xistar_inv, hsum;
  int i,j;

  cv_mem->cv_l[0] = cv_mem->cv_l[1] = xi_inv = xistar_inv = ONE;
  for (i=2; i <= cv_mem->cv_q; i++) cv_mem->cv_l[i] = ZERO;
  alpha0 = alpha0_hat = -ONE;
  hsum = cv_mem->cv_h;
  if (cv_mem->cv_q > 1) {
    for (j=2; j < cv_mem->cv_q; j++) {
      hsum += cv_mem->cv_tau[j-1];
      xi_inv = cv_mem->cv_h / hsum;
      alpha0 -= ONE / j;
      for (i=j; i >= 1; i--) cv_mem->cv_l[i] += cv_mem->cv_l[i-1]*xi_inv;
      /* The l[i] are coefficients of product(1 to j) (1 + x/xi_i) */
    }

    /* j = q */
    alpha0 -= ONE / cv_mem->cv_q;
    xistar_inv = -cv_mem->cv_l[1] - alpha0;
    hsum += cv_mem->cv_tau[cv_mem->cv_q-1];
    xi_inv = cv_mem->cv_h / hsum;
    alpha0_hat = -cv_mem->cv_l[1] - xi_inv;
    for (i=cv_mem->cv_q; i >= 1; i--)
      cv_mem->cv_l[i] += cv_mem->cv_l[i-1]*xistar_inv;
  }

  cvSetTqBDF_gpu2(cv_mem, hsum, alpha0, alpha0_hat, xi_inv, xistar_inv);
}

/*
 * cvSetTqBDF
 *
 * This routine sets the test quantity array tq in the case
 * lmm == CV_BDF.
 */

void cvSetTqBDF_gpu2(CVodeMem cv_mem, realtype hsum, realtype alpha0,
                       realtype alpha0_hat, realtype xi_inv, realtype xistar_inv)
{
  realtype A1, A2, A3, A4, A5, A6;
  realtype C, Cpinv, Cppinv;

  A1 = ONE - alpha0_hat + alpha0;
  A2 = ONE + cv_mem->cv_q * A1;
  cv_mem->cv_tq[2] = SUNRabs(A1 / (alpha0 * A2));
  cv_mem->cv_tq[5] = SUNRabs(A2 * xistar_inv / (cv_mem->cv_l[cv_mem->cv_q] * xi_inv));
  if (cv_mem->cv_qwait == 1) {
    if (cv_mem->cv_q > 1) {
      C = xistar_inv / cv_mem->cv_l[cv_mem->cv_q];
      A3 = alpha0 + ONE / cv_mem->cv_q;
      A4 = alpha0_hat + xi_inv;
      Cpinv = (ONE - A4 + A3) / A3;
      cv_mem->cv_tq[1] = SUNRabs(C * Cpinv);
    }
    else cv_mem->cv_tq[1] = ONE;
    hsum += cv_mem->cv_tau[cv_mem->cv_q];
    xi_inv = cv_mem->cv_h / hsum;
    A5 = alpha0 - (ONE / (cv_mem->cv_q+1));
    A6 = alpha0_hat - xi_inv;
    Cppinv = (ONE - A6 + A5) / A2;
    cv_mem->cv_tq[3] = SUNRabs(Cppinv / (xi_inv * (cv_mem->cv_q+2) * A5));
  }
  cv_mem->cv_tq[4] = cv_mem->cv_nlscoef / cv_mem->cv_tq[2];
}


/*
 * cvHandleNFlag
 *
 * This routine takes action on the return value nflag = *nflagPtr
 * returned by cvNls, as follows:
 *
 * If cvNls succeeded in solving the nonlinear system, then
 * cvHandleNFlag returns the constant DO_ERROR_TEST, which tells cvStep
 * to perform the error test.
 *
 * If the nonlinear system was not solved successfully, then ncfn and
 * ncf = *ncfPtr are incremented and Nordsieck array zn is restored.
 *
 * If the solution of the nonlinear system failed due to an
 * unrecoverable failure by setup, we return the value CV_LSETUP_FAIL.
 *
 * If it failed due to an unrecoverable failure in solve, then we return
 * the value CV_LSOLVE_FAIL.
 *
 * If it failed due to an unrecoverable failure in rhs, then we return
 * the value CV_RHSFUNC_FAIL.
 *
 * Otherwise, a recoverable failure occurred when solving the
 * nonlinear system (cvNls returned nflag == CONV_FAIL or RHSFUNC_RECVR).
 * In this case, if ncf is now equal to maxncf or |h| = hmin,
 * we return the value CV_CONV_FAILURE (if nflag=CONV_FAIL) or
 * CV_REPTD_RHSFUNC_ERR (if nflag=RHSFUNC_RECVR).
 * If not, we set *nflagPtr = PREV_CONV_FAIL and return the value
 * PREDICT_AGAIN, telling cvStep to reattempt the step.
 *
 */

int cvHandleNFlag_gpu2(CVodeMem cv_mem, int *nflagPtr, realtype saved_t,
                         int *ncfPtr)
{
  int nflag;

  nflag = *nflagPtr;

  if (nflag == CV_SUCCESS) return(DO_ERROR_TEST);

  /* The nonlinear soln. failed; increment ncfn and restore zn */
  cv_mem->cv_ncfn++;
  cvRestore_gpu2(cv_mem, saved_t);

  /* Return if lsetup, lsolve, or rhs failed unrecoverably */
  if (nflag == CV_LSETUP_FAIL)  return(CV_LSETUP_FAIL);
  if (nflag == CV_LSOLVE_FAIL)  return(CV_LSOLVE_FAIL);
  if (nflag == CV_RHSFUNC_FAIL) return(CV_RHSFUNC_FAIL);

  /* At this point, nflag = CONV_FAIL or RHSFUNC_RECVR; increment ncf */

  (*ncfPtr)++;
  cv_mem->cv_etamax = ONE;

  /* If we had maxncf failures or |h| = hmin,
     return CV_CONV_FAILURE or CV_REPTD_RHSFUNC_ERR. */

  if ((SUNRabs(cv_mem->cv_h) <= cv_mem->cv_hmin*ONEPSM) ||
      (*ncfPtr == cv_mem->cv_maxncf)) {
    if (nflag == CONV_FAIL)     return(CV_CONV_FAILURE);
    if (nflag == RHSFUNC_RECVR) return(CV_REPTD_RHSFUNC_ERR);
  }

  /* Reduce step size; return to reattempt the step */

  cv_mem->cv_eta = SUNMAX(ETACF, cv_mem->cv_hmin / SUNRabs(cv_mem->cv_h));
  *nflagPtr = PREV_CONV_FAIL;
  cvRescale_gpu2(cv_mem);

  return(PREDICT_AGAIN);
}

/*
 * cvRestore
 *
 * This routine restores the value of tn to saved_t and undoes the
 * prediction.  After execution of cvRestore, the Nordsieck array zn has
 * the same values as before the call to cvPredict.
 */

void cvRestore_gpu2(CVodeMem cv_mem, realtype saved_t)
{
  int j, k;

  cv_mem->cv_tn = saved_t;
  for (k = 1; k <= cv_mem->cv_q; k++)
    for (j = cv_mem->cv_q; j >= k; j--)
      N_VLinearSum(ONE, cv_mem->cv_zn[j-1], -ONE,
                   cv_mem->cv_zn[j], cv_mem->cv_zn[j-1]);
  N_VScale(ONE, cv_mem->cv_last_yn, cv_mem->cv_zn[0]);
}

/*
 * cvDoErrorTest
 *
 * This routine performs the local error test.
 * The weighted local error norm dsm is loaded into *dsmPtr, and
 * the test dsm ?<= 1 is made.
 *
 * If the test passes, cvDoErrorTest returns CV_SUCCESS.
 *
 * If the test fails, we undo the step just taken (call cvRestore) and
 *
 *   - if maxnef error test failures have occurred or if SUNRabs(h) = hmin,
 *     we return CV_ERR_FAILURE.
 *
 *   - if more than MXNEF1 error test failures have occurred, an order
 *     reduction is forced. If already at order 1, restart by reloading
 *     zn from scratch. If f() fails we return either CV_RHSFUNC_FAIL
 *     or CV_UNREC_RHSFUNC_ERR (no recovery is possible at this stage).
 *
 *   - otherwise, set *nflagPtr to PREV_ERR_FAIL, and return TRY_AGAIN.
 *
 */
booleantype cvDoErrorTest_gpu2(CVodeMem cv_mem, int *nflagPtr,
                                 realtype saved_t, int *nefPtr, realtype *dsmPtr)
{
  realtype dsm;
  realtype min_val;
  int retval;

  /* Find the minimum concentration and if it's small and negative, make it
   * positive */
  N_VLinearSum(cv_mem->cv_l[0], cv_mem->cv_acor, ONE, cv_mem->cv_zn[0],
               cv_mem->cv_ftemp);
  min_val = N_VMin(cv_mem->cv_ftemp);
  if (min_val < ZERO && min_val > -CAMP_TINY) {
    N_VAbs(cv_mem->cv_ftemp, cv_mem->cv_ftemp);
    N_VLinearSum(-cv_mem->cv_l[0], cv_mem->cv_acor, ONE, cv_mem->cv_ftemp,
                 cv_mem->cv_zn[0]);
    min_val = ZERO;
  }

  dsm = cv_mem->cv_acnrm * cv_mem->cv_tq[2];

  /* If est. local error norm dsm passes test and there are no negative values,
   * return CV_SUCCESS */
  *dsmPtr = dsm;
  if (dsm <= ONE && min_val >= ZERO) return(CV_SUCCESS);

  /* Test failed; increment counters, set nflag, and restore zn array */
  (*nefPtr)++;
  cv_mem->cv_netf++;
  *nflagPtr = PREV_ERR_FAIL;
  cvRestore_gpu2(cv_mem, saved_t);

  /* At maxnef failures or |h| = hmin, return CV_ERR_FAILURE */
  if ((SUNRabs(cv_mem->cv_h) <= cv_mem->cv_hmin*ONEPSM) ||
      (*nefPtr == cv_mem->cv_maxnef)) return(CV_ERR_FAILURE);

  /* Set etamax = 1 to prevent step size increase at end of this step */
  cv_mem->cv_etamax = ONE;

  /* Set h ratio eta from dsm, rescale, and return for retry of step */
  if (*nefPtr <= MXNEF1) {
    cv_mem->cv_eta = ONE / (SUNRpowerR(BIAS2*dsm,ONE/cv_mem->cv_L) + ADDON);
    cv_mem->cv_eta = SUNMAX(ETAMIN, SUNMAX(cv_mem->cv_eta,
                                           cv_mem->cv_hmin / SUNRabs(cv_mem->cv_h)));
    if (*nefPtr >= SMALL_NEF) cv_mem->cv_eta = SUNMIN(cv_mem->cv_eta, ETAMXF);
    cvRescale_gpu2(cv_mem);
    return(TRY_AGAIN);
  }

  /* After MXNEF1 failures, force an order reduction and retry step */
  if (cv_mem->cv_q > 1) {
    cv_mem->cv_eta = SUNMAX(ETAMIN, cv_mem->cv_hmin / SUNRabs(cv_mem->cv_h));

    //cvAdjustOrder_gpu2(cv_mem,-1);
    cvDecreaseBDF_gpu2(cv_mem);

    cv_mem->cv_L = cv_mem->cv_q;
    cv_mem->cv_q--;
    cv_mem->cv_qwait = cv_mem->cv_L;
    cvRescale_gpu2(cv_mem);
    return(TRY_AGAIN);
  }

  /* If already at order 1, restart: reload zn from scratch */

  cv_mem->cv_eta = SUNMAX(ETAMIN, cv_mem->cv_hmin / SUNRabs(cv_mem->cv_h));
  cv_mem->cv_h *= cv_mem->cv_eta;
  cv_mem->cv_next_h = cv_mem->cv_h;
  cv_mem->cv_hscale = cv_mem->cv_h;
  cv_mem->cv_qwait = LONG_WAIT;
  cv_mem->cv_nscon = 0;


  //retval = cv_mem->cv_f(cv_mem->cv_tn, cv_mem->cv_zn[0],
  //                      cv_mem->cv_tempv, cv_mem->cv_user_data);
  retval = f(cv_mem->cv_tn, cv_mem->cv_zn[0],cv_mem->cv_tempv, cv_mem->cv_user_data);

  cv_mem->cv_nfe++;
  if (retval < 0)  return(CV_RHSFUNC_FAIL);
  if (retval > 0)  return(CV_UNREC_RHSFUNC_ERR);

  N_VScale(cv_mem->cv_h, cv_mem->cv_tempv, cv_mem->cv_zn[1]);

  return(TRY_AGAIN);
}

/*
 * -----------------------------------------------------------------
 * Functions called after succesful step
 * -----------------------------------------------------------------
 */

/*
 * cvCompleteStep
 *
 * This routine performs various update operations when the solution
 * to the nonlinear system has passed the local error test.
 * We increment the step counter nst, record the values hu and qu,
 * update the tau array, and apply the corrections to the zn array.
 * The tau[i] are the last q values of h, with tau[1] the most recent.
 * The counter qwait is decremented, and if qwait == 1 (and q < qmax)
 * we save acor and cv_mem->cv_tq[5] for a possible order increase.
 */
void cvCompleteStep_gpu2(CVodeMem cv_mem)
{
  int i, j;

  cv_mem->cv_nst++;
  cv_mem->cv_nscon++;
  cv_mem->cv_hu = cv_mem->cv_h;
  cv_mem->cv_qu = cv_mem->cv_q;

  for (i=cv_mem->cv_q; i >= 2; i--)  cv_mem->cv_tau[i] = cv_mem->cv_tau[i-1];
  if ((cv_mem->cv_q==1) && (cv_mem->cv_nst > 1))
    cv_mem->cv_tau[2] = cv_mem->cv_tau[1];
  cv_mem->cv_tau[1] = cv_mem->cv_h;

  /* Apply correction to column j of zn: l_j * Delta_n */
  for (j=0; j <= cv_mem->cv_q; j++)
    N_VLinearSum(cv_mem->cv_l[j], cv_mem->cv_acor, ONE,
                 cv_mem->cv_zn[j], cv_mem->cv_zn[j]);
  cv_mem->cv_qwait--;
  if ((cv_mem->cv_qwait == 1) && (cv_mem->cv_q != cv_mem->cv_qmax)) {
    N_VScale(ONE, cv_mem->cv_acor, cv_mem->cv_zn[cv_mem->cv_qmax]);
    cv_mem->cv_saved_tq5 = cv_mem->cv_tq[5];
    cv_mem->cv_indx_acor = cv_mem->cv_qmax;
  }
}


/*
 * cvPrepareNextStep
 *
 * This routine handles the setting of stepsize and order for the
 * next step -- hprime and qprime.  Along with hprime, it sets the
 * ratio eta = hprime/h.  It also updates other state variables
 * related to a change of step size or order.
 */
void cvPrepareNextStep_gpu2(CVodeMem cv_mem, realtype dsm)
{
  /* If etamax = 1, defer step size or order changes */
  if (cv_mem->cv_etamax == ONE) {
    cv_mem->cv_qwait = SUNMAX(cv_mem->cv_qwait, 2);
    cv_mem->cv_qprime = cv_mem->cv_q;
    cv_mem->cv_hprime = cv_mem->cv_h;
    cv_mem->cv_eta = ONE;
    return;
  }

  /* etaq is the ratio of new to old h at the current order */
  cv_mem->cv_etaq = ONE /(SUNRpowerR(BIAS2*dsm,ONE/cv_mem->cv_L) + ADDON);

  /* If no order change, adjust eta and acor in cvSetEta and return */
  if (cv_mem->cv_qwait != 0) {
    cv_mem->cv_eta = cv_mem->cv_etaq;
    cv_mem->cv_qprime = cv_mem->cv_q;
    cvSetEta_gpu2(cv_mem);
    return;
  }

  /* If qwait = 0, consider an order change.   etaqm1 and etaqp1 are
     the ratios of new to old h at orders q-1 and q+1, respectively.
     cvChooseEta selects the largest; cvSetEta adjusts eta and acor */
  cv_mem->cv_qwait = 2;

  //cv_mem->cv_etaqm1 = cvComputeEtaqm1_gpu2(cv_mem);
  //compute cv_etaqm1
  realtype ddn;
  cv_mem->cv_etaqm1 = ZERO;
  if (cv_mem->cv_q > 1) {
    ddn = N_VWrmsNorm(cv_mem->cv_zn[cv_mem->cv_q], cv_mem->cv_ewt) * cv_mem->cv_tq[1];
    cv_mem->cv_etaqm1 = ONE/(SUNRpowerR(BIAS1*ddn, ONE/cv_mem->cv_q) + ADDON);
  }

  //cv_mem->cv_etaqp1 = cvComputeEtaqp1_gpu2(cv_mem);
  //compute cv_etaqp1
  realtype dup, cquot;
  cv_mem->cv_etaqp1 = ZERO;
  if (cv_mem->cv_q != cv_mem->cv_qmax && cv_mem->cv_saved_tq5 != ZERO) {
    //if (cv_mem->cv_saved_tq5 != ZERO) return(cv_mem->cv_etaqp1);
    cquot = (cv_mem->cv_tq[5] / cv_mem->cv_saved_tq5) *
            SUNRpowerI(cv_mem->cv_h/cv_mem->cv_tau[2], cv_mem->cv_L);
    N_VLinearSum(-cquot, cv_mem->cv_zn[cv_mem->cv_qmax], ONE,
                 cv_mem->cv_acor, cv_mem->cv_tempv);
    dup = N_VWrmsNorm(cv_mem->cv_tempv, cv_mem->cv_ewt) * cv_mem->cv_tq[3];
    cv_mem->cv_etaqp1 = ONE / (SUNRpowerR(BIAS3*dup, ONE/(cv_mem->cv_L+1)) + ADDON);
  }

  cvChooseEta_gpu2(cv_mem);
  cvSetEta_gpu2(cv_mem);
}

/*
 * cvSetEta
 *
 * This routine adjusts the value of eta according to the various
 * heuristic limits and the optional input hmax.
 */

void cvSetEta_gpu2(CVodeMem cv_mem)
{

  /* If eta below the threshhold THRESH, reject a change of step size */
  if (cv_mem->cv_eta < THRESH) {
    cv_mem->cv_eta = ONE;
    cv_mem->cv_hprime = cv_mem->cv_h;
  } else {
    /* Limit eta by etamax and hmax, then set hprime */
    cv_mem->cv_eta = SUNMIN(cv_mem->cv_eta, cv_mem->cv_etamax);
    cv_mem->cv_eta /= SUNMAX(ONE, SUNRabs(cv_mem->cv_h)*cv_mem->cv_hmax_inv*cv_mem->cv_eta);
    cv_mem->cv_hprime = cv_mem->cv_h * cv_mem->cv_eta;
    if (cv_mem->cv_qprime < cv_mem->cv_q) cv_mem->cv_nscon = 0;
  }
}

/*
 * cvChooseEta
 * Given etaqm1, etaq, etaqp1 (the values of eta for qprime =
 * q - 1, q, or q + 1, respectively), this routine chooses the
 * maximum eta value, sets eta to that value, and sets qprime to the
 * corresponding value of q.  If there is a tie, the preference
 * order is to (1) keep the same order, then (2) decrease the order,
 * and finally (3) increase the order.  If the maximum eta value
 * is below the threshhold THRESH, the order is kept unchanged and
 * eta is set to 1.
 */
void cvChooseEta_gpu2(CVodeMem cv_mem)
{
  realtype etam;

  etam = SUNMAX(cv_mem->cv_etaqm1, SUNMAX(cv_mem->cv_etaq, cv_mem->cv_etaqp1));

  if (etam < THRESH) {
    cv_mem->cv_eta = ONE;
    cv_mem->cv_qprime = cv_mem->cv_q;
    return;
  }

  if (etam == cv_mem->cv_etaq) {

    cv_mem->cv_eta = cv_mem->cv_etaq;
    cv_mem->cv_qprime = cv_mem->cv_q;

  } else if (etam == cv_mem->cv_etaqm1) {

    cv_mem->cv_eta = cv_mem->cv_etaqm1;
    cv_mem->cv_qprime = cv_mem->cv_q - 1;

  } else {

    cv_mem->cv_eta = cv_mem->cv_etaqp1;
    cv_mem->cv_qprime = cv_mem->cv_q + 1;

    if (cv_mem->cv_lmm == CV_BDF) {
      /*
       * Store Delta_n in zn[qmax] to be used in order increase
       *
       * This happens at the last step of order q before an increase
       * to order q+1, so it represents Delta_n in the ELTE at q+1
       */

      N_VScale(ONE, cv_mem->cv_acor, cv_mem->cv_zn[cv_mem->cv_qmax]);

    }
  }
}

/*
 * -----------------------------------------------------------------
 * Functions for BDF Stability Limit Detection
 * -----------------------------------------------------------------
 */

/*
 * cvBDFStab
 *
 * This routine handles the BDF Stability Limit Detection Algorithm
 * STALD.  It is called if lmm = CV_BDF and the SLDET option is on.
 * If the order is 3 or more, the required norm data is saved.
 * If a decision to reduce order has not already been made, and
 * enough data has been saved, cvSLdet is called.  If it signals
 * a stability limit violation, the order is reduced, and the step
 * size is reset accordingly.
 */

void cvBDFStab_gpu2(CVodeMem cv_mem)
{
  int i,k, ldflag, factorial;
  realtype sq, sqm1, sqm2;

  /* If order is 3 or greater, then save scaled derivative data,
     push old data down in i, then add current values to top.    */

  if (cv_mem->cv_q >= 3) {
    for (k = 1; k <= 3; k++)
      for (i = 5; i >= 2; i--)
        cv_mem->cv_ssdat[i][k] = cv_mem->cv_ssdat[i-1][k];
    factorial = 1;
    for (i = 1; i <= cv_mem->cv_q-1; i++) factorial *= i;
    sq = factorial * cv_mem->cv_q * (cv_mem->cv_q+1) *
         cv_mem->cv_acnrm / SUNMAX(cv_mem->cv_tq[5],TINY);
    sqm1 = factorial * cv_mem->cv_q *
           N_VWrmsNorm(cv_mem->cv_zn[cv_mem->cv_q], cv_mem->cv_ewt);
    sqm2 = factorial * N_VWrmsNorm(cv_mem->cv_zn[cv_mem->cv_q-1], cv_mem->cv_ewt);
    cv_mem->cv_ssdat[1][1] = sqm2*sqm2;
    cv_mem->cv_ssdat[1][2] = sqm1*sqm1;
    cv_mem->cv_ssdat[1][3] = sq*sq;
  }


  if (cv_mem->cv_qprime >= cv_mem->cv_q) {

    /* If order is 3 or greater, and enough ssdat has been saved,
       nscon >= q+5, then call stability limit detection routine.  */

    if ( (cv_mem->cv_q >= 3) && (cv_mem->cv_nscon >= cv_mem->cv_q+5) ) {
      ldflag = cvSLdet_gpu2(cv_mem);
      if (ldflag > 3) {
        /* A stability limit violation is indicated by
           a return flag of 4, 5, or 6.
           Reduce new order.                     */
        cv_mem->cv_qprime = cv_mem->cv_q-1;
        cv_mem->cv_eta = cv_mem->cv_etaqm1;
        cv_mem->cv_eta = SUNMIN(cv_mem->cv_eta,cv_mem->cv_etamax);
        cv_mem->cv_eta = cv_mem->cv_eta /
                         SUNMAX(ONE,SUNRabs(cv_mem->cv_h)*cv_mem->cv_hmax_inv*cv_mem->cv_eta);
        cv_mem->cv_hprime = cv_mem->cv_h*cv_mem->cv_eta;
        cv_mem->cv_nor = cv_mem->cv_nor + 1;
      }
    }
  }
  else {
    /* Otherwise, let order increase happen, and
       reset stability limit counter, nscon.     */
    cv_mem->cv_nscon = 0;
  }
}

/*
 * cvSLdet
 *
 * This routine detects stability limitation using stored scaled
 * derivatives data. cvSLdet returns the magnitude of the
 * dominate characteristic root, rr. The presence of a stability
 * limit is indicated by rr > "something a little less then 1.0",
 * and a positive kflag. This routine should only be called if
 * order is greater than or equal to 3, and data has been collected
 * for 5 time steps.
 *
 * Returned values:
 *    kflag = 1 -> Found stable characteristic root, normal matrix case
 *    kflag = 2 -> Found stable characteristic root, quartic solution
 *    kflag = 3 -> Found stable characteristic root, quartic solution,
 *                 with Newton correction
 *    kflag = 4 -> Found stability violation, normal matrix case
 *    kflag = 5 -> Found stability violation, quartic solution
 *    kflag = 6 -> Found stability violation, quartic solution,
 *                 with Newton correction
 *
 *    kflag < 0 -> No stability limitation,
 *                 or could not compute limitation.
 *
 *    kflag = -1 -> Min/max ratio of ssdat too small.
 *    kflag = -2 -> For normal matrix case, vmax > vrrt2*vrrt2
 *    kflag = -3 -> For normal matrix case, The three ratios
 *                  are inconsistent.
 *    kflag = -4 -> Small coefficient prevents elimination of quartics.
 *    kflag = -5 -> R value from quartics not consistent.
 *    kflag = -6 -> No corrected root passes test on qk values
 *    kflag = -7 -> Trouble solving for sigsq.
 *    kflag = -8 -> Trouble solving for B, or R via B.
 *    kflag = -9 -> R via sigsq[k] disagrees with R from data.
 */

int cvSLdet_gpu2(CVodeMem cv_mem)
{
  int i, k, j, it, kmin = 0, kflag = 0;
  realtype rat[5][4], rav[4], qkr[4], sigsq[4], smax[4], ssmax[4];
  realtype drr[4], rrc[4],sqmx[4], qjk[4][4], vrat[5], qc[6][4], qco[6][4];
  realtype rr, rrcut, vrrtol, vrrt2, sqtol, rrtol;
  realtype smink, smaxk, sumrat, sumrsq, vmin, vmax, drrmax, adrr;
  realtype tem, sqmax, saqk, qp, s, sqmaxk, saqj, sqmin;
  realtype rsa, rsb, rsc, rsd, rd1a, rd1b, rd1c;
  realtype rd2a, rd2b, rd3a, cest1, corr1;
  realtype ratp, ratm, qfac1, qfac2, bb, rrb;

  /* The following are cutoffs and tolerances used by this routine */

  rrcut  = RCONST(0.98);
  vrrtol = RCONST(1.0e-4);
  vrrt2  = RCONST(5.0e-4);
  sqtol  = RCONST(1.0e-3);
  rrtol  = RCONST(1.0e-2);

  rr = ZERO;

  /*  Index k corresponds to the degree of the interpolating polynomial. */
  /*      k = 1 -> q-1          */
  /*      k = 2 -> q            */
  /*      k = 3 -> q+1          */

  /*  Index i is a backward-in-time index, i = 1 -> current time, */
  /*      i = 2 -> previous step, etc    */

  /* get maxima, minima, and variances, and form quartic coefficients  */

  for (k=1; k<=3; k++) {
    smink = cv_mem->cv_ssdat[1][k];
    smaxk = ZERO;

    for (i=1; i<=5; i++) {
      smink = SUNMIN(smink,cv_mem->cv_ssdat[i][k]);
      smaxk = SUNMAX(smaxk,cv_mem->cv_ssdat[i][k]);
    }

    if (smink < TINY*smaxk) {
      kflag = -1;
      return(kflag);
    }
    smax[k] = smaxk;
    ssmax[k] = smaxk*smaxk;

    sumrat = ZERO;
    sumrsq = ZERO;
    for (i=1; i<=4; i++) {
      rat[i][k] = cv_mem->cv_ssdat[i][k]/cv_mem->cv_ssdat[i+1][k];
      sumrat = sumrat + rat[i][k];
      sumrsq = sumrsq + rat[i][k]*rat[i][k];
    }
    rav[k] = FOURTH*sumrat;
    vrat[k] = SUNRabs(FOURTH*sumrsq - rav[k]*rav[k]);

    qc[5][k] = cv_mem->cv_ssdat[1][k] * cv_mem->cv_ssdat[3][k] -
               cv_mem->cv_ssdat[2][k] * cv_mem->cv_ssdat[2][k];
    qc[4][k] = cv_mem->cv_ssdat[2][k] * cv_mem->cv_ssdat[3][k] -
               cv_mem->cv_ssdat[1][k] * cv_mem->cv_ssdat[4][k];
    qc[3][k] = ZERO;
    qc[2][k] = cv_mem->cv_ssdat[2][k] * cv_mem->cv_ssdat[5][k] -
               cv_mem->cv_ssdat[3][k] * cv_mem->cv_ssdat[4][k];
    qc[1][k] = cv_mem->cv_ssdat[4][k] * cv_mem->cv_ssdat[4][k] -
               cv_mem->cv_ssdat[3][k] * cv_mem->cv_ssdat[5][k];

    for (i=1; i<=5; i++)
      qco[i][k] = qc[i][k];

  }                            /* End of k loop */

  /* Isolate normal or nearly-normal matrix case. The three quartics will
     have a common or nearly-common root in this case.
     Return a kflag = 1 if this procedure works. If the three roots
     differ more than vrrt2, return error kflag = -3.    */

  vmin = SUNMIN(vrat[1],SUNMIN(vrat[2],vrat[3]));
  vmax = SUNMAX(vrat[1],SUNMAX(vrat[2],vrat[3]));

  if (vmin < vrrtol*vrrtol) {

    if (vmax > vrrt2*vrrt2) {
      kflag = -2;
      return(kflag);
    } else {
      rr = (rav[1] + rav[2] + rav[3])/THREE;
      drrmax = ZERO;
      for (k = 1;k<=3;k++) {
        adrr = SUNRabs(rav[k] - rr);
        drrmax = SUNMAX(drrmax, adrr);
      }
      if (drrmax > vrrt2) {kflag = -3; return(kflag);}

      kflag = 1;

      /*  can compute charactistic root, drop to next section   */
    }

  } else {

    /* use the quartics to get rr. */

    if (SUNRabs(qco[1][1]) < TINY*ssmax[1]) {
      kflag = -4;
      return(kflag);
    }

    tem = qco[1][2]/qco[1][1];
    for (i=2; i<=5; i++) {
      qco[i][2] = qco[i][2] - tem*qco[i][1];
    }

    qco[1][2] = ZERO;
    tem = qco[1][3]/qco[1][1];
    for (i=2; i<=5; i++) {
      qco[i][3] = qco[i][3] - tem*qco[i][1];
    }
    qco[1][3] = ZERO;

    if (SUNRabs(qco[2][2]) < TINY*ssmax[2]) {
      kflag = -4;
      return(kflag);
    }

    tem = qco[2][3]/qco[2][2];
    for (i=3; i<=5; i++) {
      qco[i][3] = qco[i][3] - tem*qco[i][2];
    }

    if (SUNRabs(qco[4][3]) < TINY*ssmax[3]) {
      kflag = -4;
      return(kflag);
    }

    rr = -qco[5][3]/qco[4][3];

    if (rr < TINY || rr > HUNDRED) {
      kflag = -5;
      return(kflag);
    }

    for (k=1; k<=3; k++)
      qkr[k] = qc[5][k] + rr*(qc[4][k] + rr*rr*(qc[2][k] + rr*qc[1][k]));

    sqmax = ZERO;
    for (k=1; k<=3; k++) {
      saqk = SUNRabs(qkr[k])/ssmax[k];
      if (saqk > sqmax) sqmax = saqk;
    }

    if (sqmax < sqtol) {
      kflag = 2;

      /*  can compute charactistic root, drop to "given rr,etc"   */

    } else {

      /* do Newton corrections to improve rr.  */

      for (it=1; it<=3; it++) {
        for (k=1; k<=3; k++) {
          qp = qc[4][k] + rr*rr*(THREE*qc[2][k] + rr*FOUR*qc[1][k]);
          drr[k] = ZERO;
          if (SUNRabs(qp) > TINY*ssmax[k]) drr[k] = -qkr[k]/qp;
          rrc[k] = rr + drr[k];
        }

        for (k=1; k<=3; k++) {
          s = rrc[k];
          sqmaxk = ZERO;
          for (j=1; j<=3; j++) {
            qjk[j][k] = qc[5][j] + s*(qc[4][j] + s*s*(qc[2][j] + s*qc[1][j]));
            saqj = SUNRabs(qjk[j][k])/ssmax[j];
            if (saqj > sqmaxk) sqmaxk = saqj;
          }
          sqmx[k] = sqmaxk;
        }

        sqmin = sqmx[1] + ONE;
        for (k=1; k<=3; k++) {
          if (sqmx[k] < sqmin) {
            kmin = k;
            sqmin = sqmx[k];
          }
        }
        rr = rrc[kmin];

        if (sqmin < sqtol) {
          kflag = 3;
          /*  can compute charactistic root   */
          /*  break out of Newton correction loop and drop to "given rr,etc" */
          break;
        } else {
          for (j=1; j<=3; j++) {
            qkr[j] = qjk[j][kmin];
          }
        }
      } /*  end of Newton correction loop  */

      if (sqmin > sqtol) {
        kflag = -6;
        return(kflag);
      }
    } /*  end of if (sqmax < sqtol) else   */
  } /*  end of if (vmin < vrrtol*vrrtol) else, quartics to get rr. */

  /* given rr, find sigsq[k] and verify rr.  */
  /* All positive kflag drop to this section  */

  for (k=1; k<=3; k++) {
    rsa = cv_mem->cv_ssdat[1][k];
    rsb = cv_mem->cv_ssdat[2][k]*rr;
    rsc = cv_mem->cv_ssdat[3][k]*rr*rr;
    rsd = cv_mem->cv_ssdat[4][k]*rr*rr*rr;
    rd1a = rsa - rsb;
    rd1b = rsb - rsc;
    rd1c = rsc - rsd;
    rd2a = rd1a - rd1b;
    rd2b = rd1b - rd1c;
    rd3a = rd2a - rd2b;

    if (SUNRabs(rd1b) < TINY*smax[k]) {
      kflag = -7;
      return(kflag);
    }

    cest1 = -rd3a/rd1b;
    if (cest1 < TINY || cest1 > FOUR) {
      kflag = -7;
      return(kflag);
    }
    corr1 = (rd2b/cest1)/(rr*rr);
    sigsq[k] = cv_mem->cv_ssdat[3][k] + corr1;
  }

  if (sigsq[2] < TINY) {
    kflag = -8;
    return(kflag);
  }

  ratp = sigsq[3]/sigsq[2];
  ratm = sigsq[1]/sigsq[2];
  qfac1 = FOURTH*(cv_mem->cv_q*cv_mem->cv_q - ONE);
  qfac2 = TWO/(cv_mem->cv_q - ONE);
  bb = ratp*ratm - ONE - qfac1*ratp;
  tem = ONE - qfac2*bb;

  if (SUNRabs(tem) < TINY) {
    kflag = -8;
    return(kflag);
  }

  rrb = ONE/tem;

  if (SUNRabs(rrb - rr) > rrtol) {
    kflag = -9;
    return(kflag);
  }

  /* Check to see if rr is above cutoff rrcut  */
  if (rr > rrcut) {
    if (kflag == 1) kflag = 4;
    if (kflag == 2) kflag = 5;
    if (kflag == 3) kflag = 6;
  }

  /* All positive kflag returned at this point  */

  return(kflag);

}

/*
 * cvEwtSetSV
 *
 * This routine sets ewt as decribed above in the case tol_type = CV_SV.
 * It tests for non-positive components before inverting. cvEwtSetSV
 * returns 0 if ewt is successfully set to a positive vector
 * and -1 otherwise. In the latter case, ewt is considered undefined.
 */
int cvEwtSetSV_gpu2(CVodeMem cv_mem, N_Vector cv_ewt, N_Vector weight)
{
  N_VAbs(cv_ewt, cv_mem->cv_tempv);
  N_VLinearSum(cv_mem->cv_reltol, cv_mem->cv_tempv, ONE,
               cv_mem->cv_Vabstol, cv_mem->cv_tempv);
  if (N_VMin(cv_mem->cv_tempv) <= ZERO) return(-1);
  N_VInv(cv_mem->cv_tempv, weight);
  return(0);
}


__global__ void cudaGlobalVWRMS_Norm(double *g_idata1, double *g_idata2, double *g_odata, int n, int n_shr)
{
  extern __shared__ double sdata[];
  unsigned int tid = threadIdx.x;
  //unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;
  unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

  if (tid == 0){
    for (int j=0; j<n_shr; j++)
      sdata[j] = 0.;
  }

/*
  double mySum = (i < n) ? g_idata1[i]*g_idata1[i]*g_idata2[i]*g_idata2[i] : 0;

  if (i + blockDim.x < n)
    mySum += g_idata1[i+blockDim.x]*g_idata1[i+blockDim.x]*g_idata2[i+blockDim.x]*g_idata2[i+blockDim.x];
*/

  __syncthreads();
  double mySum=g_idata1[i]*g_idata1[i]*g_idata2[i]*g_idata2[i];
  sdata[tid] = mySum;
  __syncthreads();

  for (unsigned int s=n_shr/2; s>0; s>>=1)
  {
    if (tid < s)
      sdata[tid] = mySum = mySum + sdata[tid + s];

    __syncthreads();
  }

  //if (tid == 0) g_odata[blockIdx.x] = sdata[0];
  g_odata[0] = sqrt(sdata[0]/n);
  __syncthreads();
}


int cvNlsNewton_gpu2(SolverData *sd, CVodeMem cv_mem, int nflag)
{
  itsolver *bicg = &(sd->bicg);
  ModelData *md = &(sd->model_data);
  ModelDataGPU *mGPU = &sd->mGPU;
  N_Vector vtemp1, vtemp2, vtemp3;
  int convfail, retval, ier;
  booleantype callSetup;

#ifdef CAMP_DEBUG_GPU
  //clock_t start;
  //start=clock();
#endif

  cudaMemcpy(mGPU->cv_tq,cv_mem->cv_tq,5*sizeof(double),cudaMemcpyHostToDevice);
  //double *acor_init = NV_DATA_S(cv_mem->cv_acor_init); //user-supplied value to improve guesses for zn(0)
  double *acor = NV_DATA_S(cv_mem->cv_acor);
  double *cv_y = NV_DATA_S(cv_mem->cv_y);
  double *tempv = NV_DATA_S(cv_mem->cv_tempv);
  double *ftemp = NV_DATA_S(cv_mem->cv_ftemp);

  double *J_deriv = N_VGetArrayPointer(md->J_deriv);

  int znUsedOnNewtonIt=2;//Only used zn[0] and zn[1] //0.01s
  for(int i=0; i<znUsedOnNewtonIt; i++){//cv_qmax+1
    double *zn = NV_DATA_S(cv_mem->cv_zn[i]);
    cudaMemcpy((i*mGPU->nrows+mGPU->dzn),zn,mGPU->nrows*sizeof(double),cudaMemcpyHostToDevice);
  }

#ifdef CAMP_DEBUG_GPU
  //bicg->counterSendInit++;
#endif

  /* Set flag convfail, input to lsetup for its evaluation decision */
  convfail = ((nflag == FIRST_CALL) || (nflag == PREV_ERR_FAIL)) ?
             CV_NO_FAILURES : CV_FAIL_OTHER;

  /* Decide whether or not to call setup routine (if one exists) */
  if (cv_mem->cv_lsetup) {
    callSetup = (nflag == PREV_CONV_FAIL) || (nflag == PREV_ERR_FAIL) ||
                (cv_mem->cv_nst == 0) ||
                (cv_mem->cv_nst >= cv_mem->cv_nstlp + MSBP) ||
                (SUNRabs(cv_mem->cv_gamrat-ONE) > DGMAX);
  } else {
    cv_mem->cv_crate = ONE;
    callSetup = SUNFALSE;
  }

  /* Call a user-supplied function to improve guesses for zn(0), if one exists */
  //if not, set to zero
  //N_VConst(ZERO, cv_mem->cv_acor_init);
  //cudaMemcpyDToGpu(acor_init, bicg->cv_acor_init, mGPU->nrows);

  if (cv_mem->cv_ghfun) {

  //all are cpu pointers and gpu pointers are dftemp etc
  N_VLinearSum(ONE, cv_mem->cv_zn[0], -ONE, cv_mem->cv_last_yn, cv_mem->cv_ftemp);
  retval = cv_mem->cv_ghfun(cv_mem->cv_tn, cv_mem->cv_h, cv_mem->cv_zn[0],
                            cv_mem->cv_last_yn, cv_mem->cv_ftemp, cv_mem->cv_user_data,
                            cv_mem->cv_tempv, cv_mem->cv_acor_init);
  if (retval<0) return(RHSFUNC_RECVR);
  }

  cudaMemcpy(mGPU->cv_acor,acor,mGPU->nrows*sizeof(double),cudaMemcpyHostToDevice);
  cudaMemcpy(mGPU->dtempv,tempv,mGPU->nrows*sizeof(double),cudaMemcpyHostToDevice);
  cudaMemcpy(mGPU->dftemp,ftemp,mGPU->nrows*sizeof(double),cudaMemcpyHostToDevice);

  //remove temps, not used in jac
  vtemp1 = cv_mem->cv_acor;  /* rename acor as vtemp1 for readability  */
  vtemp2 = cv_mem->cv_acor;  /* rename y as vtemp2 for readability     */
  vtemp3 = cv_mem->cv_acor;  /* rename tempv as vtemp3 for readability */

  /* Looping point for the solution of the nonlinear system.
     Evaluate f at the predicted y, call lsetup if indicated, and
     call cvNewtonIteration for the Newton iteration itself.      */
  for(;;) {

    /* Load prediction into y vector */
    //N_VLinearSum(ONE, cv_mem->cv_zn[0], ONE, cv_mem->cv_acor_init, cv_mem->cv_y);
    //gpu_zaxpby(1.0, mGPU->dzn, 1.0, bicg->cv_acor_init, mGPU->dcv_y, mGPU->nrows, mGPU->blocks, mGPU->threads);
    //gpu_yequalsx is not thread safe (need cuda_devicesync previously!)
    cudaDeviceSynchronize();
    gpu_yequalsx(mGPU->dcv_y,mGPU->dzn, mGPU->nrows, mGPU->blocks, mGPU->threads);//Consider acor_init=0
    cudaDeviceSynchronize();

    //copy cv_y to enable debug on cpu
    cudaMemcpy(cv_y,mGPU->dcv_y,mGPU->nrows*sizeof(double),cudaMemcpyDeviceToHost);
    //retval = cv_mem->cv_f(cv_mem->cv_tn, cv_mem->cv_y,
    //                      cv_mem->cv_ftemp, cv_mem->cv_user_data);
    //int f(realtype t, N_Vector y, N_Vector deriv, void *solver_data)

#ifdef CAMP_DEBUG_GPU
    //start=clock();
    cudaEventRecord(bicg->startDerivNewton);
#endif

    /*if(sd->counterDerivCPU<=5){
      for (int i = 0; i < NV_LENGTH_S(cv_mem->cv_y); i++) {
        //printf("(%d) %-le ", i + 1, NV_DATA_S(deriv)[i]);
        if(cv_y[i]!=md->deriv_aux[i]) {
          printf("(%d) dy %-le y %-le\n", i + 1, md->deriv_aux[i], cv_y[i]);
        }
      }
    }*/

    //retval = f(cv_mem->cv_tn, cv_mem->cv_y, cv_mem->cv_ftemp, cv_mem->cv_user_data);
    retval = f_gpu(cv_mem->cv_tn, cv_mem->cv_y, cv_mem->cv_ftemp, cv_mem->cv_user_data);

#ifdef CAMP_DEBUG_GPU
    cudaEventRecord(bicg->stopDerivNewton);
    cudaEventSynchronize(bicg->stopDerivNewton);
    float msDerivNewton = 0.0;
    cudaEventElapsedTime(&msDerivNewton, bicg->startDerivNewton, bicg->stopDerivNewton);
    bicg->timeDerivNewton+= msDerivNewton;

    //bicg->timeDerivNewton+= clock() - start;
    bicg->counterDerivNewton++;
#endif

    //Not needed because mGPU->dftemp=md->deriv_data_gpu;
    //cudaMemcpy(cv_ftemp_data,mGPU->dftemp,mGPU->nrows*sizeof(double),cudaMemcpyDeviceToHost);

    if (retval < 0) return(CV_RHSFUNC_FAIL);
    if (retval > 0) return(RHSFUNC_RECVR);

    cv_mem->cv_nfe++;
    if (callSetup)
    {

#ifdef CAMP_DEBUG_GPU
      //start=clock();
      cudaEventRecord(bicg->startLinSolSetup);
#endif

      ier = linsolsetup_gpu2(sd, cv_mem, convfail, vtemp1, vtemp2, vtemp3);

#ifdef CAMP_DEBUG_GPU
    cudaEventRecord(bicg->stopLinSolSetup);

    cudaEventSynchronize(bicg->stopLinSolSetup);
    float msLinSolSetup = 0.0;
    cudaEventElapsedTime(&msLinSolSetup, bicg->startLinSolSetup, bicg->stopLinSolSetup);
    bicg->timeLinSolSetup+= msLinSolSetup;

    //bicg->timeLinSolSetup+= clock() - start;
    bicg->counterLinSolSetup++;
#endif

      cv_mem->cv_nsetups++;
      callSetup = SUNFALSE;
      cv_mem->cv_gamrat = cv_mem->cv_crate = ONE;
      cv_mem->cv_gammap = cv_mem->cv_gamma;
      cv_mem->cv_nstlp = cv_mem->cv_nst;
      // Return if lsetup failed
      if (ier < 0) return(CV_LSETUP_FAIL);
      if (ier > 0) return(CONV_FAIL);
    }

    // Set acor to the initial guess for adjustments to the y vector
    //N_VScale(ONE, cv_mem->cv_acor_init, cv_mem->cv_acor);
    //gpu_yequalsx(mGPU->cv_acor, bicg->cv_acor_init, mGPU->nrows, mGPU->blocks, mGPU->threads);
    cudaMemset(mGPU->cv_acor, 0.0, mGPU->nrows*sizeof(double));

#ifdef CAMP_DEBUG_GPU
    cudaEventRecord(bicg->startLinSolSolve);
#endif

    // Do the Newton iteration
    //ier = cvNewtonIteration(cv_mem);
    ier = linsolsolve_gpu2(sd, cv_mem);

#ifdef CAMP_DEBUG_GPU
    cudaEventRecord(bicg->stopLinSolSolve);

    cudaEventSynchronize(bicg->stopLinSolSolve);
    float msLinSolSolve = 0.0;
    cudaEventElapsedTime(&msLinSolSolve, bicg->startLinSolSolve, bicg->stopLinSolSolve);
    bicg->timeLinSolSolve+= msLinSolSolve;
    bicg->counterLinSolSolve++;
#endif
    // If there is a convergence failure and the Jacobian-related
    //   data appears not to be current, loop again with a call to lsetup
    //   in which convfail=CV_FAIL_BAD_J.  Otherwise return.
    if (ier != TRY_AGAIN) return(ier);

    callSetup = SUNTRUE;
    convfail = CV_FAIL_BAD_J;

  }
}

__global__
void cvcheck_input_globald(double *x, int len, const char* s)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i<len)
  {
    printf("%s[%d]=%-le\n",s,i,x[i]);
  }
}

void check_input(double *dx, int len, int var_id){

  double *x=(double*)malloc(len*sizeof(double));

  cudaMemcpy(x, dx, len*sizeof(double), cudaMemcpyDeviceToHost);

  int n_zeros=0;
  for (int i=0; i<50; i++){
    if(x[i]==0.0)
      n_zeros++;
    printf("%d[%d]=%-le check_input\n",var_id,i,x[i]);
  }
  if(n_zeros==len)
    printf("%d is all zeros\n",var_id);

  free(x);
}

int linsolsetup_gpu2(SolverData *sd, CVodeMem cv_mem,int convfail,N_Vector vtemp1,N_Vector vtemp2,N_Vector vtemp3)
{
  itsolver *bicg = &(sd->bicg);
  booleantype jbad, jok;
  realtype dgamma;
  CVDlsMem cvdls_mem = (CVDlsMem) cv_mem->cv_lmem;;
  int retval = 0;

  ModelDataGPU *mGPU = &sd->mGPU;

  /* Use nst, gamma/gammap, and convfail to set J eval. flag jok */
  dgamma = fabs((cv_mem->cv_gamma/cv_mem->cv_gammap) - ONE); //In GPU is fabs too
  //dgamma = SUNRabs((cv_mem->cv_gamma/cv_mem->cv_gammap) - ONE);
  jbad = (cv_mem->cv_nst == 0) ||
         (cv_mem->cv_nst > cvdls_mem->nstlj + CVD_MSBJ) ||
         ((convfail == CV_FAIL_BAD_J) && (dgamma < CVD_DGMAX)) ||
         (convfail == CV_FAIL_OTHER);
  jok = !jbad;

  // If jok = SUNTRUE, use saved copy of J
  if (jok) {
    //if (0) {
    cv_mem->cv_jcur = SUNFALSE;
    retval = SUNMatCopy(cvdls_mem->savedJ, cvdls_mem->A);

    /* If jok = SUNFALSE, reset J, call jac routine for new J value and save a copy */
  } else {
    cvdls_mem->nje++;
    cvdls_mem->nstlj = cv_mem->cv_nst;
    cv_mem->cv_jcur = SUNTRUE;
    //retval = SUNMatZero(cvdls_mem->A);//we already set this to zero in our calc_jac

    //clock_t start = clock();

    //int Jac(realtype t, N_Vector y, N_Vector deriv, SUNMatrix J, void *solver_data,
    //        N_Vector tmp1, N_Vector tmp2, N_Vector tmp3) {

    //retval = cvdls_mem->jac(cv_mem->cv_tn, cv_mem->cv_y,cv_mem->cv_ftemp, cvdls_mem->A,
    //                        cvdls_mem->J_data, vtemp1, vtemp2, vtemp3);

#ifdef CAMP_DEBUG_GPU
    cudaEventRecord(bicg->startJac);
#endif

    //Ensure cv_y is loaded
    double *cv_y = NV_DATA_S(cv_mem->cv_y);
    cudaMemcpy(cv_y,mGPU->dcv_y,mGPU->nrows*sizeof(double),cudaMemcpyDeviceToHost);

#ifdef DEBUG_linsolsetup_gpu2
    check_isnand(bicg->A,mGPU->nnz,"prejac");
#endif

    retval = Jac(cv_mem->cv_tn, cv_mem->cv_y,cv_mem->cv_ftemp, cvdls_mem->A,cvdls_mem->J_data, vtemp1, vtemp2, vtemp3);

    //wrong
    //retval = Jac_gpu(cv_mem->cv_tn, cv_mem->cv_y,cv_mem->cv_ftemp, cvdls_mem->A,cvdls_mem->J_data, vtemp1, vtemp2, vtemp3);


#ifdef DEBUG_linsolsetup_gpu2
    check_isnand(bicg->A,mGPU->nnz,"postjac");
#endif

#ifdef CAMP_DEBUG_GPU
    cudaEventRecord(bicg->stopJac);

    cudaEventSynchronize(bicg->stopJac);
    float msJac = 0.0;
    cudaEventElapsedTime(&msJac, bicg->startJac, bicg->stopJac);
    bicg->timeJac+= msJac;
    bicg->counterJac++;
#endif

    if (retval < 0) {
      cvProcessError(cv_mem, CVDLS_JACFUNC_UNRECVR, "CVDLS",
                     "cvDlsSetup",  MSGD_JACFUNC_FAILED);
      cvdls_mem->last_flag = CVDLS_JACFUNC_UNRECVR;
      return(-1);
    }
    if (retval > 0) {
      cvdls_mem->last_flag = CVDLS_JACFUNC_RECVR;
      return(1);
    }

#ifdef CAMP_DEBUG_GPU
    //clock_t start = clock();
#endif

    retval = SUNMatCopy(cvdls_mem->A, cvdls_mem->savedJ);

#ifdef CAMP_DEBUG_GPU
    //bicg->timeMatCopy+= clock() - start;
    //bicg->counterMatCopy++;
#endif

    if (retval) {
      cvProcessError(cv_mem, CVDLS_SUNMAT_FAIL, "CVDLS",
                     "cvDlsSetup",  MSGD_MATCOPY_FAILED);
      cvdls_mem->last_flag = CVDLS_SUNMAT_FAIL;
      return(-1);
    }

  }

#ifdef CAMP_DEBUG_GPU
  //clock_t start = clock();
#endif

#ifndef LINSOLSOLVEGPU_INCLUDE_CUDAMEMCPY
  cudaEventRecord(bicg->startBCGMemcpy);
#endif

  cudaMemcpy(mGPU->diA,bicg->iA,(mGPU->nrows+1)*sizeof(int),cudaMemcpyHostToDevice);
  cudaMemcpy(mGPU->djA,bicg->jA,mGPU->nnz*sizeof(int),cudaMemcpyHostToDevice);
  cudaMemcpy(mGPU->dA,bicg->A,mGPU->nnz*sizeof(double),cudaMemcpyHostToDevice);

#ifndef LINSOLSOLVEGPU_INCLUDE_CUDAMEMCPY
  cudaEventRecord(bicg->stopBCGMemcpy);
  cudaEventSynchronize(bicg->stopBCGMemcpy);
  float msBiConjGradMemcpy = 0.0;
  cudaEventElapsedTime(&msBiConjGradMemcpy, bicg->startBCGMemcpy, bicg->stopBCGMemcpy);
  bicg->timeBiConjGradMemcpy+= msBiConjGradMemcpy/1000;
  bicg->timeBiConjGrad+= msBiConjGradMemcpy/1000;
#endif

#ifdef FAILURE_DETAIL
  //check if jac is correct
  int flag = check_jac_status_error_gpu2(cvdls_mem->A);
  //printf("Jac returned error flag %d\n",flag);
#endif

#ifdef CAMP_DEBUG_GPU
  //bicg->timeMatScaleAddISendA+= clock() - start;
  //bicg->counterMatScaleAddISendA++;

  //clock_t start2 = clock();
#endif

  gpu_matScaleAddI(mGPU->nrows,mGPU->dA,mGPU->djA,mGPU->diA,-cv_mem->cv_gamma,mGPU->blocks,mGPU->threads);

  cudaMemcpy(bicg->A,mGPU->dA,mGPU->nnz*sizeof(double),cudaMemcpyDeviceToHost);

#ifdef CAMP_DEBUG_GPU
  //bicg->timeMatScaleAddI+= clock() - start2;
  //bicg->counterMatScaleAddI++;
#endif

  gpu_diagprecond(mGPU->nrows,mGPU->dA,mGPU->djA,mGPU->diA,mGPU->ddiag,mGPU->blocks,mGPU->threads); //Setup linear solver

#ifdef DEBUG_linsolsetup_gpu2

  cvcheck_input_globald<<<mGPU->blocks,mGPU->threads>>>(mGPU->ddiag,mGPU->nrows,"mGPU->ddiag");

#endif


  //return(cvdls_mem->last_flag);
  return retval;
}


int linsolsolve_gpu2(SolverData *sd, CVodeMem cv_mem)
{
  itsolver *bicg = &(sd->bicg);
  ModelDataGPU *mGPU = &sd->mGPU;
  int m, retval;
  realtype del, delp, dcon;

  cv_mem->cv_mnewt = m = 0;

  //Delp = del from last iter (reduce iterations)
  del = delp = 0.0;

  double *acor = NV_DATA_S(cv_mem->cv_acor);
  double *cv_y = NV_DATA_S(cv_mem->cv_y);
  double *tempv = NV_DATA_S(cv_mem->cv_tempv);
  double *cv_ftemp = NV_DATA_S(cv_mem->cv_ftemp);
  //CVDlsMem cvdls_mem = (CVDlsMem) cv_mem->cv_lmem;
  //double *x = NV_DATA_S(cvdls_mem->x);

  N_Vector b;
  b=cv_mem->cv_tempv;
  double *b_ptr=NV_DATA_S(b);

  // Looping point for Newton iteration
  for(;;) {

    // Evaluate the residual of the nonlinear system
    // a*x + b*y = z
    gpu_zaxpby(cv_mem->cv_rl1, (mGPU->dzn+1*mGPU->nrows), 1.0, mGPU->cv_acor, mGPU->dtempv, mGPU->nrows, mGPU->blocks, mGPU->threads);
    gpu_zaxpby(cv_mem->cv_gamma, mGPU->dftemp, -1.0, mGPU->dtempv, mGPU->dtempv, mGPU->nrows, mGPU->blocks, mGPU->threads);
    //N_VLinearSum(cv_mem->cv_rl1, cv_mem->cv_zn[1], ONE,
    //             cv_mem->cv_acor, cv_mem->cv_tempv);
    //N_VLinearSum(cv_mem->cv_gamma, cv_mem->cv_ftemp, -ONE,
    //             cv_mem->cv_tempv, cv_mem->cv_tempv);

#ifndef CSR_SPMV_CPU

    swapCSC_CSR_BCG(sd);
    //cudaGlobalswapCSC_CSR
    //CSCtoCSR(bicg);

#endif

#ifndef LINSOLSOLVEGPU_INCLUDE_CUDAMEMCPY

    cudaEventRecord(bicg->startBCGMemcpy);

    //Simulate data movement cost of copy of tempv to dtempv by copying to empty array (daux)
    cudaMemcpy(mGPU->daux,tempv,mGPU->nrows*sizeof(double),cudaMemcpyHostToDevice);

    cudaEventRecord(bicg->stopBCGMemcpy);
    cudaEventSynchronize(bicg->stopBCGMemcpy);
    float msBiConjGradMemcpy = 0.0;
    cudaEventElapsedTime(&msBiConjGradMemcpy, bicg->startBCGMemcpy, bicg->stopBCGMemcpy);
    bicg->timeBiConjGradMemcpy+= msBiConjGradMemcpy/1000;
    bicg->timeBiConjGrad+= msBiConjGradMemcpy/1000;

#endif

#ifdef CAMP_DEBUG_GPU
    cudaEventRecord(bicg->startBCG);
#endif

#ifdef DEBUG_LINSOLSOLVEGPU

  //int k=0;
  //check_isnand(bicg->A,mGPU->nnz,k++);
  //check_isnand_global<<<mGPU->blocks, mGPU->threads>>>(mGPU->dA,mGPU->nnz,k++);
  //check_isnand_global0<<<1, 1>>>(mGPU->dA,mGPU->nnz,k++);

#endif

    if(bicg->cells_method==1) {//Sync with CPU
      solveGPU(sd,mGPU->dA,mGPU->djA,mGPU->diA,mGPU->dx,mGPU->dtempv);
    }
    else if(bicg->cells_method==4){
      solveCuSolver(sd);
    }
    else {
      solveGPU_block(sd,mGPU->dA,mGPU->djA,mGPU->diA,mGPU->dx,mGPU->dtempv);
    }

#ifdef CAMP_DEBUG_GPU
    cudaEventRecord(bicg->stopBCG);

    cudaEventSynchronize(bicg->stopBCG); //at the end is the same that cudadevicesynchronyze
    float msBiConjGrad = 0.0;
    cudaEventElapsedTime(&msBiConjGrad, bicg->startBCG, bicg->stopBCG);
    bicg->timeBiConjGrad+= msBiConjGrad/1000;

    bicg->counterBiConjGrad++;
#endif

#ifndef LINSOLSOLVEGPU_INCLUDE_CUDAMEMCPY

    cudaEventRecord(bicg->startBCGMemcpy);

    //Simulate data movement cost of copy of tempv to dtempv by copying to empty array (aux)
    cudaMemcpy(tempv,mGPU->dtempv,mGPU->nrows*sizeof(double),cudaMemcpyDeviceToHost);

    cudaEventRecord(bicg->stopBCGMemcpy);
    cudaEventSynchronize(bicg->stopBCGMemcpy);
    cudaEventElapsedTime(&msBiConjGradMemcpy, bicg->startBCGMemcpy, bicg->stopBCGMemcpy);
    bicg->timeBiConjGradMemcpy+= msBiConjGradMemcpy/1000;
    bicg->timeBiConjGrad+= msBiConjGradMemcpy/1000;

#endif

#ifndef CSR_SPMV_CPU

    //Swap is fine here, but not in in GPU device (problems of sync one-kernel)
    swapCSC_CSR_BCG(sd);
    //cudaGlobalswapCSC_CSR
    //CSRtoCSC(bicg);

#endif

    // Get WRMS norm of correction
    del = gpu_VWRMS_Norm(mGPU->nrows, mGPU->dx, mGPU->dewt, bicg->aux, mGPU->daux, (mGPU->blocks + 1) / 2, mGPU->threads);


    cudaMemcpy(cv_ftemp,mGPU->dftemp,mGPU->nrows*sizeof(double),cudaMemcpyDeviceToHost);
    cudaMemcpy(cv_y,mGPU->dcv_y,mGPU->nrows*sizeof(double),cudaMemcpyDeviceToHost);
    cudaMemcpy(b_ptr,mGPU->dx,mGPU->nrows*sizeof(double),cudaMemcpyDeviceToHost);

    if (cv_mem->cv_ghfun) {
      N_VLinearSum(ONE, cv_mem->cv_y, ONE, b, cv_mem->cv_ftemp);
      retval = cv_mem->cv_ghfun(cv_mem->cv_tn, ZERO, cv_mem->cv_ftemp,
                                cv_mem->cv_y, b, cv_mem->cv_user_data,
                                cv_mem->cv_tempv1, cv_mem->cv_tempv2);

      if (retval==1) {
      } else if (retval<0) {
        if ((!cv_mem->cv_jcur) && (cv_mem->cv_lsetup))
          return(TRY_AGAIN);
        else
          return(RHSFUNC_RECVR);
      }
    }
    // Check for negative concentrations
    N_VLinearSum(ONE, cv_mem->cv_y, ONE, b, cv_mem->cv_ftemp);
    if (N_VMin(cv_mem->cv_ftemp) < -CAMP_TINY) {
      return(CONV_FAIL);
    }

    cudaMemcpy(mGPU->dftemp,cv_ftemp,mGPU->nrows*sizeof(double),cudaMemcpyHostToDevice);

    //cudaMemcpy(mGPU->dftemp,cv_mem->cv_tempv2,mGPU->nrows*sizeof(double),cudaMemcpyHostToDevice);

    //add correction to acor and y
    // a*x + b*y = z
    gpu_zaxpby(1.0, mGPU->cv_acor, 1.0, mGPU->dx, mGPU->cv_acor, mGPU->nrows, mGPU->blocks, mGPU->threads);
    gpu_zaxpby(1.0, mGPU->dzn, 1.0, mGPU->cv_acor, mGPU->dcv_y, mGPU->nrows, mGPU->blocks, mGPU->threads);

    //(T a, const T *X, T b, const T *Y, T *Z, I n)
    //Z[i] = a*X[i] + b*Y[i];
    //N_VLinearSum(ONE, cv_mem->cv_acor, ONE, b, cv_mem->cv_acor);
    //N_VLinearSum(ONE, cv_mem->cv_zn[0], ONE, cv_mem->cv_acor, cv_mem->cv_y);


    // Test for convergence.  If m > 0, an estimate of the convergence
    // rate constant is stored in crate, and used in the test.
    if (m > 0) {
      cv_mem->cv_crate = SUNMAX(0.3 * cv_mem->cv_crate, del / delp);
    }

    dcon = del * SUNMIN(1.0, cv_mem->cv_crate) / cv_mem->cv_tq[4];

    if (dcon <= 1.0) {
      //cv_mem->cv_acnrm = N_VWrmsNorm(cv_mem->cv_acor, cv_mem->cv_ewt);
      cv_mem->cv_acnrm = gpu_VWRMS_Norm(mGPU->nrows, mGPU->cv_acor, mGPU->dewt, bicg->aux,
                                        mGPU->daux, (mGPU->blocks + 1) / 2, mGPU->threads);
      cv_mem->cv_jcur = SUNFALSE;

      cudaMemcpy(acor,mGPU->cv_acor,mGPU->nrows*sizeof(double),cudaMemcpyDeviceToHost);
      cudaMemcpy(tempv,mGPU->dx,mGPU->nrows*sizeof(double),cudaMemcpyDeviceToHost);
      return (CV_SUCCESS);
    }
    cv_mem->cv_mnewt = ++m;

    // Stop at maxcor iterations or if iter. seems to be diverging.
    //     If still not converged and Jacobian data is not current,
    //     signal to try the solution again
    if ((m == cv_mem->cv_maxcor) || ((m >= 2) && (del > RDIV * delp))) {
      cudaMemcpy(acor,mGPU->cv_acor,mGPU->nrows*sizeof(double),cudaMemcpyDeviceToHost);
      cudaMemcpy(tempv,mGPU->dx,mGPU->nrows*sizeof(double),cudaMemcpyDeviceToHost);
      if ((!cv_mem->cv_jcur) && (cv_mem->cv_lsetup)) {
        return (TRY_AGAIN);
      } else {
        return (CONV_FAIL);
      }
    }

    // Save norm of correction, evaluate f, and loop again
    delp = del;

    //check if its needed (i think only for f_CPU case, for f_gpu not)
    cudaMemcpy(cv_y,mGPU->dcv_y,mGPU->nrows*sizeof(double),cudaMemcpyDeviceToHost);
    //retval = cv_mem->cv_f(cv_mem->cv_tn, cv_mem->cv_y,
    //                      cv_mem->cv_ftemp, cv_mem->cv_user_data);
    //int f(realtype t, N_Vector y, N_Vector deriv, void *solver_data)

#ifdef CAMP_DEBUG_GPU
    //start=clock();
    cudaEventRecord(bicg->startDerivSolve);
#endif

    /*
    if(sd->counterDerivCPU<=5){
      for (int i = 0; i < NV_LENGTH_S(cv_mem->cv_y); i++) {
        //printf("(%d) %-le ", i + 1, NV_DATA_S(deriv)[i]);
        if(cv_y[i]!=md->deriv_aux[i]) {
          printf("(%d) dy %-le y %-le\n", i + 1, md->deriv_aux[i], cv_y[i]);
        }
      }
    }*/

    //retval = f(cv_mem->cv_tn, cv_mem->cv_y, cv_mem->cv_ftemp, cv_mem->cv_user_data);
    retval = f_gpu(cv_mem->cv_tn, cv_mem->cv_y, cv_mem->cv_ftemp, cv_mem->cv_user_data);

#ifdef CAMP_DEBUG_GPU
    cudaEventRecord(bicg->stopDerivSolve);

    cudaEventSynchronize(bicg->stopDerivSolve);
    float msDerivSolve = 0.0;
    cudaEventElapsedTime(&msDerivSolve, bicg->startDerivSolve, bicg->stopDerivSolve);
    bicg->timeDerivSolve+= msDerivSolve;

    //bicg->timeDerivSolve+= clock() - start;
    bicg->counterDerivSolve++;
#endif

    //Transfer cv_ftemp() not needed because mGPU->dftemp=mGPU->deriv_data;
    //cudaMemcpy(cv_ftemp_data,mGPU->dftemp,mGPU->nrows*sizeof(double),cudaMemcpyDeviceToHost);

    //N_VLinearSum(ONE, cv_mem->cv_y, -ONE, cv_mem->cv_zn[0], cv_mem->cv_acor);
    // a*x + b*y = z
    gpu_zaxpby(1.0, mGPU->dcv_y, -1.0, mGPU->dzn, mGPU->cv_acor, mGPU->nrows, mGPU->blocks, mGPU->threads);

    cv_mem->cv_nfe++;
    if (retval < 0){
      cudaMemcpy(acor,mGPU->cv_acor,mGPU->nrows*sizeof(double),cudaMemcpyDeviceToHost);
      cudaMemcpy(tempv,mGPU->dx,mGPU->nrows*sizeof(double),cudaMemcpyDeviceToHost);
      return(CV_RHSFUNC_FAIL);
    }
    if (retval > 0) {
      cudaMemcpy(acor,mGPU->cv_acor,mGPU->nrows*sizeof(double),cudaMemcpyDeviceToHost);
      cudaMemcpy(tempv,mGPU->dx,mGPU->nrows*sizeof(double),cudaMemcpyDeviceToHost);
      if ((!cv_mem->cv_jcur) && (cv_mem->cv_lsetup))
        return(TRY_AGAIN);
      else
        return(RHSFUNC_RECVR);
    }

  }
}

void free_ode_gpu2(SolverData *sd)
{
  ModelDataGPU *mGPU = &sd->mGPU;

  //ODE aux variables
  cudaFree(mGPU->dewt);
  cudaFree(mGPU->cv_acor);
  cudaFree(mGPU->dtempv);
  cudaFree(mGPU->dzn);

  //ODE concs arrays
  cudaFree(mGPU->dcv_y);
  cudaFree(mGPU->dx);

  free_itsolver(sd);

  //HANDLE_ERROR(cudaFree(cv_mem->indexvals_gpu2));
  //HANDLE_ERROR(cudaFree(cv_mem->indexptrs_gpu2));
  //HANDLE_ERROR(cudaFree(cv_mem->jac_data_gpu2));

  //In principle, C++ guarantee destroy the classes when they go out of scope, so don't need to call destructor
  //bicg->~itsolver(){};

}

void solver_get_statistics_gpu(SolverData *sd){

  ModelDataGPU *mGPU = &sd->mGPU;

  cudaMemcpy(&sd->mdv,mGPU->mdvo,sizeof(ModelDataVariable),cudaMemcpyDeviceToHost);

  //printf("solver_get_statistics_gpu\n");

#ifdef CAMP_DEBUG_GPU

#ifdef cudaGlobalCVode_timers_max_blocks

  printf("PENDING cudaGlobalCVode_timers_max_blocks");

#else

  cudaMemcpy(&sd->tguessNewton,mGPU->tguessNewton,sizeof(double),cudaMemcpyDeviceToHost);
  cudaMemcpy(&sd->timeNewtonIteration,mGPU->dtNewtonIteration,sizeof(double),cudaMemcpyDeviceToHost);
  cudaMemcpy(&sd->timeJac,mGPU->dtJac,sizeof(double),cudaMemcpyDeviceToHost);
  cudaMemcpy(&sd->timelinsolsetup,mGPU->dtlinsolsetup,sizeof(double),cudaMemcpyDeviceToHost);
  cudaMemcpy(&sd->timecalc_Jac,mGPU->dtcalc_Jac,sizeof(double),cudaMemcpyDeviceToHost);
  cudaMemcpy(&sd->timeRXNJac,mGPU->dtRXNJac,sizeof(double),cudaMemcpyDeviceToHost);
  cudaMemcpy(&sd->timef,mGPU->dtf,sizeof(double),cudaMemcpyDeviceToHost);
  cudaMemcpy(&sd->timeguess_helper,mGPU->dtguess_helper,sizeof(double),cudaMemcpyDeviceToHost);

#endif

#endif

}

void printSolverCounters_gpu2(SolverData *sd)
{

#ifdef CAMP_DEBUG_GPU

  itsolver *bicg = &(sd->bicg);

  solver_get_statistics_gpu(sd);

  //Upgraded GPU-CPU counters (Sync with GPU and CPU)
  printf("timecvStep %lf, countercvStep %d\n",bicg->timecvStep,sd->mdv.countercvStep);
  printf("timesolveCVODEGPU %lf, countersolveCVODEGPU %d\n",bicg->timesolveCVODEGPU,bicg->countersolveCVODEGPU);
  printf("time CudaGlobalguess_helper %lf\n",bicg->timeNewtonIt);
  printf("timef %lf \n",sd->timef);
  printf("timeguess_helper %lf \n",sd->timeguess_helper);
  printf("tguessNewton %lf \n",sd->tguessNewton);
  //printf("timeLinSolSolve %lf, counterLinSolSolve %d\n",bicg->timeLinSolSolve/1000,bicg->counterLinSolSolve);
  //printf("timeDerivNewton %lf, counterDerivNewton %d\n",bicg->timeDerivNewton/1000,bicg->counterDerivNewton);
  //printf("timeLinSolSetup %lf, counterLinSolSetup %d\n",bicg->timeLinSolSetup/1000,bicg->counterLinSolSetup);
  //printf("timeDerivSolve %lf, counterDerivSolve %d\n",bicg->timeDerivSolve/1000,bicg->counterDerivSolve);
  //printf("timeJac %lf, counterJac %d\n",bicg->timeJac/1000,bicg->counterJac);
  printf("timeBiConjGrad %lf, timeBiConjGradMemcpy %lf, counterBiConjGrad %d, counterBCGInternal %d "
         "avgCounterBiConjGrad %lf, avgTimeBCGIter %lf %%timeBiConjGradMemcpy %lf%%\n",
          bicg->timeBiConjGrad,
          bicg->timeBiConjGradMemcpy, bicg->counterBiConjGrad,sd->mdv.counterBCGInternal,
          sd->mdv.counterBCGInternal/(double)bicg->counterBiConjGrad,
          bicg->timeBiConjGrad/(double)bicg->counterBiConjGrad,
          bicg->timeBiConjGradMemcpy/bicg->timeBiConjGrad*100);
#ifdef cudaGlobalCVode_timers_max_blocks

  for(int i=1;i<blocks;i++){
    if(dtBCG[i]>dtBCG[0])
      dtBCG[0]=dtBCG[i];
    if(dtcudaDeviceCVode[i]>dtcudaDeviceCVode[0])
      dtcudaDeviceCVode[0]=dtcudaDeviceCVode[i];
    if(dtPostBCG[i]>dtPostBCG[0])
      dtPostBCG[0]=dtPostBCG[i];
  }

  printf("dtcudaDeviceCVode %lf dtBCG %lf dtPostBCG %lf\n",sd->mdv.dtcudaDeviceCVode[0],
          sd->mdv.dtBCG[0],sd->mdv.dtPostBCG[0]);

#else

  printf("dtcudaDeviceCVode %lf dtBCG %lf dtPostBCG %lf\n",sd->mdv.dtcudaDeviceCVode,
          sd->mdv.dtBCG,sd->mdv.dtPostBCG);

#endif

#endif

}
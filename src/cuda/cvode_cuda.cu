/* Copyright (C) 2021 Barcelona Supercomputing Center and University of
* Illinois at Urbana-Champaign
* SPDX-License-Identifier: MIT
*/

#include "cvode_cuda.h"
extern "C" {
#include "rxns_gpu.h"
#include "time_derivative_gpu.h"
#include "Jacobian_gpu.h"
}

#ifdef DEBUG_CVODE_GPU
__device__
void printmin(ModelDataGPU *md,double* y, const char *s) {

  __syncthreads();
  extern __shared__ double flag_shr2[];
  int tid= threadIdx.x + blockDim.x*blockIdx.x;
  __syncthreads();

  double min;
  cudaDevicemin(&min, y[tid], flag_shr2, md->n_shr_empty);
  __syncthreads();
  if(tid==0)printf("%s min %le\n",s,min);
  __syncthreads();

}
#endif
/* */
__device__ void warpReduce_2(volatile double *sdata, unsigned int tid) {
  unsigned int blockSize = blockDim.x;
  if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
  if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
  if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
  if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
  if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
  if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
}

__device__ void cudaDevicedotxy_2(double *g_idata1, double *g_idata2,
                                 double *g_odata, int n_shr_empty){
  extern __shared__ double sdata[];
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
  __syncthreads();
  if(tid<n_shr_empty)
    sdata[tid+blockDim.x]=0.;
  __syncthreads();
  sdata[tid] = g_idata1[i]*g_idata2[i];
  __syncthreads();
  unsigned int blockSize = blockDim.x+n_shr_empty;
  if ((blockSize >= 1024) && (tid < 512)) {
    sdata[tid] += sdata[tid + 512];
  }
  __syncthreads();
  if ((blockSize >= 512) && (tid < 256)) {
    sdata[tid] += sdata[tid + 256];
  }
  __syncthreads();
  if ((blockSize >= 256) && (tid < 128)) {
    sdata[tid] += sdata[tid + 128];
  }
  __syncthreads();
  if ((blockSize >= 128) && (tid < 64)) {
    sdata[tid] += sdata[tid + 64];
  }
  __syncthreads();
  if (tid < 32) warpReduce_2(sdata, tid);
  __syncthreads();
  *g_odata = sdata[0];
  __syncthreads();
}

__device__ void cudaDeviceVWRMS_Norm_2(double *g_idata1, double *g_idata2, double *g_odata, int n, int n_shr_empty){
  extern __shared__ double sdata[];
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
  __syncthreads();
  if(tid<n_shr_empty)
    sdata[tid+blockDim.x]=0.;
  __syncthreads();
  sdata[tid] = g_idata1[i]*g_idata1[i]*g_idata2[i]*g_idata2[i];
  __syncthreads();
  for (unsigned int s=(blockDim.x+n_shr_empty)/2; s>0; s>>=1)
  {
    if (tid < s)
      sdata[tid] += sdata[tid + s];

    __syncthreads();
  }
  g_odata[0] = sqrt(sdata[0]/n);
  __syncthreads();
}

__device__
void cudaDeviceJacCopy(int* Ap, double* Ax, double* Bx) {
  __syncthreads();
  int nnz=Ap[blockDim.x];
  for(int j=Ap[threadIdx.x]; j<Ap[threadIdx.x+1]; j++){
    Bx[j+nnz*blockIdx.x]=Ax[j+nnz*blockIdx.x];
  }
  __syncthreads();
}

__device__
int cudaDevicecamp_solver_check_model_state(ModelDataGPU *md, ModelDataVariable *dmdv, double *y, int *flag)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  __syncthreads();
  extern __shared__ int flag_shr[];
  flag_shr[0] = 0;
  __syncthreads();
  if (y[tid] < md->threshhold) {
    flag_shr[0] = CAMP_SOLVER_FAIL;
#ifdef DEBUG_cudaDevicecamp_solver_check_model_state
    printf("Failed model state update gpu:[spec %d] = %le flag_shr %d\n",tid,y[tid],flag_shr[0]);
#endif
  } else {
    md->state[md->map_state_deriv[tid]] =
            y[tid] <= md->threshhold ?
            md->replacement_value : y[tid];
  }
  __syncthreads();
  *flag = (int)flag_shr[0];
  __syncthreads();
#ifdef DEBUG_printmin
  printmin(md,md->state,"cudaDevicecamp_solver_check_model_state end state");
#endif
#ifdef DEBUG_cudaDevicecamp_solver_check_model_state
  __syncthreads();if(tid==0)printf("flag %d flag_shr %d\n",*flag,flag_shr2[0]);
#endif
  return *flag;
}

__device__ void solveRXN(
        int i_rxn, int i_cell,TimeDerivativeGPU deriv_data,
        double time_step,
        ModelDataGPU *md, ModelDataVariable *dmdv
){
#ifdef REVERSE_INT_FLOAT_MATRIX
  double *rxn_float_data = &( md->rxn_double[i_rxn]);
  int *int_data = &(md->rxn_int[i_rxn]);
  int rxn_type = int_data[0];
  int *rxn_int_data = (int *) &(int_data[1*md->n_rxn]);
#else
  double *rxn_float_data = (double *)&( md->rxn_double[md->rxn_float_indices[i_rxn]]);
  int *int_data = (int *)&(md->rxn_int[md->rxn_int_indices[i_rxn]]);
  int rxn_type = int_data[0];
  int *rxn_int_data = (int *) &(int_data[1]);
#endif
  double *rxn_env_data = &(md->rxn_env_data
  [md->n_rxn_env_data*i_cell+md->rxn_env_data_idx[i_rxn]]);
#ifdef DEBUG_DERIV_GPU
  if(tid==0){
    printf("[DEBUG] GPU solveRXN tid %d, \n", tid);
  }
#endif
  switch (rxn_type) {
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
    case RXN_FIRST_ORDER_LOSS:
    rxn_gpu_first_order_loss_calc_deriv_contrib(md, deriv_data, rxn_int_data,
                                    rxn_float_data, rxn_env_data,time_step);
      break;
    case RXN_PHOTOLYSIS :
      rxn_gpu_photolysis_calc_deriv_contrib(md, deriv_data, rxn_int_data,
                                            rxn_float_data, rxn_env_data,time_step);
      break;
    case RXN_TROE :
      rxn_gpu_troe_calc_deriv_contrib(md, deriv_data, rxn_int_data,
                                      rxn_float_data, rxn_env_data,time_step);
      break;
  }
}

__device__ void cudaDevicecalc_deriv(
        double time_step, double *y,
        double *yout, ModelDataGPU *md, ModelDataVariable *dmdv
) //Interface CPU/GPU
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int deriv_length_cell = md->nrows / md->n_cells;
  int tid_cell=tid%deriv_length_cell;
  int state_size_cell = md->state_size_cell;
#ifdef DEBUG_DERIV_GPU
  if(tid==0){
    printf("[DEBUG] GPU solveDerivative tid %d, \n", tid);
    printf("md->nrows %d, \n", md->nrows);
    printf("md->deriv_length_cell %d, \n", md->deriv_length_cell);
    printf("blockDim.x %d, \n", blockDim.x);
  }__syncthreads();
#endif
#ifdef DEBUG_printmin
  //__syncthreads();//no effect, but printmin yes
  printmin(md,yout,"cudaDevicecalc_deriv start end yout");
  printmin(md,md->J_tmp,"cudaDevicecalc_deriv start end J_tmp");
  printmin(md,md->J_state,"cudaDevicecalc_deriv start end J_state");
#endif
  md->J_tmp[tid]=y[tid]-md->J_state[tid];
  cudaDeviceSpmv(md->J_tmp2, md->J_tmp, md->J_solver, md->djA, md->diA);
  md->J_tmp[tid]=md->J_deriv[tid]+md->J_tmp2[tid];
  cudaDevicesetconst(md->J_tmp2, 0.0); //Reset for next iter
#ifdef DEBUG_printmin
    printmin(md,md->J_tmp,"cudaDevicecalc_deriv start end J_tmp");
    printmin(md,md->J_state,"cudaDevicecalc_deriv start end J_state");
#endif
    TimeDerivativeGPU deriv_data;
    deriv_data.num_spec = deriv_length_cell*gridDim.x;
#ifdef AEROS_CPU
#else
    deriv_data.production_rates = md->production_rates;
    deriv_data.loss_rates = md->loss_rates;
    time_derivative_reset_gpu(deriv_data);
    __syncthreads();
#endif
    int i_cell = tid/deriv_length_cell;
    deriv_data.production_rates = &( md->production_rates[deriv_length_cell*i_cell]);
    deriv_data.loss_rates = &( md->loss_rates[deriv_length_cell*i_cell]);
    md->grid_cell_state = &( md->state[state_size_cell*i_cell]);
    md->grid_cell_env = &( md->env[CAMP_NUM_ENV_PARAM_*i_cell]);
    int n_rxn = md->n_rxn;
    if( tid_cell < n_rxn) {
      int n_iters = n_rxn / deriv_length_cell;
      for (int i = 0; i < n_iters; i++) {
        int i_rxn = tid_cell + i*deriv_length_cell;
        solveRXN(i_rxn, i_cell,deriv_data, time_step, md, dmdv);
      }
      int residual=n_rxn-(deriv_length_cell*n_iters);
      if(tid_cell < residual){
        int i_rxn = tid_cell + deriv_length_cell*n_iters;
        solveRXN(i_rxn, i_cell, deriv_data, time_step, md, dmdv);
      }
    }
    __syncthreads();
    deriv_data.production_rates = md->production_rates;
    deriv_data.loss_rates = md->loss_rates;
#ifdef DEBUG_printmin
    printmin(md,yout,"cudaDevicecalc_deriv start end yout");
#endif
    __syncthreads();
    time_derivative_output_gpu(deriv_data, yout, md->J_tmp,0);
#ifdef DEBUG_printmin
    printmin(md,yout,"cudaDevicecalc_deriv start end yout");
#endif
  __syncthreads();
}

__device__
int cudaDevicef(
        double time_step, double *y,
        double *yout, ModelDataGPU *md, ModelDataVariable *dmdv, int *flag
)
{
#ifdef CAMP_DEBUG_GPU
#ifdef CAMP_PROFILE_DEVICE_FUNCTIONS
  int clock_khz=md->clock_khz;
  clock_t start;
  start = clock();
#endif
#endif
#ifdef DEBUG_printmin
  printmin(md,y,"cudaDevicef Start y");
#endif
  time_step = time_step > 0. ? time_step : md->init_time_step;
#ifdef DEBUG_printmin
  printmin(md,md->state,"cudaDevicef start state");
#endif
  int checkflag=cudaDevicecamp_solver_check_model_state(md, dmdv, y, flag);
  __syncthreads();
  if(checkflag==CAMP_SOLVER_FAIL){
    *flag=CAMP_SOLVER_FAIL;
#ifdef DEBUG_printmin
    printmin(md,y,"cudaDevicef End y");
#endif
    __syncthreads();
#ifdef CAMP_DEBUG_GPU
#ifdef CAMP_PROFILE_DEVICE_FUNCTIONS
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(threadIdx.x==0) md->s->timef += ((double)(int)(clock() - start))/(clock_khz*1000);
#endif
#endif
#ifdef DEBUG_cudaDevicef
    if(i==0)printf("cudaDevicef CAMP_SOLVER_FAIL %d\n",i);
#endif
    return CAMP_SOLVER_FAIL;
  }
#ifdef DEBUG_printmin
  printmin(md,yout,"cudaDevicef End yout");
#endif
  cudaDevicecalc_deriv(time_step, y,
          yout, md, dmdv
  );
  //printmin(md,yout,"cudaDevicef End yout");
  //printmin(md,y,"cudaDevicef End y");
#ifdef CAMP_DEBUG_GPU
#ifdef CAMP_PROFILE_DEVICE_FUNCTIONS
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(threadIdx.x==0) md->s->timef += ((double)(int)(clock() - start))/(clock_khz*1000);
#endif
#endif
  __syncthreads();
  *flag=0;
  __syncthreads();
  return 0;
}

__device__
int CudaDeviceguess_helper(double cv_tn, double cv_h, double* y_n,
                           double* y_n1, double* hf, double* dtempv1,
                           double* dtempv2, int *flag,
                           ModelDataGPU *md, ModelDataVariable *dmdv
) {
  extern __shared__ double flag_shr2[];
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int tid = threadIdx.x;
  int n_shr_empty = md->n_shr_empty;
#ifdef DEBUG_CudaDeviceguess_helper
  if(i==0)printf("CudaDeviceguess_helper start gpu\n");
#endif
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
  __syncthreads();
#ifdef CAMP_DEBUG_GPU
#ifdef CAMP_PROFILE_DEVICE_FUNCTIONS
  int clock_khz=md->clock_khz;
  clock_t start;
  start = clock();
#endif
#endif
  dtempv1[i]=y_n1[i];
  __syncthreads();
  if (cv_h > 0.) {
    dtempv2[i]=(1./cv_h)*hf[i];
  } else {
    dtempv2[i]=hf[i];
  }
  double t_0 = cv_h > 0. ? cv_tn - cv_h : cv_tn - 1.;
  double t_j = 0.;
  int GUESS_MAX_ITER = 5; //5 //reduce this to reduce time
  __syncthreads();
  for (int iter = 0; iter < GUESS_MAX_ITER && t_0 + t_j < cv_tn; iter++) {
    __syncthreads();
    double h_j = cv_tn - (t_0 + t_j);
    __syncthreads();
    double t_star;
    double h_j_init=h_j;
    if(dtempv2[i]==0){
      t_star=h_j;
    }else{
      t_star = -dtempv1[i] / dtempv2[i];
    }
    if( !(t_star > 0. || (t_star == 0. && dtempv2[i] < 0.)) ){//&&dtempv2[i]==0.)
      t_star=h_j;
    }
    __syncthreads();
    flag_shr2[tid]=h_j_init;
    cudaDevicemin(&h_j, t_star, flag_shr2, n_shr_empty);
    flag_shr2[0]=1;
    __syncthreads();
#ifdef DEBUG_CudaDeviceguess_helper
    //if(tid==0 && iter<=5) printf("CudaDeviceguess_helper h_j %le h_j_init %le t_star %le block %d iter %d\n",h_j,h_j_init,t_star,blockIdx.x,iter);
#endif
    if (cv_h > 0.)
      h_j *= 0.95 + 0.1 * iter / (double)GUESS_MAX_ITER;
    h_j = cv_tn < t_0 + t_j + h_j ? cv_tn - (t_0 + t_j) : h_j;
    __syncthreads();
    if (cv_h == 0. &&
        cv_tn - (h_j + t_j + t_0) > md->cv_reltol) {
#ifdef DEBUG_CudaDeviceguess_helper
      if(i==0)printf("CudaDeviceguess_helper small changes \n");
#endif
      __syncthreads();
#ifdef CAMP_DEBUG_GPU
#ifdef CAMP_PROFILE_DEVICE_FUNCTIONS
    if(threadIdx.x==0) md->s->timeguess_helper += ((double)(clock() - start))/(clock_khz*1000);
#endif
#endif
    return -1;
    }
    dtempv1[i]+=h_j*dtempv2[i];
    __syncthreads();
    t_j += h_j;
#ifdef DEBUG_CudaDeviceguess_helper
    //  printf("dcorr[%d] %le dhf %le dt_star %le dh_j %le dh_n %le\n",
    //         i,dtempv2[i],hf[i],t_star,h_j,cv_h);
    //if(i==0)
    //  for(int j=0;j<nrows;j++)
    //    printf("dcorr[%d] %le dtmp1 %le dhf %le dt_star %le dh_j %le dh_n %le\n",
    //           j,dtempv2[j],dtempv1[j],hf[j],t_star,h_j,cv_h);
#endif
#ifdef DEBUG_printmin
    printmin(md,md->state,"cudaDevicef start state");
#endif
    int aux_flag=0;
    int fflag=cudaDevicef(
            t_0 + t_j, dtempv1, dtempv2,md,dmdv,&aux_flag);
#ifdef DEBUG_printmin
    printmin(md,dtempv1,"cudaDevicef end dtempv1");
#endif
    __syncthreads();
    if (fflag == CAMP_SOLVER_FAIL) {
      dtempv2[i] = 0.;
#ifdef DEBUG_CudaDeviceguess_helper
      if(i==0)printf("CudaDeviceguess_helper df(t)\n");
#endif
      __syncthreads();
#ifdef CAMP_DEBUG_GPU
#ifdef CAMP_PROFILE_DEVICE_FUNCTIONS
    if(threadIdx.x==0) md->s->timeguess_helper += ((double)(clock() - start))/(clock_khz*1000);
#endif
#endif
     return -1;
    }
    if (iter == GUESS_MAX_ITER - 1 && t_0 + t_j < cv_tn) {
      if (cv_h == 0.){
        __syncthreads();
#ifdef CAMP_DEBUG_GPU
#ifdef CAMP_PROFILE_DEVICE_FUNCTIONS
    if(threadIdx.x==0) md->s->timeguess_helper += ((double)(clock() - start))/(clock_khz*1000);
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
  dtempv2[i]=dtempv1[i]-y_n[i];
  if (cv_h > 0.) dtempv2[i]=dtempv2[i]*0.999;
  hf[i]=dtempv1[i]-y_n1[i];
  __syncthreads();
#ifdef CAMP_DEBUG_GPU
#ifdef CAMP_PROFILE_DEVICE_FUNCTIONS
    if(threadIdx.x==0)  md->s->timeguess_helper += ((double)(clock() - start))/(clock_khz*1000);
#endif
#endif
  __syncthreads();
  return 1;
}

__device__ void solveRXNJac(
        int i_rxn, int i_cell, JacobianGPU jac,
        ModelDataGPU *md, ModelDataVariable *dmdv
){
  double cv_next_h = md->s->cv_next_h;
#ifdef REVERSE_INT_FLOAT_MATRIX
  double *rxn_float_data = &( md->rxn_double[i_rxn]);
  int *int_data = &(md->rxn_int[i_rxn]);
  int rxn_type = int_data[0];
  int *rxn_int_data = (int *) &(int_data[1*md->n_rxn]);
#else
  double *rxn_float_data = (double *)&( md->rxn_double[md->rxn_float_indices[i_rxn]]);
  int *int_data = (int *)&(md->rxn_int[md->rxn_int_indices[i_rxn]]);
  int rxn_type = int_data[0];
  int *rxn_int_data = (int *) &(int_data[1]);
#endif
  double *rxn_env_data = &(md->rxn_env_data
  [md->n_rxn_env_data*i_cell+md->rxn_env_data_idx[i_rxn]]);
#ifdef DEBUG_solveRXNJac
  if(tid==0){
    printf("[DEBUG] GPU solveRXN tid %d, \n", tid);
  }
#endif
  switch (rxn_type) {
    case RXN_ARRHENIUS :
      rxn_gpu_arrhenius_calc_jac_contrib(md, jac, rxn_int_data,
                                         rxn_float_data, rxn_env_data,cv_next_h);
      break;
    case RXN_CMAQ_H2O2 :
      rxn_gpu_CMAQ_H2O2_calc_jac_contrib(md, jac, rxn_int_data,
                                         rxn_float_data, rxn_env_data,cv_next_h);
      break;
    case RXN_CMAQ_OH_HNO3 :
      rxn_gpu_CMAQ_OH_HNO3_calc_jac_contrib(md, jac, rxn_int_data,
                                            rxn_float_data, rxn_env_data,cv_next_h);
      break;
  case RXN_FIRST_ORDER_LOSS :
    rxn_gpu_first_order_loss_calc_jac_contrib(md, jac, rxn_int_data,
                                        rxn_float_data, rxn_env_data,cv_next_h);
    break;
    case RXN_PHOTOLYSIS :
      rxn_gpu_photolysis_calc_jac_contrib(md, jac, rxn_int_data,
                                          rxn_float_data, rxn_env_data,cv_next_h);
      break;
    case RXN_TROE :
      rxn_gpu_troe_calc_jac_contrib(md, jac, rxn_int_data,
                                    rxn_float_data, rxn_env_data,cv_next_h);
      break;
  }
}

__device__ void cudaDevicecalc_Jac(double *y,ModelDataGPU *md, ModelDataVariable *dmdv
){
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int deriv_length_cell = md->nrows / md->n_cells;
  int state_size_cell = md->state_size_cell;
  int tid_cell=tid%deriv_length_cell;
  int active_threads = md->nrows;
  __syncthreads();
#ifdef CAMP_DEBUG_GPU
#ifdef CAMP_PROFILE_DEVICE_FUNCTIONS
  int clock_khz=md->clock_khz;
  clock_t start;
  start = clock();
#endif
#endif
#ifdef DEBUG_cudaDeviceJac
  if(tid==0){
    printf("[DEBUG] GPU solveDerivative tid %d, \n", tid);
  }__syncthreads();
#endif
  if(tid<active_threads){
    __syncthreads();
    JacobianGPU *jac = &md->jac;
    JacobianGPU jacBlock;
    __syncthreads();
    int i_cell = tid/deriv_length_cell;
    jacBlock.num_elem = jac->num_elem;
    jacBlock.production_partials = &( jac->production_partials[jacBlock.num_elem[0]*blockIdx.x]);
    jacBlock.loss_partials = &( jac->loss_partials[jacBlock.num_elem[0]*blockIdx.x]);
    __syncthreads();
    md->grid_cell_state = &( md->state[state_size_cell*i_cell]);
    md->grid_cell_env = &( md->env[CAMP_NUM_ENV_PARAM_*i_cell]);
#ifdef DEBUG_cudaDevicecalc_Jac
    if(tid==0)printf("cudaDevicecalc_Jac01\n");
#endif
    __syncthreads();
    int n_rxn = md->n_rxn;
    if( tid_cell < n_rxn) {
      int n_iters = n_rxn / deriv_length_cell;
      for (int i = 0; i < n_iters; i++) {
        int i_rxn = tid_cell + i*deriv_length_cell;
        solveRXNJac(i_rxn,i_cell,jacBlock, md, dmdv);
      }
      int residual=n_rxn-(deriv_length_cell*n_iters);
      if(tid_cell < residual){
        int i_rxn = tid_cell + deriv_length_cell*n_iters;
        solveRXNJac(i_rxn,i_cell,jacBlock, md, dmdv);
      }
    }
    __syncthreads();
  JacMap *jac_map = md->jac_map;
  int nnz = md->n_mapped_values[0];
  int n_iters = nnz / blockDim.x;
  for (int i = 0; i < n_iters; i++) {
    int j = threadIdx.x + i*blockDim.x;
  md->dA[jac_map[j].solver_id + nnz * blockIdx.x] =
      jacBlock.production_partials[jac_map[j].rxn_id] - jacBlock.loss_partials[jac_map[j].rxn_id];
    jacBlock.production_partials[jac_map[j].rxn_id] = 0.0;
    jacBlock.loss_partials[jac_map[j].rxn_id] = 0.0;
  }
  int residual=nnz-(blockDim.x*n_iters);
  if(threadIdx.x < residual){
    int j = threadIdx.x + n_iters*blockDim.x;
  md->dA[jac_map[j].solver_id + nnz * blockIdx.x] =
      jacBlock.production_partials[jac_map[j].rxn_id] - jacBlock.loss_partials[jac_map[j].rxn_id];
    jacBlock.production_partials[jac_map[j].rxn_id] = 0.0;
    jacBlock.loss_partials[jac_map[j].rxn_id] = 0.0;
  }
    __syncthreads();
#ifdef CAMP_DEBUG_GPU
#ifdef CAMP_PROFILE_DEVICE_FUNCTIONS
    if(threadIdx.x==0) md->s->timecalc_Jac += ((double)(clock() - start))/(clock_khz*1000);
#endif
#endif
  }
}

__device__
int cudaDeviceJac(int *flag, ModelDataGPU *md, ModelDataVariable *dmdv
) //Interface CPU/GPU
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  double* dftemp = md->dftemp;
  double* dcv_y = md->dcv_y;
  __syncthreads();
#ifdef CAMP_DEBUG_GPU
#ifdef CAMP_PROFILE_DEVICE_FUNCTIONS
  int clock_khz=md->clock_khz;
  clock_t start;
  start = clock();
#endif
#endif
#ifdef DEBUG_printmin
  printmin(md,md->state,"cudaDeviceJac start state");
#endif
  int aux_flag=0;
  int retval=cudaDevicef(
          md->s->cv_next_h, dcv_y, dftemp,md,dmdv,&aux_flag
  );__syncthreads();
#ifdef DEBUG_cudaDevicef
  printmin(md,dftemp,"cudaDeviceJac dftemp");
#endif
  if(retval==CAMP_SOLVER_FAIL)
    return CAMP_SOLVER_FAIL;
#ifdef DEBUG_printmin
  printmin(md,dcv_y,"cudaDeviceJac dcv_y");
  printmin(md,md->state,"cudaDeviceJac start state");
#endif
  //debug
/*
  int checkflag=cudaDevicecamp_solver_check_model_state(md, dmdv, dcv_y, flag);
  __syncthreads();
  if(checkflag==CAMP_SOLVER_FAIL){
    *flag=CAMP_SOLVER_FAIL;
    //printf("cudaDeviceJac cudaDevicecamp_solver_check_model_state *flag==CAMP_SOLVER_FAIL\n");
    //printmin(md,dcv_y,"cudaDeviceJac end dcv_y");
    return CAMP_SOLVER_FAIL;
  }
*/
#ifdef DEBUG_printmin
  printmin(md,dcv_y,"cudaDeviceJac end dcv_y");
#endif
  //printmin(md,dftemp,"cudaDeviceJac end dftemp");
  cudaDevicecalc_Jac(dcv_y,md, dmdv);
  __syncthreads();
#ifdef DEBUG_printmin
 printmin(md,dftemp,"cudaDevicecalc_Jac end dftemp");
#endif
    __syncthreads();
  int nnz = md->n_mapped_values[0];
  int n_iters = nnz / blockDim.x;
  for (int i = 0; i < n_iters; i++) {
    int j = threadIdx.x + i*blockDim.x;
    md->J_solver[j]=md->dA[j];
  }
  int residual=nnz-(blockDim.x*n_iters);
  if(threadIdx.x < residual){
    int j = threadIdx.x + n_iters*blockDim.x;
    md->J_solver[j]=md->dA[j];
  }
    __syncthreads();
    md->J_state[tid]=dcv_y[tid];
    md->J_deriv[tid]=dftemp[tid];
  __syncthreads();
#ifdef CAMP_DEBUG_GPU
#ifdef CAMP_PROFILE_DEVICE_FUNCTIONS
    //if(tid==0)printf("md->s->timeJac %lf\n",md->s->timeJac);
    if(threadIdx.x==0)  md->s->timeJac += ((double)(clock() - start))/(clock_khz*1000);
#endif
#endif
  __syncthreads();
  *flag = 0;
  __syncthreads();
  return 0;
}

__device__
int cudaDevicelinsolsetup(
        ModelDataGPU *md, ModelDataVariable *dmdv,
        int convfail
) {
  extern __shared__ int flag_shr[];
  double* dA = md->dA;
  int* djA = md->djA;
  int* diA = md->diA;
  double* ddiag = md->ddiag;
  double* dsavedJ = md->dsavedJ;
  double dgamma;
  int jbad, jok;
#ifdef DEBUG_printmin
  printmin(md,dcv_y,"cudaDevicelinsolsetup Start dcv_y");
#endif
  dgamma = fabs((md->s->cv_gamma / md->s->cv_gammap) - 1.);//SUNRabs
  jbad = (md->s->cv_nst == 0) ||
         (md->s->cv_nst > md->s->nstlj + CVD_MSBJ) ||
         ((convfail == CV_FAIL_BAD_J) && (dgamma < CVD_DGMAX)) ||
         (convfail == CV_FAIL_OTHER);
  jok = !jbad;
  if (jok==1) {
    __syncthreads();
    md->s->cv_jcur = 0;
    cudaDeviceJacCopy(diA, dsavedJ, dA);
    __syncthreads();
  } else {
  __syncthreads();
    md->s->nstlj = md->s->cv_nst;
    md->s->cv_jcur = 1;
  __syncthreads();
    int aux_flag=0;
    int guess_flag=cudaDeviceJac(&aux_flag,md,dmdv);
    __syncthreads();
    if (guess_flag < 0) {
      return -1;}
    if (guess_flag > 0) {
      return 1;}
   cudaDeviceJacCopy(diA, dA, dsavedJ);
  }
  __syncthreads();
  cudaDeviceBCGprecond(dA, djA, diA, ddiag, -md->s->cv_gamma);
  __syncthreads();
  return 0;
}

__device__
void solveBcgCudaDeviceCVODE(ModelDataGPU *md, ModelDataVariable *dmdv)
{
#ifdef DEBUG_printmin
  printmin(md,dtempv,"solveBcgCudaDeviceCVODEStart dtempv");
#endif
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  double* dA = md->dA;
  int* djA = md->djA;
  int* diA = md->diA;
  double* dx = md->dx;
  double* dtempv = md->dtempv;
  int nrows = md->nrows;
  int n_shr_empty = md->n_shr_empty;
  int maxIt = md->maxIt;
  double tolmax = md->tolmax;
  double* ddiag = md->ddiag;
  double* dr0 = md->dr0;
  double* dr0h = md->dr0h;
  double* dn0 = md->dn0;
  double* dp0 = md->dp0;
  double* dt = md->dt;
  double* ds = md->ds;
  double* dy = md->dy;
  double alpha,rho0,omega0,beta,rho1,temp1,temp2;
  alpha=rho0=omega0=beta=rho1=temp1=temp2=1.0;
  cudaDevicesetconst(dn0, 0.0);
  cudaDevicesetconst(dp0, 0.0);
  cudaDeviceSpmv(dr0,dx,dA,djA,diA);
  cudaDeviceaxpby(dr0,dtempv,1.0,-1.0,nrows);
  cudaDeviceyequalsx(dr0h,dr0,nrows);
  int it=0;
  do{
    cudaDevicedotxy_2(dr0, dr0h, &rho1, n_shr_empty);
    beta = (rho1 / rho0) * (alpha / omega0);
    dp0[i]=beta*dp0[i]+dr0[i]-dn0[i]*omega0*beta;
    dy[i]=ddiag[i]*dp0[i];
    cudaDeviceSpmv(dn0, dy, dA, djA, diA);
    cudaDevicedotxy_2(dr0h, dn0, &temp1, n_shr_empty);
    alpha = rho1 / temp1;
    ds[i]=dr0[i]-alpha*dn0[i];
    dx[i]+=alpha*dy[i];
    dy[i]=ddiag[i]*ds[i];
    cudaDeviceSpmv(dt, dy, dA, djA, diA);
    dr0[i]=ddiag[i]*dt[i];
    cudaDevicedotxy_2(dy, dr0, &temp1, n_shr_empty);
    cudaDevicedotxy_2(dr0, dr0, &temp2, n_shr_empty);
    omega0 = temp1 / temp2;
    dx[i]+=omega0*dy[i];
    dr0[i]=ds[i]-omega0*dt[i];
    cudaDevicesetconst(dt, 0.0);
    cudaDevicedotxy_2(dr0, dr0, &temp1, n_shr_empty);
    temp1 = sqrtf(temp1);
    rho0 = rho1;
    it++;
  } while(it<maxIt && temp1>tolmax);//while(it<maxIt && temp1>tolmax);//while(0);
  __syncthreads();
#ifdef CAMP_DEBUG_GPU
#ifdef CAMP_PROFILE_DEVICE_FUNCTIONS
  if(threadIdx.x==0) md->s->counterBCGInternal += it;
  if(threadIdx.x==0) md->s->counterBCG++;
#endif
#endif
}

__device__
int cudaDevicecvNewtonIteration(ModelDataGPU *md, ModelDataVariable *dmdv){
  extern __shared__ double flag_shr2[];
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int aux_flag=0;
  double* dx = md->dx;
  double* dtempv = md->dtempv;
  int nrows = md->nrows;
  double cv_tn = md->s->cv_tn;
  double* dftemp = md->dftemp;
  double* dcv_y = md->dcv_y;
  double* dtempv1 = md->dtempv1;
  double* dtempv2 = md->dtempv2;
  double cv_next_h = md->s->cv_next_h;
  int n_shr_empty = md->n_shr_empty;
  double* cv_acor = md->cv_acor;
  double* dzn = md->dzn;
  double* dewt = md->dewt;
  double del, delp, dcon, m;
  del = delp = 0.0;
  __syncthreads();
#ifdef CAMP_DEBUG_GPU
#ifdef CAMP_PROFILE_DEVICE_FUNCTIONS
  int clock_khz=md->clock_khz;
  clock_t start;
#endif
#endif
#ifdef DEBUG_printmin
  printmin(md,dtempv,"cudaDevicecvNewtonIterationStart dtempv");
#endif
  for(;;) {
#ifdef DEBUG_printmin
    printmin(md,dftemp,"cudaDevicecvNewtonIteration dftemp");
#endif
    __syncthreads();
#ifdef CAMP_DEBUG_GPU
#ifdef CAMP_PROFILE_DEVICE_FUNCTIONS
    start = clock();
#endif
#endif
    dtempv[i]=md->s->cv_rl1*(dzn[i+nrows])+cv_acor[i];
    dtempv[i]=md->s->cv_gamma*dftemp[i]-dtempv[i];
    solveBcgCudaDeviceCVODE(md, dmdv);
    __syncthreads();
#ifdef CAMP_DEBUG_GPU
#ifdef CAMP_PROFILE_DEVICE_FUNCTIONS
    if(threadIdx.x==0) md->s->dtBCG += ((double)(int)(clock() - start))/(clock_khz*1000);
#endif
#endif
    __syncthreads();
    dtempv[i] = dx[i];
    __syncthreads();
#ifdef DEBUG_printmin
    printmin(md,dcv_y,"cudaDevicecvNewtonIteration dcv_y");
    printmin(md,dtempv,"cudaDevicecvNewtonIteration dtempv");
#endif
    dftemp[i]=dcv_y[i]+dtempv[i];
#ifdef DEBUG_cudaDevicecvNewtonIteration
    //if(i==0)printf("cudaDevicecvNewtonIteration dftemp %le dtempv %le dcv_y %le it %d block %d\n",
    //               dftemp[(blockDim.x-1)*0],dtempv[(blockDim.x-1)*0],dcv_y[(blockDim.x-1)*0],it,blockIdx.x);
#endif
#ifdef DEBUG_printmin
    printmin(md,dftemp,"cudaDevicecvNewtonIteration dftemp");
#endif
    __syncthreads();
    int guessflag=CudaDeviceguess_helper(cv_tn, 0., dftemp,
                           dcv_y, dtempv, dtempv1,
                           dtempv2, &aux_flag, md, dmdv
    );
    __syncthreads();
    if (guessflag < 0) {
      if (!(md->s->cv_jcur)) { //Bool set up during linsolsetup just before Jacobian
        return TRY_AGAIN;
      } else {
        return RHSFUNC_RECVR;
      }
    }
    dftemp[i]=dcv_y[i]+dtempv[i];
    double min;
    cudaDevicemin(&min, dftemp[i], flag_shr2, md->n_shr_empty);
    if (min < -CAMP_TINY) {
      //if (dftemp[i] < -CAMP_TINY) {
      return CONV_FAIL;
    }
    __syncthreads();
    cv_acor[i]+=dx[i];
    dcv_y[i]=dzn[i]+cv_acor[i];
    cudaDeviceVWRMS_Norm_2(dx, dewt, &del, nrows, n_shr_empty);
    if (m > 0) {
      md->s->cv_crate = SUNMAX(0.3 * md->s->cv_crate, del / delp);
    }
    dcon = del * SUNMIN(1.0, md->s->cv_crate) / md->cv_tq[4+blockIdx.x*(NUM_TESTS + 1)];
    flag_shr2[0]=0;
    __syncthreads();
    if (dcon <= 1.0) {
      cudaDeviceVWRMS_Norm_2(cv_acor, dewt, &md->s->cv_acnrm, nrows, n_shr_empty);
      __syncthreads();
      md->s->cv_jcur = 0;
      __syncthreads();
      return CV_SUCCESS;
    }
    if ((m == md->cv_maxcor) || ((m >= 2) && (del > RDIV * delp))) {
      if (!(md->s->cv_jcur)) {
        return TRY_AGAIN;
      } else {
        return RHSFUNC_RECVR;
      }
    }
    delp = del;
    __syncthreads();
#ifdef DEBUG_printmin
    printmin(md,md->state,"cudaDevicef start state");
#endif
    int retval=cudaDevicef(
            cv_next_h, dcv_y, dftemp, md, dmdv, &aux_flag
    );
    __syncthreads();
    cv_acor[i]=dcv_y[i]+dzn[i];
    if (retval < 0) {
      return CV_RHSFUNC_FAIL;
    }
    if (retval > 0) {
      if (!(md->s->cv_jcur)) {
        return TRY_AGAIN;
      } else {
        return RHSFUNC_RECVR;
      }
    }
    __syncthreads();
#ifdef CAMP_DEBUG_GPU
#ifdef CAMP_PROFILE_DEVICE_FUNCTIONS
    if(threadIdx.x==0) md->s->dtPostBCG += ((double)(clock() - start))/(clock_khz*1000);
#endif
#endif
#ifdef DEBUG_cudaDevicecvNewtonIteration
    if(i==0)printf("cudaDevicecvNewtonIteration dzn[(blockDim.x*(blockIdx.x+1)-1)*0] %le it %d block %d\n",dzn[(blockDim.x*(blockIdx.x+1)-1)*0],it,blockIdx.x);
#endif
  }
}

__device__
int cudaDevicecvNlsNewton(int nflag,
        ModelDataGPU *md, ModelDataVariable *dmdv
) {
  extern __shared__ int flag_shr[];
  int flagDevice = 0;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  double* dcv_y = md->dcv_y;
  double* cv_acor = md->cv_acor;
  double* dzn = md->dzn;
  double* dftemp = md->dftemp;
  double cv_tn = md->s->cv_tn;
  double cv_h = md->s->cv_h;
  double* dtempv = md->dtempv;
  double cv_next_h = md->s->cv_next_h;
#ifdef DEBUG_printmin
  printmin(md,dtempv,"cudaDevicecvNlsNewtonStart dtempv");
#endif
  __syncthreads();
#ifdef CAMP_DEBUG_GPU
#ifdef CAMP_PROFILE_DEVICE_FUNCTIONS
  int clock_khz=md->clock_khz;
  clock_t start;
#endif
#endif
  int convfail = ((nflag == FIRST_CALL) || (nflag == PREV_ERR_FAIL)) ?
                 CV_NO_FAILURES : CV_FAIL_OTHER;
  int dgamrat=fabs(md->s->cv_gamrat - 1.);
  int callSetup = (nflag == PREV_CONV_FAIL) || (nflag == PREV_ERR_FAIL) ||
                  (md->s->cv_nst == 0) ||
                  (md->s->cv_nst >= md->s->cv_nstlp + MSBP) ||
                  (dgamrat > DGMAX);
  dftemp[i]=dzn[i]+(-md->cv_last_yn[i]);
  __syncthreads();
  int guessflag=CudaDeviceguess_helper(cv_tn, cv_h, dzn,
             md->cv_last_yn, dftemp, dtempv,
             md->cv_acor_init,  &flagDevice,
             md, dmdv
  );
  __syncthreads();
#ifdef DEBUG_printmin
  printmin(md,dtempv,"cudaDevicecvSet after guess_helper dtempv");
#endif
  if(guessflag<0){
    return RHSFUNC_RECVR;
  }
  for(;;) {
    __syncthreads();
    dcv_y[i] = dzn[i];
#ifdef DEBUG_printmin
    //printmin(md,md->state,"cudaDevicef start state");
#endif
    int aux_flag=0;
    int retval=cudaDevicef(cv_next_h, dcv_y,
            dftemp,md,dmdv,&aux_flag
    );
    if (retval < 0) {
      return CV_RHSFUNC_FAIL;
    }
    if (retval> 0) {
      return RHSFUNC_RECVR;
    }
    if (callSetup==1) {
      __syncthreads();
#ifdef CAMP_DEBUG_GPU
#ifdef CAMP_PROFILE_DEVICE_FUNCTIONS
      start = clock();
#endif
#endif
      __syncthreads();
      int linflag=cudaDevicelinsolsetup(md, dmdv,convfail);
      __syncthreads();
#ifdef CAMP_DEBUG_GPU
#ifdef CAMP_PROFILE_DEVICE_FUNCTIONS
      if(threadIdx.x==0) md->s->timelinsolsetup += ((double)(clock() - start))/(clock_khz*1000);
#endif
#endif
      callSetup = 0;
      md->s->cv_gamrat = md->s->cv_crate = 1.0;
      md->s->cv_gammap = md->s->cv_gamma;
      //if(threadIdx.x==0)
        md->s->cv_nstlp = md->s->cv_nst;
      __syncthreads();
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
    __syncthreads();
#ifdef CAMP_DEBUG_GPU
#ifdef CAMP_PROFILE_DEVICE_FUNCTIONS
    start = clock();
#endif
#endif
    __syncthreads();
    int nItflag=cudaDevicecvNewtonIteration(md, dmdv);
    __syncthreads();
#ifdef CAMP_DEBUG_GPU
#ifdef CAMP_PROFILE_DEVICE_FUNCTIONS
    if(threadIdx.x==0)  md->s->timeNewtonIteration += ((double)(clock() - start))/(clock_khz*1000);
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
  } //for(;;)
  __syncthreads();
  return nflag;
}

__device__
void cudaDevicecvRescale(ModelDataGPU *md, ModelDataVariable *dmdv) {
  extern __shared__ double dzn[];
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  double factor;
  __syncthreads();
  factor = md->s->cv_eta;
  for (int j=1; j <= md->s->cv_q; j++) {
    md->dzn[i+md->nrows*j]*=factor;
    __syncthreads();
    factor *= md->s->cv_eta;
    __syncthreads();
  }
  md->s->cv_h = md->s->cv_hscale * md->s->cv_eta;
  md->s->cv_next_h = md->s->cv_h;
  md->s->cv_hscale = md->s->cv_h;
  __syncthreads();
}

__device__
void cudaDevicecvRestore(ModelDataGPU *md, ModelDataVariable *dmdv, double saved_t) {
  extern __shared__ double dzn[];
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j, k;
  __syncthreads();
  md->s->cv_tn=saved_t;
  for (k = 1; k <= md->s->cv_q; k++){
    for (j = md->s->cv_q; j >= k; j--) {
      md->dzn[i+md->nrows*(j-1)]-=md->dzn[i+md->nrows*j];
    }
  }
  md->dzn[i]=md->cv_last_yn[i];
  __syncthreads();
}

__device__
int cudaDevicecvHandleNFlag(ModelDataGPU *md, ModelDataVariable *dmdv, int *nflagPtr, double saved_t,
                             int *ncfPtr) {
  extern __shared__ int flag_shr[];
  if (*nflagPtr == CV_SUCCESS){
    return(DO_ERROR_TEST);
  }
  cudaDevicecvRestore(md, dmdv, saved_t);
  if (*nflagPtr == CV_LSETUP_FAIL)  return(CV_LSETUP_FAIL);
  if (*nflagPtr == CV_LSOLVE_FAIL)  return(CV_LSOLVE_FAIL);
  if (*nflagPtr == CV_RHSFUNC_FAIL) return(CV_RHSFUNC_FAIL);
  (*ncfPtr)++;
  md->s->cv_etamax = 1.;
  __syncthreads();
  if ((fabs(md->s->cv_h) <= md->s->cv_hmin*ONEPSM) ||
      (*ncfPtr == md->s->cv_maxncf)) {
    if (*nflagPtr == CONV_FAIL)     return(CV_CONV_FAILURE);
    if (*nflagPtr == RHSFUNC_RECVR) return(CV_REPTD_RHSFUNC_ERR);
  }
  __syncthreads();
  md->s->cv_eta = SUNMAX(ETACF,
          md->s->cv_hmin / fabs(md->s->cv_h));
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
  A2 = 1. + md->s->cv_q * A1;
  md->cv_tq[2+blockIdx.x*(NUM_TESTS + 1)] = fabs(A1 / (alpha0 * A2));
  md->cv_tq[5+blockIdx.x*(NUM_TESTS + 1)] = fabs(A2 * xistar_inv / (md->cv_l[md->s->cv_q+blockIdx.x*L_MAX] * xi_inv));
  if (md->s->cv_qwait == 1) {
    if (md->s->cv_q > 1) {
      C = xistar_inv / md->cv_l[md->s->cv_q+blockIdx.x*L_MAX];
      A3 = alpha0 + 1. / md->s->cv_q;
      A4 = alpha0_hat + xi_inv;
      Cpinv = (1. - A4 + A3) / A3;
      md->cv_tq[1+blockIdx.x*(NUM_TESTS + 1)] = fabs(C * Cpinv);
    }
    else md->cv_tq[1+blockIdx.x*(NUM_TESTS + 1)] = 1.;
    __syncthreads();
    hsum += md->cv_tau[md->s->cv_q+blockIdx.x*(L_MAX + 1)];
    xi_inv = md->s->cv_h / hsum;
    A5 = alpha0 - (1. / (md->s->cv_q+1));
    A6 = alpha0_hat - xi_inv;
    Cppinv = (1. - A6 + A5) / A2;
    md->cv_tq[3+blockIdx.x*(NUM_TESTS + 1)] = fabs(Cppinv / (xi_inv * (md->s->cv_q+2) * A5));
    __syncthreads();
  }
  md->cv_tq[4+blockIdx.x*(NUM_TESTS + 1)] = md->cv_nlscoef / md->cv_tq[2+blockIdx.x*(NUM_TESTS + 1)];
}

__device__
void cudaDevicecvSetBDF(ModelDataGPU *md, ModelDataVariable *dmdv) {
  extern __shared__ int flag_shr[];
  double alpha0, alpha0_hat, xi_inv, xistar_inv, hsum;
  int z,j;
  __syncthreads();
  md->cv_l[0+blockIdx.x*L_MAX] = md->cv_l[1+blockIdx.x*L_MAX] = xi_inv = xistar_inv = 1.;
  for (z=2; z <= md->s->cv_q; z++) md->cv_l[z+blockIdx.x*L_MAX] = 0.;
  alpha0 = alpha0_hat = -1.;
  hsum = md->s->cv_h;
  __syncthreads();
  if (md->s->cv_q > 1) {
    for (j=2; j < md->s->cv_q; j++) {
      hsum += md->cv_tau[j-1+blockIdx.x*(L_MAX + 1)];
      xi_inv = md->s->cv_h / hsum;
      alpha0 -= 1. / j;
      for (z=j; z >= 1; z--) md->cv_l[z+blockIdx.x*L_MAX] += md->cv_l[z-1+blockIdx.x*L_MAX]*xi_inv;
    }
    __syncthreads();
    alpha0 -= 1. / md->s->cv_q;
    xistar_inv = -md->cv_l[1+blockIdx.x*L_MAX] - alpha0;
    hsum += md->cv_tau[md->s->cv_q-1+blockIdx.x*(L_MAX + 1)];
    xi_inv = md->s->cv_h / hsum;
    alpha0_hat = -md->cv_l[1+blockIdx.x*L_MAX] - xi_inv;
    for (z=md->s->cv_q; z >= 1; z--)
      md->cv_l[z+blockIdx.x*L_MAX] += md->cv_l[z-1+blockIdx.x*L_MAX]*xistar_inv;
  }
  __syncthreads();
  cudaDevicecvSetTqBDFt(md, dmdv, hsum, alpha0, alpha0_hat, xi_inv, xistar_inv);
}

__device__
void cudaDevicecvSet(ModelDataGPU *md, ModelDataVariable *dmdv) {
  extern __shared__ int flag_shr[];
#ifdef DEBUG_printmin
  printmin(md,md->dtempv,"cudaDevicecvSet Start dtempv");
#endif
  __syncthreads();
  cudaDevicecvSetBDF(md,dmdv);
  __syncthreads();
  md->s->cv_rl1 = 1.0 / md->cv_l[1+blockIdx.x*L_MAX];
  md->s->cv_gamma = md->s->cv_h * md->s->cv_rl1;
  __syncthreads();
  if (md->s->cv_nst == 0){
    md->s->cv_gammap = md->s->cv_gamma;
  }
  __syncthreads();
  md->s->cv_gamrat = (md->s->cv_nst > 0) ?
                    md->s->cv_gamma / md->s->cv_gammap : 1.;  // protect x / x != 1.0
  __syncthreads();
}

__device__
void cudaDevicecvPredict(ModelDataGPU *md, ModelDataVariable *dmdv) {
  extern __shared__ double dzn[];
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j, k;
#ifdef DEBUG_printmin
  printmin(md,md->dtempv,"cudaDevicecvPredict start dtempv");
#endif
  __syncthreads();
  md->s->cv_tn += md->s->cv_h;
  __syncthreads();
  if (md->cv_tstopset) {
    if ((md->s->cv_tn - md->cv_tstop)*md->s->cv_h > 0.)
      md->s->cv_tn = md->cv_tstop;
  }
  md->cv_last_yn[i]=md->dzn[i];
  for (k = 1; k <= md->s->cv_q; k++){
    __syncthreads();
    for (j = md->s->cv_q; j >= k; j--){
      __syncthreads();
      md->dzn[i+md->nrows*(j-1)]+=md->dzn[i+md->nrows*j];
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
  for (z=0; z <= md->cv_qmax; z++) md->cv_l[z+blockIdx.x*L_MAX] = 0.;
  md->cv_l[2+blockIdx.x*L_MAX] = 1.;
  hsum = 0.;
  for (j=1; j <= md->s->cv_q-2; j++) {
    hsum += md->cv_tau[j+blockIdx.x*(L_MAX + 1)];
    xi = hsum /md->s->cv_hscale;
    for (z=j+2; z >= 2; z--)
      md->cv_l[z+blockIdx.x*L_MAX] = md->cv_l[z+blockIdx.x*L_MAX]*xi + md->cv_l[z-1+blockIdx.x*L_MAX];
  }
  for (j=2; j < md->s->cv_q; j++){
    md->dzn[md->nrows*j]-=md->cv_l[j+blockIdx.x*L_MAX]*md->dzn[md->nrows*(md->s->cv_q)];
  }
}

__device__
int cudaDevicecvDoErrorTest(ModelDataGPU *md, ModelDataVariable *dmdv,
                             int *nflagPtr,
                             double saved_t, int *nefPtr, double *dsmPtr) {
  extern __shared__ double dzn[];
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  double dsm;
  double min_val;
  int retval;
  md->dftemp[i]=md->cv_l[blockIdx.x*L_MAX]*md->cv_acor[i]+md->dzn[i];
  cudaDevicemin(&min_val, md->dftemp[i], dzn, md->n_shr_empty);
  if (min_val < 0. && min_val > -CAMP_TINY) {
    md->dftemp[i]=fabs(md->dftemp[i]);
    md->dzn[i]=md->dftemp[i]-md->cv_l[0+blockIdx.x*L_MAX]*md->cv_acor[i];
    min_val = 0.;
  }
  dsm = md->s->cv_acnrm * md->cv_tq[2+blockIdx.x*(NUM_TESTS + 1)];
  *dsmPtr = dsm;
  if (dsm <= 1. && min_val >= 0.) return(CV_SUCCESS);
  (*nefPtr)++;
  *nflagPtr = PREV_ERR_FAIL;
  cudaDevicecvRestore(md, dmdv, saved_t);
  __syncthreads();
  if ((fabs(md->s->cv_h) <= md->s->cv_hmin*ONEPSM) ||
      (*nefPtr == md->cv_maxnef)) return(CV_ERR_FAILURE);
  md->s->cv_etamax = 1.;
  __syncthreads();
  if (*nefPtr <= MXNEF1) {
    md->s->cv_eta = 1. / (pow(BIAS2*dsm,1./md->s->cv_L) + ADDON);
    __syncthreads();
    md->s->cv_eta = SUNMAX(ETAMIN, SUNMAX(md->s->cv_eta,
                           md->s->cv_hmin / fabs(md->s->cv_h)));
    __syncthreads();
    if (*nefPtr >= SMALL_NEF)
      md->s->cv_eta = SUNMIN(md->s->cv_eta, ETAMXF);
    __syncthreads();

    cudaDevicecvRescale(md, dmdv);
    return(TRY_AGAIN);
  }
  __syncthreads();
  if (md->s->cv_q > 1) {
    md->s->cv_eta = SUNMAX(ETAMIN,
    md->s->cv_hmin / fabs(md->s->cv_h));
    cudaDevicecvDecreaseBDF(md, dmdv);
    md->s->cv_L = md->s->cv_q;
    md->s->cv_q--;
    md->s->cv_qwait = md->s->cv_L;
    cudaDevicecvRescale(md, dmdv);
    __syncthreads();
    return(TRY_AGAIN);
  }
  __syncthreads();
  md->s->cv_eta = SUNMAX(ETAMIN, md->s->cv_hmin / fabs(md->s->cv_h));
  __syncthreads();
  md->s->cv_h *= md->s->cv_eta;
  md->s->cv_next_h = md->s->cv_h;
  md->s->cv_hscale = md->s->cv_h;
  __syncthreads();
  md->s->cv_qwait = 10;
#ifdef DEBUG_printmin
  printmin(md,md->state,"cudaDevicef start state");
#endif
  int aux_flag=0;
  retval=cudaDevicef(
          md->s->cv_tn, md->dzn, md->dtempv,md,dmdv, &aux_flag
  );
  if (retval < 0)  return(CV_RHSFUNC_FAIL);
  if (retval > 0)  return(CV_UNREC_RHSFUNC_ERR);
  md->dzn[1*md->nrows+i]=md->s->cv_h*md->dtempv[i];
  return(TRY_AGAIN);
}

__device__
void cudaDevicecvCompleteStep(ModelDataGPU *md, ModelDataVariable *dmdv) {
  extern __shared__ double dzn[];
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int z, j;
#ifdef DEBUG_printmin
  printmin(md,md->dtempv,"cudaDevicecvCompleteStep start dtempv");
#endif
  __syncthreads();
  if(threadIdx.x==0) md->s->cv_nst++;
  __syncthreads();
  md->s->cv_hu = md->s->cv_h;
  for (z=md->s->cv_q; z >= 2; z--)  md->cv_tau[z+blockIdx.x*(L_MAX + 1)] = md->cv_tau[z-1+blockIdx.x*(L_MAX + 1)];
  if ((md->s->cv_q==1) && (md->s->cv_nst > 1))
    md->cv_tau[2+blockIdx.x*(L_MAX + 1)] = md->cv_tau[1+blockIdx.x*(L_MAX + 1)];
  md->cv_tau[1+blockIdx.x*(L_MAX + 1)] = md->s->cv_h;
  __syncthreads();
  for (j=0; j <= md->s->cv_q; j++){
    md->dzn[i+md->nrows*j]+=md->cv_l[j+blockIdx.x*L_MAX]*md->cv_acor[i];
  }
  md->s->cv_qwait--;
  if ((md->s->cv_qwait == 1) && (md->s->cv_q != md->cv_qmax)) {
    md->dzn[i+md->nrows*(md->cv_qmax)]=md->cv_acor[i];
    md->s->cv_saved_tq5 = md->cv_tq[5+blockIdx.x*(NUM_TESTS + 1)];
    md->s->cv_indx_acor = md->cv_qmax;
  }
}

__device__
void cudaDevicecvChooseEta(ModelDataGPU *md, ModelDataVariable *dmdv) {
  extern __shared__ double dzn[];
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  double etam;
  etam = SUNMAX(md->s->cv_etaqm1, SUNMAX(md->s->cv_etaq, md->s->cv_etaqp1));
  __syncthreads();
  if (etam < THRESH) {
    md->s->cv_eta = 1.;
    md->s->cv_qprime = md->s->cv_q;
    return;
  }
  __syncthreads();
  if (etam == md->s->cv_etaq) {
    md->s->cv_eta = md->s->cv_etaq;
    md->s->cv_qprime = md->s->cv_q;
  } else if (etam == md->s->cv_etaqm1) {
    md->s->cv_eta = md->s->cv_etaqm1;
    md->s->cv_qprime = md->s->cv_q - 1;
  } else {
    md->s->cv_eta = md->s->cv_etaqp1;
    md->s->cv_qprime = md->s->cv_q + 1;
    __syncthreads();
    md->dzn[md->nrows*(md->cv_qmax)+i]=md->cv_acor[i];
  }
  __syncthreads();
}

__device__
void cudaDevicecvSetEta(ModelDataGPU *md, ModelDataVariable *dmdv) {
  extern __shared__ int flag_shr[];
  __syncthreads();
  if (md->s->cv_eta < THRESH) {
    md->s->cv_eta = 1.;
    md->s->cv_hprime = md->s->cv_h;
  } else {
    __syncthreads();
    md->s->cv_eta = SUNMIN(md->s->cv_eta, md->s->cv_etamax);
    __syncthreads();
    md->s->cv_eta /= SUNMAX(ONE,
            fabs(md->s->cv_h)*md->cv_hmax_inv*md->s->cv_eta);
    __syncthreads();
    md->s->cv_hprime = md->s->cv_h * md->s->cv_eta;
    __syncthreads();
  }
  __syncthreads();
}

__device__
int cudaDevicecvPrepareNextStep(ModelDataGPU *md, ModelDataVariable *dmdv, double dsm) {
  extern __shared__ double sdata[];
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  __syncthreads();
#ifdef DEBUG_printmin
  printmin(md,md->dtempv,"cudaDevicecvPrepareNextStep start dtempv");
#endif
  if (md->s->cv_etamax == 1.) {
    md->s->cv_qwait = SUNMAX(md->s->cv_qwait, 2);
    md->s->cv_qprime = md->s->cv_q;
    md->s->cv_hprime = md->s->cv_h;
    md->s->cv_eta = 1.;
    return 0;
  }
  __syncthreads();
  md->s->cv_etaq = 1. /(pow(BIAS2*dsm,1./md->s->cv_L) + ADDON);
  __syncthreads();
  if (md->s->cv_qwait != 0) {
    md->s->cv_eta = md->s->cv_etaq;
    md->s->cv_qprime = md->s->cv_q;
    cudaDevicecvSetEta(md, dmdv);
    return 0;
  }
  __syncthreads();
  md->s->cv_qwait = 2;
  double ddn;
  md->s->cv_etaqm1 = 0.;
  __syncthreads();
  if (md->s->cv_q > 1) {
    cudaDeviceVWRMS_Norm_2(&md->dzn[md->nrows*(md->s->cv_q)],
                         md->dewt, &ddn, md->nrows, md->n_shr_empty);
    __syncthreads();
    ddn *= md->cv_tq[1+blockIdx.x*(NUM_TESTS + 1)];
    __syncthreads();
    md->s->cv_etaqm1 = 1./(pow(BIAS1*ddn, 1./md->s->cv_q) + ADDON);
  }
  double dup, cquot;
  md->s->cv_etaqp1 = 0.;
  __syncthreads();
  if (md->s->cv_q != md->cv_qmax && md->s->cv_saved_tq5 != 0.) {
    cquot = (md->cv_tq[5+blockIdx.x*(NUM_TESTS + 1)] / md->s->cv_saved_tq5) *
            pow(double(md->s->cv_h/md->cv_tau[2+blockIdx.x*(L_MAX + 1)]), double(md->s->cv_L));
    md->dtempv[i]=md->cv_acor[i]-cquot*md->dzn[i+md->nrows*md->cv_qmax];
    cudaDeviceVWRMS_Norm_2(md->dtempv, md->dewt, &dup, md->nrows, md->n_shr_empty);
    __syncthreads();
    dup *= md->cv_tq[3+blockIdx.x*(NUM_TESTS + 1)];
    __syncthreads();
    md->s->cv_etaqp1 = 1. / (pow(BIAS3*dup, 1./(md->s->cv_L+1)) + ADDON);
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
  int tid = threadIdx.x;
  double alpha0, alpha1, prod, xi, xiold, hsum, A1;
  int z, j;
  for (z=0; z <= md->cv_qmax; z++) md->cv_l[z+blockIdx.x*L_MAX] = 0.;
  md->cv_l[2+blockIdx.x*L_MAX] = alpha1 = prod = xiold = 1.;
  alpha0 = -1.;
  hsum = md->s->cv_hscale;
  if (md->s->cv_q > 1) {
    for (j=1; j < md->s->cv_q; j++) {
      hsum += md->cv_tau[j+1+blockIdx.x*(L_MAX + 1)];
      xi = hsum / md->s->cv_hscale;
      prod *= xi;
      alpha0 -= 1. / (j+1);
      alpha1 += 1. / xi;
      for (z=j+2; z >= 2; z--)
        md->cv_l[z+blockIdx.x*L_MAX] = md->cv_l[z+blockIdx.x*L_MAX]*xiold + md->cv_l[z-1+blockIdx.x*L_MAX];
      xiold = xi;
    }
  }
  A1 = (-alpha0 - alpha1) / prod;
  dzn[tid]=md->dzn[md->nrows*(md->s->cv_L)+i];
  dzn[tid]=A1*md->dzn[md->nrows*(md->s->cv_indx_acor)+i];
  md->dzn[md->nrows*(md->s->cv_L)+i]=dzn[tid];
  for (j=2; j <= md->s->cv_q; j++){
    md->dzn[i+md->nrows*j]+=md->cv_l[j+blockIdx.x*L_MAX]*md->dzn[i+md->nrows*(md->s->cv_L)];
  }
}

__device__
void cudaDevicecvAdjustParams(ModelDataGPU *md, ModelDataVariable *dmdv) {
  if (md->s->cv_qprime != md->s->cv_q) {
    int deltaq = md->s->cv_qprime-md->s->cv_q;
    switch(deltaq) {
      case 1:
        cudaDevicecvIncreaseBDF(md, dmdv);
        break;
      case -1:
        cudaDevicecvDecreaseBDF(md, dmdv);
        break;
    }
    md->s->cv_q = md->s->cv_qprime;
    md->s->cv_L = md->s->cv_q+1;
    md->s->cv_qwait = md->s->cv_L;
  }
  cudaDevicecvRescale(md, dmdv);
}

__device__
int cudaDevicecvStep(ModelDataGPU *md, ModelDataVariable *dmdv) {
  extern __shared__ double sdata[];
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  double saved_t = md->s->cv_tn;
  int ncf = 0;
  int nef = 0;
  int nflag=FIRST_CALL;
  double dsm;
  __syncthreads();
  if ((md->s->cv_nst > 0) && (md->s->cv_hprime != md->s->cv_h)){
    cudaDevicecvAdjustParams(md, dmdv);
  }
  __syncthreads();
  for (;;) {
    __syncthreads();
    cudaDevicecvPredict(md, dmdv);
    __syncthreads();
    cudaDevicecvSet(md, dmdv);
    __syncthreads();
    nflag = cudaDevicecvNlsNewton(nflag,md, dmdv);
    __syncthreads();
#ifdef DEBUG_cudaDevicecvStep
    if(threadIdx.x==0)printf("DEBUG_cudaDevicecvStep nflag %d block %d\n",nflag, blockIdx.x);
#endif
    int kflag = cudaDevicecvHandleNFlag(md, dmdv, &nflag, saved_t, &ncf);
    __syncthreads();
#ifdef DEBUG_cudaDevicecvStep
    if(threadIdx.x==0)printf("DEBUG_cudaDevicecvStep kflag %d block %d\n",kflag, blockIdx.x);
#endif
    if (kflag == PREDICT_AGAIN) {
      continue;
    }
    if (kflag != DO_ERROR_TEST) {
      return (kflag);
    }
    __syncthreads();
    int eflag=cudaDevicecvDoErrorTest(md,dmdv,&nflag,saved_t,&nef,&dsm);
    __syncthreads();
#ifdef DEBUG_cudaDevicecvStep
    if(threadIdx.x==0)printf("DEBUG_cudaDevicecvStep nflag %d eflag %d block %d\n",nflag, eflag, blockIdx.x);    //if(i==0)printf("eflag %d\n", eflag);
#endif
    if (eflag == TRY_AGAIN){
      continue;
    }
    if (eflag != CV_SUCCESS){
      return (eflag);
    }
    break;
  }
  __syncthreads();
  cudaDevicecvCompleteStep(md, dmdv);
  __syncthreads();
  cudaDevicecvPrepareNextStep(md, dmdv, dsm);
  __syncthreads();
  md->s->cv_etamax=10.;
  md->cv_acor[i]*=md->cv_tq[2+blockIdx.x*(NUM_TESTS + 1)];
  __syncthreads();
  return(CV_SUCCESS);
  }

__device__
int cudaDeviceCVodeGetDky(ModelDataGPU *md, ModelDataVariable *dmdv,
                           double t, int k, double *dky) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  double s, c, r;
  double tfuzz, tp, tn1;
  int z, j;
  __syncthreads();
   tfuzz = FUZZ_FACTOR * md->cv_uround * (fabs(md->s->cv_tn) + fabs(md->s->cv_hu));
   if (md->s->cv_hu < 0.) tfuzz = -tfuzz;
   tp = md->s->cv_tn - md->s->cv_hu - tfuzz;
   tn1 = md->s->cv_tn + tfuzz;
   if ((t-tp)*(t-tn1) > 0.) {
     return(CV_BAD_T);
   }
  __syncthreads();
   s = (t - md->s->cv_tn) / md->s->cv_h;
   for (j=md->s->cv_q; j >= k; j--) {
     c = 1.;
     for (z=j; z >= j-k+1; z--) c *= z;
     if (j == md->s->cv_q) {
       dky[i]=c*md->dzn[i+md->nrows*j];
     } else {
        dky[i]=c*md->dzn[i+md->nrows*j]+s*dky[i];
     }
   }
  __syncthreads();
   if (k == 0) return(CV_SUCCESS);
  __syncthreads();
   r = pow(double(md->s->cv_h),double(-k));
  __syncthreads();
   dky[i]=dky[i]*r;
   return(CV_SUCCESS);
}

__device__
int cudaDevicecvEwtSetSV(ModelDataGPU *md, ModelDataVariable *dmdv,
                         double *dzn, double *weight) {
  extern __shared__ double flag_shr2[];
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  md->dtempv[i]=fabs(dzn[i]);
  double min;
  md->dtempv[i]=md->cv_reltol*md->dtempv[i]+md->cv_Vabstol[i];
  cudaDevicemin(&min, md->dtempv[i], flag_shr2, md->n_shr_empty);
__syncthreads();
  if (min <= 0.) return(-1);
  weight[i]= 1./md->dtempv[i];
  return(0);
}

__device__
int cudaDeviceCVode(ModelDataGPU *md, ModelDataVariable *dmdv) {
  extern __shared__ int flag_shr[];
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int kflag2;
#ifdef DEBUG_printmin
  printmin(md,md->state,"cudaDeviceCVode start state");
#endif
  for(;;) {
    __syncthreads();
#ifdef CAMP_DEBUG_GPU
#ifdef CAMP_PROFILE_DEVICE_FUNCTIONS
    if(threadIdx.x==0) md->s->countercvStep++;
#endif
#endif
    flag_shr[0] = 0;
    __syncthreads();
    md->s->cv_next_h = md->s->cv_h;
    md->s->cv_next_q = md->s->cv_q;
    int ewtsetOK = 0;
    if (md->s->cv_nst > 0) {
      ewtsetOK = cudaDevicecvEwtSetSV(md, dmdv, md->dzn, md->dewt);
      if (ewtsetOK != 0) {
        md->s->cv_tretlast = md->s->tret = md->s->cv_tn;
        md->yout[i] = md->dzn[i];
        if(i==0) printf("ERROR: ewtsetOK\n");
        return CV_ILL_INPUT;
      }
    }
    if ((md->cv_mxstep > 0) && (md->s->nstloc >= md->cv_mxstep)) {
      md->s->cv_tretlast = md->s->tret = md->s->cv_tn;
      md->yout[i] = md->dzn[i];
      if(i==0) printf("ERROR: cv_mxstep\n");
      return CV_TOO_MUCH_WORK;
    }

    double nrm;
    cudaDeviceVWRMS_Norm_2(md->dzn,
                         md->dewt, &nrm, md->nrows, md->n_shr_empty);
    md->s->cv_tolsf = md->cv_uround * nrm;
    if (md->s->cv_tolsf > 1.) {
      md->s->cv_tretlast = md->s->tret = md->s->cv_tn;
      md->yout[i] = md->dzn[i];
      md->s->cv_tolsf *= 2.;
      if(i==0) printf("ERROR: cv_tolsf\n");
      __syncthreads();
      return CV_TOO_MUCH_ACC;
    } else {
      md->s->cv_tolsf = 1.;
    }
#ifdef ODE_WARNING
    if (md->s->cv_tn + md->s->cv_h == md->s->cv_tn) {
      if(threadIdx.x==0) md->s->cv_nhnil++;
      if ((md->s->cv_nhnil <= md->s->cv_mxhnil) ||
              (md->s->cv_nhnil == md->s->cv_mxhnil))
        if(i==0)printf("WARNING: h below roundoff level in tn");
    }
#endif

    kflag2 = cudaDevicecvStep(md, dmdv);

    __syncthreads();
#ifdef DEBUG_cudaDeviceCVode
    if(i==0){
      printf("DEBUG_cudaDeviceCVode%d thread %d\n", i);
      printf("md->s->cv_tn %le md->tout %le md->s->cv_h %le md->s->cv_hprime %le\n",
             md->s->cv_tn,md->tout,md->s->cv_h,md->s->cv_hprime);
    }
#endif
    if (kflag2 != CV_SUCCESS) {
      md->s->cv_tretlast = md->s->tret = md->s->cv_tn;
      md->yout[i] = md->dzn[i];
      if(i==0) printf("ERROR: dmdv->kflag != CV_SUCCESS\n");
      return kflag2;
    }
    md->s->nstloc++;
    if ((md->s->cv_tn - md->tout) * md->s->cv_h >= 0.) {
      md->s->cv_tretlast = md->s->tret = md->tout;
      cudaDeviceCVodeGetDky(md, dmdv, md->tout, 0, md->yout);
      return CV_SUCCESS;
    }
    if (md->cv_tstopset) {//needed?
      double troundoff = FUZZ_FACTOR * md->cv_uround * (fabs(md->s->cv_tn) + fabs(md->s->cv_h));
      if (fabs(md->s->cv_tn - md->cv_tstop) <= troundoff) {
        cudaDeviceCVodeGetDky(md, dmdv, md->cv_tstop, 0, md->yout);
        md->s->cv_tretlast = md->s->tret = md->cv_tstop;
        md->cv_tstopset = SUNFALSE;
        if(i==0) printf("ERROR: cv_tstopset\n");
        __syncthreads();
        return CV_TSTOP_RETURN;
      }
      if ((md->s->cv_tn + md->s->cv_hprime - md->cv_tstop) * md->s->cv_h > 0.) {
        md->s->cv_hprime = (md->cv_tstop - md->s->cv_tn) * (1.0 - 4.0 * md->cv_uround);
        if(i==0) printf("ERROR: dmdv->cv_tn + dmdv->cv_hprime - dmdv->cv_tstop\n");
        md->s->cv_eta = md->s->cv_hprime / md->s->cv_h;
      }
    }
  }
}

__global__
void cudaGlobalCVode(ModelDataGPU md_object) {
  extern __shared__ int flag_shr[];
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  ModelDataGPU *md = &md_object;
  md->s=&md->sCells[blockIdx.x];
  int active_threads = md->nrows;
  int istate;
  __syncthreads();
  if(tid<active_threads){
    __syncthreads();
#ifdef CAMP_DEBUG_GPU
#ifdef CAMP_PROFILE_DEVICE_FUNCTIONS
    int clock_khz=md->clock_khz;
    clock_t start;
    start = clock();
    __syncthreads();
#endif
#endif
    istate=cudaDeviceCVode(md,md->s);//dmdv as a function parameter seems faster than removing it
    __syncthreads();
#ifdef CAMP_DEBUG_GPU
#ifdef CAMP_PROFILE_DEVICE_FUNCTIONS
  if(threadIdx.x==0) md->s->dtcudaDeviceCVode += ((double)(int)(clock() - start))/(clock_khz*1000);
  __syncthreads();
#endif
#endif
  }
  __syncthreads();
  if(threadIdx.x==0) md->flagCells[blockIdx.x]=istate;
#ifdef CAMP_PROFILE_DEVICE_FUNCTIONS
  ModelDataVariable *mdvo = md->mdvo;
  *mdvo = *md->s;
#endif
}

int nextPowerOfTwoCVODE2(int v){
  v--;
  v |= v >> 1;
  v |= v >> 2;
  v |= v >> 4;
  v |= v >> 8;
  v |= v >> 16;
  v++;
  return v;
}

/*

 void cvodeRun(ModelDataGPU *mGPU, cudaStream_t stream){
  int len_cell = mGPU->nrows / mGPU->n_cells;
  int threads_block = len_cell;
  int blocks = mGPU->n_cells;
  int n_shr_memory = nextPowerOfTwoCVODE2(len_cell);
  int n_shr_empty = mGPU->n_shr_empty = n_shr_memory - threads_block;
  cudaGlobalCVode <<<blocks, threads_block, n_shr_memory * sizeof(double), stream>>>(*mGPU);
}

*/
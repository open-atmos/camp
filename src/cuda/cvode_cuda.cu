/* Copyright (C) 2021 Barcelona Supercomputing Center and University of
* Illinois at Urbana-Champaign
* SPDX-License-Identifier: MIT
*/

#include "cvode_cuda.h"
extern "C" {
#include "new.h"
}

__device__
void print_double(double *x, int len, const char *s){
#ifndef USE_PRINT_ARRAYS
  __syncthreads();
  if(threadIdx.x==0 && blockIdx.x==0){
    for (int i=0; i<len; i++){
      printf("%s[%d]=%.17le\n",s,i,x[i]);
    }
  }
  __syncthreads();
#endif
}

__device__
void print_int(int *x, int len, const char *s){
#ifndef USE_PRINT_ARRAYS
  __syncthreads();
  if(threadIdx.x==0 && blockIdx.x==0){
    for (int i=0; i<len; i++){
      printf("%s[%d]=%d\n",s,i,x[i]);
    }
  }
  __syncthreads();
#endif
}

__device__
double dSUNRpowerR(double base, double exponent){
  if (base <= ZERO) return(ZERO);
  if(exponent==0.5) return sqrt(base);
  return(pow(base, exponent));
}

#ifndef DEV_removeAtomic

__device__
void time_derivative_add_value_gpu(TimeDerivativeGPU time_deriv, unsigned int spec_id,
                               double rate_contribution) {
  if (rate_contribution > 0.0) {
    time_deriv.production_rates[spec_id] += rate_contribution;
  } else {
    time_deriv.loss_rates[spec_id] += -rate_contribution;
  }
}

#else

__device__
void time_derivative_add_value_gpu(TimeDerivativeGPU time_deriv, unsigned int spec_id,
                               double rate_contribution) {
  if (rate_contribution > 0.0) {
    atomicAdd_block(&(time_deriv.production_rates[spec_id]),rate_contribution);
  } else {
    atomicAdd_block(&(time_deriv.loss_rates[spec_id]),-rate_contribution);
  }
}

#endif

__device__
void rxn_gpu_first_order_loss_calc_deriv_contrib(ModelDataVariable *sc, TimeDerivativeGPU time_deriv, int *rxn_int_data,
          double *rxn_float_data, double *rxn_env_data, double time_step){
  int *int_data = rxn_int_data;
  double rate = rxn_env_data[0] * sc->grid_cell_state[int_data[1]-1];
  if (int_data[2] >= 0) time_derivative_add_value_gpu(time_deriv, int_data[2], -rate);
}

__device__
void rxn_gpu_CMAQ_H2O2_calc_deriv_contrib(ModelDataVariable *sc, TimeDerivativeGPU time_deriv, int *rxn_int_data,
          double *rxn_float_data, double *rxn_env_data, double time_step){
  int *int_data = rxn_int_data;
  double *float_data = rxn_float_data;
  double rate = rxn_env_data[0];
  for (int i_spec=0; i_spec<int_data[0]; i_spec++) rate *= sc->grid_cell_state[int_data[(2 + i_spec)]-1];
  if (rate!=0.) {
    int i_dep_var = 0;
    for (int i_spec=0; i_spec<int_data[0]; i_spec++, i_dep_var++) {
      if (int_data[(2 + int_data[0]+int_data[1]+i_dep_var)] < 0) continue;
      time_derivative_add_value_gpu(time_deriv, int_data[(2 + int_data[0]+int_data[1]+i_dep_var)], -rate);
    }
    for (int i_spec=0; i_spec<int_data[1]; i_spec++, i_dep_var++) {
      if (int_data[(2 + int_data[0]+int_data[1]+i_dep_var)] < 0) continue;
      if (-rate*float_data[(7 + i_spec)]*time_step <= sc->grid_cell_state[int_data[(2 + int_data[0]+ i_spec)]-1]) {
        time_derivative_add_value_gpu(time_deriv, int_data[(2 + int_data[0]+int_data[1]+i_dep_var)],rate*float_data[(7 + i_spec)]);
      }
    }
  }
}

__device__
void rxn_gpu_CMAQ_OH_HNO3_calc_deriv_contrib(ModelDataVariable *sc, TimeDerivativeGPU time_deriv, int *rxn_int_data,
          double *rxn_float_data, double *rxn_env_data, double time_step){
  int *int_data = rxn_int_data;
  double *float_data = rxn_float_data;
  double rate = rxn_env_data[0];
  for (int i_spec=0; i_spec<int_data[0]; i_spec++) rate *= sc->grid_cell_state[int_data[(2 + i_spec)]-1];
  if (rate!=0.) {
    int i_dep_var = 0;
    for (int i_spec=0; i_spec<int_data[0]; i_spec++, i_dep_var++) {
      if (int_data[(2 + int_data[0] + int_data[1] + i_dep_var)] < 0) continue;
      time_derivative_add_value_gpu(time_deriv, int_data[(2 + int_data[0] + int_data[1] + i_dep_var)], -rate);
    }
    for (int i_spec=0; i_spec<int_data[1]; i_spec++, i_dep_var++) {
      if (int_data[(2 + int_data[0] + int_data[1] + i_dep_var)] < 0) continue;
      if (-rate*float_data[(11 + i_spec)]*time_step <= sc->grid_cell_state[int_data[(2 + int_data[0] + i_spec)]-1]) {
        time_derivative_add_value_gpu(time_deriv, int_data[(2 + int_data[0] + int_data[1] + i_dep_var)],rate*float_data[(11 + i_spec)]);
      }
    }
  }
}

__device__
void rxn_gpu_arrhenius_calc_deriv_contrib(ModelDataVariable *sc, TimeDerivativeGPU time_deriv,
                                int *rxn_int_data, double *rxn_float_data,
                                double *rxn_env_data, double time_step){
  int *int_data = rxn_int_data;
  double *float_data = rxn_float_data;
  double rate = rxn_env_data[0];
  for (int i_spec=0; i_spec<int_data[0]; i_spec++)
    rate *= sc->grid_cell_state[int_data[(2 + i_spec)]-1];
  if (rate!=0.) {
    int i_dep_var = 0;
    for (int i_spec=0; i_spec<int_data[0]; i_spec++, i_dep_var++) {
      if (int_data[2 + int_data[0] + int_data[1] + i_dep_var] < 0) continue;
      time_derivative_add_value_gpu(time_deriv, int_data[2 + int_data[0] + int_data[1] + i_dep_var], -rate);
    }
    for (int i_spec=0; i_spec<int_data[1]; i_spec++, i_dep_var++) {
      if (int_data[2 + int_data[0] + int_data[1] + i_dep_var] < 0) continue;
      if (-rate*float_data[6+i_spec]*time_step <= sc->grid_cell_state[int_data[(2 + int_data[0] + i_spec)]-1]) {
        time_derivative_add_value_gpu(time_deriv, int_data[2 + int_data[0] + int_data[1] + i_dep_var],rate*float_data[6+i_spec]);
      }
    }
  }
}

__device__
void rxn_gpu_troe_calc_deriv_contrib(ModelDataVariable *sc, TimeDerivativeGPU time_deriv, int *rxn_int_data,
          double *rxn_float_data, double *rxn_env_data, double time_step){
  int *int_data = rxn_int_data;
  double *float_data = rxn_float_data;
  double rate = rxn_env_data[0];
  for (int i_spec=0; i_spec<int_data[0]; i_spec++)
          rate *= sc->grid_cell_state[int_data[(2 + i_spec)]-1];
  if (rate!=ZERO) {
    int i_dep_var = 0;
    for (int i_spec=0; i_spec<int_data[0]; i_spec++, i_dep_var++) {
      if (int_data[(2 + int_data[0]+int_data[1]+i_dep_var)] < 0) continue;
      time_derivative_add_value_gpu(time_deriv, int_data[(2 + int_data[0]+int_data[1]+i_dep_var)], -rate);
    }
    for (int i_spec=0; i_spec<int_data[1]; i_spec++, i_dep_var++) {
      if (int_data[(2 + int_data[0]+int_data[1]+i_dep_var)] < 0) continue;
      if (-rate * float_data[(10 + i_spec)] * time_step <= sc->grid_cell_state[int_data[(2 + int_data[0]+ i_spec)]-1]) {
        time_derivative_add_value_gpu(time_deriv, int_data[(2 + int_data[0]+int_data[1]+i_dep_var)],rate*float_data[(10 + i_spec)]);
      }
    }
  }
}

__device__
void rxn_gpu_photolysis_calc_deriv_contrib(ModelDataVariable *sc, TimeDerivativeGPU time_deriv, int *rxn_int_data,
          double *rxn_float_data, double *rxn_env_data, double time_step){
  int *int_data = rxn_int_data;
  double *float_data = rxn_float_data;
  double rate = rxn_env_data[0];
  for (int i_spec=0; i_spec<int_data[0]; i_spec++)
          rate *= sc->grid_cell_state[int_data[(3 + i_spec)]-1];
  if (rate!=ZERO) {
    int i_dep_var = 0;
    for (int i_spec=0; i_spec<int_data[0]; i_spec++, i_dep_var++) {
      if (int_data[(3 + int_data[0]+int_data[1]+i_dep_var)] < 0) continue;
      time_derivative_add_value_gpu(time_deriv, int_data[(3 + int_data[0]+int_data[1]+i_dep_var)], -rate);
    }
    for (int i_spec=0; i_spec<int_data[1]; i_spec++, i_dep_var++) {
      if (int_data[(3 + int_data[0]+int_data[1]+i_dep_var)] < 0) continue;
        if (-rate * float_data[(1 + i_spec)] * time_step <= sc->grid_cell_state[int_data[(3 + int_data[0]+ i_spec)]-1]){
        time_derivative_add_value_gpu(time_deriv, int_data[(3 + int_data[0]+int_data[1]+i_dep_var)],rate*float_data[(1 + i_spec)]);
      }
    }
  }
}

#ifndef DEV_removeAtomic
__device__
void jacobian_add_value_gpu(JacobianGPU jac, unsigned int elem_id,
                            int prod_or_loss,
                            double jac_contribution) {
  if (prod_or_loss == JACOBIAN_PRODUCTION) {
    jac.production_partials[elem_id] += jac_contribution;
  }
  else{ //(prod_or_loss == JACOBIAN_LOSS){
    jac.loss_partials[elem_id] += jac_contribution;
  }
}
#else
__device__
void jacobian_add_value_gpu(JacobianGPU jac, unsigned int elem_id,
                                   int prod_or_loss,
                                   double jac_contribution) {
  if (prod_or_loss == JACOBIAN_PRODUCTION) {
    atomicAdd_block(&(jac.production_partials[elem_id]), jac_contribution);
  }
  else{ //(prod_or_loss == JACOBIAN_LOSS){
    atomicAdd_block(&(jac.loss_partials[elem_id]),jac_contribution);
  }
}
#endif

__device__
void rxn_gpu_first_order_loss_calc_jac_contrib(ModelDataVariable *sc, JacobianGPU jac, int *rxn_int_data,
          double *rxn_float_data, double *rxn_env_data, double time_step){
  int *int_data = rxn_int_data;
if (int_data[3] >= 0) jacobian_add_value_gpu(jac, int_data[3], JACOBIAN_LOSS,
                                         rxn_env_data[0]);
}

__device__
void rxn_gpu_CMAQ_H2O2_calc_jac_contrib(ModelDataVariable *sc, JacobianGPU jac, int *rxn_int_data,
          double *rxn_float_data, double *rxn_env_data, double time_step){
  int *int_data = rxn_int_data;
  double *float_data = rxn_float_data;
  int i_elem = 0;
  for (int i_ind = 0; i_ind < int_data[0]; i_ind++) {
    double rate = rxn_env_data[0];
    for (int i_spec = 0; i_spec < int_data[0]; i_spec++)
      if (i_ind != i_spec) rate *= sc->grid_cell_state[int_data[(2 + i_spec)]-1];
    for (int i_dep = 0; i_dep < int_data[0]; i_dep++, i_elem++) {
      if (int_data[(2 + 2*(int_data[0]+int_data[1])+i_elem)] < 0) continue;
      jacobian_add_value_gpu(jac, int_data[(2 + 2*(int_data[0]+int_data[1])+i_elem)], JACOBIAN_LOSS,
             rate);
    }
    for (int i_dep = 0; i_dep < int_data[1]; i_dep++, i_elem++) {
      if (int_data[(2 + 2*(int_data[0]+int_data[1])+i_elem)] < 0) continue;
      if (-rate * sc->grid_cell_state[int_data[(2 + i_ind)]-1] * float_data[(7 + i_dep)] * time_step <=
          sc->grid_cell_state[int_data[(2 + int_data[0]+ i_dep)]-1]) {
        jacobian_add_value_gpu(jac, int_data[(2 + 2*(int_data[0]+int_data[1])+i_elem)],
                   JACOBIAN_PRODUCTION, float_data[(7 + i_dep)] * rate);
      }
    }
  }
}


__device__
void rxn_gpu_CMAQ_OH_HNO3_calc_jac_contrib(ModelDataVariable *sc, JacobianGPU jac, int *rxn_int_data,
          double *rxn_float_data, double *rxn_env_data, double time_step){
  int *int_data = rxn_int_data;
  double *float_data = rxn_float_data;
  int i_elem = 0;
  for (int i_ind = 0; i_ind < int_data[0]; i_ind++) {
    double rate = rxn_env_data[0];
    for (int i_spec = 0; i_spec < int_data[0]; i_spec++)
      if (i_ind != i_spec) rate *= sc->grid_cell_state[int_data[(2 + i_spec)]-1];
    for (int i_dep = 0; i_dep < int_data[0]; i_dep++, i_elem++) {
      if (int_data[(2 + 2*(int_data[0]+int_data[1]) + i_elem)] < 0) continue;
      jacobian_add_value_gpu(jac, int_data[(2 + 2*(int_data[0]+int_data[1]) + i_elem)], JACOBIAN_LOSS,
                   rate);
    }
    for (int i_dep = 0; i_dep < int_data[1]; i_dep++, i_elem++) {
      if (int_data[(2 + 2*(int_data[0]+int_data[1]) + i_elem)] < 0) continue;
      if (-rate * sc->grid_cell_state[int_data[(2 + i_ind)]-1] * float_data[(11 + i_dep)] * time_step <=
          sc->grid_cell_state[int_data[(2 + int_data[0] + i_dep)]-1]) {
        jacobian_add_value_gpu(jac, int_data[(2 + 2*(int_data[0]+int_data[1]) + i_elem)],
         JACOBIAN_PRODUCTION, float_data[(11 + i_dep)] * rate);
      }
    }
  }
}


__device__
void rxn_gpu_arrhenius_calc_jac_contrib(ModelDataVariable *sc, JacobianGPU jac, int *rxn_int_data,
          double *rxn_float_data, double *rxn_env_data, double time_step){
  int *int_data = rxn_int_data;
  double *float_data = rxn_float_data;
  int i_elem = 0;
  for (int i_ind = 0; i_ind < int_data[0]; i_ind++) {
    double rate = rxn_env_data[0];
    for (int i_spec = 0; i_spec < int_data[0]; i_spec++)
      if (i_spec != i_ind) rate *= sc->grid_cell_state[int_data[(2 + i_spec)]-1];
    for (int i_dep = 0; i_dep < int_data[0]; i_dep++, i_elem++) {
      if (int_data[2 + 2*(int_data[0]+int_data[1]) + i_elem] < 0) continue;
      jacobian_add_value_gpu(jac, int_data[2 + 2*(int_data[0]+int_data[1]) + i_elem], JACOBIAN_LOSS,
                    rate);
    }
    for (int i_dep = 0; i_dep < int_data[1]; i_dep++, i_elem++) {
      if (int_data[2 + 2*(int_data[0]+int_data[1]) + i_elem] < 0) continue;
      if (-rate * sc->grid_cell_state[int_data[(2 + i_ind)]-1] * float_data[6+i_dep] * time_step <=
        sc->grid_cell_state[int_data[(2 + int_data[0] + i_dep)]-1]) {
        jacobian_add_value_gpu(jac, int_data[2 + 2*(int_data[0]+int_data[1]) + i_elem],
                           JACOBIAN_PRODUCTION, float_data[6+i_dep] * rate);
      }
    }
  }
}

__device__
void rxn_gpu_troe_calc_jac_contrib(ModelDataVariable *sc, JacobianGPU jac, int *rxn_int_data,
          double *rxn_float_data, double *rxn_env_data, double time_step){
  int *int_data = rxn_int_data;
  double *float_data = rxn_float_data;
  int i_elem = 0;
  for (int i_ind = 0; i_ind < int_data[0]; i_ind++) {
    double rate = rxn_env_data[0];
    for (int i_spec = 0; i_spec < int_data[0]; i_spec++)
      if (i_ind != i_spec) rate *= sc->grid_cell_state[int_data[(2 + i_spec)]-1];
    for (int i_dep = 0; i_dep < int_data[0]; i_dep++, i_elem++) {
      if (int_data[(2 + 2*(int_data[0]+int_data[1])+i_elem)] < 0) continue;
        jacobian_add_value_gpu(jac, int_data[(2 + 2*(int_data[0]+int_data[1])+i_elem)], JACOBIAN_LOSS,
                   rate);
        }
    for (int i_dep = 0; i_dep < int_data[1]; i_dep++, i_elem++) {
      if (int_data[(2 + 2*(int_data[0]+int_data[1])+i_elem)] < 0) continue;
      if (-rate * sc->grid_cell_state[int_data[(2 + i_ind)]-1] * float_data[(10 + i_dep)] * time_step <=
        sc->grid_cell_state[int_data[(2 + int_data[0]+ i_dep)]-1]) {
        jacobian_add_value_gpu(jac, int_data[(2 + 2*(int_data[0]+int_data[1])+i_elem)], JACOBIAN_PRODUCTION,
                               float_data[(10 + i_dep)] * rate);
      }
    }
  }
}

__device__
void rxn_gpu_photolysis_calc_jac_contrib(ModelDataVariable *sc, JacobianGPU jac, int *rxn_int_data,
          double *rxn_float_data, double *rxn_env_data, double time_step){
  int *int_data = rxn_int_data;
  double *float_data = rxn_float_data;
  int i_elem = 0;
  for (int i_ind = 0; i_ind < int_data[0]; i_ind++) {
    double rate = rxn_env_data[0];
    for (int i_spec = 0; i_spec < int_data[0]; i_spec++)
      if (i_spec != i_ind) rate *= sc->grid_cell_state[int_data[(3 + i_spec)]-1];
    for (int i_dep = 0; i_dep < int_data[0]; i_dep++, i_elem++) {
      if (int_data[(3 + 2*(int_data[0]+int_data[1])+i_elem)] < 0) continue;
      jacobian_add_value_gpu(jac, int_data[(3 + 2*(int_data[0]+int_data[1])+i_elem)], JACOBIAN_LOSS,
                   rate);
    }
    for (int i_dep = 0; i_dep < int_data[1]; i_dep++, i_elem++) {
      if (int_data[(3 + 2*(int_data[0]+int_data[1])+i_elem)] < 0) continue;
      if (-rate * sc->grid_cell_state[int_data[(3 + i_ind)]-1] * float_data[(1 + i_dep)] * time_step <=
          sc->grid_cell_state[int_data[(3 + int_data[0]+ i_dep)]-1]) {
      jacobian_add_value_gpu(jac, int_data[(3 + 2*(int_data[0]+int_data[1])+i_elem)],
              JACOBIAN_PRODUCTION, float_data[(1 + i_dep)] * rate);
      }
    }
  }
}

__device__ void cudaDevicemin_2(double *g_odata, double in, volatile double *sdata, int n_shr_empty){
  unsigned int tid = threadIdx.x;
  __syncthreads();
  sdata[tid] = in;
  __syncthreads();
  if(tid<n_shr_empty)
    sdata[tid+blockDim.x]=sdata[tid];
  __syncthreads();
  for (unsigned int s=(blockDim.x+n_shr_empty)/2; s>0; s>>=1)
  {
    if (tid < s){
      if(sdata[tid + s] < sdata[tid] ) sdata[tid]=sdata[tid + s];
    }
    __syncthreads();
  }
  *g_odata = sdata[0];
  __syncthreads();
}

#ifdef DEBUG_CVODE_GPU
__device__
void printmin(ModelDataGPU *md,double* y, const char *s) {
  __syncthreads();
  extern __shared__ double flag_shr2[];
  int tid= threadIdx.x + blockDim.x*blockIdx.x;
  __syncthreads();
  double min;
  cudaDevicemin_2(&min, y[tid], flag_shr2, md->n_shr_empty);
  __syncthreads();
  if(tid==0)printf("%s min %le\n",s,min);
  __syncthreads();
}
#endif

__device__ void cudaDeviceBCGprecond_2(double* dA, int* djA, int* diA, double* ddiag, double alpha){
  int row= threadIdx.x + blockDim.x*blockIdx.x;
  int nnz=diA[blockDim.x];
  for(int j=diA[threadIdx.x];j<diA[threadIdx.x+1];j++){
    if(djA[j]==threadIdx.x){
      dA[j+nnz*blockIdx.x] = 1.0 + alpha*dA[j+nnz*blockIdx.x];
      if(dA[j+nnz*blockIdx.x]!=0.0){
        ddiag[row]= 1.0/dA[j+nnz*blockIdx.x];
       }else{
        ddiag[row]= 1.0;
      }
    }else{
      dA[j+nnz*blockIdx.x] = alpha*dA[j+nnz*blockIdx.x];
    }
  }
}

__device__ void cudaDeviceSpmv_2CSR(double* dx, double* db, double* dA, int* djA, int* diA){
  __syncthreads();
  int row= threadIdx.x + blockDim.x*blockIdx.x;
  double sum = 0.0;
  int nnz=diA[blockDim.x];
  for(int j=diA[threadIdx.x]; j<diA[threadIdx.x+1]; j++){
    sum+= db[djA[j]+blockDim.x*blockIdx.x]*dA[j+nnz*blockIdx.x];
  }
  __syncthreads();
  dx[row]=sum;
  __syncthreads();
}

__device__ void cudaDeviceSpmv_2CSC_block(double* dx, double* db, double* dA, int* djA, int* diA){
  int row = threadIdx.x + blockDim.x*blockIdx.x;
  __syncthreads();
  dx[row]=0.0;
  __syncthreads();
  int nnz=diA[blockDim.x];
  for(int j=diA[threadIdx.x]; j<diA[threadIdx.x+1]; j++){
    double mult = db[row]*dA[j+nnz*blockIdx.x];
    atomicAdd_block(&(dx[djA[j]+blockDim.x*blockIdx.x]),mult);
  }
  __syncthreads();
}

__device__ void cudaDeviceSpmv_2(double* dx, double* db, double* dA, int* djA, int* diA){
#ifndef USE_CSR_ODE_GPU
  cudaDeviceSpmv_2CSR(dx,db,dA,djA,diA);
#else
  cudaDeviceSpmv_2CSC_block(dx,db,dA,djA,diA);
#endif
}

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
  __syncthreads();
  if(tid<n_shr_empty)
  sdata[tid+blockDim.x]=0.;
  __syncthreads();
    //print_double(sdata,73,"sdata");
#ifndef DEV_cudaDevicedotxy_2
  //used for compare with cpu
  sdata[0]=0.;
  __syncthreads();
  if(tid==0){
    for(int j=0;j<blockDim.x;j++){
      sdata[0]+=g_idata1[j+blockIdx.x*blockDim.x]*g_idata2[j+blockIdx.x*blockDim.x];
    }
  }
#else
  unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
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
  //print_double(sdata,1,"sdata");
#endif
  __syncthreads();
  *g_odata = sdata[0];
  __syncthreads();
}

__device__ void cudaDeviceVWRMS_Norm_2(double *g_idata1, double *g_idata2, double *g_odata, int n_shr_empty){
  extern __shared__ double sdata[];
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
  __syncthreads();
  if(tid<n_shr_empty)
    sdata[tid+blockDim.x]=0.;
  __syncthreads();
  sdata[tid] = g_idata1[i]*g_idata2[i];
  sdata[tid] = sdata[tid]*sdata[tid];
  __syncthreads();
#ifndef DEBUG_cudaDevicedotxy_2
  //used for compare with cpu
  if(tid==0){
    double sum=0.;
    for(int j=0;j<blockDim.x;j++){
      sum+=sdata[j];
    }
    sdata[0]=sum;
  }
  __syncthreads();
#else
  for (unsigned int s=(blockDim.x+n_shr_empty)/2; s>0; s>>=1){
    if (tid < s)
      sdata[tid] += sdata[tid + s];
    __syncthreads();
  }
#endif
  g_odata[0] = sqrt(sdata[0]/blockDim.x);
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
int cudaDevicecamp_solver_check_model_state(ModelDataGPU *md, ModelDataVariable *sc, double *y, int *flag)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  __syncthreads();
  extern __shared__ int flag_shr[];
  flag_shr[0] = 0;
  __syncthreads();
  if (y[i] < -SMALL) {
    flag_shr[0] = CAMP_SOLVER_FAIL;
  } else {
    md->state[md->map_state_deriv[i]] =
            y[i] <= -SMALL ?
            TINY : y[i];
  }
  __syncthreads();
  *flag = flag_shr[0];
  __syncthreads();
  return *flag;
}

__device__ void solveRXN(
  int i_rxn,TimeDerivativeGPU deriv_data,
  double time_step,ModelDataGPU *md, ModelDataVariable *sc
){
  double *rxn_float_data = (double *)&( md->rxn_double[md->rxn_float_indices[i_rxn]]);
  int *int_data = (int *)&(md->rxn_int[md->rxn_int_indices[i_rxn]]);
  int *rxn_int_data = (int *) &(int_data[1]);
  double *rxn_env_data = &(md->rxn_env_data
  [md->n_rxn_env_data*blockIdx.x+md->rxn_env_data_idx[i_rxn]]);
  switch (int_data[0]) {
    case RXN_ARRHENIUS :
      rxn_gpu_arrhenius_calc_deriv_contrib(sc, deriv_data, rxn_int_data,
                                           rxn_float_data, rxn_env_data,time_step);
      break;
    case RXN_CMAQ_H2O2 :
      rxn_gpu_CMAQ_H2O2_calc_deriv_contrib(sc, deriv_data, rxn_int_data,
                                           rxn_float_data, rxn_env_data,time_step);
      break;
    case RXN_CMAQ_OH_HNO3 :
      rxn_gpu_CMAQ_OH_HNO3_calc_deriv_contrib(sc, deriv_data, rxn_int_data,
                                              rxn_float_data, rxn_env_data,time_step);
      break;
    case RXN_FIRST_ORDER_LOSS:
    rxn_gpu_first_order_loss_calc_deriv_contrib(sc, deriv_data, rxn_int_data,
                                    rxn_float_data, rxn_env_data,time_step);
      break;
    case RXN_PHOTOLYSIS :
      rxn_gpu_photolysis_calc_deriv_contrib(sc, deriv_data, rxn_int_data,
                                            rxn_float_data, rxn_env_data,time_step);
      break;
    case RXN_TROE :
      rxn_gpu_troe_calc_deriv_contrib(sc, deriv_data, rxn_int_data,
                                      rxn_float_data, rxn_env_data,time_step);
      break;
  }
}

__device__ void cudaDevicecalc_deriv(double time_step, double *y,
        double *yout, ModelDataGPU *md, ModelDataVariable *sc)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  md->J_tmp[i]=y[i]-md->J_state[i];
  cudaDeviceSpmv_2(md->J_tmp2, md->J_tmp, md->J_solver, md->djA, md->diA);
  md->J_tmp[i]=md->J_deriv[i]+md->J_tmp2[i];
  TimeDerivativeGPU deriv_data;
  deriv_data.production_rates = md->production_rates;
  deriv_data.loss_rates = md->loss_rates;
  __syncthreads();
  deriv_data.production_rates[i] = 0.0;
  deriv_data.loss_rates[i] = 0.0;
  __syncthreads();
  deriv_data.production_rates = &( md->production_rates[blockDim.x*blockIdx.x]);
  deriv_data.loss_rates = &( md->loss_rates[blockDim.x*blockIdx.x]);
  sc->grid_cell_state = &( md->state[md->state_size_cell*blockIdx.x]);
  int n_rxn = md->n_rxn;
  __syncthreads();
#ifndef DEV_removeAtomic
  if(threadIdx.x==0){
    for (int j = 0; j < n_rxn; j++){
      //printf("n_rxn %d i %d j %d \n",n_rxn,i,j);
      solveRXN(j,deriv_data, time_step, md, sc);
    }
  }
#else
  if( threadIdx.x < n_rxn) {
    int n_iters = n_rxn / blockDim.x;
    for (int j = 0; j < n_iters; j++) {
      int i_rxn = threadIdx.x + j*blockDim.x;
      solveRXN(i_rxn,deriv_data, time_step, md, sc);
    }
    int residual=n_rxn%blockDim.x;
    if(threadIdx.x < residual){
      int i_rxn = threadIdx.x + blockDim.x*n_iters;
      solveRXN(i_rxn, deriv_data, time_step, md, sc);
    }
  }
#endif
  __syncthreads();
  deriv_data.production_rates = md->production_rates;
  deriv_data.loss_rates = md->loss_rates;
  __syncthreads();
  double *r_p = deriv_data.production_rates;
  double *r_l = deriv_data.loss_rates;
  if (r_p[i] + r_l[i] != 0.0) {
    if (md->use_deriv_est==1) {
      double scale_fact = 1.0 / (r_p[i] + r_l[i]) /
          (1.0 / (r_p[i] + r_l[i]) + MAX_PRECISION_LOSS / fabs(r_p[i]- r_l[i]));
      yout[i] = scale_fact * (r_p[i] - r_l[i]) + (1.0 - scale_fact) * (md->J_tmp[i]);
    }else {
      yout[i] = r_p[i] - r_l[i];
    }
  } else {
    yout[i] = 0.0;
  }
  //print_double(y,73,"y646");
  //print_double(md->J_state,73,"J_state644");
  //print_double(md->J_tmp,73,"J_tmp643");
  //print_double(md->J_deriv,73,"J_deriv644");
  //print_double(md->J_tmp2,73,"J_tmp2645");
  //print_double(deriv_data.loss_rates,73,"loss_rates");
  //print_double(deriv_data.production_rates,73,"production_rates");
  //print_double(yout,73,"deriv_data645");
  __syncthreads();
}

__device__
int cudaDevicef(double time_step, double *y,
        double *yout, ModelDataGPU *md, ModelDataVariable *sc, int *flag)
{
  __syncthreads();
#ifdef CAMP_PROFILE_DEVICE_FUNCTIONS
  int clock_khz=md->clock_khz;
  clock_t start;
  start = clock();
#endif
  time_step = sc->cv_next_h;
  time_step = time_step > 0. ? time_step : md->init_time_step;
  //print_double(&time_step,1,"time_step661");
  //print_double(md->state,md->state_size_cell,"state661");
  int checkflag=cudaDevicecamp_solver_check_model_state(md, sc, y, flag);
  //print_double(md->state,md->state_size_cell,"state663");
  __syncthreads();
  if(checkflag==CAMP_SOLVER_FAIL){
    *flag=CAMP_SOLVER_FAIL;
    __syncthreads();
#ifdef CAMP_PROFILE_DEVICE_FUNCTIONS
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(threadIdx.x==0) sc->timef += ((double)(int)(clock() - start))/(clock_khz*1000);
#endif
    return CAMP_SOLVER_FAIL;
  }
  cudaDevicecalc_deriv(time_step, y, yout, md, sc);
#ifdef CAMP_PROFILE_DEVICE_FUNCTIONS
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(threadIdx.x==0) sc->timef += ((double)(int)(clock() - start))/(clock_khz*1000);
#endif
  __syncthreads();
  *flag=0;
  __syncthreads();
  return 0;
}

__device__
int CudaDeviceguess_helper(double h_n, double* y_n,
   double* y_n1, double* hf, double* atmp1,
   double* acorr, int *flag, ModelDataGPU *md, ModelDataVariable *sc
) {
  extern __shared__ double flag_shr2[];
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  __syncthreads();
  double min;
  cudaDevicemin_2(&min, y_n[i], flag_shr2, md->n_shr_empty);
  if(min>-SMALL){
    return 0;
  }
  __syncthreads();
#ifdef CAMP_PROFILE_DEVICE_FUNCTIONS
  int clock_khz=md->clock_khz;
  clock_t start;
  start = clock();
#endif
  atmp1[i]=y_n1[i];
  __syncthreads();
  if (h_n > 0.) {
    acorr[i]=(1./h_n)*hf[i];
  } else {
    acorr[i]=hf[i];
  }
  //print_double(&h_n,1,"h_n711");
  //print_double(hf,73,"hf711");
  //print_double(acorr,73,"acorr711");
  double t_0 = h_n > 0. ? sc->cv_tn - h_n : sc->cv_tn - 1.;
  double t_j = 0.;
  __syncthreads();
  for (int iter = 0; iter < GUESS_MAX_ITER && t_0 + t_j < sc->cv_tn; iter++) {
    __syncthreads();
    double h_j = sc->cv_tn - (t_0 + t_j);
    //print_double(atmp1,73,"atmp720");
#ifndef DEV_CudaDeviceguess_helper
    if(threadIdx.x==0){
    int i_fast = -1;
    for (int j = 0; j < blockDim.x; j++) {
      double t_star = -atmp1[j+blockIdx.x*blockDim.x] / acorr[j+blockIdx.x*blockDim.x];
      if ((t_star > 0. || (t_star == 0. && acorr[j+blockIdx.x*blockDim.x] < 0.)) &&
          t_star < h_j) {
        h_j = t_star;
        i_fast = j;
      }
    }
    if (i_fast >= 0 && h_n > 0.)
      h_j *= 0.95 + 0.1 * iter / (double)GUESS_MAX_ITER;
    flag_shr2[0]=h_j;
    }
    __syncthreads();
    h_j=flag_shr2[0];
    __syncthreads();
#else
    double t_star = h_j;
    if(acorr[i]!=0.){
      t_star=-atmp1[i] / acorr[i];
    }
    flag_shr2[0] = 0;
    __syncthreads();
    if( (t_star > 0. || (t_star == 0. && acorr[i] < 0.)) ){
      flag_shr2[0] = 1;
    }else{
      t_star=h_j;
    }
    __syncthreads();
    int i_fast = flag_shr2[0];
    cudaDevicemin_2(&h_j, t_star, flag_shr2, md->n_shr_empty);
    if (i_fast == 1 && h_n > 0.)
      h_j *= 0.95 + 0.1 * iter / (double)GUESS_MAX_ITER;
#endif
    //print_double(&h_j,1,"h_j756");
    h_j = sc->cv_tn < t_0 + t_j + h_j ? sc->cv_tn - (t_0 + t_j) : h_j;
    //print_double(&h_j,1,"h_j758");
    __syncthreads();
    if (h_n == 0. && sc->cv_tn - (h_j + t_j + t_0) > md->cv_reltol) {
      __syncthreads();
#ifdef CAMP_PROFILE_DEVICE_FUNCTIONS
    if(threadIdx.x==0) sc->timeguess_helper += ((double)(clock() - start))/(clock_khz*1000);
#endif
    return -1;
    }
    atmp1[i]+=h_j*acorr[i];
    __syncthreads();
    t_j += h_j;
    int aux_flag=0;
    //print_double(atmp1,73,"atmp1766");
    int fflag=cudaDevicef(t_0 + t_j, atmp1, acorr,md,sc,&aux_flag);
    //print_double(acorr,73,"acorr721");
    __syncthreads();
    if (fflag == CAMP_SOLVER_FAIL) {
      acorr[i] = 0.;
      __syncthreads();
#ifdef CAMP_PROFILE_DEVICE_FUNCTIONS
    if(threadIdx.x==0) sc->timeguess_helper += ((double)(clock() - start))/(clock_khz*1000);
#endif
     return -1;
    }
    if (iter == GUESS_MAX_ITER - 1 && t_0 + t_j < sc->cv_tn) {
      if (h_n == 0.){
        __syncthreads();
#ifdef CAMP_PROFILE_DEVICE_FUNCTIONS
    if(threadIdx.x==0) sc->timeguess_helper += ((double)(clock() - start))/(clock_khz*1000);
#endif
        return -1;
      }
    }
    __syncthreads();
  }
  __syncthreads();
  acorr[i]=atmp1[i]-y_n[i];
  if (h_n > 0.) acorr[i]=acorr[i]*0.999;
  hf[i]=atmp1[i]-y_n1[i];
  __syncthreads();
#ifdef CAMP_PROFILE_DEVICE_FUNCTIONS
    if(threadIdx.x==0)  sc->timeguess_helper += ((double)(clock() - start))/(clock_khz*1000);
#endif
  __syncthreads();
  return 1;
}

__device__ void solveRXNJac(
        int i_rxn, JacobianGPU jac,
        ModelDataGPU *md, ModelDataVariable *sc
){
  double *rxn_float_data = (double *)&( md->rxn_double[md->rxn_float_indices[i_rxn]]);
  int *int_data = (int *)&(md->rxn_int[md->rxn_int_indices[i_rxn]]);
  int *rxn_int_data = (int *) &(int_data[1]);
  double *rxn_env_data = &(md->rxn_env_data
  [md->n_rxn_env_data*blockIdx.x+md->rxn_env_data_idx[i_rxn]]);
  switch (int_data[0]) {
    case RXN_ARRHENIUS :
      rxn_gpu_arrhenius_calc_jac_contrib(sc, jac, rxn_int_data,
                                         rxn_float_data, rxn_env_data,sc->cv_next_h);
      break;
    case RXN_CMAQ_H2O2 :
      rxn_gpu_CMAQ_H2O2_calc_jac_contrib(sc, jac, rxn_int_data,
                                         rxn_float_data, rxn_env_data,sc->cv_next_h);
      break;
    case RXN_CMAQ_OH_HNO3 :
      rxn_gpu_CMAQ_OH_HNO3_calc_jac_contrib(sc, jac, rxn_int_data,
                                            rxn_float_data, rxn_env_data,sc->cv_next_h);
      break;
    case RXN_FIRST_ORDER_LOSS :
      rxn_gpu_first_order_loss_calc_jac_contrib(sc, jac, rxn_int_data,
                                        rxn_float_data, rxn_env_data,sc->cv_next_h);
      break;
    case RXN_PHOTOLYSIS :
      rxn_gpu_photolysis_calc_jac_contrib(sc, jac, rxn_int_data,
                                          rxn_float_data, rxn_env_data,sc->cv_next_h);
      break;
    case RXN_TROE :
      rxn_gpu_troe_calc_jac_contrib(sc, jac, rxn_int_data,
                                    rxn_float_data, rxn_env_data,sc->cv_next_h);
      break;
  }
}

__device__ void cudaDevicecalc_Jac(double *y,ModelDataGPU *md, ModelDataVariable *sc
){
  __syncthreads();
#ifdef CAMP_PROFILE_DEVICE_FUNCTIONS
  int clock_khz=md->clock_khz;
  clock_t start;
  start = clock();
  __syncthreads();
#endif
  JacobianGPU *jac = &md->jac;
  JacobianGPU jacBlock;
  __syncthreads();
  jacBlock.num_elem = jac->num_elem;
  jacBlock.production_partials = &( jac->production_partials[jacBlock.num_elem[0]*blockIdx.x]);
  jacBlock.loss_partials = &( jac->loss_partials[jacBlock.num_elem[0]*blockIdx.x]);
  __syncthreads();
  sc->grid_cell_state = &( md->state[md->state_size_cell*blockIdx.x]);
  __syncthreads();
  int n_rxn = md->n_rxn;
#ifndef DEV_removeAtomic
  if(threadIdx.x==0){
    for (int j = 0; j < n_rxn; j++){
      solveRXNJac(j,jacBlock, md, sc);
    }
  }
#else
  if( threadIdx.x < n_rxn) {
    int n_iters = n_rxn / blockDim.x;
    for (int j = 0; j < n_iters; j++) {
      int i_rxn = threadIdx.x + j*blockDim.x;
      solveRXNJac(i_rxn,jacBlock, md, sc);
    }
    int residual=n_rxn%blockDim.x;
    if(threadIdx.x < residual){
      int i_rxn = threadIdx.x + blockDim.x*n_iters;
      solveRXNJac(i_rxn,jacBlock, md, sc);
    }
  }
#endif
  __syncthreads();
  JacMap *jac_map = md->jac_map;
  int nnz = md->n_mapped_values[0];
  int n_iters = nnz / blockDim.x;
  for (int z = 0; z < n_iters; z++) {
    int j = threadIdx.x + z*blockDim.x;
    md->dA[jac_map[j].solver_id + nnz * blockIdx.x] =
    jacBlock.production_partials[jac_map[j].rxn_id] - jacBlock.loss_partials[jac_map[j].rxn_id];
    jacBlock.production_partials[jac_map[j].rxn_id] = 0.0;
    jacBlock.loss_partials[jac_map[j].rxn_id] = 0.0;
  }
  int residual=nnz%blockDim.x;
  if(threadIdx.x < residual){
    int j = threadIdx.x + n_iters*blockDim.x;
  md->dA[jac_map[j].solver_id + nnz * blockIdx.x] =
      jacBlock.production_partials[jac_map[j].rxn_id] - jacBlock.loss_partials[jac_map[j].rxn_id];
    jacBlock.production_partials[jac_map[j].rxn_id] = 0.0;
    jacBlock.loss_partials[jac_map[j].rxn_id] = 0.0;
  }
    __syncthreads();
#ifdef CAMP_PROFILE_DEVICE_FUNCTIONS
    if(threadIdx.x==0) sc->timecalc_Jac += ((double)(clock() - start))/(clock_khz*1000);
#endif
}

__device__
int cudaDeviceJac(int *flag, ModelDataGPU *md, ModelDataVariable *sc)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int retval;
  __syncthreads();
#ifdef CAMP_PROFILE_DEVICE_FUNCTIONS
  int clock_khz=md->clock_khz;
  clock_t start;
  start = clock();
#endif
  md->use_deriv_est=0;
  int aux_flag=0;
  __syncthreads();
  //print_double(md->dcv_y,73,"dcv_y914");
  retval=cudaDevicef(sc->cv_next_h, md->dcv_y, md->dftemp,md,sc,&aux_flag);
  md->use_deriv_est=1;
  __syncthreads();
  if(retval==CAMP_SOLVER_FAIL)
    return CAMP_SOLVER_FAIL;
#ifdef DEV_check_model_state
  //Pending: remove in cpu, since it is the same result in gpu
  //print_double(md->state,md->state_size_cell,"state920");
  int checkflag=cudaDevicecamp_solver_check_model_state(md, sc, md->dcv_y, &aux_flag);
  //print_double(md->state,md->state_size_cell,"state923");
  __syncthreads();
  if(checkflag==CAMP_SOLVER_FAIL){
    *flag=CAMP_SOLVER_FAIL;
#ifdef CAMP_PROFILE_DEVICE_FUNCTIONS
    if(threadIdx.x==0)  sc->timeJac += ((double)(clock() - start))/(clock_khz*1000);
#endif
    __syncthreads();
    return CAMP_SOLVER_FAIL;
  }
#endif
  cudaDevicecalc_Jac(md->dcv_y,md, sc);
  __syncthreads();
  int nnz = md->n_mapped_values[0];
  int n_iters = nnz / blockDim.x;
  for (int z = 0; z < n_iters; z++) {
    int j = threadIdx.x + z*blockDim.x + nnz * blockIdx.x;
    md->J_solver[j]=md->dA[j];
  }
  int residual=nnz%blockDim.x;
  if(threadIdx.x < residual){
    int j = threadIdx.x + n_iters*blockDim.x + nnz * blockIdx.x;
    md->J_solver[j]=md->dA[j];
  }
  __syncthreads();
  md->J_state[i]=md->dcv_y[i];
  md->J_deriv[i]=md->dftemp[i];
  __syncthreads();
#ifdef CAMP_PROFILE_DEVICE_FUNCTIONS
    if(threadIdx.x==0)  sc->timeJac += ((double)(clock() - start))/(clock_khz*1000);
#endif
  __syncthreads();
  *flag = 0;
  __syncthreads();
  return 0;
}

__device__
int cudaDevicelinsolsetup(
    ModelDataGPU *md, ModelDataVariable *sc, int convfail
) {
  extern __shared__ int flag_shr[];
  double dgamma;
  int jbad, jok;
  dgamma = fabs((sc->cv_gamma / sc->cv_gammap) - 1.);//SUNRabs
  jbad = (sc->cv_nst == 0) ||
         (sc->cv_nst > sc->nstlj + CVD_MSBJ) ||
         ((convfail == CV_FAIL_BAD_J) && (dgamma < CVD_DGMAX)) ||
         (convfail == CV_FAIL_OTHER);
  jok = !jbad;
  if (jok==1) {
    __syncthreads();
    sc->cv_jcur = 0;
    cudaDeviceJacCopy(md->diA, md->dsavedJ, md->dA);
    __syncthreads();
  } else {
    __syncthreads();
    sc->nstlj = sc->cv_nst;
    sc->cv_jcur = 1;
    __syncthreads();
    int aux_flag=0;
    __syncthreads();
    int guess_flag=cudaDeviceJac(&aux_flag,md,sc);
    __syncthreads();
    if (guess_flag < 0) {
      return -1;}
    if (guess_flag > 0) {
      return 1;}
   cudaDeviceJacCopy(md->diA, md->dA, md->dsavedJ);
  }
  __syncthreads();
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  md->dx[i]=0.;
  cudaDeviceBCGprecond_2(md->dA, md->djA, md->diA, md->ddiag, -sc->cv_gamma);
  __syncthreads();
  return 0;
}

__device__
void solveBcgCudaDeviceCVODE(ModelDataGPU *md, ModelDataVariable *sc)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  double alpha,rho0,omega0,beta,rho1,temp1,temp2;
  //print_double(md->dx,73,"dx1017");
  alpha=rho0=omega0=beta=rho1=temp1=temp2=1.0;
  md->dn0[i]=0.0;
  md->dp0[i]=0.0;
  cudaDeviceSpmv_2(md->dr0,md->dx,md->dA,md->djA,md->diA);
  md->dr0[i]=md->dtempv[i]-md->dr0[i];
  md->dr0h[i]=md->dr0[i];
  int it=0;
  while(it<BCG_MAXIT && temp1>BCG_TOLMAX){
    //print_double(md->dr0,73,"dr0");
    //print_double(md->dr0h,73,"dr0h");
    cudaDevicedotxy_2(md->dr0, md->dr0h, &rho1, md->n_shr_empty);
    //print_double(&rho1,1,"rho1");
    beta = (rho1 / rho0) * (alpha / omega0);
    //print_double(&beta,1,"beta");
    md->dp0[i]=beta*md->dp0[i]+md->dr0[i]-md->dn0[i]*omega0*beta;
    md->dy[i]=md->ddiag[i]*md->dp0[i];
    cudaDeviceSpmv_2(md->dn0, md->dy, md->dA, md->djA, md->diA);
    cudaDevicedotxy_2(md->dr0h, md->dn0, &temp1, md->n_shr_empty);
    alpha = rho1 / temp1;
    md->ds[i]=md->dr0[i]-alpha*md->dn0[i];
    md->dx[i]+=alpha*md->dy[i];
    md->dy[i]=md->ddiag[i]*md->ds[i];
    cudaDeviceSpmv_2(md->dt, md->dy, md->dA, md->djA, md->diA);
    md->dr0[i]=md->ddiag[i]*md->dt[i];
    //print_double(md->ddiag,73,"ddiag");
    cudaDevicedotxy_2(md->dy, md->dr0, &temp1, md->n_shr_empty);
    cudaDevicedotxy_2(md->dr0, md->dr0, &temp2, md->n_shr_empty);
    omega0 = temp1 / temp2;
    md->dx[i]+=omega0*md->dy[i];
    md->dr0[i]=md->ds[i]-omega0*md->dt[i];
    md->dt[i]=0.0;
    cudaDevicedotxy_2(md->dr0, md->dr0, &temp1, md->n_shr_empty);
    //print_double(md->dx,73,"dx");
    //print_double(&temp1,1,"temp1");
    temp1 = sqrt(temp1);
    //print_double(&temp1,1,"sqrt(temp1)");
    rho0 = rho1;
    it++;
  __syncthreads();
  //if(i==0)printf("end iter %d BCG GPU\n",it);
  }
  //print_double(&temp1,1,"temp1");
  //if(i==0)printf("end BCG GPU\n");
  __syncthreads();
#ifdef CAMP_PROFILE_DEVICE_FUNCTIONS
  if(threadIdx.x==0) sc->counterBCGInternal += it;
  if(threadIdx.x==0) sc->counterBCG++;
#endif
}

__device__
int cudaDevicecvNewtonIteration(ModelDataGPU *md, ModelDataVariable *sc){
  extern __shared__ double flag_shr2[];
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int aux_flag=0;
  double del, delp, dcon;
  int m = 0;
  del = delp = 0.0;
  int retval;
  __syncthreads();
#ifdef CAMP_PROFILE_DEVICE_FUNCTIONS
  int clock_khz=md->clock_khz;
  clock_t start;
#endif
  for(;;) {
    __syncthreads();
#ifdef CAMP_PROFILE_DEVICE_FUNCTIONS
    start = clock();
#endif
    //print_double(md->dtempv,73,"dtempvN_VLinearSum1");
    //print_double(md->dzn+md->nrows,73,"dzn1_1089");
    md->dtempv[i]=sc->cv_rl1*md->dzn[i+md->nrows]+md->cv_acor[i];
    //print_double(md->dtempv,73,"dtempvN_VLinearSum2");
    md->dtempv[i]=sc->cv_gamma*md->dftemp[i]-md->dtempv[i];
    print_double(md->dtempv,73,"dtempvcv_lsolve1");
    solveBcgCudaDeviceCVODE(md, sc);
    __syncthreads();
#ifdef CAMP_PROFILE_DEVICE_FUNCTIONS
    if(threadIdx.x==0) sc->dtBCG += ((double)(int)(clock() - start))/(clock_khz*1000);
#endif
    md->dtempv[i] = md->dx[i];
    //print_double(md->dtempv,73,"dtempvcv_lsolve2");
    __syncthreads();
    cudaDeviceVWRMS_Norm_2(md->dx, md->dewt, &del, md->n_shr_empty);
    //print_double(&del,1,"del1171");
    //if(i==0)printf("md->dtempv[0] %.16le\n",md->dtempv[0]);
    md->dftemp[i]=md->dcv_y[i]+md->dtempv[i];
    __syncthreads();
    //print_double(md->dcv_y,73,"dcv_y2994");
    print_double(md->dftemp,73,"cv_ftemplsolve");
    int guessflag=CudaDeviceguess_helper(0., md->dftemp,
       md->dcv_y, md->dtempv, md->dtempv1,md->dtempv2, &aux_flag, md, sc
    );
    __syncthreads();
    if (guessflag < 0) {
      if (!(sc->cv_jcur)) { //Bool set up during linsolsetup just before Jacobian
        return TRY_AGAIN;
      } else {
        return RHSFUNC_RECVR;
      }
    }
    md->dftemp[i]=md->dcv_y[i]+md->dtempv[i];
    double min;
    cudaDevicemin_2(&min, md->dftemp[i], flag_shr2, md->n_shr_empty);
    if (min < -CAMP_TINY) {
      return CONV_FAIL;
    }
    __syncthreads();
    md->cv_acor[i]+=md->dtempv[i];
    md->dcv_y[i]=md->dzn[i]+md->cv_acor[i];
    //print_double(md->cv_acor,73,"cv_acor1060");
    //print_double(md->dcv_y,73,"dcv_y1060");
    if (m > 0) {
      sc->cv_crate = SUNMAX(0.3 * sc->cv_crate, del / delp);
    }
    dcon = del * SUNMIN(1.0, sc->cv_crate) / md->cv_tq[4+blockIdx.x*(NUM_TESTS + 1)];
    flag_shr2[0]=0;
    __syncthreads();
    if (dcon <= 1.) {
      //print_double(md->cv_acor,73,"cv_acor1505");
      //print_double(md->dewt,73,"dewt1505");
      cudaDeviceVWRMS_Norm_2(md->cv_acor, md->dewt, &sc->cv_acnrm, md->n_shr_empty);
      //print_double(&sc->cv_acnrm,1,"cv_acnrm1151");
      __syncthreads();
      sc->cv_jcur = 0;
      __syncthreads();
      return CV_SUCCESS;
    }
    m++;
    if ((m == md->cv_maxcor) || ((m >= 2) && (del > RDIV * delp))) {
      if (!(sc->cv_jcur)) {
        return TRY_AGAIN;
      } else {
        return RHSFUNC_RECVR;
      }
    }
    delp = del;
    __syncthreads();
    //print_double(md->dcv_y,73,"dcv_y1137");
    retval=cudaDevicef(sc->cv_next_h, md->dcv_y, md->dftemp, md, sc, &aux_flag);
    __syncthreads();
    md->cv_acor[i]=md->dcv_y[i]+md->dzn[i];
    if (retval < 0) {
      return CV_RHSFUNC_FAIL;
    }
    if (retval > 0) {
      if (!(sc->cv_jcur)) {
        return TRY_AGAIN;
      } else {
        return RHSFUNC_RECVR;
      }
    }
    __syncthreads();
#ifdef CAMP_PROFILE_DEVICE_FUNCTIONS
    if(threadIdx.x==0) sc->dtPostBCG += ((double)(clock() - start))/(clock_khz*1000);
#endif
  }
}

__device__
int cudaDevicecvNlsNewton(int nflag,
        ModelDataGPU *md, ModelDataVariable *sc
) {
  extern __shared__ int flag_shr[];
  int flagDevice = 0;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int retval=0;
  __syncthreads();
#ifdef CAMP_PROFILE_DEVICE_FUNCTIONS
  int clock_khz=md->clock_khz;
  clock_t start;
#endif
  int convfail = ((nflag == FIRST_CALL) || (nflag == PREV_ERR_FAIL)) ?
                 CV_NO_FAILURES : CV_FAIL_OTHER;
  int dgamrat=fabs(sc->cv_gamrat - 1.);
  int callSetup = (nflag == PREV_CONV_FAIL) || (nflag == PREV_ERR_FAIL) ||
                  (sc->cv_nst == 0) ||
                  (sc->cv_nst >= sc->cv_nstlp + MSBP) ||
                  (dgamrat > DGMAX);
  __syncthreads();
  //print_double(md->dzn,73,"dzn1174");
  //print_double(md->cv_last_yn,73,"cv_last_yn1175");
  md->dftemp[i]=md->dzn[i]-md->cv_last_yn[i];
  //print_double(md->dftemp,73,"cv_ftemppN_VLinearSum2");
  md->cv_acor_init[i]=0.;
  __syncthreads();
  int guessflag=CudaDeviceguess_helper(sc->cv_h, md->dzn,
       md->cv_last_yn, md->dftemp, md->dtempv1,
       md->cv_acor_init,  &flagDevice,md, sc
  );
  __syncthreads();
  if(guessflag<0){
    return RHSFUNC_RECVR;
  }
  for(;;) {
    __syncthreads();
    //print_double(md->dzn,73,"dzn1139");
    //print_double(md->cv_acor_init,73,"cv_acor_init1140");
    md->dcv_y[i] = md->dzn[i]+md->cv_acor_init[i];
    //print_double(md->dcv_y,73,"dcv_y1139");
    int aux_flag=0;
    //print_double(md->dftemp,73,"cv_ftemppcv_f1");
    //print_double(&sc->cv_tn,1,"cv_tn1216");
    retval=cudaDevicef(sc->cv_tn, md->dcv_y,md->dftemp,md,sc,&aux_flag);
    print_double(md->dftemp,73,"cv_ftemppcv_f2");
    //print_double(md->dcv_y,73,"dcv_y1144");
    if (retval < 0) {
      return CV_RHSFUNC_FAIL;
    }
    if (retval> 0) {
      return RHSFUNC_RECVR;
    }
    if (callSetup) {
      __syncthreads();
#ifdef CAMP_PROFILE_DEVICE_FUNCTIONS
      start = clock();
#endif
      __syncthreads();
      int linflag=cudaDevicelinsolsetup(md, sc,convfail);
      print_double(md->dftemp,73,"cv_ftempp1160");
      __syncthreads();
#ifdef CAMP_PROFILE_DEVICE_FUNCTIONS
      if(threadIdx.x==0) sc->timelinsolsetup += ((double)(clock() - start))/(clock_khz*1000);
#endif
      callSetup = 0;
      sc->cv_gamrat = sc->cv_crate = 1.0;
      sc->cv_gammap = sc->cv_gamma;
      sc->cv_nstlp = sc->cv_nst;
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
    md->cv_acor[i] = md->cv_acor_init[i];
    __syncthreads();
#ifdef CAMP_PROFILE_DEVICE_FUNCTIONS
    start = clock();
#endif
    __syncthreads();
    int nItflag=cudaDevicecvNewtonIteration(md, sc);
    __syncthreads();
#ifdef CAMP_PROFILE_DEVICE_FUNCTIONS
    if(threadIdx.x==0)  sc->timeNewtonIteration += ((double)(clock() - start))/(clock_khz*1000);
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
void cudaDevicecvRescale(ModelDataGPU *md, ModelDataVariable *sc) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  double factor;
  __syncthreads();
  factor = sc->cv_eta;
  for (int j=1; j <= sc->cv_q; j++) {
    md->dzn[i+md->nrows*j]*=factor;
    __syncthreads();
    factor *= sc->cv_eta;
    __syncthreads();
  }
  print_double(&sc->cv_eta,1,"cv_eta_1290");
  //print_double(md->dzn+md->nrows,73,"dzn1_1290");
  sc->cv_h = sc->cv_hscale * sc->cv_eta;
  sc->cv_next_h = sc->cv_h;
  sc->cv_hscale = sc->cv_h;
  __syncthreads();
}

__device__
void cudaDevicecvRestore(ModelDataGPU *md, ModelDataVariable *sc, double saved_t) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j, k;
  __syncthreads();
  sc->cv_tn=saved_t;
  print_double(md->dzn,73,"dzn1299");
  for (k = 1; k <= sc->cv_q; k++){
    for (j = sc->cv_q; j >= k; j--) {
      md->dzn[i+md->nrows*(j-1)]-=md->dzn[i+md->nrows*j];
    }
  }
  md->dzn[i]=md->cv_last_yn[i];
  print_double(md->dzn,73,"dzn1306");
  __syncthreads();
}

__device__
int cudaDevicecvHandleNFlag(ModelDataGPU *md, ModelDataVariable *sc, int *nflagPtr, double saved_t,
                             int *ncfPtr) {
  extern __shared__ int flag_shr[];
  if (*nflagPtr == CV_SUCCESS){
    return(DO_ERROR_TEST);
  }
  cudaDevicecvRestore(md, sc, saved_t);
  if (*nflagPtr == CV_LSETUP_FAIL)  return(CV_LSETUP_FAIL);
  if (*nflagPtr == CV_LSOLVE_FAIL)  return(CV_LSOLVE_FAIL);
  if (*nflagPtr == CV_RHSFUNC_FAIL) return(CV_RHSFUNC_FAIL);
  (*ncfPtr)++;
  sc->cv_etamax = 1.;
  __syncthreads();
  if ((fabs(sc->cv_h) <= sc->cv_hmin*ONEPSM) ||
      (*ncfPtr == sc->cv_maxncf)) {
    if (*nflagPtr == CONV_FAIL)     return(CV_CONV_FAILURE);
    if (*nflagPtr == RHSFUNC_RECVR) return(CV_REPTD_RHSFUNC_ERR);
  }
  __syncthreads();
  sc->cv_eta = SUNMAX(ETACF,
          sc->cv_hmin / fabs(sc->cv_h));
  print_double(&sc->cv_eta,1,"cv_eta_1337");
  __syncthreads();
  *nflagPtr = PREV_CONV_FAIL;
  cudaDevicecvRescale(md, sc);
  __syncthreads();
  return (PREDICT_AGAIN);
}

__device__
void cudaDevicecvSetTqBDFt(ModelDataGPU *md, ModelDataVariable *sc,
                           double hsum, double alpha0, double alpha0_hat,
                           double xi_inv, double xistar_inv) {
  extern __shared__ int flag_shr[];
  double A1, A2, A3, A4, A5, A6;
  double C, Cpinv, Cppinv;
  __syncthreads();
  A1 = 1. - alpha0_hat + alpha0;
  A2 = 1. + sc->cv_q * A1;
  md->cv_tq[2+blockIdx.x*(NUM_TESTS + 1)] = fabs(A1 / (alpha0 * A2));
  md->cv_tq[5+blockIdx.x*(NUM_TESTS + 1)] = fabs(A2 * xistar_inv / (md->cv_l[sc->cv_q+blockIdx.x*L_MAX] * xi_inv));
  if (sc->cv_qwait == 1) {
    if (sc->cv_q > 1) {
      C = xistar_inv / md->cv_l[sc->cv_q+blockIdx.x*L_MAX];
      A3 = alpha0 + 1. / sc->cv_q;
      A4 = alpha0_hat + xi_inv;
      Cpinv = (1. - A4 + A3) / A3;
      md->cv_tq[1+blockIdx.x*(NUM_TESTS + 1)] = fabs(C * Cpinv);
    }
    else md->cv_tq[1+blockIdx.x*(NUM_TESTS + 1)] = 1.;
    __syncthreads();
    hsum += md->cv_tau[sc->cv_q+blockIdx.x*(L_MAX + 1)];
    xi_inv = sc->cv_h / hsum;
    A5 = alpha0 - (1. / (sc->cv_q+1));
    A6 = alpha0_hat - xi_inv;
    Cppinv = (1. - A6 + A5) / A2;
    md->cv_tq[3+blockIdx.x*(NUM_TESTS + 1)] = fabs(Cppinv / (xi_inv * (sc->cv_q+2) * A5));
    __syncthreads();
  }
  md->cv_tq[4+blockIdx.x*(NUM_TESTS + 1)] = md->cv_nlscoef / md->cv_tq[2+blockIdx.x*(NUM_TESTS + 1)];
}

__device__
void cudaDevicecvSetBDF(ModelDataGPU *md, ModelDataVariable *sc) {
  extern __shared__ int flag_shr[];
  double alpha0, alpha0_hat, xi_inv, xistar_inv, hsum;
  int z,j;
  __syncthreads();
  md->cv_l[0+blockIdx.x*L_MAX] = md->cv_l[1+blockIdx.x*L_MAX] = xi_inv = xistar_inv = 1.;
  for (z=2; z <= sc->cv_q; z++) md->cv_l[z+blockIdx.x*L_MAX] = 0.;
  alpha0 = alpha0_hat = -1.;
  hsum = sc->cv_h;
  __syncthreads();
  if (sc->cv_q > 1) {
    for (j=2; j < sc->cv_q; j++) {
      hsum += md->cv_tau[j-1+blockIdx.x*(L_MAX + 1)];
      xi_inv = sc->cv_h / hsum;
      alpha0 -= 1. / j;
      for (z=j; z >= 1; z--) md->cv_l[z+blockIdx.x*L_MAX] += md->cv_l[z-1+blockIdx.x*L_MAX]*xi_inv;
    }
    __syncthreads();
    alpha0 -= 1. / sc->cv_q;
    xistar_inv = -md->cv_l[1+blockIdx.x*L_MAX] - alpha0;
    hsum += md->cv_tau[sc->cv_q-1+blockIdx.x*(L_MAX + 1)];
    xi_inv = sc->cv_h / hsum;
    alpha0_hat = -md->cv_l[1+blockIdx.x*L_MAX] - xi_inv;
    for (z=sc->cv_q; z >= 1; z--)
      md->cv_l[z+blockIdx.x*L_MAX] += md->cv_l[z-1+blockIdx.x*L_MAX]*xistar_inv;
  }
  __syncthreads();
  cudaDevicecvSetTqBDFt(md, sc, hsum, alpha0, alpha0_hat, xi_inv, xistar_inv);
}

__device__
void cudaDevicecvSet(ModelDataGPU *md, ModelDataVariable *sc) {
  extern __shared__ int flag_shr[];
  __syncthreads();
  cudaDevicecvSetBDF(md,sc);
  __syncthreads();
  sc->cv_rl1 = 1.0 / md->cv_l[1+blockIdx.x*L_MAX];
  sc->cv_gamma = sc->cv_h * sc->cv_rl1;
  __syncthreads();
  if (sc->cv_nst == 0){
    sc->cv_gammap = sc->cv_gamma;
  }
  __syncthreads();
  sc->cv_gamrat = (sc->cv_nst > 0) ?
                    sc->cv_gamma / sc->cv_gammap : 1.;  // protect x / x != 1.0
  __syncthreads();
}

__device__
void cudaDevicecvPredict(ModelDataGPU *md, ModelDataVariable *sc) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j, k;
  __syncthreads();
  sc->cv_tn += sc->cv_h;
  __syncthreads();
  if (md->cv_tstopset) {
    if ((sc->cv_tn - md->cv_tstop)*sc->cv_h > 0.)
      sc->cv_tn = md->cv_tstop;
  }
  md->cv_last_yn[i]=md->dzn[i];
  __syncthreads();
  print_double(md->dzn,73,"dzn1432");
  //print_double(md->dzn+md->nrows,73,"dzn1_1432");
  //if(i==0)printf("md->nrows %d\n",md->nrows);
  //if(i==0)printf("cv_q %d\n",sc->cv_q);
  for (k = 1; k <= sc->cv_q; k++){
    for (j = sc->cv_q; j >= k; j--){
      md->dzn[i+md->nrows*(j-1)]+=md->dzn[i+md->nrows*j];
      __syncthreads();
    }
  }
  print_double(md->dzn,73,"dzn1439");
}

__device__
void cudaDevicecvDecreaseBDF(ModelDataGPU *md, ModelDataVariable *sc) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  double hsum, xi;
  int z, j;
  for (z=0; z <= md->cv_qmax; z++) md->cv_l[z+blockIdx.x*L_MAX] = 0.;
  md->cv_l[2+blockIdx.x*L_MAX] = 1.;
  hsum = 0.;
  for (j=1; j <= sc->cv_q-2; j++) {
    hsum += md->cv_tau[j+blockIdx.x*(L_MAX + 1)];
    xi = hsum /sc->cv_hscale;
    for (z=j+2; z >= 2; z--)
      md->cv_l[z+blockIdx.x*L_MAX] = md->cv_l[z+blockIdx.x*L_MAX]*xi + md->cv_l[z-1+blockIdx.x*L_MAX];
  }
  for (j=2; j < sc->cv_q; j++){
    md->dzn[i+md->nrows*j]=-md->cv_l[j+blockIdx.x*L_MAX]*
      md->dzn[i+md->nrows*sc->cv_q]+md->dzn[i+md->nrows*j];
  }
  //print_double(md->dzn+md->nrows*2,73,"dzn2_1469");
  print_double(md->dzn,73,"dzn1460");
}

__device__
int cudaDevicecvDoErrorTest(ModelDataGPU *md, ModelDataVariable *sc,
       int *nflagPtr,double saved_t, int *nefPtr, double *dsmPtr) {
  extern __shared__ double flag_shr2[];
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  double dsm;
  double min_val;
  int retval;
  md->dftemp[i]=md->cv_l[blockIdx.x*L_MAX]*md->cv_acor[i]+md->dzn[i];
  cudaDevicemin_2(&min_val, md->dftemp[i], flag_shr2, md->n_shr_empty);
  if (min_val < 0. && min_val > -CAMP_TINY) {
    md->dftemp[i]=fabs(md->dftemp[i]);
    md->dzn[i]=md->dftemp[i]-md->cv_l[0+blockIdx.x*L_MAX]*md->cv_acor[i];
    min_val = 0.;
  }
  //print_double(&md->cv_tq[2+blockIdx.x*(NUM_TESTS + 1)],1,"cv_tq_21504");
  //print_double(&sc->cv_acnrm,1,"cv_acnrm1504");
  dsm = sc->cv_acnrm * md->cv_tq[2+blockIdx.x*(NUM_TESTS + 1)];
  //print_double(&dsm,1,"dsm1504");
  *dsmPtr = dsm;
  if (dsm <= 1. && min_val >= 0.) return(CV_SUCCESS);
  (*nefPtr)++;
  *nflagPtr = PREV_ERR_FAIL;
  cudaDevicecvRestore(md, sc, saved_t);
  __syncthreads();
  if ((fabs(sc->cv_h) <= sc->cv_hmin*ONEPSM) ||
      (*nefPtr == md->cv_maxnef)) return(CV_ERR_FAILURE);
  sc->cv_etamax = 1.;
  __syncthreads();
  if (*nefPtr <= MXNEF1) {
    sc->cv_eta = 1. / (dSUNRpowerR(BIAS2*dsm,1./sc->cv_L) + ADDON);
    __syncthreads();
    sc->cv_eta = SUNMAX(ETAMIN, SUNMAX(sc->cv_eta,
                           sc->cv_hmin / fabs(sc->cv_h)));
    __syncthreads();
    if (*nefPtr >= SMALL_NEF)
      sc->cv_eta = SUNMIN(sc->cv_eta, ETAMXF);
    __syncthreads();
    //print_double(&sc->cv_eta,1,"cv_eta_1510");
    cudaDevicecvRescale(md, sc);
    return(TRY_AGAIN);
  }
  __syncthreads();
  if (sc->cv_q > 1) {
    sc->cv_eta = SUNMAX(ETAMIN,sc->cv_hmin / fabs(sc->cv_h));
    //print_double(&sc->cv_eta,1,"cv_eta_1517");
    cudaDevicecvDecreaseBDF(md, sc);
    sc->cv_L = sc->cv_q;
    //print_int(&sc->cv_L,1,"cv_L1547");
    sc->cv_q--;
    sc->cv_qwait = sc->cv_L;
    cudaDevicecvRescale(md, sc);
    __syncthreads();
    return(TRY_AGAIN);
  }
  __syncthreads();
  sc->cv_eta = SUNMAX(ETAMIN, sc->cv_hmin / fabs(sc->cv_h));
  print_double(&sc->cv_eta,1,"cv_eta_1529");
  __syncthreads();
  sc->cv_h *= sc->cv_eta;
  sc->cv_next_h = sc->cv_h;
  sc->cv_hscale = sc->cv_h;
  __syncthreads();
  sc->cv_qwait = 10;
  int aux_flag=0;
  //print_double(md->dzn,73,"dzn1505");
  retval=cudaDevicef(sc->cv_tn, md->dzn, md->dtempv,md,sc, &aux_flag);
  if (retval < 0)  return(CV_RHSFUNC_FAIL);
  if (retval > 0)  return(CV_UNREC_RHSFUNC_ERR);
  md->dzn[i+md->nrows]=sc->cv_h*md->dtempv[i];
  //print_double(md->dzn+md->nrows,73,"dzn1_1536");
  return(TRY_AGAIN);
}

__device__
void cudaDevicecvCompleteStep(ModelDataGPU *md, ModelDataVariable *sc) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int z, j;
  __syncthreads();
  sc->cv_nst++;
  __syncthreads();
  sc->cv_hu = sc->cv_h;
  for (z=sc->cv_q; z >= 2; z--)  md->cv_tau[z+blockIdx.x*(L_MAX + 1)] = md->cv_tau[z-1+blockIdx.x*(L_MAX + 1)];
  if ((sc->cv_q==1) && (sc->cv_nst > 1))
    md->cv_tau[2+blockIdx.x*(L_MAX + 1)] = md->cv_tau[1+blockIdx.x*(L_MAX + 1)];
  md->cv_tau[1+blockIdx.x*(L_MAX + 1)] = sc->cv_h;
  __syncthreads();
  for (j=0; j <= sc->cv_q; j++){
    md->dzn[i+md->nrows*j]+=md->cv_l[j+blockIdx.x*L_MAX]*md->cv_acor[i];
    __syncthreads();
  }
  //print_double(md->dzn+md->nrows,73,"dzn1_1559");
  sc->cv_qwait--;
  if ((sc->cv_qwait == 1) && (sc->cv_q != md->cv_qmax)) {
    md->dzn[i+md->nrows*md->cv_qmax]=md->cv_acor[i];
    sc->cv_saved_tq5 = md->cv_tq[5+blockIdx.x*(NUM_TESTS + 1)];
    sc->cv_indx_acor = md->cv_qmax;
  }
  //print_double(md->dzn,73,"dzn1554");
}

__device__
void cudaDevicecvChooseEta(ModelDataGPU *md, ModelDataVariable *sc) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  double etam;
  print_double(&sc->cv_etaqm1,1,"cv_etaqm11605");
  print_double(&sc->cv_etaq,1,"cv_etaq1605");
  print_double(&sc->cv_etaqp1,1,"cv_etaqp1605");
  etam = SUNMAX(sc->cv_etaqm1, SUNMAX(sc->cv_etaq, sc->cv_etaqp1));
  print_double(&etam,1,"etam1605");
  __syncthreads();
  if (etam < THRESH) {
    sc->cv_eta = 1.;
    print_double(&sc->cv_eta,1,"cv_eta1609");
    sc->cv_qprime = sc->cv_q;
    return;
  }
  __syncthreads();
  if (etam == sc->cv_etaq) {
    sc->cv_eta = sc->cv_etaq;
    print_double(&sc->cv_eta,1,"cv_eta1616");
    sc->cv_qprime = sc->cv_q;
  } else if (etam == sc->cv_etaqm1) {
    sc->cv_eta = sc->cv_etaqm1;
    print_double(&sc->cv_eta,1,"cv_eta1620");
    sc->cv_qprime = sc->cv_q - 1;
  } else {
    sc->cv_eta = sc->cv_etaqp1;
    print_double(&sc->cv_eta,1,"cv_eta1624");
    sc->cv_qprime = sc->cv_q + 1;
    __syncthreads();
    md->dzn[i+md->nrows*md->cv_qmax]=md->cv_acor[i];
  }
  __syncthreads();
  print_double(md->dzn,73,"dzn1581");
}

__device__
void cudaDevicecvSetEta(ModelDataGPU *md, ModelDataVariable *sc) {
  __syncthreads();
  if (sc->cv_eta < THRESH) {
    sc->cv_eta = 1.;
    sc->cv_hprime = sc->cv_h;
  } else {
    __syncthreads();
    sc->cv_eta = SUNMIN(sc->cv_eta, sc->cv_etamax);
    __syncthreads();
    sc->cv_eta /= SUNMAX(ONE,
            fabs(sc->cv_h)*md->cv_hmax_inv*sc->cv_eta);
    __syncthreads();
    sc->cv_hprime = sc->cv_h * sc->cv_eta;
    __syncthreads();
  }
  print_double(&sc->cv_eta,1,"cv_eta_1618");
  __syncthreads();
}

__device__
int cudaDevicecvPrepareNextStep(ModelDataGPU *md, ModelDataVariable *sc, double dsm) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  __syncthreads();
  if (sc->cv_etamax == 1.) {
    sc->cv_qwait = SUNMAX(sc->cv_qwait, 2);
    sc->cv_qprime = sc->cv_q;
    sc->cv_hprime = sc->cv_h;
    sc->cv_eta = 1.;
    print_double(&sc->cv_eta,1,"cv_eta_1631");
    return 0;
  }
  __syncthreads();
  //print_double(&dsm,1,"dsm1639");
  //print_int(&sc->cv_L,1,"cv_L1639");
  //double BIAS2dsm=BIAS2*dsm;
  //print_double(&BIAS2dsm,1,"BIAS2dsm");
  //double cv_L1=1./sc->cv_L;
  //print_double(&cv_L1,1,"1cv_L");
  //double cv_etaq_power=dSUNRpowerR(BIAS2dsm,cv_L1);
  //print_double(&cv_etaq_power,1,"cv_etaq_power");
  //double cv_etaq_sqrt=sqrt(BIAS2dsm);
  //print_double(&cv_etaq_sqrt,1,"cv_etaq_sqrt");
  sc->cv_etaq=1./(dSUNRpowerR(BIAS2*dsm,1./sc->cv_L) + ADDON);
  if(sc->cv_L!=2){
    //print_int(&sc->cv_L,1,"cv_L1674");
    /*
    if(i==0)printf("WARNING: pow is innacurate from CPU"
    " result for CUDA/10.1.105 "
    " (which is used during development at CTE-POWER) "
    " (debug by compare pow(x,0.5) and"
    " sqrt(x.0.5), double x=3.28586921557249207e-12)\n");
     */
  }
  //print_double(&sc->cv_etaq,1,"cv_etaq1639");
  __syncthreads();
  if (sc->cv_qwait != 0) {
    sc->cv_eta = sc->cv_etaq;
    //print_double(&sc->cv_eta,1,"cv_eta1639");
    sc->cv_qprime = sc->cv_q;
    cudaDevicecvSetEta(md, sc);
    return 0;
  }
  __syncthreads();
  sc->cv_qwait = 2;
  double ddn;
  sc->cv_etaqm1 = 0.;
  __syncthreads();
  if (sc->cv_q > 1) {
    cudaDeviceVWRMS_Norm_2(&md->dzn[md->nrows*sc->cv_q],
                         md->dewt, &ddn, md->n_shr_empty);
    __syncthreads();
    ddn *= md->cv_tq[1+blockIdx.x*(NUM_TESTS + 1)];
    __syncthreads();
    sc->cv_etaqm1 = 1./(dSUNRpowerR(BIAS1*ddn, 1./sc->cv_q) + ADDON);
  }
  double dup, cquot;
  sc->cv_etaqp1 = 0.;
  __syncthreads();
  if (sc->cv_q != md->cv_qmax && sc->cv_saved_tq5 != 0.) {
    cquot = (md->cv_tq[5+blockIdx.x*(NUM_TESTS + 1)] / sc->cv_saved_tq5) *
            dSUNRpowerR(sc->cv_h/md->cv_tau[2+blockIdx.x*(L_MAX + 1)],sc->cv_L);
    md->dtempv[i]=md->cv_acor[i]-cquot*md->dzn[i+md->nrows*md->cv_qmax];
    print_double(md->dtempv,73,"dtempv1658");
    cudaDeviceVWRMS_Norm_2(md->dtempv, md->dewt, &dup, md->n_shr_empty);
    __syncthreads();
    dup *= md->cv_tq[3+blockIdx.x*(NUM_TESTS + 1)];
    __syncthreads();
    sc->cv_etaqp1 = 1. / (dSUNRpowerR(BIAS3*dup, 1./(sc->cv_L+1)) + ADDON);
  }
  __syncthreads();
  cudaDevicecvChooseEta(md, sc);
  __syncthreads();
  cudaDevicecvSetEta(md, sc);
  __syncthreads();
  return CV_SUCCESS;
}

__device__
void cudaDevicecvIncreaseBDF(ModelDataGPU *md, ModelDataVariable *sc) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  double alpha0, alpha1, prod, xi, xiold, hsum, A1;
  int z, j;
  for (z=0; z <= md->cv_qmax; z++) md->cv_l[z+blockIdx.x*L_MAX] = 0.;
  md->cv_l[2+blockIdx.x*L_MAX] = alpha1 = prod = xiold = 1.;
  alpha0 = -1.;
  hsum = sc->cv_hscale;
  if (sc->cv_q > 1) {
    for (j=1; j < sc->cv_q; j++) {
      hsum += md->cv_tau[j+1+blockIdx.x*(L_MAX + 1)];
      xi = hsum / sc->cv_hscale;
      prod *= xi;
      alpha0 -= 1. / (j+1);
      alpha1 += 1. / xi;
      for (z=j+2; z >= 2; z--)
        md->cv_l[z+blockIdx.x*L_MAX] = md->cv_l[z+blockIdx.x*L_MAX]*xiold + md->cv_l[z-1+blockIdx.x*L_MAX];
      xiold = xi;
    }
  }
  A1 = (-alpha0 - alpha1) / prod;
  md->dzn[i+md->nrows*sc->cv_L]=A1*md->dzn[i+md->nrows*sc->cv_indx_acor];
  __syncthreads();
  for (j=2; j <= sc->cv_q; j++){
    md->dzn[i+md->nrows*j]+=md->cv_l[j+blockIdx.x*L_MAX]*md->dzn[i+md->nrows*(sc->cv_L)];
    __syncthreads();
  }
  print_double(md->dzn,73,"dzn1687");
}

__device__
void cudaDevicecvAdjustParams(ModelDataGPU *md, ModelDataVariable *sc) {
  if (sc->cv_qprime != sc->cv_q) {
    int deltaq = sc->cv_qprime-sc->cv_q;
    switch(deltaq) {
      case 1:
        cudaDevicecvIncreaseBDF(md, sc);
        break;
      case -1:
        cudaDevicecvDecreaseBDF(md, sc);
        break;
    }
    sc->cv_q = sc->cv_qprime;
    sc->cv_L = sc->cv_q+1;
    //print_int(&sc->cv_L,1,"cv_L1770");
    sc->cv_qwait = sc->cv_L;
  }
  cudaDevicecvRescale(md, sc);
}

__device__
int cudaDevicecvStep(ModelDataGPU *md, ModelDataVariable *sc) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int ncf = 0;
  int nef = 0;
  int nflag=FIRST_CALL;
  double saved_t=sc->cv_tn;
  double dsm;
  __syncthreads();
  if ((sc->cv_nst > 0) && (sc->cv_hprime != sc->cv_h)){
    cudaDevicecvAdjustParams(md, sc);
  }
  __syncthreads();
  for (;;) {
    __syncthreads();
    cudaDevicecvPredict(md, sc);
    __syncthreads();
    cudaDevicecvSet(md, sc);
    __syncthreads();
    nflag = cudaDevicecvNlsNewton(nflag,md, sc);
    __syncthreads();
    int kflag = cudaDevicecvHandleNFlag(md, sc, &nflag, saved_t, &ncf);
    __syncthreads();
    if (kflag == PREDICT_AGAIN) {
      continue;
    }
    if (kflag != DO_ERROR_TEST) {
      return (kflag);
    }
    __syncthreads();
    int eflag=cudaDevicecvDoErrorTest(md,sc,&nflag,saved_t,&nef,&dsm);
    __syncthreads();
    if (eflag == TRY_AGAIN){
      continue;
    }
    if (eflag != CV_SUCCESS){
      return (eflag);
    }
    break;
  }
  __syncthreads();
  cudaDevicecvCompleteStep(md, sc);
  __syncthreads();
  cudaDevicecvPrepareNextStep(md, sc, dsm);
  __syncthreads();
  sc->cv_etamax=10.;
  md->cv_acor[i]*=md->cv_tq[2+blockIdx.x*(NUM_TESTS + 1)];
  __syncthreads();
  return(CV_SUCCESS);
  }

__device__
int cudaDeviceCVodeGetDky(ModelDataGPU *md, ModelDataVariable *sc,
                           double t, int k, double *dky) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  double s, c, r;
  double tfuzz, tp, tn1;
  int z, j;
  __syncthreads();
   tfuzz = FUZZ_FACTOR * md->cv_uround * (fabs(sc->cv_tn) + fabs(sc->cv_hu));
   if (sc->cv_hu < 0.) tfuzz = -tfuzz;
   tp = sc->cv_tn - sc->cv_hu - tfuzz;
   tn1 = sc->cv_tn + tfuzz;
   if ((t-tp)*(t-tn1) > 0.) {
     return(CV_BAD_T);
   }
  __syncthreads();
   s = (t - sc->cv_tn) / sc->cv_h;
   for (j=sc->cv_q; j >= k; j--) {
     c = 1.;
     for (z=j; z >= j-k+1; z--) c *= z;
     if (j == sc->cv_q) {
       dky[i]=c*md->dzn[i+md->nrows*j];
     } else {
        dky[i]=c*md->dzn[i+md->nrows*j]+s*dky[i];
     }
   }
  __syncthreads();
   if (k == 0) return(CV_SUCCESS);
  __syncthreads();
   r = dSUNRpowerR(double(sc->cv_h),double(-k));
  __syncthreads();
   dky[i]=dky[i]*r;
   return(CV_SUCCESS);
}

__device__
int cudaDevicecvEwtSetSV(ModelDataGPU *md, ModelDataVariable *sc,double *weight) {
  extern __shared__ double flag_shr2[];
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  md->dtempv[i]=fabs(md->dzn[i]);
  double min;
  md->dtempv[i]=md->cv_reltol*md->dtempv[i]+md->cv_Vabstol[i];
  cudaDevicemin_2(&min, md->dtempv[i], flag_shr2, md->n_shr_empty);
__syncthreads();
  if (min <= 0.) return(-1);
  weight[i]= 1./md->dtempv[i];
  return(0);
}

__device__
int cudaDeviceCVode(ModelDataGPU *md, ModelDataVariable *sc) {
  extern __shared__ int flag_shr[];
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int kflag2;
  for(;;) {
    __syncthreads();
#ifdef CAMP_PROFILE_DEVICE_FUNCTIONS
    if(threadIdx.x==0) sc->countercvStep++;
#endif
    flag_shr[0] = 0;
    __syncthreads();
    sc->cv_next_h = sc->cv_h;
    sc->cv_next_q = sc->cv_q;
    int ewtsetOK = 0;
    if (sc->cv_nst > 0) {
      print_double(md->dtempv,73,"dtempvcv_efun0");
      ewtsetOK = cudaDevicecvEwtSetSV(md, sc, md->dewt);
      //print_double(md->dtempv,73,"dtempvcv_efun1");
      if (ewtsetOK != 0) {
        sc->cv_tretlast = sc->tret = sc->cv_tn;
        md->yout[i] = md->dzn[i];
        if(i==0) printf("ERROR: ewtsetOK\n");
        return CV_ILL_INPUT;
      }
    }
    if ((md->cv_mxstep > 0) && (sc->nstloc >= md->cv_mxstep)) {
      sc->cv_tretlast = sc->tret = sc->cv_tn;
      md->yout[i] = md->dzn[i];
      if(i==0) printf("ERROR: cv_mxstep\n");
      return CV_TOO_MUCH_WORK;
    }
    double nrm;
    cudaDeviceVWRMS_Norm_2(md->dzn,
     md->dewt, &nrm, md->n_shr_empty);
    sc->cv_tolsf = md->cv_uround * nrm;
    if (sc->cv_tolsf > 1.) {
      sc->cv_tretlast = sc->tret = sc->cv_tn;
      md->yout[i] = md->dzn[i];
      sc->cv_tolsf *= 2.;
      if(i==0) printf("ERROR: cv_tolsf\n");
      __syncthreads();
      if(i==0) printf("ERROR: cv_tolsf\n");
      return CV_TOO_MUCH_ACC;
    } else {
      sc->cv_tolsf = 1.;
    }
#ifdef ODE_WARNING
    if (sc->cv_tn + sc->cv_h == sc->cv_tn) {
      if(threadIdx.x==0) sc->cv_nhnil++;
      if ((sc->cv_nhnil <= sc->cv_mxhnil) ||
              (sc->cv_nhnil == sc->cv_mxhnil))
        if(i==0)printf("WARNING: h below roundoff level in tn");
    }
#endif
    //print_double(md->dzn,73,"dzn1858");
    kflag2 = cudaDevicecvStep(md, sc);
    __syncthreads();
    if (kflag2 != CV_SUCCESS) {
      sc->cv_tretlast = sc->tret = sc->cv_tn;
      md->yout[i] = md->dzn[i];
      if(i==0) printf("ERROR: kflag != CV_SUCCESS\n");
      return kflag2;
    }
    sc->nstloc++;
    if ((sc->cv_tn - md->tout) * sc->cv_h >= 0.) {
      sc->cv_tretlast = sc->tret = md->tout;
      cudaDeviceCVodeGetDky(md, sc, md->tout, 0, md->yout);
      return CV_SUCCESS;
    }
    if (md->cv_tstopset) {//needed?
      double troundoff = FUZZ_FACTOR * md->cv_uround * (fabs(sc->cv_tn) + fabs(sc->cv_h));
      if (fabs(sc->cv_tn - md->cv_tstop) <= troundoff) {
        cudaDeviceCVodeGetDky(md, sc, md->cv_tstop, 0, md->yout);
        sc->cv_tretlast = sc->tret = md->cv_tstop;
        md->cv_tstopset = SUNFALSE;
        if(i==0) printf("ERROR: cv_tstopset\n");
        __syncthreads();
        return CV_TSTOP_RETURN;
      }
      if ((sc->cv_tn + sc->cv_hprime - md->cv_tstop) * sc->cv_h > 0.) {
        sc->cv_hprime = (md->cv_tstop - sc->cv_tn) * (1.0 - 4.0 * md->cv_uround);
        if(i==0) printf("ERROR: sc->cv_tn + sc->cv_hprime - sc->cv_tstop\n");
        sc->cv_eta = sc->cv_hprime / sc->cv_h;
      }
    }
  }
}

__global__
void cudaGlobalCVode(ModelDataGPU md_object) {
  ModelDataGPU *md = &md_object;
  extern __shared__ int flag_shr[];
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  //TODO CHECK IF USING SC AS LOCAL INSTEAD OF MD->SCELLS HAS BETTER MAPE AND FINE IN MONARCH
  //IF WANT TO USE SC 1 PER BLOCK, THEN CHECK ALL SC->SOMETHING = SOMETHING AND BLOCKIDX.X CALLS AND ADD IF(THREADIDX.X==0)...SYNCTHREADS() TO AVOID OVERLAPPING
  //ModelDataVariable *sc = &md->sCells[blockIdx.x];
  ModelDataVariable sc_object = md->sCells[blockIdx.x];
  __syncthreads();
  ModelDataVariable *sc = &sc_object;
  __syncthreads();
  int istate;
  if(i<md->nrows){
#ifdef CAMP_PROFILE_DEVICE_FUNCTIONS
    int clock_khz=md->clock_khz;
    clock_t start;
    start = clock();
    __syncthreads();
#endif
    istate=cudaDeviceCVode(md,sc);
    __syncthreads();
#ifdef CAMP_PROFILE_DEVICE_FUNCTIONS
  if(threadIdx.x==0) sc->dtcudaDeviceCVode += ((double)(int)(clock() - start))/(clock_khz*1000);
  __syncthreads();
#endif
  }
  __syncthreads();
  if(threadIdx.x==0) md->flagCells[blockIdx.x]=istate;
#ifdef CAMP_PROFILE_DEVICE_FUNCTIONS
  ModelDataVariable *mdvo = md->mdvo;
  *mdvo = *sc;
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

void cvodeRun(ModelDataGPU *mGPU, cudaStream_t stream){
  int len_cell = mGPU->nrows / mGPU->n_cells;
  int threads_block = len_cell;
  int blocks = mGPU->n_cells;
  int n_shr_memory = nextPowerOfTwoCVODE2(len_cell);
  int n_shr_empty = mGPU->n_shr_empty = n_shr_memory - threads_block;
  cudaGlobalCVode <<<blocks, threads_block, n_shr_memory * sizeof(double), stream>>>(*mGPU);
}

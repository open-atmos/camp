/* Copyright (C) 2021 Barcelona Supercomputing Center and University of
* Illinois at Urbana-Champaign
* SPDX-License-Identifier: MIT
*/

#include "cvode_cuda.h"

__device__
double dSUNRpowerR(double base, double exponent){
  if (base <= ZERO) return(ZERO);
#ifdef EQUALLIZE_CPU_CUDA_POW
  if(exponent==(1./2)) return sqrt(base);
  if(exponent==(1./3)) return sqrt(sqrt(base));
  if(exponent==(1./4)) return sqrt(sqrt(base));
#endif
  return(pow(base, (double)(exponent)));
}

__device__
double dSUNRpowerI(double base, int exponent)
{
  int i, expt;
  double prod;
  prod = ONE;
  expt = abs(exponent);
  for(i = 1; i <= expt; i++) prod *= base;
  if (exponent < 0) prod = ONE/prod;
  return(prod);
}

#ifdef IS_DEBUG_MODE_removeAtomic

__device__
void time_derivative_add_value_gpu(TimeDerivativeGPU time_deriv, unsigned int spec_id,
                               double rate_contribution) {
  if (rate_contribution > 0.0) {
    time_deriv.production_rates[spec_id] += rate_contribution;
  } else {
    time_deriv.loss_rates[spec_id] += -rate_contribution;
  }
}

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
void time_derivative_add_value_gpu(TimeDerivativeGPU time_deriv, unsigned int spec_id,
                               double rate_contribution) {
  //WARNING: Atomicadd is not desirable,
  //because it leads to small deviations in the results,
  //even when scaling the number of data computed in the GPU
  //It would be desirable to remove it
  if (rate_contribution > 0.0) {
    atomicAdd_block(&(time_deriv.production_rates[spec_id]),rate_contribution);
  } else {
    atomicAdd_block(&(time_deriv.loss_rates[spec_id]),-rate_contribution);
  }
}

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

__device__ void cudaDevicemin(double *g_odata, double in, volatile double *sdata, int n_shr_empty){
  unsigned int tid = threadIdx.x;
  __syncthreads();
  sdata[tid] = in;
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

__device__ void cudaDeviceSpmv_CSR(double* dx, double* db, double* dA, int* djA, int* diA){
  __syncthreads();
  int row= threadIdx.x + blockDim.x*blockIdx.x;
  double sum = 0.0;
  int nnz=diA[blockDim.x];
  for(int j=diA[threadIdx.x]; j<diA[threadIdx.x+1]; j++){
    sum+= db[djA[j]+blockDim.x*blockIdx.x]*dA[j+nnz*blockIdx.x];
  }
  __syncthreads();
  dx[row]=sum;
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
#ifdef IS_DEBUG_MODE_cudaDevicedotxy_2
  //used for compare with cpu
  sdata[0] = 0.;
  __syncthreads();
  if(tid==0){
    for(int j=0;j<blockDim.x;j++){
      sdata[0]+=g_idata1[j+blockIdx.x*blockDim.x]*g_idata2[j+blockIdx.x*blockDim.x];
    }
  }
#else
  unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
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
  sdata[tid] = g_idata1[i]*g_idata2[i];
  sdata[tid] = sdata[tid]*sdata[tid];
  __syncthreads();
#ifdef IS_DEBUG_MODE_cudaDevicedotxy_2
  //used for compare with cpu
  if(tid==0){
    double sum=0.;
    for(int j=0;j<blockDim.x;j++){
      sum+=sdata[j];
    }
    sdata[0] = sum;
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
void cudaDeviceJacCopy(int* diA, double* Ax, double* Bx) {
  int nnz=diA[blockDim.x];
  for(int j=diA[threadIdx.x]; j<diA[threadIdx.x+1]; j++){
    Bx[j+nnz*blockIdx.x]=Ax[j+nnz*blockIdx.x];
  }
}

__device__
int cudaDevicecamp_solver_check_model_state(ModelDataGPU *md, ModelDataVariable *sc, double *y)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  extern __shared__ int flag_shr[];
  __syncthreads();
  flag_shr[0] = 0;
  __syncthreads();
  if (y[i] < -SMALL) {
    flag_shr[0] = CAMP_SOLVER_FAIL;
  } else {
    md->state[md->map_state_deriv[threadIdx.x]+blockIdx.x*md->n_per_cell_state_var] =
            y[i] <= -SMALL ?
            TINY : y[i];
  }
  __syncthreads();
  int flag = flag_shr[0];
  __syncthreads();
  return flag;
}

__device__ void solveRXN(
  int i_rxn,TimeDerivativeGPU deriv_data,
  double time_step,ModelDataGPU *md, ModelDataVariable *sc
){
  double *rxn_float_data = (double *)&( md->rxn_double[md->rxn_float_indices[i_rxn]]);
  int *int_data = (int *)&(md->rxn_int[md->rxn_int_indices[i_rxn]]);
  int *rxn_int_data = (int *) &(int_data[1]);
  double *rxn_env_data = &(md->rxn_env_data
  [md->n_rxn_env_data*blockIdx.x+md->rxn_env_idx[i_rxn]]);
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
        double *yout, bool use_deriv_est, ModelDataGPU *md, ModelDataVariable *sc)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  md->dn0[i]=y[i]-md->J_state[i];
  cudaDeviceSpmv_CSR(md->dy, md->dn0, md->J_solver, md->djA, md->diA);
  md->dn0[i]=md->J_deriv[i]+md->dy[i];
  TimeDerivativeGPU deriv_data;
  __syncthreads();
  deriv_data.production_rates = md->production_rates;
  deriv_data.loss_rates = md->loss_rates;
  __syncthreads();
  deriv_data.production_rates[i] = 0.0;
  deriv_data.loss_rates[i] = 0.0;
  __syncthreads();
  deriv_data.production_rates = &( md->production_rates[blockDim.x*blockIdx.x]);
  deriv_data.loss_rates = &( md->loss_rates[blockDim.x*blockIdx.x]);
  sc->grid_cell_state = &( md->state[md->n_per_cell_state_var*blockIdx.x]);
  __syncthreads();
#ifdef IS_DEBUG_MODE_removeAtomic
  if(threadIdx.x==0){
    for (int j = 0; j < md->n_rxn; j++){
      solveRXN(j,deriv_data, time_step, md, sc);
    }
  }
#else
  if( threadIdx.x < md->n_rxn) {
    int n_iters = md->n_rxn / blockDim.x;
    for (int j = 0; j < n_iters; j++) {
      int i_rxn = threadIdx.x + j*blockDim.x;
      solveRXN(i_rxn,deriv_data, time_step, md, sc);
    }
    int residual=md->n_rxn%blockDim.x;
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
    if (use_deriv_est) {
      double scale_fact = 1.0 / (r_p[i] + r_l[i]) /
          (1.0 / (r_p[i] + r_l[i]) + MAX_PRECISION_LOSS / fabs(r_p[i]- r_l[i]));
      yout[i] = scale_fact * (r_p[i] - r_l[i]) + (1.0 - scale_fact) * (md->dn0[i]);
    }else {
      yout[i] = r_p[i] - r_l[i];
    }
  } else {
    yout[i] = 0.0;
  }
}

__device__
int cudaDevicef(double time_step, double *y,
        double *yout, bool use_deriv_est, ModelDataGPU *md, ModelDataVariable *sc)
{
#ifdef CAMP_PROFILE_DEVICE_FUNCTIONS
  int clock_khz=md->clock_khz;
  clock_t start;
  start = clock();
#endif
  time_step = sc->cv_next_h;
  time_step = time_step > 0. ? time_step : md->init_time_step;
  int checkflag=cudaDevicecamp_solver_check_model_state(md, sc, y);
  if(checkflag==CAMP_SOLVER_FAIL){
#ifdef CAMP_PROFILE_DEVICE_FUNCTIONS
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(threadIdx.x==0) sc->timef += ((double)(int)(clock() - start))/(clock_khz*1000);
#endif
    return CAMP_SOLVER_FAIL;
  }
  cudaDevicecalc_deriv(time_step, y, yout, use_deriv_est, md, sc);
#ifdef CAMP_PROFILE_DEVICE_FUNCTIONS
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(threadIdx.x==0) sc->timef += ((double)(int)(clock() - start))/(clock_khz*1000);
#endif
  return 0;
}

__device__
int CudaDeviceguess_helper(double t_n, double h_n, double* y_n,
   double* y_n1, double* hf, double* atmp1,
   double* acorr, ModelDataGPU *md, ModelDataVariable *sc
) {
  extern __shared__ double sdata[];
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  double min;
  cudaDevicemin(&min, y_n[i], sdata, md->n_shr_empty);
  if(min>-SMALL){
    return 0;
  }
#ifdef CAMP_PROFILE_DEVICE_FUNCTIONS
  int clock_khz=md->clock_khz;
  clock_t start;
  start = clock();
#endif
  atmp1[i]=y_n1[i];
  if (h_n > 0.) {
    acorr[i]=(1./h_n)*hf[i];
  } else {
    acorr[i]=hf[i];
  }
  double t_0 = h_n > 0. ? t_n - h_n : t_n - 1.;
  double t_j = 0.;
  for (int iter = 0; iter < GUESS_MAX_ITER && t_0 + t_j < t_n; iter++) {
    double h_j = t_n - (t_0 + t_j);
#ifdef IS_DEBUG_MODE_CudaDeviceguess_helper
    if(threadIdx.x==0){
      for (int j = 0; j < blockDim.x; j++) {
        double t_star = -atmp1[j+blockIdx.x*blockDim.x] / acorr[j+blockIdx.x*blockDim.x];
        if ((t_star > 0. || (t_star == 0. && acorr[j+blockIdx.x*blockDim.x] < 0.)) &&
            t_star < h_j) {
          h_j = t_star;
        }
      }
      sdata[0] = h_j;
    }
    __syncthreads();
    h_j=sdata[0];
    __syncthreads();
#else
    double t_star = -atmp1[i] / acorr[i];
    if (t_star < 0. || (t_star == 0. && acorr[i] >= 0.)){
      t_star=h_j;
    }
    cudaDevicemin(&min, t_star, sdata, md->n_shr_empty);
    if(min<h_j){
      h_j = min;
      h_j *= 0.95 + 0.1 * iter / (double)GUESS_MAX_ITER;
    }
#endif
    h_j = t_n < t_0 + t_j + h_j ? t_n - (t_0 + t_j) : h_j;
    if (h_n == 0. && t_n - (h_j + t_j + t_0) > md->cv_reltol) {
#ifdef CAMP_PROFILE_DEVICE_FUNCTIONS
    if(threadIdx.x==0) sc->timeguess_helper += ((double)(clock() - start))/(clock_khz*1000);
#endif
    return -1;
    }
    atmp1[i]+=h_j*acorr[i];
    t_j += h_j;
    int fflag=cudaDevicef(t_0 + t_j, atmp1, acorr,true,md,sc);
    if (fflag == CAMP_SOLVER_FAIL) {
      acorr[i] = 0.;
#ifdef CAMP_PROFILE_DEVICE_FUNCTIONS
    if(threadIdx.x==0) sc->timeguess_helper += ((double)(clock() - start))/(clock_khz*1000);
#endif
     return -1;
    }
    if (iter == GUESS_MAX_ITER - 1 && t_0 + t_j < t_n) {
      if (h_n == 0.){
#ifdef CAMP_PROFILE_DEVICE_FUNCTIONS
    if(threadIdx.x==0) sc->timeguess_helper += ((double)(clock() - start))/(clock_khz*1000);
#endif
        return -1;
      }
    }
  }
  acorr[i]=atmp1[i]-y_n[i];
  if (h_n > 0.) acorr[i]=acorr[i]*0.999;
  hf[i]=atmp1[i]-y_n1[i];
#ifdef CAMP_PROFILE_DEVICE_FUNCTIONS
    if(threadIdx.x==0)  sc->timeguess_helper += ((double)(clock() - start))/(clock_khz*1000);
#endif
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
  [md->n_rxn_env_data*blockIdx.x+md->rxn_env_idx[i_rxn]]);
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
#ifdef CAMP_PROFILE_DEVICE_FUNCTIONS
  int clock_khz=md->clock_khz;
  clock_t start;
  start = clock();
#endif
  JacobianGPU *jac = &md->jac;
  JacobianGPU jacBlock;
  __syncthreads();
  jacBlock.num_elem = jac->num_elem;
  jacBlock.production_partials = &( jac->production_partials[jacBlock.num_elem[0]*blockIdx.x]);
  jacBlock.loss_partials = &( jac->loss_partials[jacBlock.num_elem[0]*blockIdx.x]);
  sc->grid_cell_state = &( md->state[md->n_per_cell_state_var*blockIdx.x]);
  __syncthreads();
  int n_rxn = md->n_rxn;
#ifdef IS_DEBUG_MODE_removeAtomic
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
int cudaDeviceJac(ModelDataGPU *md, ModelDataVariable *sc)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int retval;
#ifdef CAMP_PROFILE_DEVICE_FUNCTIONS
  int clock_khz=md->clock_khz;
  clock_t start;
  start = clock();
#endif
  retval=cudaDevicef(sc->cv_next_h, md->dcv_y, md->dftemp,false,md,sc);
  if(retval==CAMP_SOLVER_FAIL)
    return CAMP_SOLVER_FAIL;
  cudaDevicecalc_Jac(md->dcv_y,md, sc);
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
  md->J_state[i]=md->dcv_y[i];
  md->J_deriv[i]=md->dftemp[i];
#ifdef CAMP_PROFILE_DEVICE_FUNCTIONS
    if(threadIdx.x==0)  sc->timeJac += ((double)(clock() - start))/(clock_khz*1000);
#endif
  return 0;
}

__device__
int cudaDevicelinsolsetup(
    ModelDataGPU *md, ModelDataVariable *sc, int convfail
) {
  extern __shared__ int flag_shr[];
  double dgamma;
  int jbad, jok;
  dgamma = fabs((sc->cv_gamma / sc->cv_gammap) - 1.);
  jbad = (sc->cv_nst == 0) ||
         (sc->cv_nst > sc->nstlj + CVD_MSBJ) ||
         ((convfail == CV_FAIL_BAD_J) && (dgamma < CVD_DGMAX)) ||
         (convfail == CV_FAIL_OTHER);
  jok = !jbad;
  if (jok==1) {
    sc->cv_jcur = 0;
    cudaDeviceJacCopy(md->diA, md->dsavedJ, md->dA);
  } else {
    sc->nstlj = sc->cv_nst;
    sc->cv_jcur = 1;
    int guess_flag=cudaDeviceJac(md,sc);
    if (guess_flag < 0) {
      return -1;}
    if (guess_flag > 0) {
      return 1;}
   cudaDeviceJacCopy(md->diA, md->dA, md->dsavedJ);
  }
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  md->dx[i]=0.;
  cudaDeviceBCGprecond_2(md->dA, md->djA, md->diA, md->ddiag, -sc->cv_gamma);
  return 0;
}

__device__
void solveBcgCudaDeviceCVODE(ModelDataGPU *md, ModelDataVariable *sc)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  double alpha,rho0,omega0,beta,rho1,temp1,temp2;
  alpha=rho0=omega0=beta=rho1=temp1=temp2=1.0;
  md->dn0[i]=0.0;
  md->dp0[i]=0.0;
  cudaDeviceSpmv_CSR(md->dr0,md->dx,md->dA,md->djA,md->diA);
  md->dr0[i]=md->dtempv[i]-md->dr0[i];
  md->dr0h[i]=md->dr0[i];
  int it=0;
  while(it<1000 && temp1>1.0E-30){
    cudaDevicedotxy_2(md->dr0, md->dr0h, &rho1, md->n_shr_empty);
    beta = (rho1 / rho0) * (alpha / omega0);
    md->dp0[i]=beta*md->dp0[i]+md->dr0[i]-md->dn0[i]*omega0*beta;
    md->dy[i]=md->ddiag[i]*md->dp0[i];
    cudaDeviceSpmv_CSR(md->dn0, md->dy, md->dA, md->djA, md->diA);
    cudaDevicedotxy_2(md->dr0h, md->dn0, &temp1, md->n_shr_empty);
    alpha = rho1 / temp1;
    md->ds[i]=md->dr0[i]-alpha*md->dn0[i];
    md->dx[i]+=alpha*md->dy[i];
    md->dy[i]=md->ddiag[i]*md->ds[i];
    cudaDeviceSpmv_CSR(md->dt, md->dy, md->dA, md->djA, md->diA);
    md->dr0[i]=md->ddiag[i]*md->dt[i];
    cudaDevicedotxy_2(md->dy, md->dr0, &temp1, md->n_shr_empty);
    cudaDevicedotxy_2(md->dr0, md->dr0, &temp2, md->n_shr_empty);
    omega0 = temp1 / temp2;
    md->dx[i]+=omega0*md->dy[i];
    md->dr0[i]=md->ds[i]-omega0*md->dt[i];
    md->dt[i]=0.0;
    cudaDevicedotxy_2(md->dr0, md->dr0, &temp1, md->n_shr_empty);
    temp1 = sqrt(temp1);
    rho0 = rho1;
    it++;
  }
#ifdef CAMP_PROFILE_DEVICE_FUNCTIONS
  if(threadIdx.x==0) sc->counterBCGInternal += it;
  if(threadIdx.x==0) sc->counterBCG++;
#endif
}

__device__
int cudaDevicecvNewtonIteration(ModelDataGPU *md, ModelDataVariable *sc){
  extern __shared__ double flag_shr2[];
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  double del, delp, dcon;
  int m = 0;
  del = delp = 0.0;
  int retval;
#ifdef CAMP_PROFILE_DEVICE_FUNCTIONS
  int clock_khz=md->clock_khz;
  clock_t start;
#endif
  for(;;) {
#ifdef CAMP_PROFILE_DEVICE_FUNCTIONS
    start = clock();
#endif
    md->dtempv[i]=sc->cv_rl1*md->dzn[i+blockDim.x * gridDim.x]+md->cv_acor[i];
    md->dtempv[i]=sc->cv_gamma*md->dftemp[i]-md->dtempv[i];
    solveBcgCudaDeviceCVODE(md, sc);
#ifdef CAMP_PROFILE_DEVICE_FUNCTIONS
    if(threadIdx.x==0) sc->dtBCG += ((double)(int)(clock() - start))/(clock_khz*1000);
#endif
    md->dtempv[i] = md->dx[i];
    cudaDeviceVWRMS_Norm_2(md->dx, md->dewt, &del, md->n_shr_empty);
    md->dftemp[i]=md->dcv_y[i]+md->dtempv[i];
    int guessflag=CudaDeviceguess_helper(sc->cv_tn, 0., md->dftemp,
       md->dcv_y, md->dtempv, md->dtempv1,md->dp0, md, sc);
    if (guessflag < 0) {
      if (!(sc->cv_jcur)) {
        return TRY_AGAIN;
      } else {
        return RHSFUNC_RECVR;
      }
    }
    md->dftemp[i]=md->dcv_y[i]+md->dtempv[i];
    double min;
    cudaDevicemin(&min, md->dftemp[i], flag_shr2, md->n_shr_empty);
    if (min < -CAMP_TINY) {
      return CONV_FAIL;
    }
    md->cv_acor[i]+=md->dtempv[i];
    md->dcv_y[i]=md->dzn[i]+md->cv_acor[i];
    if (m > 0) {
      sc->cv_crate = SUNMAX(0.3 * sc->cv_crate, del / delp);
    }
    dcon = del * SUNMIN(1.0, sc->cv_crate) / md->cv_tq[4+blockIdx.x*(NUM_TESTS + 1)];
    __syncthreads();
    flag_shr2[0] = 0;
    __syncthreads();
    if (dcon <= 1.) {
      cudaDeviceVWRMS_Norm_2(md->cv_acor, md->dewt, &sc->cv_acnrm, md->n_shr_empty);
      sc->cv_jcur = 0;
      return CV_SUCCESS;
    }
    m++;
    if ((m == NLS_MAXCOR) || ((m >= 2) && (del > RDIV * delp))) {
      if (!(sc->cv_jcur)) {
        return TRY_AGAIN;
      } else {
        return RHSFUNC_RECVR;
      }
    }
    delp = del;
    retval=cudaDevicef(sc->cv_next_h, md->dcv_y, md->dftemp, true,md, sc);
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
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int retval=0;
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
  md->dftemp[i]=md->dzn[i]-md->cv_last_yn[i];
  md->cv_acor_init[i]=0.;
  int guessflag=CudaDeviceguess_helper(sc->cv_tn, sc->cv_h, md->dzn,
       md->cv_last_yn, md->dftemp, md->dtempv1,
       md->cv_acor_init, md, sc
  );
  if(guessflag<0){
    return RHSFUNC_RECVR;
  }
  for(;;) {
    md->dcv_y[i] = md->dzn[i]+md->cv_acor_init[i];
    retval=cudaDevicef(sc->cv_tn, md->dcv_y,md->dftemp,true,md,sc);
    if (retval < 0) {
      return CV_RHSFUNC_FAIL;
    }
    if (retval> 0) {
      return RHSFUNC_RECVR;
    }
    if (callSetup) {
#ifdef CAMP_PROFILE_DEVICE_FUNCTIONS
      start = clock();
#endif
      int linflag=cudaDevicelinsolsetup(md, sc,convfail);
#ifdef CAMP_PROFILE_DEVICE_FUNCTIONS
      if(threadIdx.x==0) sc->timelinsolsetup += ((double)(clock() - start))/(clock_khz*1000);
#endif
      callSetup = 0;
      sc->cv_gamrat = sc->cv_crate = 1.0;
      sc->cv_gammap = sc->cv_gamma;
      sc->cv_nstlp = sc->cv_nst;
      if (linflag < 0) {
        flag_shr[0] = CV_LSETUP_FAIL;
        break;
      }
      if (linflag > 0) {
        flag_shr[0] = CONV_FAIL;
        break;
      }
    }
    md->cv_acor[i] = md->cv_acor_init[i];
#ifdef CAMP_PROFILE_DEVICE_FUNCTIONS
    start = clock();
#endif
    int nItflag=cudaDevicecvNewtonIteration(md, sc);
#ifdef CAMP_PROFILE_DEVICE_FUNCTIONS
    if(threadIdx.x==0)  sc->timeNewtonIteration += ((double)(clock() - start))/(clock_khz*1000);
#endif
    if (nItflag != TRY_AGAIN) {
      return nItflag;
    }
    callSetup = 1;
    convfail = CV_FAIL_BAD_J;
  } //for(;;)
  return nflag;
}

__device__
void cudaDevicecvRescale(ModelDataGPU *md, ModelDataVariable *sc) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  double factor;
  factor = sc->cv_eta;
  for (int j=1; j <= sc->cv_q; j++) {
    md->dzn[i+blockDim.x * gridDim.x*j]*=factor;
    factor *= sc->cv_eta;
  }
  sc->cv_h = sc->cv_hscale * sc->cv_eta;
  sc->cv_next_h = sc->cv_h;
  sc->cv_hscale = sc->cv_h;
}

__device__
void cudaDevicecvRestore(ModelDataGPU *md, ModelDataVariable *sc, double saved_t) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j, k;
  sc->cv_tn=saved_t;
  for (k = 1; k <= sc->cv_q; k++){
    for (j = sc->cv_q; j >= k; j--) {
      md->dzn[i+blockDim.x * gridDim.x*(j-1)]-=md->dzn[i+blockDim.x * gridDim.x*j];
    }
  }
  md->dzn[i]=md->cv_last_yn[i];
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
  if ((fabs(sc->cv_h) <= sc->cv_hmin*ONEPSM) ||
      (*ncfPtr == CAMP_SOLVER_DEFAULT_MAX_CONV_FAILS)) {
    if (*nflagPtr == CONV_FAIL)     return(CV_CONV_FAILURE);
    if (*nflagPtr == RHSFUNC_RECVR) return(CV_REPTD_RHSFUNC_ERR);
  }
  sc->cv_eta = SUNMAX(ETACF,
          sc->cv_hmin / fabs(sc->cv_h));
  *nflagPtr = PREV_CONV_FAIL;
  cudaDevicecvRescale(md, sc);
  return (PREDICT_AGAIN);
}

__device__
void cudaDevicecvSetTqBDFt(ModelDataGPU *md, ModelDataVariable *sc,
                           double hsum, double alpha0, double alpha0_hat,
                           double xi_inv, double xistar_inv) {
  extern __shared__ int flag_shr[];
  double A1, A2, A3, A4, A5, A6;
  double C, Cpinv, Cppinv;
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
    hsum += md->cv_tau[sc->cv_q+blockIdx.x*(L_MAX + 1)];
    xi_inv = sc->cv_h / hsum;
    A5 = alpha0 - (1. / (sc->cv_q+1));
    A6 = alpha0_hat - xi_inv;
    Cppinv = (1. - A6 + A5) / A2;
    md->cv_tq[3+blockIdx.x*(NUM_TESTS + 1)] = fabs(Cppinv / (xi_inv * (sc->cv_q+2) * A5));
  }
  md->cv_tq[4+blockIdx.x*(NUM_TESTS + 1)] = CV_NLSCOEF / md->cv_tq[2+blockIdx.x*(NUM_TESTS + 1)];
}

__device__
void cudaDevicecvSetBDF(ModelDataGPU *md, ModelDataVariable *sc) {
  extern __shared__ int flag_shr[];
  double alpha0, alpha0_hat, xi_inv, xistar_inv, hsum;
  int z,j;
  md->cv_l[0+blockIdx.x*L_MAX] = md->cv_l[1+blockIdx.x*L_MAX] = xi_inv = xistar_inv = 1.;
  for (z=2; z <= sc->cv_q; z++) md->cv_l[z+blockIdx.x*L_MAX] = 0.;
  alpha0 = alpha0_hat = -1.;
  hsum = sc->cv_h;
  if (sc->cv_q > 1) {
    for (j=2; j < sc->cv_q; j++) {
      hsum += md->cv_tau[j-1+blockIdx.x*(L_MAX + 1)];
      xi_inv = sc->cv_h / hsum;
      alpha0 -= 1. / j;
      for (z=j; z >= 1; z--) md->cv_l[z+blockIdx.x*L_MAX] += md->cv_l[z-1+blockIdx.x*L_MAX]*xi_inv;
    }
    alpha0 -= 1. / sc->cv_q;
    xistar_inv = -md->cv_l[1+blockIdx.x*L_MAX] - alpha0;
    hsum += md->cv_tau[sc->cv_q-1+blockIdx.x*(L_MAX + 1)];
    xi_inv = sc->cv_h / hsum;
    alpha0_hat = -md->cv_l[1+blockIdx.x*L_MAX] - xi_inv;
    for (z=sc->cv_q; z >= 1; z--)
      md->cv_l[z+blockIdx.x*L_MAX] += md->cv_l[z-1+blockIdx.x*L_MAX]*xistar_inv;
  }
  cudaDevicecvSetTqBDFt(md, sc, hsum, alpha0, alpha0_hat, xi_inv, xistar_inv);
}

__device__
void cudaDevicecvSet(ModelDataGPU *md, ModelDataVariable *sc) {
  extern __shared__ int flag_shr[];
  cudaDevicecvSetBDF(md,sc);
  sc->cv_rl1 = 1.0 / md->cv_l[1+blockIdx.x*L_MAX];
  sc->cv_gamma = sc->cv_h * sc->cv_rl1;
  if (sc->cv_nst == 0){
    sc->cv_gammap = sc->cv_gamma;
  }
  sc->cv_gamrat = (sc->cv_nst > 0) ?
                    sc->cv_gamma / sc->cv_gammap : 1.;  // protect x / x != 1.0
}

__device__
void cudaDevicecvPredict(ModelDataGPU *md, ModelDataVariable *sc) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j, k;
  sc->cv_tn += sc->cv_h;
  md->cv_last_yn[i]=md->dzn[i];
  for (k = 1; k <= sc->cv_q; k++){
    for (j = sc->cv_q; j >= k; j--){
      md->dzn[i+blockDim.x * gridDim.x*(j-1)]+=md->dzn[i+blockDim.x * gridDim.x*j];
    }
  }
}

__device__
void cudaDevicecvDecreaseBDF(ModelDataGPU *md, ModelDataVariable *sc) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  double hsum, xi;
  int z, j;
  for (z=0; z <= BDF_Q_MAX; z++) md->cv_l[z+blockIdx.x*L_MAX] = 0.;
  md->cv_l[2+blockIdx.x*L_MAX] = 1.;
  hsum = 0.;
  for (j=1; j <= sc->cv_q-2; j++) {
    hsum += md->cv_tau[j+blockIdx.x*(L_MAX + 1)];
    xi = hsum /sc->cv_hscale;
    for (z=j+2; z >= 2; z--)
      md->cv_l[z+blockIdx.x*L_MAX] = md->cv_l[z+blockIdx.x*L_MAX]*xi + md->cv_l[z-1+blockIdx.x*L_MAX];
  }
  for (j=2; j < sc->cv_q; j++){
    md->dzn[i+blockDim.x * gridDim.x*j]=-md->cv_l[j+blockIdx.x*L_MAX]*
      md->dzn[i+blockDim.x * gridDim.x*sc->cv_q]+md->dzn[i+blockDim.x * gridDim.x*j];
  }
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
  cudaDevicemin(&min_val, md->dftemp[i], flag_shr2, md->n_shr_empty);
  if (min_val < 0. && min_val > -CAMP_TINY) {
    md->dftemp[i]=fabs(md->dftemp[i]);
    md->dzn[i]=md->dftemp[i]-md->cv_l[0+blockIdx.x*L_MAX]*md->cv_acor[i];
    min_val = 0.;
  }
  dsm = sc->cv_acnrm * md->cv_tq[2+blockIdx.x*(NUM_TESTS + 1)];
  *dsmPtr = dsm;
  if (dsm <= 1. && min_val >= 0.) return(CV_SUCCESS);
  (*nefPtr)++;
  *nflagPtr = PREV_ERR_FAIL;
  cudaDevicecvRestore(md, sc, saved_t);
  if ((fabs(sc->cv_h) <= sc->cv_hmin*ONEPSM) ||
      (*nefPtr == CAMP_SOLVER_DEFAULT_MAX_CONV_FAILS)) return(CV_ERR_FAILURE);
  sc->cv_etamax = 1.;
  if (*nefPtr <= MXNEF1) {
    sc->cv_eta = 1. / (dSUNRpowerR(BIAS2*dsm,1./sc->cv_L) + ADDON);
    sc->cv_eta = SUNMAX(ETAMIN, SUNMAX(sc->cv_eta,
                           sc->cv_hmin / fabs(sc->cv_h)));
    if (*nefPtr >= SMALL_NEF)
      sc->cv_eta = SUNMIN(sc->cv_eta, ETAMXF);
    cudaDevicecvRescale(md, sc);
    return(TRY_AGAIN);
  }
  if (sc->cv_q > 1) {
    sc->cv_eta = SUNMAX(ETAMIN,sc->cv_hmin / fabs(sc->cv_h));
    cudaDevicecvDecreaseBDF(md, sc);
    sc->cv_L = sc->cv_q;
    sc->cv_q--;
    sc->cv_qwait = sc->cv_L;
    cudaDevicecvRescale(md, sc);
    return(TRY_AGAIN);
  }
  sc->cv_eta = SUNMAX(ETAMIN, sc->cv_hmin / fabs(sc->cv_h));
  sc->cv_h *= sc->cv_eta;
  sc->cv_next_h = sc->cv_h;
  sc->cv_hscale = sc->cv_h;
  sc->cv_qwait = 10;
  retval=cudaDevicef(sc->cv_tn, md->dzn, md->dtempv,true,md,sc);
  if (retval < 0)  return(CV_RHSFUNC_FAIL);
  if (retval > 0)  return(CV_UNREC_RHSFUNC_ERR);
  md->dzn[i+blockDim.x * gridDim.x]=sc->cv_h*md->dtempv[i];
  return(TRY_AGAIN);
}

__device__
void cudaDevicecvCompleteStep(ModelDataGPU *md, ModelDataVariable *sc) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int z, j;
  sc->cv_nst++;
  sc->cv_hu = sc->cv_h;
  for (z=sc->cv_q; z >= 2; z--)  md->cv_tau[z+blockIdx.x*(L_MAX + 1)] = md->cv_tau[z-1+blockIdx.x*(L_MAX + 1)];
  if ((sc->cv_q==1) && (sc->cv_nst > 1))
    md->cv_tau[2+blockIdx.x*(L_MAX + 1)] = md->cv_tau[1+blockIdx.x*(L_MAX + 1)];
  md->cv_tau[1+blockIdx.x*(L_MAX + 1)] = sc->cv_h;
  for (j=0; j <= sc->cv_q; j++){
    md->dzn[i+blockDim.x * gridDim.x*j]+=md->cv_l[j+blockIdx.x*L_MAX]*md->cv_acor[i];
  }
  sc->cv_qwait--;
  if ((sc->cv_qwait == 1) && (sc->cv_q != BDF_Q_MAX)) {
    md->dzn[i+blockDim.x * gridDim.x*BDF_Q_MAX]=md->cv_acor[i];
    sc->cv_saved_tq5 = md->cv_tq[5+blockIdx.x*(NUM_TESTS + 1)];
  }
}

__device__
void cudaDevicecvChooseEta(double cv_etaqp1, double cv_etaq, double cv_etaqm1, ModelDataGPU *md, ModelDataVariable *sc) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  double etam;
  etam = SUNMAX(cv_etaqm1, SUNMAX(cv_etaq, cv_etaqp1));
  if (etam < THRESH) {
    sc->cv_eta = 1.;
    sc->cv_qprime = sc->cv_q;
    return;
  }
  if (etam == cv_etaq) {
    sc->cv_eta = cv_etaq;
    sc->cv_qprime = sc->cv_q;
  } else if (etam == cv_etaqm1) {
    sc->cv_eta = cv_etaqm1;
    sc->cv_qprime = sc->cv_q - 1;
  } else {
    sc->cv_eta = cv_etaqp1;
    sc->cv_qprime = sc->cv_q + 1;
    md->dzn[i+blockDim.x * gridDim.x*BDF_Q_MAX]=md->cv_acor[i];
  }
}

__device__
void cudaDevicecvSetEta(ModelDataGPU *md, ModelDataVariable *sc) {
  if (sc->cv_eta < THRESH) {
    sc->cv_eta = 1.;
    sc->cv_hprime = sc->cv_h;
  } else {
    sc->cv_eta = SUNMIN(sc->cv_eta, sc->cv_etamax);
    sc->cv_hprime = sc->cv_h * sc->cv_eta;
  }
}

__device__
int cudaDevicecvPrepareNextStep(ModelDataGPU *md, ModelDataVariable *sc, double dsm) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (sc->cv_etamax == 1.) {
    sc->cv_qwait = SUNMAX(sc->cv_qwait, 2);
    sc->cv_qprime = sc->cv_q;
    sc->cv_hprime = sc->cv_h;
    sc->cv_eta = 1.;
    return 0;
  }
  double cv_etaq=1./(dSUNRpowerR(BIAS2*dsm,1./sc->cv_L) + ADDON);
  if (sc->cv_qwait != 0) {
    sc->cv_eta = cv_etaq;
    sc->cv_qprime = sc->cv_q;
    cudaDevicecvSetEta(md, sc);
    return 0;
  }
  sc->cv_qwait = 2;
  double ddn;
  double cv_etaqm1 = 0.;
  if (sc->cv_q > 1) {
    cudaDeviceVWRMS_Norm_2(&md->dzn[blockDim.x * gridDim.x*sc->cv_q],
                         md->dewt, &ddn, md->n_shr_empty);
    ddn *= md->cv_tq[1+blockIdx.x*(NUM_TESTS + 1)];
    cv_etaqm1 = 1./(dSUNRpowerR(BIAS1*ddn, 1./sc->cv_q) + ADDON);
  }
  double dup, cquot;
  double cv_etaqp1 = 0.;
  if (sc->cv_q != BDF_Q_MAX && sc->cv_saved_tq5 != 0.) {
    cquot = (md->cv_tq[5+blockIdx.x*(NUM_TESTS + 1)] / sc->cv_saved_tq5) *
            dSUNRpowerI(sc->cv_h/md->cv_tau[2+blockIdx.x*(L_MAX + 1)],(double)sc->cv_L);
    md->dtempv[i]=md->cv_acor[i]-cquot*md->dzn[i+blockDim.x * gridDim.x*BDF_Q_MAX];
    cudaDeviceVWRMS_Norm_2(md->dtempv, md->dewt, &dup, md->n_shr_empty);
    dup *= md->cv_tq[3+blockIdx.x*(NUM_TESTS + 1)];
    cv_etaqp1 = 1. / (dSUNRpowerR(BIAS3*dup, 1./(sc->cv_L+1)) + ADDON);
  }
  cudaDevicecvChooseEta(cv_etaqp1, cv_etaq, cv_etaqm1 ,md, sc);
  cudaDevicecvSetEta(md, sc);
  return CV_SUCCESS;
}

__device__
void cudaDevicecvIncreaseBDF(ModelDataGPU *md, ModelDataVariable *sc) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  double alpha0, alpha1, prod, xi, xiold, hsum, A1;
  int z, j;
  for (z=0; z <= BDF_Q_MAX; z++) md->cv_l[z+blockIdx.x*L_MAX] = 0.;
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
  md->dzn[i+blockDim.x * gridDim.x*sc->cv_L]=A1*md->dzn[i+blockDim.x * gridDim.x*BDF_Q_MAX];
  for (j=2; j <= sc->cv_q; j++){
    md->dzn[i+blockDim.x * gridDim.x*j]+=md->cv_l[j+blockIdx.x*L_MAX]*md->dzn[i+blockDim.x * gridDim.x*(sc->cv_L)];
  }
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
  if ((sc->cv_nst > 0) && (sc->cv_hprime != sc->cv_h)){
    cudaDevicecvAdjustParams(md, sc);
  }
  for (;;) {
    cudaDevicecvPredict(md, sc);
    cudaDevicecvSet(md, sc);
    nflag = cudaDevicecvNlsNewton(nflag,md, sc);
    int kflag = cudaDevicecvHandleNFlag(md, sc, &nflag, saved_t, &ncf);
    if (kflag == PREDICT_AGAIN) {
      continue;
    }
    if (kflag != DO_ERROR_TEST) {
      return (kflag);
    }
    int eflag=cudaDevicecvDoErrorTest(md,sc,&nflag,saved_t,&nef,&dsm);
    if (eflag == TRY_AGAIN){
      continue;
    }
    if (eflag != CV_SUCCESS){
      return (eflag);
    }
    break;
  }
  cudaDevicecvCompleteStep(md, sc);
  cudaDevicecvPrepareNextStep(md, sc, dsm);
  sc->cv_etamax=10.;
  md->cv_acor[i]*=md->cv_tq[2+blockIdx.x*(NUM_TESTS + 1)];
  return(CV_SUCCESS);
  }

__device__
int cudaDeviceCVodeGetDky(ModelDataGPU *md, ModelDataVariable *sc,
                           double t, int k, double *dky) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  double s, c, r;
  double tfuzz, tp, tn1;
  int z, j;
  tfuzz = FUZZ_FACTOR * UNIT_ROUNDOFF * (fabs(sc->cv_tn) + fabs(sc->cv_hu));
  if (sc->cv_hu < 0.) tfuzz = -tfuzz;
  tp = sc->cv_tn - sc->cv_hu - tfuzz;
  tn1 = sc->cv_tn + tfuzz;
  if ((t-tp)*(t-tn1) > 0.) {
   return(CV_BAD_T);
  }
  s = (t - sc->cv_tn) / sc->cv_h;
  for (j=sc->cv_q; j >= k; j--) {
   c = 1.;
   for (z=j; z >= j-k+1; z--) c *= z;
   if (j == sc->cv_q) {
      dky[i]=c*md->dzn[i+blockDim.x * gridDim.x*j];
   } else {
      dky[i]=c*md->dzn[i+blockDim.x * gridDim.x*j]+s*dky[i];
   }
  }
  if (k == 0) return(CV_SUCCESS);
  r = dSUNRpowerI(double(sc->cv_h),double(-k));
  dky[i]=dky[i]*r;
return(CV_SUCCESS);
}

__device__
int cudaDevicecvEwtSetSV(ModelDataGPU *md, ModelDataVariable *sc,double *weight) {
  extern __shared__ double flag_shr2[];
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  md->dtempv[i]=fabs(md->dzn[i]);
  double min;
  md->dtempv[i]=md->cv_reltol*md->dtempv[i]+md->cv_Vabstol[threadIdx.x];
  cudaDevicemin(&min, md->dtempv[i], flag_shr2, md->n_shr_empty);
  if (min <= 0.) return(-1);
  weight[i]= 1./md->dtempv[i];
  return(0);
}

__device__
int cudaDeviceCVode(ModelDataGPU *md, ModelDataVariable *sc) {
  extern __shared__ int flag_shr[];
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int kflag2, retval;
  sc->cv_h = md->init_time_step; //CVodeSetInitStep
  //CVodeReInit
  sc->cv_q=1;
  sc->cv_L=2;
  sc->cv_qwait=sc->cv_L;
  sc->cv_etamax=ETAMX1;
  sc->cv_next_h=0.;
  retval = cudaDevicecvEwtSetSV(md, sc, md->dewt);
  if(retval != 0){
    return(CV_ILL_INPUT);
  }
  retval = cudaDevicef(
    sc->cv_tn, md->dzn, &md->dzn[i+blockDim.x * gridDim.x], true, md, sc);
  md->yout[i]=md->dzn[i];
  if (retval != 0) {
    return(CV_RHSFUNC_FAIL);
  }
  if (fabs(sc->cv_h) < sc->cv_hmin){
    sc->cv_h *= sc->cv_hmin/fabs(sc->cv_h);
  }
  sc->cv_hscale = sc->cv_h;
  sc->cv_hprime = sc->cv_h;
  md->dzn[i+blockDim.x * gridDim.x]*=sc->cv_h;
  md->dtempv1[i] = md->dzn[i] + md->dzn[i+blockDim.x * gridDim.x];
  CudaDeviceguess_helper(
    sc->cv_tn + sc->cv_h, sc->cv_h, md->dtempv1,
    md->dzn, &md->dzn[i+blockDim.x * gridDim.x], md->dp0,
    md->cv_acor_init, md, sc);
  int nstloc=0;
  sc->nstlj=0;
  sc->cv_nst=0;
  sc->cv_nstlp=0;
  for(;;) {
#ifdef CAMP_PROFILE_DEVICE_FUNCTIONS
    if(threadIdx.x==0) sc->countercvStep++;
#endif
    __syncthreads();
    flag_shr[0] = 0;
    __syncthreads();
    sc->cv_next_h = sc->cv_h;
    int ewtsetOK = 0;
    if (sc->cv_nst > 0) {
      ewtsetOK = cudaDevicecvEwtSetSV(md, sc, md->dewt);
      if (ewtsetOK != 0) {
        md->yout[i] = md->dzn[i];
        if(i==0) printf("ERROR: ewtsetOK\n");
        return CV_ILL_INPUT;
      }
    }
    if ((CAMP_SOLVER_DEFAULT_MAX_STEPS > 0) &&
      (nstloc >= CAMP_SOLVER_DEFAULT_MAX_STEPS)) {
      md->yout[i] = md->dzn[i];
      if(i==0) printf("ERROR: CAMP_SOLVER_DEFAULT_MAX_STEPS reached "
        "nstloc %d CAMP_SOLVER_DEFAULT_MAX_STEPS %d\n",
        nstloc,CAMP_SOLVER_DEFAULT_MAX_STEPS);
      return CV_TOO_MUCH_WORK;
    }
    double nrm;
    cudaDeviceVWRMS_Norm_2(md->dzn,
     md->dewt, &nrm, md->n_shr_empty);
    if (UNIT_ROUNDOFF * nrm > 1.) {
      md->yout[i] = md->dzn[i];
      if(i==0) printf("ERROR: cv_tolsf > 1\n");
      return CV_TOO_MUCH_ACC;
    }
    kflag2 = cudaDevicecvStep(md, sc);
    if (kflag2 != CV_SUCCESS) {
      md->yout[i] = md->dzn[i];
      if(i==0) printf("ERROR: kflag != CV_SUCCESS\n");
      return kflag2;
    }
    nstloc++;
    if ((sc->cv_tn - md->tout) * sc->cv_h >= 0.) {
      cudaDeviceCVodeGetDky(md, sc, md->tout, 0, md->yout);
      return CV_SUCCESS;
    }
  }
}

__global__
void cudaGlobalCVode(double t_initial, ModelDataGPU md_object) {
  ModelDataGPU *md = &md_object;
  extern __shared__ int flag_shr[];
  ModelDataVariable sc_object = *md->sCells;
  ModelDataVariable *sc = &sc_object;
  sc->cv_tn = t_initial;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  //Update concs from state
  md->dzn[i]=
    md->state[md->map_state_deriv[threadIdx.x]+blockIdx.x*md->n_per_cell_state_var] > TINY
    ? md->state[md->map_state_deriv[threadIdx.x]+blockIdx.x*md->n_per_cell_state_var]
    : TINY;
#ifdef CAMP_PROFILE_DEVICE_FUNCTIONS
    int clock_khz=md->clock_khz;
    clock_t start;
    start = clock();
#endif
  cudaDeviceCVode(md,sc);
  //Update state from concs
  md->state[md->map_state_deriv[threadIdx.x]+blockIdx.x*md->n_per_cell_state_var]=
    md->yout[i] > 0. ? md->yout[i] : 0.;
  //if(i==0) printf("kernel end\n");
#ifdef CAMP_PROFILE_DEVICE_FUNCTIONS
    if(threadIdx.x==0) sc->dtcudaDeviceCVode += ((double)(int)(clock() - start))/(clock_khz*1000);
#endif
#ifdef CAMP_PROFILE_DEVICE_FUNCTIONS
  ModelDataVariable *mdvo = md->mdvo;
  *mdvo = *sc;
#endif
}

static int nextPowerOfTwoCVODE2(int v){
  v--;
  v |= v >> 1;
  v |= v >> 2;
  v |= v >> 4;
  v |= v >> 8;
  v |= v >> 16;
  v++;
  return v;
}

void cvodeRun(double t_initial, ModelDataGPU *mGPU, int blocks, int threads_block, cudaStream_t stream){
  int n_shr_memory = nextPowerOfTwoCVODE2(threads_block);
  mGPU->n_shr_empty = n_shr_memory - threads_block;
  cudaGlobalCVode <<<blocks, threads_block, n_shr_memory * sizeof(double), stream>>>
    (t_initial, *mGPU);
}
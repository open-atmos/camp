/* Copyright (C) 2021 Barcelona Supercomputing Center and University of
 * Illinois at Urbana-Champaign
 * SPDX-License-Identifier: MIT
 */

#include<iostream>
#include<cuda.h>
#include<cuda_runtime.h>
#include<cuda_runtime_api.h>
#include "libsolv.h"

__device__ void cudaDeviceBCGprecond(double* dA, int* djA, int* diA, double* ddiag, double alpha){
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

__device__ void cudaDevicesetconst(double* dy,double constant){
  int row= threadIdx.x + blockDim.x*blockIdx.x;
  dy[row]=constant;
}

__device__ void cudaDeviceSpmvCSR(double* dx, double* db, double* dA, int* djA, int* diA){
  __syncthreads();
  int row= threadIdx.x + blockDim.x*blockIdx.x;
  double sum = 0.0;
  int nnz=diA[blockDim.x];
  for(int j=diA[threadIdx.x]; j<diA[threadIdx.x+1]; j++){
    sum+= db[djA[j]+blockDim.x*blockIdx.x]*dA[j+nnz*blockIdx.x];
  }
  dx[row]=sum;
  __syncthreads();
}

__device__ void cudaDeviceSpmvCSC_block(double* dx, double* db, double* dA, int* djA, int* diA){
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

__device__ void cudaDeviceSpmv(double* dx, double* db, double* dA, int* djA, int* diA){
#ifndef old_USE_CSR_ODE_GPU
  cudaDeviceSpmvCSR(dx,db,dA,djA,diA);
#else
  cudaDeviceSpmvCSC_block(dx,db,dA,djA,diA);
#endif
}

// y= a*x+ b*y
__device__ void cudaDeviceaxpby(double* dy,double* dx, double a, double b, int nrows)
{
  int row= threadIdx.x + blockDim.x*blockIdx.x;
  dy[row]= a*dx[row] + b*dy[row];
}

// y = x
__device__ void cudaDeviceyequalsx(double* dy,double* dx,int nrows)
{
  int row= threadIdx.x + blockDim.x*blockIdx.x;
    dy[row]=dx[row];
}

__device__ void cudaDevicemin(double *g_odata, double in, volatile double *sdata, int n_shr_empty){
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

__device__ void warpReduce(volatile double *sdata, unsigned int tid) {
  unsigned int blockSize = blockDim.x;
  if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
  if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
  if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
  if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
  if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
  if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
}

__device__ void cudaDevicedotxy(double *g_idata1, double *g_idata2,
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
  if (tid < 32) warpReduce(sdata, tid);
  __syncthreads();
  *g_odata = sdata[0];
  __syncthreads();
}

// z= a*z + x + b*y
__device__ void cudaDevicezaxpbypc(double* dz, double* dx,double* dy, double a, double b, int nrows)
{
  int row= threadIdx.x + blockDim.x*blockIdx.x;
  dz[row]=a*dz[row]  + dx[row] + b*dy[row];
}

// z= x*y
__device__ void cudaDevicemultxy(double* dz, double* dx,double* dy, int nrows)
{
  int row= threadIdx.x + blockDim.x*blockIdx.x;
  dz[row]=dx[row]*dy[row];
}

// z= a*x + b*y
__device__ void cudaDevicezaxpby(double a, double* dx, double b, double* dy, double* dz, int nrows)
{
  int row= threadIdx.x + blockDim.x*blockIdx.x;
  dz[row]=a*dx[row] + b*dy[row];
}

// y= a*x + y
__device__ void cudaDeviceaxpy(double* dy,double* dx, double a, int nrows)
{
  int row= threadIdx.x + blockDim.x*blockIdx.x;
  dy[row]+=a*dx[row];
}



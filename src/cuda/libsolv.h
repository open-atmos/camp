/* Copyright (C) 2021 Barcelona Supercomputing Center and University of
 * Illinois at Urbana-Champaign
 * SPDX-License-Identifier: MIT
 */

#ifndef LIBSOLV_H
#define LIBSOLV_H

#include<cuda.h>

__device__ void cudaDeviceBCGprecond(double* dA, int* djA, int* diA, double* ddiag, double alpha);
__device__ void cudaDevicesetconst(double* dy,double constant);
__device__ void cudaDeviceSpmvCSR(double* dx, double* db, double* dA, int* djA, int* diA);
__device__ void cudaDeviceSpmvCSC_block(double* dx, double* db, double* dA, int* djA, int* diA);
__device__ void cudaDeviceSpmv(double* dx, double* db, double* dA, int* djA, int* diA);
__device__ void cudaDeviceaxpby(double* dy,double* dx, double a, double b, int nrows);
__device__ void cudaDeviceyequalsx(double* dy,double* dx,int nrows);
__device__ void cudaDevicemin(double *g_odata, double in, volatile double *sdata, int n_shr_empty);
__device__ void cudaDevicedotxy(double *g_idata1, double *g_idata2, double *g_odata, int n_shr_empty);
__device__ void cudaDevicezaxpbypc(double* dz, double* dx,double* dy, double a, double b, int nrows);
__device__ void cudaDevicemultxy(double* dz, double* dx,double* dy, int nrows);
__device__ void cudaDevicezaxpby(double a, double* dx, double b, double* dy, double* dz, int nrows);
__device__ void cudaDeviceaxpy(double* dy,double* dx, double a, int nrows);



#endif
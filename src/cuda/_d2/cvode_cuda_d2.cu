/* Copyright (C) 2021 Barcelona Supercomputing Center and University of
* Illinois at Urbana-Champaign
* SPDX-License-Identifier: MIT
*/

#include "cvode_cuda_d2.h"
#include "libsolv.h"

__device__
void solveBcgCuda_d2_cvode_cuda(
        double *dA, int *djA, int *diA, double *dx, double *dtempv //Input data
        ,int nrows, int blocks, int n_shr_empty, int maxIt
        ,int n_cells, double tolmax, double *ddiag //Init variables
        ,double *dr0, double *dr0h, double *dn0, double *dp0
        ,double *dt, double *ds, double *dAx2, double *dy, double *dz// Auxiliary vectors
)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int active_threads = nrows;
  if(tid<active_threads){
    double alpha,rho0,omega0,beta,rho1,temp1,temp2;
    alpha=rho0=omega0=beta=rho1=temp1=temp2=1.0;
    dn0[tid]=0.;
    dp0[tid]=0.;
#ifndef CSR_SPMV_CPU
    cudaDeviceSpmvCSR(dr0,dx,dA,djA,diA); //y=A*x
#else
    cudaDeviceSpmvCSC_block(dr0,dx,dA,djA,diA,n_shr_empty)); //y=A*x
#endif
    //cudaDeviceaxpby(dr0,dtempv,1.0,-1.0,nrows);
    dr0[tid]=dtempv[tid]-dr0[tid];
    dr0h[tid]=dr0[tid];
    //cudaDeviceyequalsx(dr0h,dr0,nrows);
    int it=0;
    do{
      cudaDevicedotxy(dr0, dr0h, &rho1, n_shr_empty);
      beta = (rho1 / rho0) * (alpha / omega0);


cudaDevicezaxpbypc(dp0, dr0, dn0, beta, -1.0 * omega0 * beta, nrows);   //z = ax + by + c
//dp0[tid]=beta*dp0[tid]+dr0[tid]+ (-1.0)*omega0 * beta * dn0[tid];
            //cudaDevicezaxpbypc_d2(dp0, dr0, dn0, beta, -1.0 * omega0 * beta, nrows);   //z = ax + by + c



      cudaDevicemultxy(dy, ddiag, dp0, nrows);
      cudaDevicesetconst(dn0, 0.0, nrows);
#ifndef CSR_SPMV_CPU
      cudaDeviceSpmvCSR(dn0, dy, dA, djA, diA);
#else
      cudaDeviceSpmvCSC_block(dn0, dy, dA, djA, diA,n_shr_empty);
#endif
      cudaDevicedotxy(dr0h, dn0, &temp1, n_shr_empty);
      alpha = rho1 / temp1;
      cudaDevicezaxpby(1.0, dr0, -1.0 * alpha, dn0, ds, nrows);
      cudaDevicemultxy(dz, ddiag, ds, nrows); // precond z=diag*s
#ifndef CSR_SPMV_CPU
      cudaDeviceSpmvCSR(dt, dz, dA, djA, diA);
#else
      cudaDeviceSpmvCSC_block(dt, dz, dA, djA, diA,n_shr_empty);
#endif
      cudaDevicemultxy(dAx2, ddiag, dt, nrows);
      cudaDevicedotxy(dz, dAx2, &temp1, n_shr_empty);
      cudaDevicedotxy(dAx2, dAx2, &temp2, n_shr_empty);
      omega0 = temp1 / temp2;
      cudaDeviceaxpy(dx, dy, alpha, nrows); // x=alpha*y +x
      cudaDeviceaxpy(dx, dz, omega0, nrows);
      cudaDevicezaxpby(1.0, ds, -1.0 * omega0, dt, dr0, nrows);
      cudaDevicesetconst(dt, 0.0, nrows);
      cudaDevicedotxy(dr0, dr0, &temp1, n_shr_empty);
      temp1 = sqrtf(temp1);
      rho0 = rho1;
      it++;
    } while(it<maxIt && temp1>tolmax);//while(it<maxIt && temp1>tolmax);//while(0);
  }
}


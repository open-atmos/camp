/* Copyright (C) 2021 Barcelona Supercomputing Center and University of
 * Illinois at Urbana-Champaign
 * SPDX-License-Identifier: MIT
 */

#include "itsolver_gpu.h"
//#include "../debug_and_stats/camp_debug_2.h"


void read_options(itsolver *bicg){

  FILE *fp;
  char buff[255];

  char path[] = "itsolver_options.txt";

  fp = fopen("itsolver_options.txt", "r");
  if (fp == NULL){
    printf("Could not open file %s, setting itsolver to One-cell\n",path);
    bicg->cells_method=0;
  }else{

    fscanf(fp, "%s", buff);

    if(strstr(buff,"CELLS_METHOD=Block-cellsNhalf")!=NULL){
      bicg->cells_method=BLOCKCELLSNHALF;
    }
   else if(strstr(buff,"CELLS_METHOD=Block-cells1")!=NULL){
      bicg->cells_method=BLOCKCELLS1;
    }
    else if(strstr(buff,"CELLS_METHOD=Block-cellsN")!=NULL){
      bicg->cells_method=BLOCKCELLSN;
    }
    else if(strstr(buff,"CELLS_METHOD=Multi-cells")!=NULL){
      bicg->cells_method=MULTICELLS;
    }
    else if(strstr(buff,"CELLS_METHOD=One-cell")!=NULL){
      bicg->cells_method=ONECELL;
    }else{
      printf("ERROR: solveBCGBlocks unkonwn cells_method");
      exit(0);
    }
    fclose(fp);
  }

  //printf("itsolver read_options CELLS_METHOD %d\n",bicg->cells_method);

}

void createLinearSolver(SolverData *sd)
{
  itsolver *bicg = &(sd->bicg);
  ModelDataGPU *mGPU = sd->mGPU;

  //Init variables ("public")
  int nrows = mGPU->nrows;
  int blocks = mGPU->blocks;
  if(sd->use_gpu_cvode==0) read_options(bicg);
  mGPU->maxIt=1000;
  mGPU->tolmax=1.0e-30; //cv_mem->cv_reltol CAMP selected accuracy (1e-8) //1e-10;//1e-6
#ifndef CSR_SPMV_CPU
  mGPU->mattype=0;
  //printf("BCG Mattype=CSR\n");
#else
  mGPU->mattype=1; //CSC
  //printf("BCG Mattype=CSC\n");
#endif

  int len_cell=mGPU->nrows/mGPU->n_cells;
  if(len_cell>mGPU->threads){
    printf("ERROR: Size cell greater than available threads per block");
    exit(0);
  }

  //Auxiliary vectors ("private")
  double ** dr0 = &mGPU->dr0;
  double ** dr0h = &mGPU->dr0h;
  double ** dn0 = &mGPU->dn0;
  double ** dp0 = &mGPU->dp0;
  double ** dt = &mGPU->dt;
  double ** ds = &mGPU->ds;
  double ** dAx2 = &mGPU->dAx2;
  double ** dy = &mGPU->dy;
  double ** dz = &mGPU->dz;
  double ** daux = &mGPU->daux;
  double ** ddiag = &mGPU->ddiag;

  //Allocate
  cudaMalloc(dr0,nrows*sizeof(double));
  cudaMalloc(dr0h,nrows*sizeof(double));
  cudaMalloc(dn0,nrows*sizeof(double));
  cudaMalloc(dp0,nrows*sizeof(double));
  cudaMalloc(dt,nrows*sizeof(double));
  cudaMalloc(ds,nrows*sizeof(double));
  cudaMalloc(dAx2,nrows*sizeof(double));
  cudaMalloc(dy,nrows*sizeof(double));
  cudaMalloc(dz,nrows*sizeof(double));
  cudaMalloc(ddiag,nrows*sizeof(double));
  cudaMalloc(daux,nrows*sizeof(double));
  mGPU->aux=(double*)malloc(sizeof(double)*blocks);

}


int nextPowerOfTwo(int v){

  v--;
  v |= v >> 1;
  v |= v >> 2;
  v |= v >> 4;
  v |= v >> 8;
  v |= v >> 16;
  v++;

  //printf("nextPowerOfTwo %d", v);

  return v;
}

void exportConfBCG(SolverData *sd, const char *filepath){

  itsolver *bicg = &(sd->bicg);
  ModelDataGPU *mGPU = sd->mGPU;
  FILE *fp = fopen(filepath, "w+");

  fprintf(fp, "%d\n",  mGPU->n_cells);
  fprintf(fp, "%d\n",  mGPU->nrows);
  fprintf(fp, "%d\n",  mGPU->nnz);
  fprintf(fp, "%d\n",  mGPU->maxIt);
  fprintf(fp, "%d\n",  mGPU->mattype);
  fprintf(fp, "%le\n",  mGPU->tolmax);

  int *jA=(int*)malloc(mGPU->nnz*sizeof(int));
  int *iA=(int*)malloc((mGPU->nrows+1)*sizeof(int));
  double *A=(double*)malloc(mGPU->nnz*sizeof(double));
  double *diag=(double*)malloc(mGPU->nrows*sizeof(double));
  double *x=(double*)malloc(mGPU->nrows*sizeof(double));
  double *tempv=(double*)malloc(mGPU->nrows*sizeof(double));

  cudaMemcpy(jA, mGPU->djA,mGPU->nnz*sizeof(int),cudaMemcpyDeviceToHost);
  cudaMemcpy(iA, mGPU->diA,(mGPU->nrows+1)*sizeof(int),cudaMemcpyDeviceToHost);
  cudaMemcpy(A, mGPU->dA,mGPU->nnz*sizeof(double),cudaMemcpyDeviceToHost);
  cudaMemcpy(diag,mGPU->ddiag,mGPU->nrows*sizeof(double),cudaMemcpyDeviceToHost);
  cudaMemcpy(x,mGPU->dx,mGPU->nrows*sizeof(double),cudaMemcpyDeviceToHost);
  cudaMemcpy(tempv,mGPU->dtempv,mGPU->nrows*sizeof(double),cudaMemcpyDeviceToHost);

  for(int i=0; i<mGPU->nnz; i++){
    //printf("%d\n",mGPU->djA[i]);
    fprintf(fp, "%d ",  jA[i]);
  }
  fprintf(fp, "\n");
  for(int i=0; i<mGPU->nrows+1; i++)
    fprintf(fp, "%d ",  iA[i]);
  fprintf(fp, "\n");
  for(int i=0; i<mGPU->nnz; i++)
    fprintf(fp, "%le ",  A[i]);
  fprintf(fp, "\n");
  for(int i=0; i<mGPU->nrows; i++)
    fprintf(fp, "%le ",  diag[i]);
  fprintf(fp, "\n");
  for(int i=0; i<mGPU->nrows; i++)
    fprintf(fp, "%le ",  x[i]);
  fprintf(fp, "\n");
  for(int i=0; i<mGPU->nrows; i++)
    fprintf(fp, "%le ",  tempv[i]);

  fclose(fp);

#ifdef IS_IMPORTBCG

  fp = fopen("confBCG.txt", "r");
  if (fp == NULL) {
    printf("File not found \n");
    exit(EXIT_FAILURE);
  }

  fscanf(fp, "%d", &mGPU->n_cells);
  fscanf(fp, "%d", &mGPU->nrows);
  fscanf(fp, "%d", &mGPU->nnz);
  fscanf(fp, "%d", &mGPU->maxIt);
  fscanf(fp, "%d", &mGPU->mattype);
  fscanf(fp, "%le", &mGPU->tolmax);

  for(int i=0; i<mGPU->nnz; i++){
    fscanf(fp, "%d", &jA[i]);
    //printf("%d %d\n",i, jA[i]);
  }

  for(int i=0; i<mGPU->nrows+1; i++){
    fscanf(fp, "%d", &iA[i]);
    //printf("%d %d\n",i, iA[i]);
  }

  for(int i=0; i<mGPU->nnz; i++){
    fscanf(fp, "%le", &A[i]);
    //printf("%d %lf\n",i, A[i]);
  }

  for(int i=0; i<mGPU->nrows; i++){
    fscanf(fp, "%le", &diag[i]);
    //printf("%d %lf\n",i, diag[i]);
  }

  for(int i=0; i<mGPU->nrows; i++){
    fscanf(fp, "%le", &x[i]);
    //printf("%d %lf\n",i, x[i]);
  }

  for(int i=0; i<mGPU->nrows; i++){
    fscanf(fp, "%le", &tempv[i]);
    //printf("%d %lf\n",i, tempv[i]);
  }

  fclose(fp);

  cudaMemcpy(mGPU->djA,jA,mGPU->nnz*sizeof(int),cudaMemcpyHostToDevice);
  cudaMemcpy(mGPU->diA,iA,(mGPU->nrows+1)*sizeof(int),cudaMemcpyHostToDevice);

  cudaMemcpy(mGPU->dA,A,mGPU->nnz*sizeof(double),cudaMemcpyHostToDevice);

  cudaMemcpy(mGPU->ddiag,diag,mGPU->nrows*sizeof(double),cudaMemcpyHostToDevice);
  cudaMemcpy(mGPU->dx,x,mGPU->nrows*sizeof(double),cudaMemcpyHostToDevice);
  cudaMemcpy(mGPU->dtempv,tempv,mGPU->nrows*sizeof(double),cudaMemcpyHostToDevice);
  /**/
  printf("IMPORTBCG: Data read from %s\n",filepath);

#endif

  free(jA);
  free(iA);
  free(A);
  free(diag);
  free(x);
  free(tempv);

  printf("exportConfBCG: Data saved to %s\n",filepath);
  //exit(0);
}

void exportOutBCG(SolverData *sd, const char *filepath){

  itsolver *bicg = &(sd->bicg);
  ModelDataGPU *mGPU = sd->mGPU;
  FILE *fp = fopen(filepath, "w+");

  double *A=(double*)malloc(mGPU->nnz*sizeof(double));
  double *x=(double*)malloc(mGPU->nrows*sizeof(double));
  double *tempv=(double*)malloc(mGPU->nrows*sizeof(double));

  cudaMemcpy(A, mGPU->dA,mGPU->nnz*sizeof(double),cudaMemcpyDeviceToHost);
  cudaMemcpy(x,mGPU->dx,mGPU->nrows*sizeof(double),cudaMemcpyDeviceToHost);
  cudaMemcpy(tempv,mGPU->dtempv,mGPU->nrows*sizeof(double),cudaMemcpyDeviceToHost);

  for(int i=0; i<mGPU->nnz; i++)
    fprintf(fp, "%le ",  A[i]);
  fprintf(fp, "\n");
  for(int i=0; i<mGPU->nrows; i++)
    fprintf(fp, "%le ",  x[i]);
  fprintf(fp, "\n");
  for(int i=0; i<mGPU->nrows; i++)
    fprintf(fp, "%le ",  tempv[i]);

  free(A);
  free(x);
  free(tempv);

  printf("exportOutBCG: Data saved to %s\n",filepath);

  fclose(fp);
  exit(0);
}

//Based on
// https://github.com/scipy/scipy/blob/3b36a574dc657d1ca116f6e230be694f3de31afc/scipy/sparse/sparsetools/csr.h#L363
void swapCSC_CSR(int n_row, int n_col, int* Ap, int* Aj, double* Ax, int* Bp, int* Bi, double* Bx){

  int nnz=Ap[n_row];

  memset(Bp, 0, (n_row+1)*sizeof(int));

  for (int n = 0; n < nnz; n++){
    Bp[Aj[n]]++;
  }

  //cumsum the nnz per column to get Bp[]
  for(int col = 0, cumsum = 0; col < n_col; col++){
    int temp  = Bp[col];
    Bp[col] = cumsum;
    cumsum += temp;
  }
  Bp[n_col] = nnz;

  for(int row = 0; row < n_row; row++){
    for(int jj = Ap[row]; jj < Ap[row+1]; jj++){
      int col  = Aj[jj];
      int dest = Bp[col];

      Bi[dest] = row;
      Bx[dest] = Ax[jj];

      Bp[col]++;
    }
  }

  for(int col = 0, last = 0; col <= n_col; col++){
    int temp  = Bp[col];
    Bp[col] = last;
    last    = temp;
  }

}


void swapCSC_CSR_BCG(SolverData *sd){

  itsolver *bicg = &(sd->bicg);
  ModelDataGPU *mGPU = sd->mGPU;

#ifdef TEST_CSCtoCSR

  //Example configuration taken from KLU Sparse pdf
  int n_row=3;
  int n_col=n_row;
  int nnz=6;
  int Ap[n_row+1]={0,3,5,6};
  int Aj[nnz]={0,1,2,1,2,2};
  double Ax[nnz]={5.,4.,3.,2.,1.,8.};
  int* Bp=(int*)malloc((n_row+1)*sizeof(int));
  int* Bi=(int*)malloc(nnz*sizeof(int));
  double* Bx=(double*)malloc(nnz*sizeof(double));

#elif TEST_CSRtoCSC

  //Example configuration taken from KLU Sparse pdf
  int n_row=3;
  int n_col=n_row;
  int nnz=6;
  int Ap[n_row+1]={0,1,3,6};
  int Aj[nnz]={0,0,1,0,1,2};
  double Ax[nnz]={5.,4.,2.,3.,1.,8.};
  int* Bp=(int*)malloc((n_row+1)*sizeof(int));
  int* Bi=(int*)malloc(nnz*sizeof(int));
  double* Bx=(int*)malloc(nnz*sizeof(double));

#else

  int n_row=mGPU->nrows;
  int n_col=mGPU->nrows;
  int nnz=mGPU->nnz;
  int* Ap=mGPU->iA;
  int* Aj=mGPU->jA;
  double* Ax=mGPU->A;
  int* Bp=(int*)malloc((mGPU->nrows+1)*sizeof(int));
  int* Bi=(int*)malloc(mGPU->nnz*sizeof(int));
  double* Bx=(double*)malloc(nnz*sizeof(double));

#endif

    swapCSC_CSR(n_row,n_col,Ap,Aj,Ax,Bp,Bi,Bx);

#ifdef TEST_CSCtoCSR

  //Correct result:
  //int Cp[n_row+1]={0,1,3,6};
  //int Ci[nnz]={0,0,1,0,1,2};
  //int Cx[nnz]={5,4,2,3,1,8};

  printf("Bp:\n");
  for(int i=0;i<=n_row;i++)
    printf("%d ",Bp[i]);
  printf("\n");
  printf("Bi:\n");
  for(int i=0;i<nnz;i++)
    printf("%d ",Bi[i]);
  printf("\n");
  printf("Bx:\n");
  for(int i=0;i<nnz;i++)
    printf("%-le ",Bx[i]);
  printf("\n");

  exit(0);

#elif TEST_CSRtoCSC

  //Correct result:
  //int Cp[n_row+1]={0,3,5,6};
  //int Ci[nnz]={0,1,2,1,2,2};
  //int Cx[nnz]={5,4,3,2,1,8};

  printf("Bp:\n");
  for(int i=0;i<=n_row;i++)
    printf("%d ",Bp[i]);
  printf("\n");
  printf("Bi:\n");
  for(int i=0;i<nnz;i++)
    printf("%d ",Bi[i]);
  printf("\n");
  printf("Bx:\n");
  for(int i=0;i<nnz;i++)
    printf("%-le ",Bx[i]);
  printf("\n");
  exit(0);

#else

  cudaMemcpyAsync(mGPU->diA,Bp,(mGPU->nrows+1)*sizeof(int),cudaMemcpyHostToDevice, 0);
  cudaMemcpyAsync(mGPU->djA,Bi,mGPU->nnz*sizeof(int),cudaMemcpyHostToDevice, 0);
  cudaMemcpyAsync(mGPU->dA,Bx,mGPU->nnz*sizeof(double),cudaMemcpyHostToDevice, 0);

#endif

  free(Bp);
  free(Bi);
  free(Bx);

}

void print_int(int *x, int len, const char *s){

  for (int i=0; i<len; i++){
    printf("%s[%d]=%d\n",s,i,x[i]);
  }

}

void print_double(double *x, int len, const char *s){

  for (int i=0; i<len; i++){
    printf("%s[%d]=%le\n",s,i,x[i]);
  }

}

__device__
void dvcheck_input_gpud(double *x, int len, const char* s)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  //if(i<2)
  if(i<len)
  {
    printf("%s[%d]=%-le\n",s,i,x[i]);
  }
}

//Algorithm: Biconjugate gradient
__global__
void solveBcgCuda(
        double *dA, int *djA, int *diA, double *dx, double *dtempv //Input data
        ,int nrows, int blocks, int n_shr_empty, int maxIt, int mattype
        ,int n_cells, double tolmax, double *ddiag //Init variables
        ,double *dr0, double *dr0h, double *dn0, double *dp0
        ,double *dt, double *ds, double *dAx2, double *dy, double *dz// Auxiliary vectors
#ifdef CAMP_DEBUG_GPU
        ,int *it_pointer, int last_blockN
#endif
)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int tid = threadIdx.x;
  int active_threads = nrows;

  //if(tid==0)printf("blockDim.x %d\n",blockDim.x);

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

#ifndef CSR_SPMV_CPU
    cudaDeviceSpmvCSR(dr0,dx,nrows,dA,djA,diA); //y=A*x
#else
    cudaDeviceSpmvCSC_block(dr0,dx,nrows,dA,djA,diA,n_shr_empty)); //y=A*x
#endif

#ifdef DEBUG_SOLVEBCGCUDA_DEEP

    //printf("%d ddiag %-le\n",i,ddiag[i]);
    //printf("%d dr0 %-le\n",i, dr0[i]);

#endif

    //gpu_axpby(dr0,dtempv,1.0,-1.0,nrows,blocks,threads); // r0=1.0*rhs+-1.0r0 //y=ax+by
    cudaDeviceaxpby(dr0,dtempv,1.0,-1.0,nrows);

    __syncthreads();
    //gpu_yequalsx(dr0h,dr0,nrows,blocks,threads);  //r0h=r0
    cudaDeviceyequalsx(dr0h,dr0,nrows);

#ifdef CAMP_DEBUG_GPU
    int it=*it_pointer;
#endif

#ifdef DEBUG_SOLVEBCGCUDA_DEEP

    if(i==0){
      //printf("%d dr0[%d] %-le\n",it,i,dr0[i]);
      printf("%d %d rho1 %-le\n",it,i,rho1);
    }

    //dvcheck_input_gpud(dx,nrows,"dx");
    //dvcheck_input_gpud(dr0,nrows,"dr0");

#endif

    do
    {
      //rho1=gpu_dotxy(dr0, dr0h, aux, daux, nrows,(blocks + 1) / 2, threads);
      __syncthreads();

      cudaDevicedotxy(dr0, dr0h, &rho1, nrows, n_shr_empty);

#ifdef DEBUG_SOLVEBCGCUDA_DEEP

      if(i==0){
      //printf("%d dr0[%d] %-le\n",it,i,dr0[i]);
      printf("%d %d rho1 rho0 %-le %-le\n",it,i,rho1,rho0);
    }
    if(isnan(rho1) || rho1==0.0){
      dvcheck_input_gpud(dx,nrows,"dx");
      dvcheck_input_gpud(dr0h,nrows,"dr0h");
      dvcheck_input_gpud(dr0,nrows,"dr0");
    }

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
#ifndef CSR_SPMV_CPU
      cudaDeviceSpmvCSR(dn0, dy, nrows, dA, djA, diA);
#else
      cudaDeviceSpmvCSC_block(dn0, dy, nrows, dA, djA, diA,n_shr_empty);
#endif

#ifdef DEBUG_SOLVEBCGCUDA_DEEP

      if(it==0){
        printf("%d %d dy dn0 ddiag %-le %-le %le\n",it,i,dy[i],dn0[i],ddiag[i]);
        //printf("%d %d dn0 %-le\n",it,i,dn0[i]);
        //printf("%d %d &temp1 %p\n",it,i,&temp1);
        //printf("%d %d &test %p\n",it,i,&test);
        //printf("%d %d &i %p\n",it,i,&i);
      }

#endif

      //temp1=gpu_dotxy(dr0h, dn0, aux, daux, nrows,(blocks + 1) / 2, threads);
      cudaDevicedotxy(dr0h, dn0, &temp1, nrows, n_shr_empty);

#ifdef DEBUG_SOLVEBCGCUDA_DEEP

      if(i==0){
        printf("%d %d temp1 %-le\n",it,i,temp1);
        //printf("%d %d &temp1 %p\n",it,i,&temp1);
        //printf("%d %d &test %p\n",it,i,&test);
        //printf("%d %d &i %p\n",it,i,&i);
      }

#endif

      __syncthreads();
      alpha = rho1 / temp1;

      //gpu_zaxpby(1.0,dr0,-1.0*alpha,dn0,ds,nrows,blocks,threads); // a*x + b*y = z
      cudaDevicezaxpby(1.0, dr0, -1.0 * alpha, dn0, ds, nrows);

#ifdef DEBUG_SOLVEBCGCUDA_DEEP

      if(i==0){
        printf("%d ds[%d] %-le\n",it,i,ds[i]);
      }

#endif

      __syncthreads();
      //gpu_multxy(dz,ddiag,ds,nrows,blocks,threads); // precond z=diag*s
      cudaDevicemultxy(dz, ddiag, ds, nrows); // precond z=diag*s

      //gpu_spmv(dt,dz,nrows,dA,djA,diA,mattype,blocks,threads);
#ifndef CSR_SPMV_CPU
      cudaDeviceSpmvCSR(dt, dz, nrows, dA, djA, diA);
#else
      cudaDeviceSpmvCSC_block(dt, dz, nrows, dA, djA, diA,n_shr_empty);
#endif

      __syncthreads();
      //gpu_multxy(dAx2,ddiag,dt,nrows,blocks,threads);
      cudaDevicemultxy(dAx2, ddiag, dt, nrows);

      __syncthreads();
      //temp1=gpu_dotxy(dz, dAx2, aux, daux, nrows,(blocks + 1) / 2, threads);
      cudaDevicedotxy(dz, dAx2, &temp1, nrows, n_shr_empty);

#ifdef DEBUG_SOLVEBCGCUDA_DEEP

      if(i>=0){
        //printf("%d ddiag[%d] %-le\n",it,i,ddiag[i]);
        //printf("%d dt[%d] %-le\n",it,i,dt[i]);
        //printf("%d dAx2[%d] %-le\n",it,i,dAx2[i]);
        //printf("%d dz[%d] %-le\n",it,i,dz[i]);
      }

      if(i==0){
        printf("%d %d temp1 %-le\n",it,i,temp1);
      }

#endif

      __syncthreads();
      //temp2=gpu_dotxy(dAx2, dAx2, aux, daux, nrows,(blocks + 1) / 2, threads);
      cudaDevicedotxy(dAx2, dAx2, &temp2, nrows, n_shr_empty);

#ifdef DEBUG_SOLVEBCGCUDA_DEEP

      if(i==0){
        printf("%d %d temp2 %-le\n",it,i,temp2);
      }

#endif

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
      cudaDevicesetconst(dt, 0.0, nrows);

      __syncthreads();
      //temp1=gpu_dotxy(dr0, dr0, aux, daux, nrows,(blocks + 1) / 2, threads);
      cudaDevicedotxy(dr0, dr0, &temp1, nrows, n_shr_empty);

      //temp1 = sqrt(temp1);
      temp1 = sqrtf(temp1);

      rho0 = rho1;
      /**/
      __syncthreads();
      /**/

      //if (tid==0) it++;
      it++;
    } while(it<maxIt+*it_pointer && temp1>tolmax);//while(it<maxIt && temp1>tolmax);//while(0);

#ifdef DEBUG_SOLVEBCGCUDA_DEEP
    if(tid==0)
      printf("%d %d %-le %-le\n",tid,it,temp1,tolmax);
#endif

    //if(it>=maxIt-1)
    //  dvcheck_input_gpud(dr0,nrows,999);
    //dvcheck_input_gpud(dr0,nrows,k++);

#ifdef CAMP_DEBUG_GPU

    if(tid==0) {
      if(last_blockN==1){
        if(*it_pointer-it>*it_pointer)*it_pointer = it;
    }
    else{
      *it_pointer = it;
      }
    //printf("it_pointer %d\n",*it_pointer);
    }

#endif

  }

}

void solveGPU_block_thr(int blocks, int threads_block, int n_shr_memory, int n_shr_empty, int offset_cells,
        SolverData *sd, int last_blockN)
{
  itsolver *bicg = &(sd->bicg);
  ModelDataGPU *mGPU = sd->mGPU;

  //Init variables ("public")
  int nrows = mGPU->nrows;
  int nnz = mGPU->nnz;
  int n_cells = mGPU->n_cells;
  int maxIt = mGPU->maxIt;
  int mattype = mGPU->mattype;
  double tolmax = mGPU->tolmax;

  // Auxiliary vectors ("private")
  double *dr0 = mGPU->dr0;
  double *dr0h = mGPU->dr0h;
  double *dn0 = mGPU->dn0;
  double *dp0 = mGPU->dp0;
  double *dt = mGPU->dt;
  double *ds = mGPU->ds;
  double *dAx2 = mGPU->dAx2;
  double *dy = mGPU->dy;
  double *dz = mGPU->dz;
  double *daux = mGPU->daux;

  //Input variables
  int offset_nrows=(nrows/n_cells)*offset_cells;
  int offset_nnz=(nnz/n_cells)*offset_cells;
  //int offset_nnz=0;
  //int offset_nrows=0;

  //Works always supposing the same jac structure for all cells (same reactions on all cells)
  int *djA=mGPU->djA;
  int *diA=mGPU->diA;
  double *dA=mGPU->dA+offset_nnz;
  double *ddiag=mGPU->ddiag+offset_nrows;
  double *dx=mGPU->dx+offset_nrows;
  double *dtempv=mGPU->dtempv+offset_nrows;

  int len_cell=nrows/n_cells;

#ifdef IS_EXPORTBCG
  exportConfBCG(sd,"confBCG.txt");
#endif

#ifdef DEBUG_SOLVEBCGCUDA
  if(bicg->counterBiConjGrad==0) {
    printf("solveGPU_block_thr n_cells %d len_cell %d nrows %d nnz %d max_threads_block %d blocks %d threads_block %d n_shr_empty %d offset_cells %d\n",
           mGPU->n_cells,len_cell,mGPU->nrows,mGPU->nnz,n_shr_memory,blocks,threads_block,n_shr_empty,offset_cells);

    //print_double(mGPU->A,nnz,"A");
    //print_int(mGPU->jA,nnz,"jA");
    //print_int(mGPU->iA,nrows+1,"iA");

  }
#endif

  solveBcgCuda << < blocks, threads_block, n_shr_memory * sizeof(double) >> >
                                           //solveBcgCuda << < blocks, threads_block, threads_block * sizeof(double) >> >
                                           (dA, djA, diA, dx, dtempv, nrows, blocks, n_shr_empty, maxIt, mattype, n_cells,
                                                   tolmax, ddiag, dr0, dr0h, dn0, dp0, dt, ds, dAx2, dy, dz
#ifdef CAMP_DEBUG_GPU
                                                   ,&mGPU->mdvo->counterBCGInternal
                                                   , last_blockN
#endif
                                           );

#ifdef IS_EXPORTBCG
  exportOutBCG(sd,"outBCG.txt");
#endif


}

//solveBCGBlocks: Each block will compute only a cell/group of cells
//Algorithm: Biconjugate gradient
// dx: Input and output RHS
// dtempv: Input preconditioner RHS
void solveBCGBlocks(SolverData *sd, double *dA, int *djA, int *diA, double *dx, double *dtempv)
{

  itsolver *bicg = &(sd->bicg);
  ModelDataGPU *mGPU = sd->mGPU;

#ifdef DEBUG_SOLVEBCGCUDA
  if(bicg->counterBiConjGrad==0) {
    printf("solveGPUBlock\n");
  }
#endif

  int len_cell = mGPU->nrows/mGPU->n_cells;
  int max_threads_block=nextPowerOfTwo(len_cell);
  if(bicg->cells_method==BLOCKCELLSN) {
    max_threads_block = mGPU->threads;//1024;
  }else if(bicg->cells_method==BLOCKCELLSNHALF){
    max_threads_block = mGPU->threads/2;
  }

  int n_cells_block =  max_threads_block/len_cell;
  int threads_block = n_cells_block*len_cell;
  int n_shr_empty = max_threads_block-threads_block;
  int blocks = (mGPU->nrows+threads_block-1)/threads_block;

  int offset_cells=0;
  int last_blockN=0;

  //Common kernel (Launch all blocks except the last)
  if(bicg->cells_method==BLOCKCELLSN ||
  bicg->cells_method==BLOCKCELLSNHALF
  ) {

    blocks=blocks-1;

    if(blocks!=0){
      solveGPU_block_thr(blocks, threads_block, max_threads_block, n_shr_empty, offset_cells,
                       sd, last_blockN);
      last_blockN = 1;
    }
#ifdef DEBUG_SOLVEBCGCUDA
    else{
      if(bicg->counterBiConjGrad==0){
        printf("solveBCGBlocks blocks==0\n");
      }
    }
#endif

    //Update vars to launch last kernel
    offset_cells=n_cells_block*blocks;
    int n_cells_last_block=mGPU->n_cells-offset_cells;
    threads_block=n_cells_last_block*len_cell;
    max_threads_block=nextPowerOfTwo(threads_block);
    n_shr_empty = max_threads_block-threads_block;
    blocks=1;

  }

  solveGPU_block_thr(blocks, threads_block, max_threads_block, n_shr_empty, offset_cells,
           sd, last_blockN);

}

//Algorithm: Biconjugate gradient
void solveBCG(SolverData *sd, double *dA, int *djA, int *diA, double *dx, double *dtempv)
{
  //Init variables ("public")

  itsolver *bicg = &(sd->bicg);
  ModelDataGPU *mGPU = sd->mGPU;

  int nrows = mGPU->nrows;
  int blocks = mGPU->blocks;
  int threads = mGPU->threads;
  int maxIt = mGPU->maxIt;
  int mattype = mGPU->mattype;
  double tolmax = mGPU->tolmax;
  double *ddiag = mGPU->ddiag;

  // Auxiliary vectors ("private")
  double *dr0 = mGPU->dr0;
  double *dr0h = mGPU->dr0h;
  double *dn0 = mGPU->dn0;
  double *dp0 = mGPU->dp0;
  double *dt = mGPU->dt;
  double *ds = mGPU->ds;
  double *dAx2 = mGPU->dAx2;
  double *dy = mGPU->dy;
  double *dz = mGPU->dz;
  double *aux = mGPU->aux;
  double *daux = mGPU->daux;

#ifdef DEBUG_SOLVEBCGCUDA
  if(bicg->counterBiConjGrad==0) {
    printf("solveBCG\n");
  }
#endif

  //Function private variables
  double alpha,rho0,omega0,beta,rho1,temp1,temp2;

  gpu_spmv(dr0,dx,nrows,dA,djA,diA,mattype,blocks,threads);  // r0= A*x

  gpu_axpby(dr0,dtempv,1.0,-1.0,nrows,blocks,threads); // r0=1.0*rhs+-1.0r0 //y=ax+by

  gpu_yequalsx(dr0h,dr0,nrows,blocks,threads);  //r0h=r0

  gpu_yequalsconst(dn0,0.0,nrows,blocks,threads);  //n0=0.0
  gpu_yequalsconst(dp0,0.0,nrows,blocks,threads);  //p0=0.0

  alpha  = 1.0;
  rho0   = 1.0;
  omega0 = 1.0;

#ifdef DEBUG_SOLVEBCGCUDA_DEEP

  double *aux_x1;
  aux_x1=(double*)malloc(mGPU->nrows*sizeof(double));

#endif

  //for(int it=0;it<maxIt;it++){
  int it=0;
  do {

    rho1=gpu_dotxy(dr0, dr0h, aux, daux, nrows,(blocks + 1) / 2, threads);//rho1 =<r0,r0h>
    //rho1=gpu_dotxy(dr0, dr0h, aux, daux, nrows,blocks, threads);//rho1 =<r0,r0h>

#ifdef DEBUG_SOLVEBCGCUDA_DEEP
    //good here first iter
    printf("%d rho1 %-le\n",it,rho1);
#endif

    beta=(rho1/rho0)*(alpha/omega0);

    gpu_zaxpbypc(dp0,dr0,dn0,beta,-1.0*omega0*beta,nrows,blocks,threads);   //z = ax + by + c

    gpu_multxy(dy,ddiag,dp0,nrows,blocks,threads);  // precond y= p0*diag

    gpu_spmv(dn0,dy,nrows,dA,djA,diA,mattype,blocks,threads);  // n0= A*y

    temp1=gpu_dotxy(dr0h, dn0, aux, daux, nrows,(blocks + 1) / 2, threads);
    //temp1=gpu_dotxy(dr0h, dn0, aux, daux, nrows, blocks, threads);

#ifdef DEBUG_SOLVEBCGCUDA_DEEP
    printf("%d temp1 %-le\n",it,temp1);
#endif

    alpha=rho1/temp1;

    gpu_zaxpby(1.0,dr0,-1.0*alpha,dn0,ds,nrows,blocks,threads);

#ifdef DEBUG_SOLVEBCGCUDA_DEEP
    cudaMemcpy(aux_x1,ds,mGPU->nrows*sizeof(double),cudaMemcpyDeviceToHost);

    printf("%d ds[0] %-le\n",it,aux_x1[0]);

#endif

    gpu_multxy(dz,ddiag,ds,nrows,blocks,threads); // precond z=diag*s

    gpu_spmv(dt,dz,nrows,dA,djA,diA,mattype,blocks,threads);

    gpu_multxy(dAx2,ddiag,dt,nrows,blocks,threads);

    temp1=gpu_dotxy(dz, dAx2, aux, daux, nrows,(blocks + 1) / 2, threads);

#ifdef DEBUG_SOLVEBCGCUDA_DEEP
    cudaMemcpy(aux_x1,dAx2,mGPU->nrows*sizeof(double),cudaMemcpyDeviceToHost);

    for(int i=0; i<mGPU->nrows; i++){
      //printf("%d ddiag[%i] %-le\n",it,i,aux_x1[i]);
      //printf("%d dt[%i] %-le\n",it,i,aux_x1[i]);
      //printf("%d dAx2[%i] %-le\n",it,i,aux_x1[i]);
      //printf("%d dz[%i] %-le\n",it,i,aux_x1[i]);
    }

    printf("%d temp1 %-le\n",it,temp1);
#endif

    temp2=gpu_dotxy(dAx2, dAx2, aux, daux, nrows,(blocks + 1) / 2, threads);

#ifdef DEBUG_SOLVEBCGCUDA_DEEP
    printf("%d temp2 %-le\n",it,temp2);
#endif

    omega0= temp1/temp2;

    gpu_axpy(dx,dy,alpha,nrows,blocks,threads); // x=alpha*y +x

    gpu_axpy(dx,dz,omega0,nrows,blocks,threads);

    gpu_zaxpby(1.0,ds,-1.0*omega0,dt,dr0,nrows,blocks,threads);

    temp1=gpu_dotxy(dr0, dr0, aux, daux, nrows,(blocks + 1) / 2, threads);
    temp1=sqrt(temp1);

    rho0=rho1;
    it++;
  }while(it<maxIt && temp1>tolmax);

#ifdef DEBUG_SOLVEBCGCUDA_DEEP
  free(aux_x1);
#endif

}

void free_itsolver(SolverData *sd)
{
  itsolver *bicg = &(sd->bicg);
  ModelDataGPU *mGPU = sd->mGPU;

  //Auxiliary vectors ("private")
  double ** dr0 = &mGPU->dr0;
  double ** dr0h = &mGPU->dr0h;
  double ** dn0 = &mGPU->dn0;
  double ** dp0 = &mGPU->dp0;
  double ** dt = &mGPU->dt;
  double ** ds = &mGPU->ds;
  double ** dAx2 = &mGPU->dAx2;
  double ** dy = &mGPU->dy;
  double ** dz = &mGPU->dz;
  double ** daux = &mGPU->daux;
  double ** ddiag = &mGPU->ddiag;

  cudaFree(dr0);
  cudaFree(dr0h);
  cudaFree(dn0);
  cudaFree(dp0);
  cudaFree(dt);
  cudaFree(ds);
  cudaFree(dAx2);
  cudaFree(dy);
  cudaFree(dz);
  cudaFree(ddiag);
  cudaFree(daux);
  free(mGPU->aux);

}


#include "itsolver_gpu.h"
//#include "../debug_and_stats/camp_debug_2.h"

void read_options(itsolver *bicg){

  FILE *fp;
  char buff[255];

  //print_current_directory();

  char path[] = "itsolver_options.txt";

  fp = fopen("itsolver_options.txt", "r");
  if (fp == NULL){
    printf("Could not open file %s",path);
  }
  fscanf(fp, "%s", buff);

  if(strstr(buff,"USE_MULTICELLS=ON")!=NULL){
    //printf("itsolver read_options USE_MULTICELLS=ON\n");
    bicg->use_multicells=1; //One-cell per block (Independent cells)
  }
  else{
    //printf("itsolver read_options USE_MULTICELLS=OFF\n");
    bicg->use_multicells=0;
  }

}

void createSolver(itsolver *bicg)
{
  //Init variables ("public")
  int nrows = bicg->nrows;
  int blocks = bicg->blocks;
  read_options(bicg);
  bicg->maxIt=1000;
  bicg->tolmax=1.0e-30; //cv_mem->cv_reltol CAMP selected accuracy (1e-8) //1e-10;//1e-6
#ifndef CSR_SPMV
  bicg->mattype=0;
  printf("BCG Mattype=CSR\n");
#else
  bicg->mattype=1; //CSC
  printf("BCG Mattype=CSC\n");
#endif

  //todo previous check to exception if len_cell>bicg_threads
  int len_cell=bicg->nrows/bicg->n_cells;
  if(len_cell>bicg->threads){
    printf("ERROR: Size cell greater than available threads per block");
    exit(0);
  }

  //Auxiliary vectors ("private")
  double ** dr0 = &bicg->dr0;
  double ** dr0h = &bicg->dr0h;
  double ** dn0 = &bicg->dn0;
  double ** dp0 = &bicg->dp0;
  double ** dt = &bicg->dt;
  double ** ds = &bicg->ds;
  double ** dAx2 = &bicg->dAx2;
  double ** dy = &bicg->dy;
  double ** dz = &bicg->dz;
  double ** daux = &bicg->daux;
  double ** ddiag = &bicg->ddiag;

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
  bicg->aux=(double*)malloc(sizeof(double)*blocks);

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

__device__
void cudaDeviceswapCSC_CSR1Thread(int n_row, int n_col, int* Ap, int* Aj, double* Ax, int* Bp, int* Bi, double* Bx) {

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int tid = threadIdx.x;

  //if(i==0) printf("start cudaDeviceswapCSC_CSR1\n");

  int nnz=Ap[n_row];

  if(i==0){ //good
    //if(tid==0){//wrong
    //if(i<n_row){//wrong

    memset(Bp, 0, (n_row+1)*sizeof(int));

    //for (int n = 0; n < n_row+1; n++){
    //  Bp[n]=0;}

    for (int n = 0; n < nnz; n++){
      Bp[Aj[n]]++;
    }

    if(i==0) printf("start cudaDeviceswapCSC_CSR1Thread2\n");
    if(i==0) {
      printf("Bp:\n");
      for (int n = 0; n <= n_row; n++)
        printf("%d ", Bp[n]);
      printf("\n");
    }

    for(int col = 0, cumsum = 0; col < n_col; col++){
      int temp  = Bp[col];
      Bp[col] = cumsum;
      cumsum += temp;
    }
    Bp[n_col] = nnz;

    if(i==0) printf("start cudaDeviceswapCSC_CSR1Thread3\n");
    if(i==0) {
      printf("Bp:\n");
      for (int n = 0; n <= n_row; n++)
        printf("%d ", Bp[n]);
      printf("\n");
    }

    //int row=i;
    for(int row = 0; row < n_row; row++){
      for(int jj = Ap[row]; jj < Ap[row+1]; jj++){
        int col  = Aj[jj];
        int dest = Bp[col];

        Bi[dest] = row;
        Bx[dest] = Ax[jj];

        Bp[col]++;
      }
    }

    if(i==0) printf("start cudaDeviceswapCSC_CSR1Thread4\n");
    if(i==0) {
      printf("Bp:\n");
      for (int n = 0; n <= n_row; n++)
        printf("%d ", Bp[n]);
      printf("\n");
    }

    for(int col = 0, last = 0; col <= n_col; col++){
      int temp  = Bp[col];
      Bp[col] = last;
      last    = temp;
    }

    if(i==0) printf("start cudaDeviceswapCSC_CSR1Thread5\n");
    if(i==0) {
      printf("Bp:\n");
      for (int n = 0; n <= n_row; n++)
        printf("%d ", Bp[n]);
      printf("\n");
    }

    //copy to A
    for (int n = 0; n < n_row+1; n++){
      Ap[n]=Bp[n];
    }

    for (int n = 0; n < nnz; n++){
      Aj[n]=Bi[n];
      Ax[n]=Bx[n];
    }
  }
}

//Based on
// https://github.com/scipy/scipy/blob/3b36a574dc657d1ca116f6e230be694f3de31afc/scipy/sparse/sparsetools/csr.h#L363
__device__
void cudaDeviceswapCSC_CSR1ThreadBlock(int n_row, int n_col, int* Ap, int* Aj, double* Ax, int* BpGlobal, int* Bi, double* Bx) {

  extern __shared__ int Bp[];
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int tid = threadIdx.x;
  int nnz=Ap[n_row];
#ifdef DEBUG_cudaGlobalswapCSC_CSR
  int iprint=0;
  if(gridDim.x>1)iprint=blockDim.x;//block 2
#endif

#ifdef DEBUG_cudaGlobalswapCSC_CSR
  if(i==0) printf("start cudaDeviceswapCSC_CSR1ThreadBlock nnz %d n_row %d blockdim %d "
                  "gridDim.x %d \n",nnz,n_row,blockDim.x,gridDim.x);
#endif

  //if(tid==0){
  if(i<n_row){

    if(tid==0) {
#ifdef DEBUG_cudaGlobalswapCSC_CSR
      printf("blockDim.x*blockIdx.x %d %d\n",blockDim.x*blockIdx.x,blockDim.x*(blockIdx.x+1));
#endif
    }
    Bp[tid]=0; //todo dont needed? only first value is init to zero, not the whole array
    //Bp[2*tid]=0;
    if(blockIdx.x==gridDim.x-1) Bp[blockDim.x]=0;

#ifdef DEBUG_cudaGlobalswapCSC_CSR
    if(i==iprint) printf("start cudaDeviceswapCSC_CSR1ThreadBlock1\n");
    /*if(i==iprint) {
    printf("Bp %d:\n",blockIdx);
      for (int n = blockDim.x*blockIdx.x; n <= blockDim.x*(blockIdx.x+1); n++)
        printf("%d[%d] ",Bp[n],n);
      printf("\n");
    }__syncthreads();*/
#endif

    if(tid==0){
      for (int n = (nnz/gridDim.x)*blockIdx.x; n < (nnz/gridDim.x)*(blockIdx.x+1); n++){
        Bp[Aj[n]-blockIdx.x*blockDim.x]++;
      }
    }

#ifdef DEBUG_cudaGlobalswapCSC_CSR
    if(i==iprint) printf("start cudaDeviceswapCSC_CSR1ThreadBlock2\n");
    if(i==iprint) {
      printf("Bp %d:\n",blockIdx);
      //for (int n = blockDim.x*blockIdx.x; n <= blockDim.x*(blockIdx.x+1); n++)
      for (int n = 0; n <= blockDim.x; n++)
        printf("%d ", Bp[n]);
      printf("\n");
    }
#endif

    //TODO efficient cumsum http://www.eecs.umich.edu/courses/eecs570/hw/parprefix.pdf
    /*int offset = 1;
    for (int d = n_col>>1; d > 0; d >>= 1) // build sum in place up the tree
    {
      __syncthreads();
      if (tid < d)
      {
        int ai = offset*(2*tid+1)-1;
        int bi = offset*(2*tid+2)-1;
        Bp[bi] += Bp[ai];
      }
      offset *= 2;
    }
    if (tid == 0) { Bp[n_col - 1] = 0; } // clear the last element
    for (int d = 1; d < n_col; d *= 2) // traverse down tree & build scan
    {
      offset >>= 1;
      __syncthreads();
      if (tid < d)
      {
        int ai = offset*(2*tid+1)-1;
        int bi = offset*(2*tid+2)-1;
        float t = Bp[ai];
        Bp[ai] = Bp[bi];
        Bp[bi] += t;
      }
    }
    __syncthreads();*/

    if(tid==0){
      int cumsum=Ap[blockDim.x*blockIdx.x];
      for(int n = 0; n < blockDim.x; n++){
        //printf("%d ", Bp[n]);//el compilador me optimiza esto o algo y me trolea cuando printea da diferente wtf xd
        int temp  = Bp[n];
        //printf("%d ", temp);
        Bp[n] = cumsum;
        cumsum += temp;
      }
      if(blockIdx.x==gridDim.x-1) Bp[blockDim.x]=nnz;
    }


#ifdef DEBUG_cudaGlobalswapCSC_CSR
    if(i==iprint) printf("start cudaDeviceswapCSC_CSR1ThreadBlock3\n");
    if(i==iprint) {
      //printf("Bp %d:\n",blockIdx);
      //for (int n = blockDim.x*blockIdx.x; n <= blockDim.x*(blockIdx.x+1); n++)
      for (int n = 0; n <= blockDim.x; n++)
        printf("%d ", Bp[n]);
      printf("\n");
    }
#endif

    if(tid==0) {
      for(int row=n_row/gridDim.x*blockIdx.x;row<n_row/gridDim.x*(blockIdx.x+1);row++){
        for (int jj = Ap[row]; jj < Ap[row + 1]; jj++) {
          int col = Aj[jj];
          int dest = Bp[col-blockIdx.x*blockDim.x];

          Bi[dest] = row;
          Bx[dest] = Ax[jj];

          Bp[col-blockIdx.x*blockDim.x]++;
        }
      }
    }

#ifdef DEBUG_cudaGlobalswapCSC_CSR
    if(i==iprint) printf("start cudaDeviceswapCSC_CSR1ThreadBlock4\n");
    if(i==iprint) {
      //printf("Bp %d:\n",blockIdx);
      //for (int n = blockDim.x*blockIdx.x; n <= blockDim.x*(blockIdx.x+1); n++)
      for (int n = 0; n <= blockDim.x; n++)
        printf("%d ", Bp[n]);
      printf("\n");
    }
#endif

    if(tid==0) {
      int last=Ap[n_row/gridDim.x*blockIdx.x];
      int limit=blockDim.x;
      if(blockIdx.x==gridDim.x-1) limit++;
      for (int col = 0; col < blockDim.x; col++) {
        int temp = Bp[col];
        Bp[col] = last;
        last = temp;
      }
    }

#ifdef DEBUG_cudaGlobalswapCSC_CSR
    if(i==iprint) printf("start cudaDeviceswapCSC_CSR1ThreadBlock5\n");
    if(i==iprint) {
      //printf("Bp %d:\n",blockIdx);
      //for (int n = blockDim.x*blockIdx.x; n <= blockDim.x*(blockIdx.x+1); n++)
      for (int n = 0; n <= blockDim.x; n++)
        printf("%d ", Bp[n]);
      printf("\n");
    }
#endif

    //Copy to A
    if(tid==0){
      for (int n = (nnz/gridDim.x)*blockIdx.x; n < (nnz/gridDim.x)*(blockIdx.x+1); n++) {
        Aj[n]=Bi[n];
        Ax[n]=Bx[n];
      }
    }

    if(tid==0){
      for(int n=0;n<n_row/gridDim.x;n++){
        Ap[n+blockIdx.x*blockDim.x]=Bp[n];
        BpGlobal[n+blockIdx.x*blockDim.x]=Bp[n];
      }
      if(blockIdx.x==gridDim.x-1){
        Ap[n_row]=nnz;
        BpGlobal[n_row]=nnz;}
      }
    }

}

__global__
void cudaGlobalswapCSC_CSR(int n_row, int n_col, int* Ap, int* Aj, double* Ax, int* Bp, int* Bi, double* Bx){

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int tid = threadIdx.x;

  //if(i==0) printf("start cudaDeviceswapCSC_CSR\n");

#ifdef TEST_DEVICECSCtoCSR

  //Example configuration taken from KLU Sparse pdf

  const int n_row2=3;
  const int nnz=6;
  int Cp[n_row2+1]={0,3,5,6};
  int Cj[nnz]={0,1,2,1,2,2};
  double Cx[nnz]={5.,4.,3.,2.,1.,8.};

  int* Dp=(int*)malloc((n_row2+1)*sizeof(int));
  int* Di=(int*)malloc(nnz*sizeof(int));
  double* Dx=(double*)malloc(nnz*sizeof(double));

  //cudaDeviceswapCSC_CSR1Thread(n_row2,n_row2,Cp,Cj,Cx,Dp,Di,Dx);
  cudaDeviceswapCSC_CSR1ThreadBlock(n_row2,n_row2,Cp,Cj,Cx,Dp,Di,Dx);

  //Correct result:
  //int Cp[n_row+1]={0,1,3,6};
  //int Ci[nnz]={0,0,1,0,1,2};
  //int Cx[nnz]={5,4,2,3,1,8};

  if(i==0) {
    printf("Bp:\n");
    for (int i = 0; i <= n_row2; i++)
      printf("%d ", Dp[i]);
    printf("\n");
    printf("Bi:\n");
    for (int i = 0; i < nnz; i++)
      printf("%d ", Di[i]);
    printf("\n");
    printf("Bx:\n");
    for (int i = 0; i < nnz; i++)
      printf("%-le ", Dx[i]);
    printf("\n");
  }

  //exit(0);

#endif

#ifdef DEBUG_cudaGlobalswapCSC_CSR
  int nnz=Ap[n_row];
  int iprint=0;
  if(gridDim.x>1)iprint=blockDim.x;//block 2
  if(i==iprint) printf("end cudaGlobalswapCSC_CSR\n");
  if(i==iprint) {
    printf("Ap:\n");
    for (int n = 0; n <= n_row; n++)
      printf("%d ", Ap[n]);
    printf("\n");
    printf("Aj:\n");
    for (int i = 0; i < nnz; i++)
      printf("%d ", Aj[i]);
    printf("\n");
    //printf("Ax:\n");
    //for (int i = 0; i < nnz; i++)
    //  printf("%-le ", Ax[i]);
    //printf("\n");
  }
#endif

  //cudaDeviceswapCSC_CSR1Thread(n_row,n_col,Ap,Aj,Ax,Bp,Bi,Bx);
  cudaDeviceswapCSC_CSR1ThreadBlock(n_row,n_col,Ap,Aj,Ax,Bp,Bi,Bx);

#ifdef DEBUG_cudaGlobalswapCSC_CSR
  if(gridDim.x>1)iprint=blockDim.x;//block 2
  if(i==iprint) printf("end cudaGlobalswapCSC_CSR\n");
  if(i==iprint) {
    printf("Ap:\n");
    for (int n = 0; n <= n_row; n++)
      printf("%d ", Ap[n]);
    printf("\n");
    printf("Aj:\n");
    for (int i = 0; i < nnz; i++)
      printf("%d ", Aj[i]);
    printf("\n");
    /*printf("Ax:\n");
    for (int i = 0; i < nnz; i++)
      printf("%-le ", Ax[i]);
    printf("\n");
     */
  }
#endif

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


void swapCSC_CSR_BCG(itsolver *bicg){

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

  int n_row=bicg->nrows;
  int n_col=bicg->nrows;
  int nnz=bicg->nnz;
  int* Ap=bicg->iA;
  int* Aj=bicg->jA;
  double* Ax=bicg->A;
  int* Bp=(int*)malloc((bicg->nrows+1)*sizeof(int));
  int* Bi=(int*)malloc(bicg->nnz*sizeof(int));
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

  cudaMemcpy(bicg->diA,Bp,(bicg->nrows+1)*sizeof(int),cudaMemcpyHostToDevice);
  cudaMemcpy(bicg->djA,Bi,bicg->nnz*sizeof(int),cudaMemcpyHostToDevice);
  cudaMemcpy(bicg->dA,Bx,bicg->nnz*sizeof(double),cudaMemcpyHostToDevice);

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
__device__
void solveBcgCudaDevice(
        double *dA, int *djA, int *diA, double *dx, double *dtempv //Input data
        ,int nrows, int blocks, int n_shr_empty, int maxIt, int mattype
        ,int n_cells, double tolmax, double *ddiag //Init variables
        ,double *dr0, double *dr0h, double *dn0, double *dp0
        ,double *dt, double *ds, double *dAx2, double *dy, double *dz// Auxiliary vectors
#ifdef PMC_DEBUG_GPU
        ,int *it_pointer
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

#ifdef BASIC_SPMV
    cudaDevicesetconst(dr0, 0.0, nrows);
    __syncthreads();
    cudaDeviceSpmvCSC(dr0,dx,nrows,dA,djA,diA); //y=A*x
#else
    cudaDeviceSpmv(dr0,dx,nrows,dA,djA,diA,n_shr_empty); //y=A*x
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

#ifdef PMC_DEBUG_GPU
    //int it=*it_pointer;
    int it=0;
#else
    int it=0;
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
#ifdef BASIC_SPMV
      cudaDevicesetconst(dn0, 0.0, nrows);
      __syncthreads();
      cudaDeviceSpmvCSC(dn0, dy, nrows, dA, djA, diA);
#else
      cudaDeviceSpmv(dn0, dy, nrows, dA, djA, diA,n_shr_empty);
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
#ifdef BASIC_SPMV
      cudaDevicesetconst(dt, 0.0, nrows);
      __syncthreads();
      cudaDeviceSpmvCSC(dt, dz, nrows, dA, djA, diA);
#else
      cudaDeviceSpmv(dt, dz, nrows, dA, djA, diA,n_shr_empty);
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
    } while(it<maxIt && temp1>tolmax);//while(it<maxIt && temp1>tolmax);//while(0);

#ifdef DEBUG_SOLVEBCGCUDA_DEEP
    if(tid==0)
      printf("%d %d %-le %-le\n",tid,it,temp1,tolmax);
#endif

    //if(it>=maxIt-1)
    //  dvcheck_input_gpud(dr0,nrows,999);

    //dvcheck_input_gpud(dr0,nrows,k++);

#ifdef PMC_DEBUG_GPU

#ifdef solveBcgCuda_sum_it

  //printf("it %d %d\n",i,it);
  if(tid==0)
    it_pointer[blockIdx.x]=it;

#else

  *it_pointer = it;

#endif

#endif

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
#ifdef PMC_DEBUG_GPU
        ,int *it_pointer
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

#ifdef BASIC_SPMV
    cudaDevicesetconst(dr0, 0.0, nrows);
    __syncthreads();
    cudaDeviceSpmvCSC(dr0,dx,nrows,dA,djA,diA); //y=A*x
#else
    cudaDeviceSpmv(dr0,dx,nrows,dA,djA,diA,n_shr_empty); //y=A*x
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

#ifdef PMC_DEBUG_GPU
    //int it=*it_pointer;
    int it=0;
#else
    int it=0;
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
#ifdef BASIC_SPMV
      cudaDevicesetconst(dn0, 0.0, nrows);
      __syncthreads();
      cudaDeviceSpmvCSC(dn0, dy, nrows, dA, djA, diA);
#else
      cudaDeviceSpmv(dn0, dy, nrows, dA, djA, diA,n_shr_empty);
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
#ifdef BASIC_SPMV
      cudaDevicesetconst(dt, 0.0, nrows);
      __syncthreads();
      cudaDeviceSpmvCSC(dt, dz, nrows, dA, djA, diA);
#else
      cudaDeviceSpmv(dt, dz, nrows, dA, djA, diA,n_shr_empty);
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
    } while(it<maxIt && temp1>tolmax);//while(it<maxIt && temp1>tolmax);//while(0);

#ifdef DEBUG_SOLVEBCGCUDA_DEEP
    if(tid==0)
      printf("%d %d %-le %-le\n",tid,it,temp1,tolmax);
#endif

    //if(it>=maxIt-1)
    //  dvcheck_input_gpud(dr0,nrows,999);

    //dvcheck_input_gpud(dr0,nrows,k++);

#ifdef PMC_DEBUG_GPU

    #ifdef solveBcgCuda_sum_it

  //printf("it %d %d\n",i,it);
  if(tid==0)
    it_pointer[blockIdx.x]=it;

#else

  *it_pointer = it;

#endif

#endif

  }

}

void solveGPU_block_thr(int blocks, int threads_block, int n_shr_memory, int n_shr_empty, int offset_cells,
        itsolver *bicg)
{

  //Init variables ("public")
  int nrows = bicg->nrows;
  int nnz = bicg->nnz;
  int n_cells = bicg->n_cells;
  int maxIt = bicg->maxIt;
  int mattype = bicg->mattype;
  double tolmax = bicg->tolmax;

  // Auxiliary vectors ("private")
  double *dr0 = bicg->dr0;
  double *dr0h = bicg->dr0h;
  double *dn0 = bicg->dn0;
  double *dp0 = bicg->dp0;
  double *dt = bicg->dt;
  double *ds = bicg->ds;
  double *dAx2 = bicg->dAx2;
  double *dy = bicg->dy;
  double *dz = bicg->dz;
  double *daux = bicg->daux;

  //Input variables
  int offset_nrows=(nrows/n_cells)*offset_cells;
  int offset_nnz=(nnz/n_cells)*offset_cells;
  //int offset_nnz=0;
  //int offset_nrows=0;


  //Works always supposing the same jac structure for all cells (same reactions on all cells)
  int *djA=bicg->djA;
  int *diA=bicg->diA;

  double *dA=bicg->dA+offset_nnz;
  double *ddiag=bicg->ddiag+offset_nrows;
  double *dx=bicg->dx+offset_nrows;
  double *dtempv=bicg->dtempv+offset_nrows;

  int len_cell=nrows/n_cells;

#ifndef DEBUG_SOLVEBCGCUDA
  if(bicg->counterBiConjGrad==0) {
    printf("n_cells %d len_cell %d nrows %d nnz %d max_threads_block %d blocks %d threads_block %d n_shr_empty %d offset_cells %d\n",
           bicg->n_cells,len_cell,bicg->nrows,bicg->nnz,n_shr_memory,blocks,threads_block,n_shr_empty,offset_cells);

    //print_double(bicg->A,nnz,"A");
    //print_int(bicg->jA,nnz,"jA");
    //print_int(bicg->iA,nrows+1,"iA");

  }
#endif

#ifdef PMC_DEBUG_GPU
  int *dit_ptr;

#ifdef solveBcgCuda_sum_it

  //cudaMalloc((void**)&dit_ptr,nrows*sizeof(int));
  //cudaMemset(dit_ptr, 0, bicg->nrows*sizeof(int));

  cudaMalloc((void**)&dit_ptr,blocks*sizeof(int));
  cudaMemset(dit_ptr, 0, blocks*sizeof(int));

#else

  //int *dit_ptr=bicg->counterBiConjGradInternalGPU;

  cudaMalloc((void**)&dit_ptr,sizeof(int));
  cudaMemset(dit_ptr, 0, sizeof(int));

#endif

#endif

  //max_threads_block = nextPowerOfTwo(nrows);
  //n_shr_empty = max_threads_block-threads_block;

  solveBcgCuda << < blocks, threads_block, n_shr_memory * sizeof(double) >> >
                                           //solveBcgCuda << < blocks, threads_block, threads_block * sizeof(double) >> >
                                           (dA, djA, diA, dx, dtempv, nrows, blocks, n_shr_empty, maxIt, mattype, n_cells,
                                                   tolmax, ddiag, dr0, dr0h, dn0, dp0, dt, ds, dAx2, dy, dz
#ifdef PMC_DEBUG_GPU
                                                   ,dit_ptr
#endif
                                           );

#ifdef PMC_DEBUG_GPU


  int it=0;

#ifdef solveBcgCuda_sum_it

  int *it_ptr=(int*)malloc(blocks*sizeof(int));
  cudaMemcpy(it_ptr,dit_ptr,blocks*sizeof(int),cudaMemcpyDeviceToHost);

  for(int i=0;i<blocks;i++){
    it+=it_ptr[i];
  }

#ifdef solveBcgCuda_avg_it
  it=it/blocks;
  //it=it/nrows;
#endif

  free(it_ptr);

  bicg->counterBiConjGradInternal += it;

#else

  cudaMemcpy(&it,dit_ptr,sizeof(int),cudaMemcpyDeviceToHost);

  if(offset_cells==0)
    bicg->counterBiConjGradInternal += it;

#endif

#ifndef DEBUG_SOLVEBCGCUDA
  if(bicg->counterBiConjGrad==0) {
    printf("solveGPUBlock it %d\n",
           it);
  }
#endif



  cudaFree(dit_ptr);

#endif


}

//solveGPU_block: Each block will compute only a cell/group of cells
//Algorithm: Biconjugate gradient
void solveGPU_block(itsolver *bicg, double *dA, int *djA, int *diA, double *dx, double *dtempv)
{

#ifndef DEBUG_SOLVEBCGCUDA
  if(bicg->counterBiConjGrad==0) {
    printf("solveGPUBlock\n");
  }
#endif

  int len_cell = bicg->nrows/bicg->n_cells;
  int max_threads_block;
  if(bicg->use_multicells) {

    max_threads_block = bicg->threads;//bicg->threads; 128;

  }else{

    max_threads_block = nextPowerOfTwo(len_cell);

  }

  int n_cells_block =  max_threads_block/len_cell;
  int threads_block = n_cells_block*len_cell;
  int n_shr_empty = max_threads_block-threads_block;
  int blocks = (bicg->nrows+threads_block-1)/threads_block;

  int offset_cells=0;

#ifndef ALL_BLOCKS_EQUAL_SIZE

  //Common kernel (Launch all blocks except the last)
  //blocks=blocks-1;
  if(bicg->use_multicells
  //&& blocks!=0
  ) {

    blocks=blocks-1;

    if(blocks!=0)//myb not needed
    solveGPU_block_thr(blocks, threads_block, max_threads_block, n_shr_empty, offset_cells,
                       bicg);

    //todo fix case one-cell updating vars

    //Update vars to launch last kernel
    offset_cells=n_cells_block*blocks;
    int n_cells_last_block=bicg->n_cells-offset_cells;
    threads_block=n_cells_last_block*len_cell;
    max_threads_block=nextPowerOfTwo(threads_block);
    n_shr_empty = max_threads_block-threads_block;
    blocks=1;

  }

#endif

  solveGPU_block_thr(blocks, threads_block, max_threads_block, n_shr_empty, offset_cells,
           bicg);

}

//Algorithm: Biconjugate gradient
void solveGPU(itsolver *bicg, double *dA, int *djA, int *diA, double *dx, double *dtempv)
{
  //Init variables ("public")
  int nrows = bicg->nrows;
  int blocks = bicg->blocks;
  int threads = bicg->threads;
  int maxIt = bicg->maxIt;
  int mattype = bicg->mattype;
  double tolmax = bicg->tolmax;
  double *ddiag = bicg->ddiag;

  // Auxiliary vectors ("private")
  double *dr0 = bicg->dr0;
  double *dr0h = bicg->dr0h;
  double *dn0 = bicg->dn0;
  double *dp0 = bicg->dp0;
  double *dt = bicg->dt;
  double *ds = bicg->ds;
  double *dAx2 = bicg->dAx2;
  double *dy = bicg->dy;
  double *dz = bicg->dz;
  double *aux = bicg->aux;
  double *daux = bicg->daux;

#ifndef DEBUG_SOLVEBCGCUDA
  if(bicg->counterBiConjGrad==0) {
    printf("solveGPU\n");
  }
#endif

  //Function private variables
  double alpha,rho0,omega0,beta,rho1,temp1,temp2;

  gpu_spmv(dr0,dx,nrows,dA,djA,diA,mattype,blocks,threads);  // r0= A*x

  gpu_axpby(dr0,dtempv,1.0,-1.0,nrows,blocks,threads); // r0=1.0*rhs+-1.0r0 //y=ax+by

  gpu_yequalsx(dr0h,dr0,nrows,blocks,threads);  //r0h=r0

  gpu_yequalsconst(dn0,0.0,nrows,blocks,threads);  //n0=0.0 //memset???
  gpu_yequalsconst(dp0,0.0,nrows,blocks,threads);  //p0=0.0

  alpha  = 1.0;
  rho0   = 1.0;
  omega0 = 1.0;

  //printf("temp1 %-le", temp1);
  //printf("rho1 %f", rho1);

#ifdef DEBUG_SOLVEBCGCUDA_DEEP

  double *aux_x1;
  aux_x1=(double*)malloc(bicg->nrows*sizeof(double));

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

    //    cout<<"rho1 "<<rho1<<" beta "<<beta<<endl;

    gpu_zaxpbypc(dp0,dr0,dn0,beta,-1.0*omega0*beta,nrows,blocks,threads);   //z = ax + by + c

    gpu_multxy(dy,ddiag,dp0,nrows,blocks,threads);  // precond y= p0*diag

    gpu_spmv(dn0,dy,nrows,dA,djA,diA,mattype,blocks,threads);  // n0= A*y

    temp1=gpu_dotxy(dr0h, dn0, aux, daux, nrows,(blocks + 1) / 2, threads);
    //temp1=gpu_dotxy(dr0h, dn0, aux, daux, nrows, blocks, threads);

#ifdef DEBUG_SOLVEBCGCUDA_DEEP
    printf("%d temp1 %-le\n",it,temp1);
#endif

    alpha=rho1/temp1;

    //       cout<<"temp1 "<<temp1<<" alpha "<<alpha<<endl;

    gpu_zaxpby(1.0,dr0,-1.0*alpha,dn0,ds,nrows,blocks,threads);

#ifdef DEBUG_SOLVEBCGCUDA_DEEP
    cudaMemcpy(aux_x1,ds,bicg->nrows*sizeof(double),cudaMemcpyDeviceToHost);

    printf("%d ds[0] %-le\n",it,aux_x1[0]);

#endif

    gpu_multxy(dz,ddiag,ds,nrows,blocks,threads); // precond z=diag*s

    gpu_spmv(dt,dz,nrows,dA,djA,diA,mattype,blocks,threads);

    gpu_multxy(dAx2,ddiag,dt,nrows,blocks,threads);

    temp1=gpu_dotxy(dz, dAx2, aux, daux, nrows,(blocks + 1) / 2, threads);
    //temp1=gpu_dotxy(dz, dAx2, aux, daux, nrows,blocks, threads);

#ifdef DEBUG_SOLVEBCGCUDA_DEEP
    cudaMemcpy(aux_x1,dAx2,bicg->nrows*sizeof(double),cudaMemcpyDeviceToHost);

    for(int i=0; i<bicg->nrows; i++){
      //printf("%d ddiag[%i] %-le\n",it,i,aux_x1[i]);
      //printf("%d dt[%i] %-le\n",it,i,aux_x1[i]);
      //printf("%d dAx2[%i] %-le\n",it,i,aux_x1[i]);
      //printf("%d dz[%i] %-le\n",it,i,aux_x1[i]);
    }

    printf("%d temp1 %-le\n",it,temp1);
#endif

    temp2=gpu_dotxy(dAx2, dAx2, aux, daux, nrows,(blocks + 1) / 2, threads);
    //temp2=gpu_dotxy(dAx2, dAx2, aux, daux, nrows,blocks, threads);

#ifdef DEBUG_SOLVEBCGCUDA_DEEP
    printf("%d temp2 %-le\n",it,temp2);
#endif

    omega0= temp1/temp2;

    gpu_axpy(dx,dy,alpha,nrows,blocks,threads); // x=alpha*y +x

    gpu_axpy(dx,dz,omega0,nrows,blocks,threads);

    gpu_zaxpby(1.0,ds,-1.0*omega0,dt,dr0,nrows,blocks,threads);

    temp1=gpu_dotxy(dr0, dr0, aux, daux, nrows,(blocks + 1) / 2, threads);
    //temp1=gpu_dotxy(dr0, dr0, aux, daux, nrows,blocks, threads);
    temp1=sqrt(temp1);

  //cout<<it<<": "<<temp1<<endl;

    rho0=rho1;

    //if(temp1<tolmax){
    //  break;}}

    it++;
  }while(it<maxIt && temp1>tolmax);

#ifdef PMC_DEBUG_GPU
  bicg->counterBiConjGradInternal += it;

#ifndef DEBUG_SOLVEBCGCUDA
  if(bicg->counterBiConjGrad==0) {
    printf("counterBiConjGradInternal %d\n",
           bicg->counterBiConjGradInternal);
  }
#endif

#endif

#ifdef DEBUG_SOLVEBCGCUDA_DEEP
  free(aux_x1);
#endif

}

void free_itsolver(itsolver *bicg)
{
  //Auxiliary vectors ("private")
  double ** dr0 = &bicg->dr0;
  double ** dr0h = &bicg->dr0h;
  double ** dn0 = &bicg->dn0;
  double ** dp0 = &bicg->dp0;
  double ** dt = &bicg->dt;
  double ** ds = &bicg->ds;
  double ** dAx2 = &bicg->dAx2;
  double ** dy = &bicg->dy;
  double ** dz = &bicg->dz;
  double ** daux = &bicg->daux;
  double ** ddiag = &bicg->ddiag;

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
  free(bicg->aux);

}

 /*
void setUpSolver(itsolver *bicg, double reltol, double *ewt, int tnrows,int tnnz,double *tA, int *tjA, int *tiA, int tmattype, int qmax, double *dACamp, double *dftempCamp);
{

  bicg.tolmax=reltol;

}
*/
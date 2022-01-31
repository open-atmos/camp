#include "cusolver.h"

void createCuSolver(SolverData *sd){

  ModelDataGPU *mGPU = &sd->mGPU;
  itsolver *bicg = &(sd->bicg);

  printf("createCuSolver start");

  CuSolver cus_object;
  sd->cus = &cus_object;
  //CuSolver *cus = (CuSolver *)sd->cus
  //SolverData *sd = (SolverData *)solver_data;

  CuSolver *cus = (CuSolver *)sd->cus;
  cus->test = 1;


  cus->info = NULL;
  cus->descrA = NULL;
  cus->buffer_qr = NULL; // working space for numerical factorization
  cusparseStatus_t cusparse_status = CUSPARSE_STATUS_SUCCESS;
  cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;
  cudaError_t cudaStat1 = cudaSuccess;
  size_t size_qr = 0;
  size_t size_internal = 0;

  // step 1: create a descriptor
  cusolver_status = cusolverSpCreate(&cus->handle);//todo needed?

#ifdef ISSUE_26

  assert (cusolver_status == CUSOLVER_STATUS_SUCCESS);
  cusparse_status = cusparseCreateMatDescr(&cus->descrA);
  assert(cusparse_status == CUSPARSE_STATUS_SUCCESS);
  cusparseSetMatIndexBase(cus->descrA, CUSPARSE_INDEX_BASE_ONE); // A is base-1
  cusparseSetMatType(cus->descrA, CUSPARSE_MATRIX_TYPE_GENERAL); // A is a general matrix

  // step 2: create empty info structure
  cusolver_status = cusolverSpCreateCsrqrInfo(&cus->info);
  assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);

  // step 3: symbolic analysis
  cusolverSpXcsrqrAnalysisBatched(
          cus->handle, mGPU->nrows, mGPU->nrows, mGPU->nnz,
          cus->descrA, bicg->jA, bicg->iA, cus->info);

  // step 4: allocate working space for Aj*xj=bj
  cusolver_status = cusolverSpDcsrqrBufferInfoBatched(
          cus->handle, mGPU->nrows, mGPU->nrows, mGPU->nnz,
          cus->descrA, bicg->A, bicg->jA, bicg->iA,
          mGPU->n_cells,
          cus->info,
          &size_internal,
          &size_qr);
  assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);
#ifndef DEBUG_CUSOLVER
  printf("numerical factorization needs internal data %lld bytes\n",
         (long long)size_internal);
  printf("numerical factorization needs working space %lld bytes\n",
         (long long)size_qr);
#endif
  cudaStat1 = cudaMalloc((void**)&cus->buffer_qr, size_qr);
  assert(cudaStat1 == cudaSuccess);

#endif

  printf("createCuSolver end");
}

#ifdef EXAMPLE_APPENDIX_1

void solveCuSolver(SolverData *sd){

  CuSolver *cus = (CuSolver *)sd->cus;
  ModelDataGPU *mGPU = &sd->mGPU;
  itsolver *bicg = &(sd->bicg);

  printf("solvecuSolver start\n");
  printf("WARNING: Pending to implement\n");

  cusolverSpHandle_t cusolverH = NULL;
// GPU does batch QR
  csrqrInfo_t info = NULL;
  cusparseMatDescr_t descrA = NULL;
  cusparseStatus_t cusparse_status = CUSPARSE_STATUS_SUCCESS;
  cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;
  cudaError_t cudaStat1 = cudaSuccess;
  cudaError_t cudaStat2 = cudaSuccess;
  cudaError_t cudaStat3 = cudaSuccess;
  cudaError_t cudaStat4 = cudaSuccess;
  cudaError_t cudaStat5 = cudaSuccess;

  // GPU does batch QR
// d_A is CSR format, d_csrValA is of size nnzA*batchSize
// d_x is a matrix of size batchSize * m
// d_b is a matrix of size batchSize * m

  int *d_csrRowPtrA = mGPU->djA ; //each Aj has the same csrRowPtrA
  int *d_csrColIndA = mGPU->diA; //// each Aj has the same csrColIndA
  double *d_csrValA = mGPU->dA;
  double *d_b = NULL; // batchSize * m
  double *d_x = NULL; // batchSize * m
  size_t size_qr = 0;
  size_t size_internal = 0;
  void *buffer_qr = NULL; // working space for numerical factorization

    /*
  const int m = 4 ;
  const int nnzA = 7;
  const int csrRowPtrA[m+1] = { 1, 2, 3, 4, 8};
  const int csrColIndA[nnzA] = { 1, 2, 3, 1, 2, 3, 4};
  const double csrValA[nnzA] = { 1.0, 2.0, 3.0, 0.1, 0.1, 0.1, 4.0};
  const double b[m] = {1.0, 1.0, 1.0, 1.0};
  const int batchSize = 17;
  double *csrValABatch = (double*)malloc(sizeof(double)*nnzA*batchSize);
  double *bBatch = (double*)malloc(sizeof(double)*m*batchSize);
  double *xBatch = (double*)malloc(sizeof(double)*m*batchSize);
  assert( NULL != csrValABatch );
  assert( NULL != bBatch );
  assert( NULL != xBatch );
   */

   int m = mGPU->nrows ; // number of rows and columns of each Aj
   int nnzA = mGPU->nnz; // number of nonzeros of each Aj
   int csrRowPtrA = bicg->jA; //each Aj has the same csrRowPtrA
   int csrColIndA = bicg->iA; //// each Aj has the same csrColIndA
   double csrValA = bicg->A;
   double b[m] = {1.0, 1.0, 1.0, 1.0};
   int batchSize = 17;
   *csrValABatch = (double*)malloc(sizeof(double)*nnzA*batchSize);
   *bBatch = (double*)malloc(sizeof(double)*m*batchSize);
   *xBatch = (double*)malloc(sizeof(double)*m*batchSize);
  assert( NULL != csrValABatch );
  assert( NULL != bBatch );
  assert( NULL != xBatch );




  // step 1: prepare Aj and bj on host
// Aj is a small perturbation of A
// bj is a small perturbation of b
// csrValABatch = [A0, A1, A2, ...]
// bBatch = [b0, b1, b2, ...]
  for(int colidx = 0 ; colidx < nnzA ; colidx++){
    double Areg = csrValA[colidx];
    for (int batchId = 0 ; batchId < batchSize ; batchId++){
      double eps = ((double)((rand() % 100) + 1)) * 1.e-4;
      csrValABatch[batchId*nnzA + colidx] = Areg + eps;
    }
  }
  for(int j = 0 ; j < m ; j++){
    double breg = b[j];
    for (int batchId = 0 ; batchId < batchSize ; batchId++){
      double eps = ((double)((rand() % 100) + 1)) * 1.e-4;
      bBatch[batchId*m + j] = breg + eps;
    }
  }
// step 2: create cusolver handle, qr info and matrix descriptor
  cusolver_status = cusolverSpCreate(&cusolverH);
  assert (cusolver_status == CUSOLVER_STATUS_SUCCESS);
  cusparse_status = cusparseCreateMatDescr(&descrA);
  assert(cusparse_status == CUSPARSE_STATUS_SUCCESS);
  cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);

  cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ONE); // base-1
  cusolver_status = cusolverSpCreateCsrqrInfo(&info);
  assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);

  // step 3: copy Aj and bj to device
  cudaStat1 = cudaMalloc ((void**)&d_csrValA , sizeof(double) * nnzA *
                                               batchSize);
  cudaStat2 = cudaMalloc ((void**)&d_csrColIndA, sizeof(int) * nnzA);
  cudaStat3 = cudaMalloc ((void**)&d_csrRowPtrA, sizeof(int) * (m+1));
  cudaStat4 = cudaMalloc ((void**)&d_b , sizeof(double) * m * batchSize);
  cudaStat5 = cudaMalloc ((void**)&d_x , sizeof(double) * m * batchSize);
  assert(cudaStat1 == cudaSuccess);
  assert(cudaStat2 == cudaSuccess);
  assert(cudaStat3 == cudaSuccess);
  assert(cudaStat4 == cudaSuccess);
  assert(cudaStat5 == cudaSuccess);
  cudaStat1 = cudaMemcpy(d_csrValA , csrValABatch, sizeof(double) * nnzA *
                                                   batchSize, cudaMemcpyHostToDevice);
  cudaStat2 = cudaMemcpy(d_csrColIndA, csrColIndA, sizeof(int) * nnzA,
                         cudaMemcpyHostToDevice);
  cudaStat3 = cudaMemcpy(d_csrRowPtrA, csrRowPtrA, sizeof(int) * (m+1),
                         cudaMemcpyHostToDevice);
  cudaStat4 = cudaMemcpy(d_b, bBatch, sizeof(double) * m * batchSize,
                         cudaMemcpyHostToDevice);
  assert(cudaStat1 == cudaSuccess);
  assert(cudaStat2 == cudaSuccess);
  assert(cudaStat3 == cudaSuccess);
  assert(cudaStat4 == cudaSuccess);
// step 4: symbolic analysis
  cusolver_status = cusolverSpXcsrqrAnalysisBatched(
          cusolverH, m, m, nnzA,
          descrA, d_csrRowPtrA, d_csrColIndA,
          info);
  assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);
// step 5: prepare working space
  cusolver_status = cusolverSpDcsrqrBufferInfoBatched(
          cusolverH, m, m, nnzA,
          descrA, d_csrValA, d_csrRowPtrA, d_csrColIndA,
          batchSize,
          info,
          &size_internal,
          &size_qr);
  assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);
  printf("numerical factorization needs internal data %lld bytes\n",
         (long long)size_internal);
  printf("numerical factorization needs working space %lld bytes\n",
         (long long)size_qr);
  cudaStat1 = cudaMalloc((void**)&buffer_qr, size_qr);
  assert(cudaStat1 == cudaSuccess);

// step 6: numerical factorization
// assume device memory is big enough to compute all matrices.
  cusolver_status = cusolverSpDcsrqrsvBatched(
          cusolverH, m, m, nnzA,
          descrA, d_csrValA, d_csrRowPtrA, d_csrColIndA,
          d_b, d_x,
          batchSize,
          info,
          buffer_qr);
  assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);

  // step 7: check residual
// xBatch = [x0, x1, x2, ...]
  cudaStat1 = cudaMemcpy(xBatch, d_x, sizeof(double)*m*batchSize,
                         cudaMemcpyDeviceToHost);
  assert(cudaStat1 == cudaSuccess);
  const int baseA = (CUSPARSE_INDEX_BASE_ONE == cusparseGetMatIndexBase(descrA))?
                    1:0 ;
  for(int batchId = 0 ; batchId < batchSize; batchId++){
    // measure |bj - Aj*xj|
    double *csrValAj = csrValABatch + batchId * nnzA;
    double *xj = xBatch + batchId * m;
    double *bj = bBatch + batchId * m;
    // sup| bj - Aj*xj|
    double sup_res = 0;
    for(int row = 0 ; row < m ; row++){
      const int start = csrRowPtrA[row ] - baseA;
      const int end = csrRowPtrA[row+1] - baseA;
      double Ax = 0.0; // Aj(row,:)*xj
      for(int colidx = start ; colidx < end ; colidx++){
        const int col = csrColIndA[colidx] - baseA;
        const double Areg = csrValAj[colidx];
        const double xreg = xj[col];
        Ax = Ax + Areg * xreg;
      }
      double r = bj[row] - Ax;
      sup_res = (sup_res > fabs(r))? sup_res : fabs(r);
    }
    printf("batchId %d: sup|bj - Aj*xj| = %E \n", batchId, sup_res);
  }
  for(int batchId = 0 ; batchId < batchSize; batchId++){
    double *xj = xBatch + batchId * m;
    for(int row = 0 ; row < m ; row++){
      printf("x%d[%d] = %E\n", batchId, row, xj[row]);
    }
    printf("\n");
  }

  //  printf("WARNING: Pending to implement, running solveGPU_block instead\n");
  //  solveGPU_block(sd,mGPU->dA,mGPU->djA,mGPU->diA,mGPU->dx,mGPU->dtempv);
  //printf("solveCuSolver test %d\n",cus->test);//fine

  printf("solvecuSolver end\n");

}

#else

void solveCuSolver(SolverData *sd){

  CuSolver *cus = (CuSolver *)sd->cus;
  ModelDataGPU *mGPU = &sd->mGPU;
  itsolver *bicg = &(sd->bicg);

  printf("solvecuSolver start\n");
  printf("WARNING: Pending to implement\n");

  cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;

#ifdef ISSUE_26

  // step 5: solve Aj*xj = bj
  // assume device memory is big enough to compute all matrices. //todo add checking (check cusolver manual appendix 2 example 2)
  cusolver_status = cusolverSpDcsrqrsvBatched(
          cus->handle, mGPU->nrows, mGPU->nrows, mGPU->nnz,
          cus->descrA, bicg->A, bicg->jA, bicg->iA,
          mGPU->dx, mGPU->dtempv1,
          mGPU->n_cells,
          cus->info,
          cus->buffer_qr);
  assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);

  cudaMemcpy(mGPU->dx,mGPU->dtempv1,mGPU->nrows*sizeof(double),cudaMemcpyDeviceToDevice); //todo test set dx as output and input

#endif

  // step 7: destroy info
  //cusolverSpDestroyCsrqrInfo(info); //todo add deconstructor function and call from cvode_gpu


  //  printf("WARNING: Pending to implement, running solveGPU_block instead\n");
  //  solveGPU_block(sd,mGPU->dA,mGPU->djA,mGPU->diA,mGPU->dx,mGPU->dtempv);
  //printf("solveCuSolver test %d\n",cus->test);//fine

  printf("solvecuSolver end\n");

}

/*

void solveCuSolver_wrong_alloc_included(SolverData *sd){

  CuSolver *cus = (CuSolver *)sd->cus;
  ModelDataGPU *mGPU = &sd->mGPU;
  itsolver *bicg = &(sd->bicg);

  printf("solvecuSolver start\n");
  printf("WARNING: Pending to implement\n");

  int m = mGPU->nrows ; // number of rows and columns of each Aj
  int nnzA = mGPU->nnz; // number of nonzeros of each Aj
  int csrRowPtrA = bicg->jA; //each Aj has the same csrRowPtrA
  int csrColIndA = bicg->iA; //// each Aj has the same csrColIndA
  double csrValA = bicg->A; // aggregation of A0,A1,...,A9
  int batchSize = mGPU->n_cells; // number of linear systems
  double *d_b = mGPU->dx; // RHS batchSize * m
  double *d_x = mGPU->dtempv1;

  cusolverSpHandle_t handle; // handle to cusolver library
  csrqrInfo_t info = NULL;
  cusparseMatDescr_t descrA = NULL;
  void *buffer_qr = NULL; // working space for numerical factorization

  cusparseStatus_t cusparse_status = CUSPARSE_STATUS_SUCCESS;
  cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;
  cudaError_t cudaStat1 = cudaSuccess;
  cudaError_t cudaStat2 = cudaSuccess;
  cudaError_t cudaStat3 = cudaSuccess;
  cudaError_t cudaStat4 = cudaSuccess;
  cudaError_t cudaStat5 = cudaSuccess;
  size_t size_qr = 0;
  size_t size_internal = 0;

  // step 1: create a descriptor
  cusolver_status = cusolverSpCreate(&handle);//todo needed?
  assert (cusolver_status == CUSOLVER_STATUS_SUCCESS);
  cusparse_status = cusparseCreateMatDescr(&descrA);
  assert(cusparse_status == CUSPARSE_STATUS_SUCCESS);
  cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ONE); // A is base-1
  cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL); // A is a general matrix

  // step 2: create empty info structure
  cusolver_status = cusolverSpCreateCsrqrInfo(&info);
  assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);

  // step 3: symbolic analysis
  cusolverSpXcsrqrAnalysisBatched(
          handle, m, m, nnzA,
          descrA, csrRowPtrA, csrColIndA, info);

  // step 4: allocate working space for Aj*xj=bj
  cusolver_status = cusolverSpDcsrqrBufferInfoBatched(
          handle, m, m, nnzA,
          descrA, csrValA, csrRowPtrA, csrColIndA,
          batchSize,
          info,
          &size_internal,
          &size_qr);
  assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);
#ifndef DEBUG_CUSOLVER
  printf("numerical factorization needs internal data %lld bytes\n",
         (long long)size_internal);
  printf("numerical factorization needs working space %lld bytes\n",
         (long long)size_qr);
#endif
  cudaStat1 = cudaMalloc((void**)&buffer_qr, size_qr);
  assert(cudaStat1 == cudaSuccess);

  // step 5: solve Aj*xj = bj
  // assume device memory is big enough to compute all matrices. //todo add checking (check cusolver manual appendix 2 example 2)
  cusolver_status = cusolverSpDcsrqrsvBatched(
          handle, m, m, nnzA,
          descrA, csrValA, csrRowPtrA, csrColIndA,
          d_b, d_x,
          batchSize,
          info,
          buffer_qr);
  assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);

  cudaMemcpy(d_b,d_x,mGPU->nrows*sizeof(double),cudaMemcpyDeviceToDevice); //todo test set dx as output and input

  // step 7: destroy info
  //cusolverSpDestroyCsrqrInfo(info); //todo add deconstructor function and call from cvode_gpu


  //  printf("WARNING: Pending to implement, running solveGPU_block instead\n");
  //  solveGPU_block(sd,mGPU->dA,mGPU->djA,mGPU->diA,mGPU->dx,mGPU->dtempv);
  //printf("solveCuSolver test %d\n",cus->test);//fine

  printf("solvecuSolver end\n");

}
*/

#endif
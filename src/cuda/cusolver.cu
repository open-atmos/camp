#include "cusolver.h"

void solveCuSolver(SolverData *sd){

  CuSolver *cus = (CuSolver *)sd->cus;

  printf("solveCuSolver test %d\n",cus->test);//fine

}

void createCuSolver(SolverData *sd){

  printf("createCuSolver start");

  CuSolver cus;
  sd->cus = &cus;
  //CuSolver *cus = (CuSolver *)sd->cus
  //SolverData *sd = (SolverData *)solver_data;

  CuSolver *cus2 = (CuSolver *)sd->cus;
  cus2->test = 1;

  cusolverSpHandle_t handle; // handle to cusolver library
  csrqrInfo_t info = NULL;
  cusparseMatDescr_t descrA = NULL;
  void *pBuffer = NULL; // working space for numerical factorization

  solveCuSolver(sd);
  printf("createCuSolver end");
}

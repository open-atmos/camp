#ifndef CVODE_cpu2_SOLVER_H_
#define CVODE_cpu2_SOLVER_H_

#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include "../camp_common.h"

//#include<cublas.h>
//#include<cublas_v2.h>

//#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

void alloc_solver_cpu2(CVodeMem cv_mem, SolverData *sd);
int CVode_cpu2(void *cvode_mem, realtype tout, N_Vector yout,
              realtype *tret, int itask, SolverData *sd);
int CVodeGetDky_cpu2(void *cvode_mem, realtype t, int k, N_Vector dky);
void CVodeFree_cpu2(void **cvode_mem);
booleantype cvCheckNvector_cpu2(N_Vector tmpl);
booleantype cvAllocVectors_cpu2(CVodeMem cv_mem, N_Vector tmpl);
void cvFreeVectors_cpu2(CVodeMem cv_mem);
int cvInitialSetup_cpu2(CVodeMem cv_mem);
int cvHin_cpu2(CVodeMem cv_mem, realtype tout);
realtype cvUpperBoundH0_cpu2(CVodeMem cv_mem, realtype tdist);
int cvYddNorm_cpu2(CVodeMem cv_mem, realtype hg, realtype *yddnrm);
int cvRcheck1_cpu2(CVodeMem cv_mem);
int cvRcheck2_cpu2(CVodeMem cv_mem);
int cvRcheck3_cpu2(CVodeMem cv_mem);
int cvRootfind_cpu2(CVodeMem cv_mem);

void set_data_cpu2(CVodeMem cv_mem, SolverData *sd);
int cvStep_cpu2(SolverData *sd, CVodeMem cv_mem);
void cvAdjustParams_cpu2(CVodeMem cv_mem);
void cvIncreaseBDF_cpu2(CVodeMem cv_mem);
void cvDecreaseBDF_cpu2(CVodeMem cv_mem);
void cvRescale_cpu2(CVodeMem cv_mem);
void cvPredict_cpu2(CVodeMem cv_mem);
void cvSet_cpu2(CVodeMem cv_mem);
void cvSetBDF_cpu2(CVodeMem cv_mem);
void cvSetTqBDF_cpu2(CVodeMem cv_mem, realtype hsum, realtype alpha0,
                    realtype alpha0_hat, realtype xi_inv, realtype xistar_inv);
int cvHandleNFlag_cpu2(CVodeMem cv_mem, int *nflagPtr, realtype saved_t,
                      int *ncfPtr);
void cvRestore_cpu2(CVodeMem cv_mem, realtype saved_t);
booleantype cvDoErrorTest_cpu2(CVodeMem cv_mem, int *nflagPtr,
                              realtype saved_t, int *nefPtr, realtype *dsmPtr);
void cvCompleteStep_cpu2(CVodeMem cv_mem);
void cvPrepareNextStep_cpu2(CVodeMem cv_mem, realtype dsm);
void cvSetEta_cpu2(CVodeMem cv_mem);
void cvChooseEta_cpu2(CVodeMem cv_mem);
void cvBDFStab_cpu2(CVodeMem cv_mem);
int cvSLdet_cpu2(CVodeMem cv_mem);
int cvEwtSetSV_cpu2(CVodeMem cv_mem, N_Vector cv_ewt, N_Vector weight);
int cvNlsNewton_cpu2(SolverData *sd, CVodeMem cv_mem, int nflag);
//int linsolsetup_cpu2(CVodeMem cv_mem);
int linsolsetup_cpu2(SolverData *sd, CVodeMem cv_mem, int convfail, N_Vector vtemp1, N_Vector vtemp2, N_Vector vtemp3);
void free_ode_cpu2(SolverData *sd);
int linsolsolve_cpu2(SolverData *sd, CVodeMem cv_mem);
//int linsolsolve_cpu2(int *m, double *del, double *delp, double *dcon, SUNMatrix J, CVodeMem cv_mem, double *x, double *b);

//void setUpSolver(itsolver bicg, double reltol, double *ewt, int tnrows,int tnnz,double *tA, int *tjA, int *tiA, int tmattype, int qmax, double *dACamp, double *dftempCamp);

int check_jac_status_error(SUNMatrix A);
int cvHandleFailure_cpu2(CVodeMem cv_mem, int flag);

void printSolverCounters_cpu2(SolverData *sd);

#endif

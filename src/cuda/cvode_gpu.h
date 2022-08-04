/* Copyright (C) 2021 Barcelona Supercomputing Center and University of
 * Illinois at Urbana-Champaign
 * SPDX-License-Identifier: MIT
 */

#ifndef CVODE_gpu_SOLVER_H_
#define CVODE_gpu_SOLVER_H_

#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include "../camp_common.h"

void constructor_cvode_gpu(CVodeMem cv_mem, SolverData *sd);
int CVode_gpu(void *cvode_mem, realtype tout, N_Vector yout,
              realtype *tret, int itask, SolverData *sd);
int cudaCVode(void *cvode_mem, realtype tout, N_Vector yout,
               realtype *tret, int itask, SolverData *sd);
int cvInitialSetup_gpu(CVodeMem cv_mem);
int cvHin_gpu(CVodeMem cv_mem, realtype tout);
realtype cvUpperBoundH0_gpu(CVodeMem cv_mem, realtype tdist);
int cvYddNorm_gpu(CVodeMem cv_mem, realtype hg, realtype *yddnrm);
int cvRcheck1_gpu(CVodeMem cv_mem);
int cvRcheck2_gpu(CVodeMem cv_mem);
int cvRcheck3_gpu(CVodeMem cv_mem);
int cvRootfind_gpu(CVodeMem cv_mem);

void set_data_gpu(CVodeMem cv_mem, SolverData *sd);
int cvStep_gpu(SolverData *sd, CVodeMem cv_mem);
void cvAdjustParams_gpu(CVodeMem cv_mem);
void cvIncreaseBDF_gpu(CVodeMem cv_mem);
void cvDecreaseBDF_gpu(CVodeMem cv_mem);
void cvRescale_gpu(CVodeMem cv_mem);
void cvPredict_gpu(CVodeMem cv_mem);
void cvSet_gpu(CVodeMem cv_mem);
void cvSetBDF_gpu(CVodeMem cv_mem);
void cvSetTqBDF_gpu(CVodeMem cv_mem, realtype hsum, realtype alpha0,
                    realtype alpha0_hat, realtype xi_inv, realtype xistar_inv);
int cvHandleNFlag_gpu(CVodeMem cv_mem, int *nflagPtr, realtype saved_t,
                      int *ncfPtr);
void cvRestore_gpu(CVodeMem cv_mem, realtype saved_t);
booleantype cvDoErrorTest_gpu(CVodeMem cv_mem, int *nflagPtr,
                              realtype saved_t, int *nefPtr, realtype *dsmPtr);
void cvCompleteStep_gpu(CVodeMem cv_mem);
void cvPrepareNextStep_gpu(CVodeMem cv_mem, realtype dsm);
void cvSetEta_gpu(CVodeMem cv_mem);
void cvChooseEta_gpu(CVodeMem cv_mem);
void cvBDFStab_gpu(CVodeMem cv_mem);
int cvSLdet_gpu(CVodeMem cv_mem);
int cvNlsNewton_gpu(SolverData *sd, CVodeMem cv_mem, int nflag);
int linsolsetup_gpu(SolverData *sd, CVodeMem cv_mem, int convfail, N_Vector vtemp1, N_Vector vtemp2, N_Vector vtemp3);
int linsolsolve_gpu(SolverData *sd, CVodeMem cv_mem);

int check_jac_status_error_gpu(SUNMatrix A);
int cvHandleFailure_gpu(CVodeMem cv_mem, int flag);

void solver_get_statistics_gpu(SolverData *sd);
void solver_reset_statistics_gpu(SolverData *sd);
void printSolverCounters_gpu(SolverData *sd);

#endif

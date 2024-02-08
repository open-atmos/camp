/* Copyright (C) 2021 Barcelona Supercomputing Center and University of
 * Illinois at Urbana-Champaign
 * SPDX-License-Identifier: MIT
 */
#include "cvode_cuda.h"

extern "C" {
#include "cvode_gpu.h"
}

#ifdef CAMP_USE_MPI
#include <mpi.h>
#endif


static int cvHandleFailure_gpu(CVodeMem cv_mem, int flag){
  switch (flag) {
    case CV_ERR_FAILURE:
      cvProcessError(cv_mem, CV_ERR_FAILURE, "CVODE", "CVode", MSGCV_ERR_FAILS,
                     cv_mem->cv_tn, cv_mem->cv_h);
      break;
    case CV_CONV_FAILURE:
      cvProcessError(cv_mem, CV_CONV_FAILURE, "CVODE", "CVode", MSGCV_CONV_FAILS,
                     cv_mem->cv_tn, cv_mem->cv_h);
      break;
    case CV_LSETUP_FAIL:
      cvProcessError(cv_mem, CV_LSETUP_FAIL, "CVODE", "CVode", MSGCV_SETUP_FAILED,
                     cv_mem->cv_tn);
      break;
    case CV_LSOLVE_FAIL:
      cvProcessError(cv_mem, CV_LSOLVE_FAIL, "CVODE", "CVode", MSGCV_SOLVE_FAILED,
                     cv_mem->cv_tn);
      break;
    case CV_RHSFUNC_FAIL:
      cvProcessError(cv_mem, CV_RHSFUNC_FAIL, "CVODE", "CVode", MSGCV_RHSFUNC_FAILED,
                     cv_mem->cv_tn);
      break;
    case CV_UNREC_RHSFUNC_ERR:
      cvProcessError(cv_mem, CV_UNREC_RHSFUNC_ERR, "CVODE", "CVode", MSGCV_RHSFUNC_UNREC,
                     cv_mem->cv_tn);
      break;
    case CV_REPTD_RHSFUNC_ERR:
      cvProcessError(cv_mem, CV_REPTD_RHSFUNC_ERR, "CVODE", "CVode", MSGCV_RHSFUNC_REPTD,
                     cv_mem->cv_tn);
      break;
    case CV_RTFUNC_FAIL:
      cvProcessError(cv_mem, CV_RTFUNC_FAIL, "CVODE", "CVode", MSGCV_RTFUNC_FAILED,
                     cv_mem->cv_tn);
      break;
    case CV_TOO_CLOSE:
      cvProcessError(cv_mem, CV_TOO_CLOSE, "CVODE", "CVode", MSGCV_TOO_CLOSE);
      break;
    default:
      return(CV_SUCCESS);
  }
  return(flag);
}

static int cvYddNorm_gpu(CVodeMem cv_mem, realtype hg, realtype *yddnrm){
  int retval;
  N_VLinearSum(hg, cv_mem->cv_zn[1], ONE, cv_mem->cv_zn[0], cv_mem->cv_y);
  retval = f(cv_mem->cv_tn+hg, cv_mem->cv_y, cv_mem->cv_tempv, cv_mem->cv_user_data);
  cv_mem->cv_nfe++;
  if (retval < 0) return(CV_RHSFUNC_FAIL);
  if (retval > 0) return(RHSFUNC_RECVR);
  N_VLinearSum(ONE, cv_mem->cv_tempv, -ONE, cv_mem->cv_zn[1], cv_mem->cv_tempv);
  N_VScale(ONE/hg, cv_mem->cv_tempv, cv_mem->cv_tempv);
  *yddnrm = N_VWrmsNorm(cv_mem->cv_tempv, cv_mem->cv_ewt);
  return(CV_SUCCESS);
}

static realtype cvUpperBoundH0_gpu(CVodeMem cv_mem, realtype tdist){
  realtype hub_inv, hub;
  N_Vector temp1, temp2;
  temp1 = cv_mem->cv_tempv;
  temp2 = cv_mem->cv_acor;
  N_VAbs(cv_mem->cv_zn[0], temp2);
  cv_mem->cv_efun(cv_mem->cv_zn[0], temp1, cv_mem->cv_e_data);
  N_VInv(temp1, temp1);
  N_VLinearSum(HUB_FACTOR, temp2, ONE, temp1, temp1);
  N_VAbs(cv_mem->cv_zn[1], temp2);
  N_VDiv(temp2, temp1, temp1);
  hub_inv = N_VMaxNorm(temp1);
  hub = HUB_FACTOR*tdist;
  if (hub*hub_inv > ONE) hub = ONE/hub_inv;
  return(hub);
}

static int cvHin_gpu(CVodeMem cv_mem, realtype tout){
  int retval, sign, count1, count2;
  realtype tdiff, tdist, tround, hlb, hub;
  realtype hg, hgs, hs, hnew, hrat, h0, yddnrm;
  booleantype hgOK, hnewOK;
  if ((tdiff = tout-cv_mem->cv_tn) == ZERO) return(CV_TOO_CLOSE);
  sign = (tdiff > ZERO) ? 1 : -1;
  tdist = SUNRabs(tdiff);
  tround = cv_mem->cv_uround * SUNMAX(SUNRabs(cv_mem->cv_tn), SUNRabs(tout));
  if (tdist < TWO*tround) return(CV_TOO_CLOSE);
  hlb = HLB_FACTOR * tround;
  hub = cvUpperBoundH0_gpu(cv_mem, tdist);
  hg  = SUNRsqrt(hlb*hub);
  if (hub < hlb) {
    if (sign == -1) cv_mem->cv_h = -hg;
    else            cv_mem->cv_h =  hg;
    return(CV_SUCCESS);
  }
  hnewOK = SUNFALSE;
  hs = hg;
  for(count1 = 1; count1 <= MAX_ITERS; count1++) {
    hgOK = SUNFALSE;
    for (count2 = 1; count2 <= MAX_ITERS; count2++) {
      hgs = hg*sign;
      retval = cvYddNorm_gpu(cv_mem, hgs, &yddnrm);
      if (retval < 0) return(CV_RHSFUNC_FAIL);
      if (retval == CV_SUCCESS) {hgOK = SUNTRUE; break;}
      hg *= POINT2;
    }
    if (!hgOK) {
      if (count1 <= 2) return(CV_REPTD_RHSFUNC_ERR);
      hnew = hs;
      break;
    }
    hs = hg;
    if ( (hnewOK) || (count1 == MAX_ITERS))  {hnew = hg; break;}
    hnew = (yddnrm*hub*hub > TWO) ? SUNRsqrt(TWO/yddnrm) : SUNRsqrt(hg*hub);
    hrat = hnew/hg;
    if ((hrat > HALF) && (hrat < TWO)) {
      hnewOK = SUNTRUE;
    }
    if ((count1 > 1) && (hrat > TWO)) {
      hnew = hg;
      hnewOK = SUNTRUE;
    }
    hg = hnew;
  }
  h0 = H_BIAS*hnew;
  if (h0 < hlb) h0 = hlb;
  if (h0 > hub) h0 = hub;
  if (sign == -1) h0 = -h0;
  cv_mem->cv_h = h0;
  return(CV_SUCCESS);
}

static int cvRcheck1_gpu(CVodeMem cv_mem){
  int i, retval;
  realtype smallh, hratio, tplus;
  booleantype zroot;
  for (i = 0; i < cv_mem->cv_nrtfn; i++) cv_mem->cv_iroots[i] = 0;
  cv_mem->cv_tlo = cv_mem->cv_tn;
  cv_mem->cv_ttol = (SUNRabs(cv_mem->cv_tn) + SUNRabs(cv_mem->cv_h)) *
                    cv_mem->cv_uround*HUNDRED;
  retval = cv_mem->cv_gfun(cv_mem->cv_tlo, cv_mem->cv_zn[0],
                           cv_mem->cv_glo, cv_mem->cv_user_data);
  cv_mem->cv_nge = 1;
  if (retval != 0) return(CV_RTFUNC_FAIL);
  zroot = SUNFALSE;
  for (i = 0; i < cv_mem->cv_nrtfn; i++) {
    if (SUNRabs(cv_mem->cv_glo[i]) == ZERO) {
      zroot = SUNTRUE;
      cv_mem->cv_gactive[i] = SUNFALSE;
    }
  }
  if (!zroot) return(CV_SUCCESS);
  hratio = SUNMAX(cv_mem->cv_ttol/SUNRabs(cv_mem->cv_h), PT1);
  smallh = hratio*cv_mem->cv_h;
  tplus = cv_mem->cv_tlo + smallh;
  N_VLinearSum(ONE, cv_mem->cv_zn[0], hratio,
               cv_mem->cv_zn[1], cv_mem->cv_y);
  retval = cv_mem->cv_gfun(tplus, cv_mem->cv_y,
                           cv_mem->cv_ghi, cv_mem->cv_user_data);
  cv_mem->cv_nge++;
  if (retval != 0) return(CV_RTFUNC_FAIL);
  for (i = 0; i < cv_mem->cv_nrtfn; i++) {
    if (!cv_mem->cv_gactive[i] && SUNRabs(cv_mem->cv_ghi[i]) != ZERO) {
      cv_mem->cv_gactive[i] = SUNTRUE;
      cv_mem->cv_glo[i] = cv_mem->cv_ghi[i];
    }
  }
  return(CV_SUCCESS);
}

static int cvRcheck2_gpu(CVodeMem cv_mem){
  int i, retval;
  realtype smallh, hratio, tplus;
  booleantype zroot;
  if (cv_mem->cv_irfnd == 0) return(CV_SUCCESS);
  (void) CVodeGetDky(cv_mem, cv_mem->cv_tlo, 0, cv_mem->cv_y);
  retval = cv_mem->cv_gfun(cv_mem->cv_tlo, cv_mem->cv_y,
                           cv_mem->cv_glo, cv_mem->cv_user_data);
  cv_mem->cv_nge++;
  if (retval != 0) return(CV_RTFUNC_FAIL);
  zroot = SUNFALSE;
  for (i = 0; i < cv_mem->cv_nrtfn; i++) cv_mem->cv_iroots[i] = 0;
  for (i = 0; i < cv_mem->cv_nrtfn; i++) {
    if (!cv_mem->cv_gactive[i]) continue;
    if (SUNRabs(cv_mem->cv_glo[i]) == ZERO) {
      zroot = SUNTRUE;
      cv_mem->cv_iroots[i] = 1;
    }
  }
  if (!zroot) return(CV_SUCCESS);
  cv_mem->cv_ttol = (SUNRabs(cv_mem->cv_tn) + SUNRabs(cv_mem->cv_h)) *
                    cv_mem->cv_uround * HUNDRED;
  smallh = (cv_mem->cv_h > ZERO) ? cv_mem->cv_ttol : -cv_mem->cv_ttol;
  tplus = cv_mem->cv_tlo + smallh;
  if ( (tplus - cv_mem->cv_tn)*cv_mem->cv_h >= ZERO) {
    hratio = smallh/cv_mem->cv_h;
    N_VLinearSum(ONE, cv_mem->cv_y, hratio, cv_mem->cv_zn[1], cv_mem->cv_y);
  } else {
    (void) CVodeGetDky(cv_mem, tplus, 0, cv_mem->cv_y);
  }
  retval = cv_mem->cv_gfun(tplus, cv_mem->cv_y,
                           cv_mem->cv_ghi, cv_mem->cv_user_data);
  cv_mem->cv_nge++;
  if (retval != 0) return(CV_RTFUNC_FAIL);
  zroot = SUNFALSE;
  for (i = 0; i < cv_mem->cv_nrtfn; i++) {
    if (!cv_mem->cv_gactive[i]) continue;
    if (SUNRabs(cv_mem->cv_ghi[i]) == ZERO) {
      if (cv_mem->cv_iroots[i] == 1) return(CLOSERT);
      zroot = SUNTRUE;
      cv_mem->cv_iroots[i] = 1;
    } else {
      if (cv_mem->cv_iroots[i] == 1)
        cv_mem->cv_glo[i] = cv_mem->cv_ghi[i];
    }
  }
  if (zroot) return(RTFOUND);
  return(CV_SUCCESS);
}

static int cvRootfind_gpu(CVodeMem cv_mem){
  realtype alph, tmid, gfrac, maxfrac, fracint, fracsub;
  int i, retval, imax, side, sideprev;
  booleantype zroot, sgnchg;
  imax = 0;
  maxfrac = ZERO;
  zroot = SUNFALSE;
  sgnchg = SUNFALSE;
  for (i = 0;  i < cv_mem->cv_nrtfn; i++) {
    if(!cv_mem->cv_gactive[i]) continue;
    if (SUNRabs(cv_mem->cv_ghi[i]) == ZERO) {
      if(cv_mem->cv_rootdir[i]*cv_mem->cv_glo[i] <= ZERO) {
        zroot = SUNTRUE;
      }
    } else {
      if ( (cv_mem->cv_glo[i]*cv_mem->cv_ghi[i] < ZERO) &&
           (cv_mem->cv_rootdir[i]*cv_mem->cv_glo[i] <= ZERO) ) {
        gfrac = SUNRabs(cv_mem->cv_ghi[i]/(cv_mem->cv_ghi[i] - cv_mem->cv_glo[i]));
        if (gfrac > maxfrac) {
          sgnchg = SUNTRUE;
          maxfrac = gfrac;
          imax = i;
        }
      }
    }
  }
  if (!sgnchg) {
    cv_mem->cv_trout = cv_mem->cv_thi;
    for (i = 0; i < cv_mem->cv_nrtfn; i++) cv_mem->cv_grout[i] = cv_mem->cv_ghi[i];
    if (!zroot) return(CV_SUCCESS);
    for (i = 0; i < cv_mem->cv_nrtfn; i++) {
      cv_mem->cv_iroots[i] = 0;
      if(!cv_mem->cv_gactive[i]) continue;
      if ( (SUNRabs(cv_mem->cv_ghi[i]) == ZERO) &&
           (cv_mem->cv_rootdir[i]*cv_mem->cv_glo[i] <= ZERO) )
        cv_mem->cv_iroots[i] = cv_mem->cv_glo[i] > 0 ? -1 : 1;
    }
    return(RTFOUND);
  }
  alph = ONE;
  side = 0;  sideprev = -1;
  for(;;) {
    if (SUNRabs(cv_mem->cv_thi - cv_mem->cv_tlo) <= cv_mem->cv_ttol) break;
    if (sideprev == side) {
      alph = (side == 2) ? alph*TWO : alph*HALF;
    } else {
      alph = ONE;
    }
    tmid = cv_mem->cv_thi - (cv_mem->cv_thi - cv_mem->cv_tlo) *
                            cv_mem->cv_ghi[imax] / (cv_mem->cv_ghi[imax] - alph*cv_mem->cv_glo[imax]);
    if (SUNRabs(tmid - cv_mem->cv_tlo) < HALF*cv_mem->cv_ttol) {
      fracint = SUNRabs(cv_mem->cv_thi - cv_mem->cv_tlo)/cv_mem->cv_ttol;
      fracsub = (fracint > FIVE) ? PT1 : HALF/fracint;
      tmid = cv_mem->cv_tlo + fracsub*(cv_mem->cv_thi - cv_mem->cv_tlo);
    }
    if (SUNRabs(cv_mem->cv_thi - tmid) < HALF*cv_mem->cv_ttol) {
      fracint = SUNRabs(cv_mem->cv_thi - cv_mem->cv_tlo)/cv_mem->cv_ttol;
      fracsub = (fracint > FIVE) ? PT1 : HALF/fracint;
      tmid = cv_mem->cv_thi - fracsub*(cv_mem->cv_thi - cv_mem->cv_tlo);
    }
    (void) CVodeGetDky(cv_mem, tmid, 0, cv_mem->cv_y);
    retval = cv_mem->cv_gfun(tmid, cv_mem->cv_y, cv_mem->cv_grout,
                             cv_mem->cv_user_data);
    cv_mem->cv_nge++;
    if (retval != 0) return(CV_RTFUNC_FAIL);
    maxfrac = ZERO;
    zroot = SUNFALSE;
    sgnchg = SUNFALSE;
    sideprev = side;
    for (i = 0;  i < cv_mem->cv_nrtfn; i++) {
      if(!cv_mem->cv_gactive[i]) continue;
      if (SUNRabs(cv_mem->cv_grout[i]) == ZERO) {
        if(cv_mem->cv_rootdir[i]*cv_mem->cv_glo[i] <= ZERO) zroot = SUNTRUE;
      } else {
        if ( (cv_mem->cv_glo[i]*cv_mem->cv_grout[i] < ZERO) &&
             (cv_mem->cv_rootdir[i]*cv_mem->cv_glo[i] <= ZERO) ) {
          gfrac = SUNRabs(cv_mem->cv_grout[i]/(cv_mem->cv_grout[i] - cv_mem->cv_glo[i]));
          if (gfrac > maxfrac) {
            sgnchg = SUNTRUE;
            maxfrac = gfrac;
            imax = i;
          }
        }
      }
    }
    if (sgnchg) {
      cv_mem->cv_thi = tmid;
      for (i = 0; i < cv_mem->cv_nrtfn; i++)
        cv_mem->cv_ghi[i] = cv_mem->cv_grout[i];
      side = 1;
      if (SUNRabs(cv_mem->cv_thi - cv_mem->cv_tlo) <= cv_mem->cv_ttol) break;
      continue;
    }
    if (zroot) {
      cv_mem->cv_thi = tmid;
      for (i = 0; i < cv_mem->cv_nrtfn; i++)
        cv_mem->cv_ghi[i] = cv_mem->cv_grout[i];
      break;
    }
    cv_mem->cv_tlo = tmid;
    for (i = 0; i < cv_mem->cv_nrtfn; i++)
      cv_mem->cv_glo[i] = cv_mem->cv_grout[i];
    side = 2;
    if (SUNRabs(cv_mem->cv_thi - cv_mem->cv_tlo) <= cv_mem->cv_ttol) break;
  }
  cv_mem->cv_trout = cv_mem->cv_thi;
  for (i = 0; i < cv_mem->cv_nrtfn; i++) {
    cv_mem->cv_grout[i] = cv_mem->cv_ghi[i];
    cv_mem->cv_iroots[i] = 0;
    if(!cv_mem->cv_gactive[i]) continue;
    if ( (SUNRabs(cv_mem->cv_ghi[i]) == ZERO) &&
         (cv_mem->cv_rootdir[i]*cv_mem->cv_glo[i] <= ZERO) )
      cv_mem->cv_iroots[i] = cv_mem->cv_glo[i] > 0 ? -1 : 1;
    if ( (cv_mem->cv_glo[i]*cv_mem->cv_ghi[i] < ZERO) &&
         (cv_mem->cv_rootdir[i]*cv_mem->cv_glo[i] <= ZERO) )
      cv_mem->cv_iroots[i] = cv_mem->cv_glo[i] > 0 ? -1 : 1;
  }
  return(RTFOUND);
}

static int cvRcheck3_gpu(CVodeMem cv_mem){
  int i, ier, retval;
  if ( (cv_mem->cv_toutc - cv_mem->cv_tn)*cv_mem->cv_h >= ZERO) {
    cv_mem->cv_thi = cv_mem->cv_tn;
    N_VScale(ONE, cv_mem->cv_zn[0], cv_mem->cv_y);
  } else {
    cv_mem->cv_thi = cv_mem->cv_toutc;
    (void) CVodeGetDky(cv_mem, cv_mem->cv_thi, 0, cv_mem->cv_y);
  }
  retval = cv_mem->cv_gfun(cv_mem->cv_thi, cv_mem->cv_y,
                           cv_mem->cv_ghi, cv_mem->cv_user_data);
  cv_mem->cv_nge++;
  if (retval != 0) return(CV_RTFUNC_FAIL);
  cv_mem->cv_ttol = (SUNRabs(cv_mem->cv_tn) + SUNRabs(cv_mem->cv_h)) *
                    cv_mem->cv_uround * HUNDRED;
  ier = cvRootfind_gpu(cv_mem);
  if (ier == CV_RTFUNC_FAIL) return(CV_RTFUNC_FAIL);
  for(i=0; i<cv_mem->cv_nrtfn; i++) {
    if(!cv_mem->cv_gactive[i] && cv_mem->cv_grout[i] != ZERO)
      cv_mem->cv_gactive[i] = SUNTRUE;
  }
  cv_mem->cv_tlo = cv_mem->cv_trout;
  for (i = 0; i < cv_mem->cv_nrtfn; i++)
    cv_mem->cv_glo[i] = cv_mem->cv_grout[i];
  if (ier == CV_SUCCESS) return(CV_SUCCESS);
  (void) CVodeGetDky(cv_mem, cv_mem->cv_trout, 0, cv_mem->cv_y);
  return(RTFOUND);
}

int cudaCVode(void *cvode_mem, realtype tout, N_Vector yout,
               realtype *tret, SolverData *sd){
  CVodeMem cv_mem;
  int retval, hflag, istate, ier;
  realtype troundoff, tout_hin, rh;
  ModelDataCPU *mCPU = &(sd->mCPU);
  ModelDataGPU *mGPU = sd->mGPU;
  ModelData *md = &(sd->model_data);
  cudaStream_t stream = 0;
#ifdef DEV_CPU_GPU
  if (n_cells==0) goto dev_cpu;
#else
  int n_cells = md->n_cells;
#endif
  HANDLE_ERROR(cudaMemcpyAsync(mGPU->rxn_env_data,md->rxn_env_data,md->n_rxn_env_data * mGPU->n_cells * sizeof(double),cudaMemcpyHostToDevice,stream));
  HANDLE_ERROR(cudaMemcpyAsync(mGPU->env,md->total_env,CAMP_NUM_ENV_PARAM_ * n_cells * sizeof(double),cudaMemcpyHostToDevice,stream));
  double *J_state = N_VGetArrayPointer(md->J_state);
  cudaMemcpyAsync(mGPU->J_state,J_state,mGPU->nrows*sizeof(double),cudaMemcpyHostToDevice,stream);
  double *J_deriv = N_VGetArrayPointer(md->J_deriv);
  cudaMemcpyAsync(mGPU->J_deriv,J_deriv,mGPU->nrows*sizeof(double),cudaMemcpyHostToDevice,stream);

  /*
  double *aux=(double*)malloc(mGPU->nrows*sizeof(double));
  for (int i = 0; i < mGPU->nrows; i++) {
    aux[i] = 0.;
  }
  cudaMemcpyAsync(mGPU->J_state,J_state,mGPU->nrows*sizeof(double),cudaMemcpyHostToDevice,stream);
  cudaMemcpyAsync(mGPU->J_deriv,J_deriv,mGPU->nrows*sizeof(double),cudaMemcpyHostToDevice,stream);
  free(aux);
*/

  int nnz = md->n_per_cell_solver_jac_elem * n_cells;
  size_t jac_size = nnz * sizeof(double);
  double *J_solver = SM_DATA_S(md->J_solver);
  cudaMemcpyAsync(mGPU->J_solver, J_solver, jac_size, cudaMemcpyHostToDevice,stream);
  double *A = SM_DATA_S(sd->J);
  HANDLE_ERROR(cudaMemcpyAsync(mGPU->dA, A, jac_size, cudaMemcpyHostToDevice, stream));
  cv_mem = (CVodeMem) cvode_mem;
  cv_mem->cv_y = yout;
  cv_mem->cv_toutc = tout;
  //Initializations performed only at the first step (nst=0):
  if (cv_mem->cv_nst == 0) {
    cv_mem->cv_tretlast = *tret = cv_mem->cv_tn;
    cv_mem->cv_e_data = cv_mem;
    cv_mem->cv_efun(cv_mem->cv_zn[0], cv_mem->cv_ewt, cv_mem->cv_e_data);
    cv_mem->cv_linit(cv_mem);
    retval = f(cv_mem->cv_tn, cv_mem->cv_zn[0], cv_mem->cv_zn[1], cv_mem->cv_user_data);
    N_VScale(ONE, cv_mem->cv_zn[0], yout);
    cv_mem->cv_nfe++;
    if (retval < 0) {
      cvProcessError(cv_mem, CV_RHSFUNC_FAIL, "CVODE", "CVode",
                     MSGCV_RHSFUNC_FAILED, cv_mem->cv_tn);
      return(CV_RHSFUNC_FAIL);
    }
    if (retval > 0) {
      cvProcessError(cv_mem, CV_FIRST_RHSFUNC_ERR, "CVODE", "CVode",
                     MSGCV_RHSFUNC_FIRST);
      return(CV_FIRST_RHSFUNC_ERR);
    }
    if (cv_mem->cv_tstopset) {
      if ( (cv_mem->cv_tstop - cv_mem->cv_tn)*(tout - cv_mem->cv_tn) <= ZERO ) {
        cvProcessError(cv_mem, CV_ILL_INPUT, "CVODE", "CVode",
                       MSGCV_BAD_TSTOP, cv_mem->cv_tstop, cv_mem->cv_tn);
        return(CV_ILL_INPUT);
      }
    }
    cv_mem->cv_h = cv_mem->cv_hin;
    if ( (cv_mem->cv_h != ZERO) && ((tout-cv_mem->cv_tn)*cv_mem->cv_h < ZERO) ) {
      cvProcessError(cv_mem, CV_ILL_INPUT, "CVODE", "CVode", MSGCV_BAD_H0);
      return(CV_ILL_INPUT);
    }
    if (cv_mem->cv_h == ZERO) {
      tout_hin = tout;
      if ( cv_mem->cv_tstopset && (tout-cv_mem->cv_tn)*(tout-cv_mem->cv_tstop) > ZERO )
        tout_hin = cv_mem->cv_tstop;
      hflag = cvHin_gpu(cv_mem, tout_hin);
      if (hflag != CV_SUCCESS) {
        istate = cvHandleFailure_gpu(cv_mem, hflag);
        return(istate);
      }
    }
    rh = SUNRabs(cv_mem->cv_h)*cv_mem->cv_hmax_inv;
    if (rh > ONE) cv_mem->cv_h /= rh;
    if (SUNRabs(cv_mem->cv_h) < cv_mem->cv_hmin)
      cv_mem->cv_h *= cv_mem->cv_hmin/SUNRabs(cv_mem->cv_h);
    if (cv_mem->cv_tstopset) {
      if ( (cv_mem->cv_tn + cv_mem->cv_h - cv_mem->cv_tstop)*cv_mem->cv_h > ZERO )
        cv_mem->cv_h = (cv_mem->cv_tstop - cv_mem->cv_tn)*(ONE-FOUR*cv_mem->cv_uround);
    }
    cv_mem->cv_hscale = cv_mem->cv_h;
    cv_mem->cv_h0u    = cv_mem->cv_h;
    cv_mem->cv_hprime = cv_mem->cv_h;
    N_VScale(cv_mem->cv_h, cv_mem->cv_zn[1], cv_mem->cv_zn[1]);
    if (cv_mem->cv_ghfun) {
      N_VLinearSum(ONE, cv_mem->cv_zn[0], ONE, cv_mem->cv_zn[1], cv_mem->cv_tempv1);
      cv_mem->cv_ghfun(cv_mem->cv_tn + cv_mem->cv_h, cv_mem->cv_h, cv_mem->cv_tempv1,
                       cv_mem->cv_zn[0], cv_mem->cv_zn[1], cv_mem->cv_user_data,
                       cv_mem->cv_tempv2, cv_mem->cv_acor_init);
    }
    if (cv_mem->cv_nrtfn > 0) {
      retval = cvRcheck1_gpu(cv_mem);
      if (retval == CV_RTFUNC_FAIL) {
        cvProcessError(cv_mem, CV_RTFUNC_FAIL, "CVODE", "cvRcheck1",
                       MSGCV_RTFUNC_FAILED, cv_mem->cv_tn);
        return(CV_RTFUNC_FAIL);
      }
    }
  }
   //At following steps, perform stop tests:
  if (cv_mem->cv_nst > 0) {
    troundoff = FUZZ_FACTOR*cv_mem->cv_uround*(SUNRabs(cv_mem->cv_tn) + SUNRabs(cv_mem->cv_h));
    if (cv_mem->cv_nrtfn > 0) {
      retval = cvRcheck2_gpu(cv_mem);
      if (retval == CLOSERT) {
        cvProcessError(cv_mem, CV_ILL_INPUT, "CVODE", "cvRcheck2",
                       MSGCV_CLOSE_ROOTS, cv_mem->cv_tlo);
        return(CV_ILL_INPUT);
      } else if (retval == CV_RTFUNC_FAIL) {
        cvProcessError(cv_mem, CV_RTFUNC_FAIL, "CVODE", "cvRcheck2",
                       MSGCV_RTFUNC_FAILED, cv_mem->cv_tlo);
        return(CV_RTFUNC_FAIL);
      } else if (retval == RTFOUND) {
        cv_mem->cv_tretlast = *tret = cv_mem->cv_tlo;
        return(CV_ROOT_RETURN);
      }
      if ( SUNRabs(cv_mem->cv_tn - cv_mem->cv_tretlast) > troundoff ) {
        retval = cvRcheck3_gpu(cv_mem);
        if (retval == CV_SUCCESS) {
          cv_mem->cv_irfnd = 0;
        } else if (retval == RTFOUND) {
          cv_mem->cv_irfnd = 1;
          cv_mem->cv_tretlast = *tret = cv_mem->cv_tlo;
          return(CV_ROOT_RETURN);
        } else if (retval == CV_RTFUNC_FAIL) {
          cvProcessError(cv_mem, CV_RTFUNC_FAIL, "CVODE", "cvRcheck3",
                         MSGCV_RTFUNC_FAILED, cv_mem->cv_tlo);
          return(CV_RTFUNC_FAIL);
        }
      }
    }
    if ((cv_mem->cv_tn-tout)*cv_mem->cv_h >= ZERO) {
      cv_mem->cv_tretlast = *tret = tout;
      ier =  CVodeGetDky(cv_mem, tout, 0, yout);
      if (ier != CV_SUCCESS) {
        cvProcessError(cv_mem, CV_ILL_INPUT, "CVODE", "CVode",
                       MSGCV_BAD_TOUT, tout);
        return(CV_ILL_INPUT);
      }
      return(CV_SUCCESS);
    }
    if ( cv_mem->cv_tstopset ) {
      if ( SUNRabs(cv_mem->cv_tn - cv_mem->cv_tstop) <= troundoff) {
        ier =  CVodeGetDky(cv_mem, cv_mem->cv_tstop, 0, yout);
        if (ier != CV_SUCCESS) {
          cvProcessError(cv_mem, CV_ILL_INPUT, "CVODE", "CVode",
                         MSGCV_BAD_TSTOP, cv_mem->cv_tstop, cv_mem->cv_tn);
          return(CV_ILL_INPUT);
        }
        cv_mem->cv_tretlast = *tret = cv_mem->cv_tstop;
        cv_mem->cv_tstopset = SUNFALSE;
        return(CV_TSTOP_RETURN);
      }
      if ( (cv_mem->cv_tn + cv_mem->cv_hprime - cv_mem->cv_tstop)*cv_mem->cv_h > ZERO ) {
        cv_mem->cv_hprime = (cv_mem->cv_tstop - cv_mem->cv_tn)*(ONE-FOUR*cv_mem->cv_uround);
        cv_mem->cv_eta = cv_mem->cv_hprime/cv_mem->cv_h;
      }
    }
  }
   //Looping point for internal steps
#ifdef CAMP_DEBUG_GPU
  cudaEventRecord(sd->startcvStep);
#endif
  for (int i = 0; i < mGPU->n_cells; i++)
    sd->flagCells[i] = 99;
#ifdef ODE_WARNING
  mCPU->mdvCPU.cv_nhnil = cv_mem->cv_nhnil;
#endif
  mCPU->mdvCPU.tret = *tret;
  mCPU->mdvCPU.cv_tretlast = cv_mem->cv_tretlast;
  mCPU->mdvCPU.cv_etaqm1 = cv_mem->cv_etaqm1;
  mCPU->mdvCPU.cv_etaq = cv_mem->cv_etaq;
  mCPU->mdvCPU.cv_etaqp1 = cv_mem->cv_etaqp1;
  mCPU->mdvCPU.cv_saved_tq5 = cv_mem->cv_saved_tq5;
  mCPU->mdvCPU.cv_tolsf = cv_mem->cv_tolsf;
  mCPU->mdvCPU.cv_indx_acor = cv_mem->cv_indx_acor;
  mCPU->mdvCPU.cv_hu = cv_mem->cv_hu;
  mCPU->mdvCPU.cv_jcur = cv_mem->cv_jcur;
  mCPU->mdvCPU.cv_nstlp = cv_mem->cv_nstlp;
  mCPU->mdvCPU.cv_L = cv_mem->cv_L;
  mCPU->mdvCPU.cv_acnrm = cv_mem->cv_acnrm;
  mCPU->mdvCPU.cv_qwait = cv_mem->cv_qwait;
  mCPU->mdvCPU.cv_crate = cv_mem->cv_crate;
  mCPU->mdvCPU.cv_gamrat = cv_mem->cv_gamrat;
  mCPU->mdvCPU.cv_gammap = cv_mem->cv_gammap;
  mCPU->mdvCPU.cv_gamma = cv_mem->cv_gamma;
  mCPU->mdvCPU.cv_rl1 = cv_mem->cv_rl1;
  mCPU->mdvCPU.cv_eta = cv_mem->cv_eta;
  mCPU->mdvCPU.cv_q = cv_mem->cv_q;
  mCPU->mdvCPU.cv_qprime = cv_mem->cv_qprime;
  mCPU->mdvCPU.cv_h = cv_mem->cv_h;
  mCPU->mdvCPU.cv_next_h = cv_mem->cv_next_h;
  mCPU->mdvCPU.cv_hscale = cv_mem->cv_hscale;
  mCPU->mdvCPU.cv_hprime = cv_mem->cv_hprime;
  mCPU->mdvCPU.cv_hmin = cv_mem->cv_hmin;
  mCPU->mdvCPU.cv_tn = cv_mem->cv_tn;
  mCPU->mdvCPU.cv_etamax = cv_mem->cv_etamax;
  mCPU->mdvCPU.cv_maxncf = cv_mem->cv_maxncf;
  double *ewt = NV_DATA_S(cv_mem->cv_ewt);
  double *acor = NV_DATA_S(cv_mem->cv_acor);
  double *tempv = NV_DATA_S(cv_mem->cv_tempv);
  double *ftemp = NV_DATA_S(cv_mem->cv_ftemp);
  double *cv_last_yn = N_VGetArrayPointer(cv_mem->cv_last_yn);
  double *cv_acor_init = N_VGetArrayPointer(cv_mem->cv_acor_init);
  double *youtArray = N_VGetArrayPointer(yout);
  double *cv_Vabstol = N_VGetArrayPointer(cv_mem->cv_Vabstol);
  cudaMemcpyAsync(mGPU->state,md->total_state,md->n_per_cell_state_var*mGPU->n_cells*sizeof(double),cudaMemcpyHostToDevice,stream);
  cudaMemcpyAsync(mGPU->dewt, ewt, mGPU->nrows * sizeof(double), cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(mGPU->cv_acor, acor, mGPU->nrows * sizeof(double), cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(mGPU->dtempv, tempv, mGPU->nrows * sizeof(double), cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(mGPU->dftemp, ftemp, mGPU->nrows * sizeof(double), cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(mGPU->cv_last_yn, cv_last_yn, mGPU->nrows * sizeof(double), cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(mGPU->cv_acor_init, cv_acor_init, mGPU->nrows * sizeof(double), cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(mGPU->yout, youtArray, mGPU->nrows * sizeof(double), cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(mGPU->cv_Vabstol, cv_Vabstol, mGPU->nrows * sizeof(double), cudaMemcpyHostToDevice, stream);
  for (int i = 0; i <= cv_mem->cv_qmax; i++) {
    double *zn = NV_DATA_S(cv_mem->cv_zn[i]);
    cudaMemcpyAsync((mGPU->dzn + i * mGPU->nrows), zn, mGPU->nrows * sizeof(double), cudaMemcpyHostToDevice, stream);
  }
  cudaMemcpyAsync(mGPU->flagCells, sd->flagCells, mGPU->n_cells * sizeof(int), cudaMemcpyHostToDevice,
                  stream);
  mGPU->cv_tstop = cv_mem->cv_tstop;
  mGPU->cv_tstopset = cv_mem->cv_tstopset;
  mGPU->use_deriv_est = sd->use_deriv_est;
  mGPU->cv_nlscoef = cv_mem->cv_nlscoef;
  mGPU->init_time_step = sd->init_time_step;
  mGPU->cv_mxstep = cv_mem->cv_mxstep;
  mGPU->cv_uround = cv_mem->cv_uround;
  mGPU->cv_hmax_inv = cv_mem->cv_hmax_inv;
  mGPU->cv_reltol = cv_mem->cv_reltol;
  mGPU->cv_maxcor = cv_mem->cv_maxcor;
  mGPU->cv_qmax = cv_mem->cv_qmax;
  mGPU->cv_maxnef = cv_mem->cv_maxnef;
  mGPU->tout = tout;
  for (int i = 0; i < mGPU->n_cells; i++) {
    cudaMemcpyAsync(mGPU->cv_l + i * L_MAX, cv_mem->cv_l, L_MAX * sizeof(double), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(mGPU->cv_tau + i * (L_MAX + 1), cv_mem->cv_tau, (L_MAX + 1) * sizeof(double),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(mGPU->cv_tq + i * (NUM_TESTS + 1), cv_mem->cv_tq, (NUM_TESTS + 1) * sizeof(double),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(&mGPU->sCells[i], &mCPU->mdvCPU, sizeof(ModelDataVariable), cudaMemcpyHostToDevice, stream);
  }
  cvodeRun(mGPU,stream);
  cudaMemcpyAsync(cv_acor_init, mGPU->cv_acor_init, mGPU->nrows * sizeof(double), cudaMemcpyDeviceToHost, stream);
  cudaMemcpyAsync(youtArray, mGPU->yout, mGPU->nrows * sizeof(double), cudaMemcpyDeviceToHost, stream);
  for (int i = 0; i <= cv_mem->cv_qmax; i++) {
    double *zn = NV_DATA_S(cv_mem->cv_zn[i]);
    cudaMemcpyAsync(zn, (i * mGPU->nrows + mGPU->dzn), mGPU->nrows * sizeof(double), cudaMemcpyDeviceToHost, stream);
  }
  cudaMemcpyAsync(sd->flagCells, mGPU->flagCells, mGPU->n_cells * sizeof(int), cudaMemcpyDeviceToHost, stream);
  mGPU = sd->mGPU;
  cudaDeviceSynchronize();
#ifdef CAMP_DEBUG_GPU
    cudaEventRecord(sd->stopcvStep);
    cudaEventSynchronize(sd->stopcvStep);
    float mscvStep = 0.0;
    cudaEventElapsedTime(&mscvStep, sd->startcvStep, sd->stopcvStep);
    cv_mem->timecvStep+= mscvStep/1000;
#ifdef CAMP_PROFILE_DEVICE_FUNCTIONS
    cudaMemcpy(&mCPU->mdvCPU, mGPU->mdvo, sizeof(ModelDataVariable), cudaMemcpyDeviceToHost);
#endif
#endif
  istate = CV_SUCCESS;
  for (int i = 0; i < mGPU->n_cells; i++) {
    if (sd->flagCells[i] != CV_SUCCESS) {
      istate = sd->flagCells[i];
      int rank;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      printf("cudaCVode2 kflag %d cell %d rank %d\n",istate,i,rank);
      istate = cvHandleFailure_gpu(cv_mem, istate);
    }
  }

#ifdef DEV_CPU_GPU
dev_cpu:
  printf("n_cells=0\n");
#endif
  return(istate);
}

void solver_get_statistics_gpu(SolverData *sd){
  ModelDataGPU *mGPU = sd->mGPU;
  ModelDataCPU *mCPU = &(sd->mCPU);
  cudaMemcpy(&mCPU->mdvCPU,mGPU->mdvo,sizeof(ModelDataVariable),cudaMemcpyDeviceToHost);
}
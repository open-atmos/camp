/*
 * Copyright (C) 2022 Barcelona Supercomputing Center and University of
 * Illinois at Urbana-Champaign
 * SPDX-License-Identifier: MIT
 */

#ifndef CAMP_CUDA_STRUCTS_D2_H
#define CAMP_CUDA_STRUCTS_D2_H

typedef struct
{
  //double* A;
  //int*    jA;
  //int*    iA;
  double* aux;

#ifdef CAMP_DEBUG_GPU
  int countercvStep;
  double timecvStep;
  cudaEvent_t startcvStep;
  cudaEvent_t stopcvStep;
#endif

} ModelDataCPU_d2;

typedef struct {
    unsigned int num_spec;          // Number of species in the derivative
    // long double is treated as double in GPU
    double *production_rates;  // Production rates for all species
    double *loss_rates;        // Loss rates for all species
#ifdef CAMP_DEBUG
    double last_max_loss_precision;  // Maximum loss of precision at last output
#endif

} TimeDerivativeGPU_d2;

#ifndef DEF_JAC_MAP
#define DEF_JAC_MAP
typedef struct {
    int solver_id;  // solver Jacobian id
    int rxn_id;     // reaction Jacobian id
    int param_id;   // sub model Jacobian id
} JacMap;
#endif

typedef struct {
#ifdef __CUDA_ARCH__
#else
#endif
#ifdef DEV_JACOBIANGPUNUMSPEC
    int num_spec;   // Number of species
#endif
    //unsigned int num_elem;   // Number of potentially non-zero Jacobian elements
    int *num_elem;   // Number of potentially non-zero Jacobian elements
    //unsigned int *row_ids;   // Row id of each Jacobian element in data array
    double *production_partials;    // Data array for productions rate partial derivs
    double *loss_partials;  // Data array for loss rate partial derivs
    //JacobianColumnElements *elements;  // Jacobian elements flagged for inclusion
} JacobianGPU_d2;

typedef struct {

    int flag;
    int nflag;
    int kflag;
    int kflag2;
    int eflag;

    //f & jac
    int i_cell;
    int i_rxn;
    int i_aero_rep;

    double init_time_step;
    int cv_mxstep;
    int cv_next_q;
    double tout;
    int cv_taskc;
    double cv_uround;
    int cv_nrtfn;
    int nstloc;
    double tret;
    double cv_tretlast;
    int istate;
    double cv_hmax_inv;
    int cv_lmm;                /* lmm = CV_ADAMS or CV_BDF                      */
    int cv_iter;               /* iter = CV_FUNCTIONAL or CV_NEWTON             */
    int cv_itol;               /* itol = CV_SS, CV_SV, CV_WF, CV_NN             */
    double cv_reltol;        /* relative tolerance                            */
    int cv_nhnil;            /* number of messages issued to the user that t + h == t for the next iternal step            */
    double cv_etaqm1;      /* ratio of new to old h for order q-1             */
    double cv_etaq;        /* ratio of new to old h for order q               */
    double cv_etaqp1;      /* ratio of new to old h for order q+1             */
    int cv_lrw1;        /* no. of realtype words in 1 N_Vector             */
    int cv_liw1;        /* no. of integer words in 1 N_Vector              */
    int cv_lrw;             /* no. of realtype words in CVODE work vectors     */
    int cv_liw;             /* no. of integer words in CVODE work vectors      */
    double cv_saved_tq5;       /* saved value of tq[5]                        */
    double cv_tolsf;           /* tolerance scale factor                      */
    int cv_qmax_alloc;           /* value of qmax used when allocating memory   */
    int cv_indx_acor;            /* index of the zn vector with saved acor      */
    int cv_qu;
    double cv_h0u;
    double cv_hu;
    int cv_jcur;
    int cv_mnewt;
    int cv_maxcor;
    int cv_nstlp;
    int cv_qmax;
    int cv_L;
    int cv_maxnef;
    int cv_netf;
    double cv_acnrm;
    double cv_tstop;
    int cv_tstopset; //Used as bool
    double cv_nlscoef;
    int cv_qwait;
    double cv_crate;
    double cv_gamrat;
    double cv_gammap;
    int cv_nst;
    double cv_gamma;
    double cv_rl1;
    double cv_eta;
    int cv_q;
    int cv_qprime;
    double cv_h;
    double cv_next_h;
    double cv_hscale;
    int cv_nscon;
    double saved_t;
    int ncf;
    int nef;
    double cv_hprime;
    double cv_hmin;
    double cv_tn;
    double cv_etamax;
    int cv_maxncf;

    //Counters (e.g. iterations of function cvnlsNewton)
    int cv_nsetups;
    int cv_nfe;
    int nje;
    int nstlj;
    int cv_ncfn;

#ifdef CAMP_DEBUG_GPU
#ifdef CAMP_PROFILE_DEVICE_FUNCTIONS
    int countercvStep;
    int counterDerivGPU;
    int counterBCGInternal;
    int counterBCG;

    double dtBCG;
    double dtcudaDeviceCVode;
    double dtPostBCG;
#endif
#endif
}ModelDataVariable_d2; //variables to pass between gpu and cpu (different data between cells)

typedef struct {

  //cudacvNewtonIteration
  double* cv_acor;
  double* dzn;
  double* dewt;

#ifndef DEV_NEEDED_VARIABLES
  //LS (BCG)
  double *dA;
  int *djA;
  int *diA;
  double *dx;
  double* dtempv;
  int nrows;
  int n_shr_empty;
  int maxIt;
  int n_cells;
  double tolmax;
  double *ddiag;
  double *dr0;
  double *dr0h;
  double *dn0;
  double *dp0;
  double *dt;
  double *ds;
  double *dy;
#endif

  //itsolver
  double* A;
  int*    jA;
  int*    iA;
  double* aux;

//Guess_helper
double* dftemp;
double* dcv_y;
//double* dtempv1;
double* dtempv2;

//CVODE
double *cv_l;
double *cv_tau;
double *cv_tq;//NUM_TESTS+1
double* dsavedJ;

//from CAMP
  int threads,blocks;
  int nnz;
  int *map_state_deriv;
  double *deriv_data;
  double *J;
  double *J_solver;
  int *jJ_solver;
  int *iJ_solver;
  double *J_state;
  double *J_deriv;
  double *J_tmp;
  double *J_tmp2;
  int *indexvals;
  int *indexptrs;
  int *rxn_int;
  double *rxn_double;
  double *state;
  double *env;
  double *rxn_env_data;
  int *rxn_env_data_idx;

  double *production_rates;
  double *loss_rates;

#ifdef REVERSE_INT_FLOAT_MATRIX
#else
  int *rxn_int_indices;
  int *rxn_float_indices;
#endif

  int n_rxn;
  int n_rxn_env_data;

  int *n_mapped_values;
  JacMap *jac_map;
  JacobianGPU jac;

  double *yout;
  double *cv_Vabstol;

  //
  size_t deriv_size;
  size_t jac_size;
  size_t state_size;
  size_t env_size;
  size_t rxn_env_data_size;
  size_t rxn_env_data_idx_size;
  size_t map_state_deriv_size;
  int max_n_gpu_thread;
  int max_n_gpu_blocks;
  int *map_state_derivCPU;

  //f_cuda
  int deriv_length_cell;
  int state_size_cell;

  //jac
  int nnz_J_solver;
  int nrows_J_solver;
  double *jac_aux;
  int *indexvals_gpu;
  int *indexptrs_gpu;

//ODE stats
#ifdef CAMP_DEBUG_GPU
#ifdef CAMP_PROFILE_DEVICE_FUNCTIONS
    int clock_khz;
    double *tguessNewton;
    double *dtNewtonIteration;
    double *dtJac;
    double *dtlinsolsetup;
    double *dtcalc_Jac;
    double *dtRXNJac;
    double *dtf;
    double *dtguess_helper;
    double *dtBCG;
    double *dtcudaDeviceCVode;
    double *dtPostBCG;
#endif
#endif

} ModelDataGPU_d2; //GPU structs

#endif //CAMP_CUDA_STRUCTS_D2_H

/* Copyright (C) 2021 Barcelona Supercomputing Center and University of
 * Illinois at Urbana-Champaign
 * SPDX-License-Identifier: MIT
 */

#ifndef CAMPGPU_CUDA_STRUCTS_H
#define CAMPGPU_CUDA_STRUCTS_H

typedef struct {
    unsigned int num_spec;          // Number of species in the derivative
    // long double is treated as double in GPU
    double *production_rates;  // Production rates for all species
    double *loss_rates;        // Loss rates for all species
#ifdef CAMP_DEBUG
    double last_max_loss_precision;  // Maximum loss of precision at last output
#endif
} TimeDerivativeGPU;

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
    int *num_elem;   // Number of potentially non-zero Jacobian elements
    double *production_partials;    // Data array for productions rate partial derivs
    double *loss_partials;  // Data array for loss rate partial derivs
} JacobianGPU;

typedef struct {
    int cv_next_q;
    int nstloc;
    double tret;
    double cv_tretlast;
#ifdef ODE_WARNING
    int cv_nhnil;            /* number of messages issued to the user that t + h == t for the next iternal step            */
#endif
    double cv_etaqm1;      /* ratio of new to old h for order q-1             */
    double cv_etaq;        /* ratio of new to old h for order q               */
    double cv_etaqp1;      /* ratio of new to old h for order q+1             */
    double cv_saved_tq5;       /* saved value of tq[5]                        */
    double cv_tolsf;           /* tolerance scale factor                      */
    int cv_indx_acor;            /* index of the zn vector with saved acor      */
    double cv_hu;
    int cv_jcur;
    int cv_nstlp;
    int cv_L;
    double cv_acnrm;
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
    double cv_hprime;
    double cv_hmin;
    double cv_tn;
    double cv_etamax;
    int cv_maxncf;
    //Counters (e.g. iterations of function cvnlsNewton)
    int nstlj;
#ifdef CAMP_DEBUG_GPU
#ifndef CAMP_PROFILE_DEVICE_FUNCTIONS
    int countercvStep;
    int counterBCGInternal;
    int counterBCG;
    double timeNewtonIteration;
    double timeJac;
    double timelinsolsetup;
    double timecalc_Jac;
    double timef;
    double timeguess_helper;
    double dtBCG;
    double dtcudaDeviceCVode;
    double dtPostBCG;
#endif
#endif
}ModelDataVariable; //variables to pass between gpu and cpu (different data between cells)

typedef struct{
  int*    jA;
  int*    iA;
  double* aux;
  int cells_method;
  int threads,blocks;
  int nnz_J_solver;
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
  ModelDataVariable mdvCPU; //cpu equivalent to gpu
  cudaStream_t *streams;
#ifdef CAMP_DEBUG_GPU
  int counterNewtonIt;
  int counterLinSolSetup;
  int counterLinSolSolve;
  int countercvStep;
  int counterDerivNewton;
  int counterBCG;
  int counterDerivSolve;
  int countersolveCVODEGPU;

  double timeNewtonIt;
  double timeLinSolSetup;
  double timeLinSolSolve;
  double timeDerivNewton;
  double timeBiConjGrad;
  double timeBiConjGradMemcpy;
  double timeDerivSolve;
  double timeJac;

  cudaEvent_t startDerivNewton;
  cudaEvent_t startDerivSolve;
  cudaEvent_t startLinSolSetup;
  cudaEvent_t startLinSolSolve;
  cudaEvent_t startNewtonIt;
  cudaEvent_t startBCG;
  cudaEvent_t startBCGMemcpy;

  cudaEvent_t stopDerivNewton;
  cudaEvent_t stopDerivSolve;
  cudaEvent_t stopLinSolSetup;
  cudaEvent_t stopLinSolSolve;
  cudaEvent_t stopNewtonIt;
  cudaEvent_t stopBCGMemcpy;
  cudaEvent_t stopBCG;

  double timecvStep;
  cudaEvent_t startcvStep;
  cudaEvent_t stopcvStep;
#endif
} ModelDataCPU;

typedef struct {

    //CPU (Needed because each GPU points a different CPU pointer
    double* A;

    //Allocated from CPU (used during CPU / need some cudamemcpy)
    int *map_state_deriv;
    double *deriv_data;
    double *J_solver;
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
    int *rxn_int_indices;
    int *rxn_float_indices;
    int n_rxn;
    int n_rxn_env_data;
    int *n_mapped_values;
    JacMap *jac_map;
    JacobianGPU jac;

    double *yout;
    double *cv_Vabstol;
    double *grid_cell_state;
    double *grid_cell_env;
    double *grid_cell_aero_rep_env_data;
    double *cv_l;
    double *cv_tau;
    double *cv_tq;//NUM_TESTS+1

    //CVODE variables only GPU
    double *cv_last_yn;
    double *cv_acor_init;

    //LS (BCG)
    double *dA;
    int *djA;
    int *diA;
    double *dx;
    double* dtempv;
    int nrows;
    int nnz;
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
    double *dAx2;
    double *dy;
    double *dz;

    //Guess_helper
    double* dftemp;
    double* dcv_y;
    double* dtempv1;
    double* dtempv2;

    //update_state
    double threshhold;
    double replacement_value;
    int *flag;
    int *flagCells;
    //f_cuda
    int state_size_cell;
    //cudacvNewtonIteration
    double* cv_acor;
    double* dzn;
    double* dewt;
    //Auxiliar variables
    double* dsavedJ;
#ifdef DEV_CSR_REACTIONS
    int *colARXN;
    int *jARXN;
    int *iARXN;
#endif
    ModelDataVariable *mdv; //device
    ModelDataVariable *mdvo; //out device
    ModelDataVariable *s;
    ModelDataVariable *sCells;
    //Constant during solving
    double init_time_step;
    int cv_mxstep;
    double tout;
    double cv_uround;
    double cv_hmax_inv;
    double cv_reltol;
    int cv_maxcor;
    int cv_qmax;
    int cv_maxnef;
    double cv_tstop;
    int cv_tstopset; //Used as bool
    double cv_nlscoef;
//ODE stats
#ifdef CAMP_DEBUG_GPU
#ifndef CAMP_PROFILE_DEVICE_FUNCTIONS
    int clock_khz;
#endif
#endif
} ModelDataGPU; //CPU and GPU structs

#endif //CAMPGPU_CUDA_STRUCTS_H



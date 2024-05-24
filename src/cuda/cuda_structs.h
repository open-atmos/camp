/* Copyright (C) 2021 Barcelona Supercomputing Center and University of
 * Illinois at Urbana-Champaign
 * SPDX-License-Identifier: MIT
 */

#ifndef CAMPGPU_CUDA_STRUCTS_H
#define CAMPGPU_CUDA_STRUCTS_H

typedef struct {
    unsigned int num_spec;          // Number of species in the derivative
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
    int *num_elem;   // Number of potentially non-zero Jacobian elements
    double *production_partials;    // Data array for productions rate partial derivs
    double *loss_partials;  // Data array for loss rate partial derivs
} JacobianGPU;

typedef struct {
    int nstloc;
    double tret;
    double cv_tretlast;
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
    double *grid_cell_state;
    int nstlj;
    int cv_nst;
#ifdef CAMP_PROFILE_DEVICE_FUNCTIONS
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
}ModelDataVariable; //variables to pass between gpu and cpu (different data between cells)

typedef struct{
  ModelDataVariable mdvCPU; // WARNING: Moving this to struct "sd" cause errors when running gpu version
} ModelDataCPU;

typedef struct {
    int *map_state_deriv;
    double *J_solver;
    double *J_state;
    double *J_deriv;
    double *J_tmp;
    int *indexvals;
    int *indexptrs;
    int *rxn_int;
    double *rxn_double;
    double *state;
    double *env;
    double *rxn_env_data;
    int *rxn_env_idx;
    double *production_rates;
    double *loss_rates;
    int *rxn_int_indices;
    int *rxn_float_indices;
    double *grid_cell_state;
    int n_rxn;
    int n_rxn_env_data;
    int *n_mapped_values;
    JacMap *jac_map;
    JacobianGPU jac;
    double *yout;
    double *cv_Vabstol;
    double *cv_l;
    double *cv_tau;
    double *cv_tq;
    double *cv_last_yn;
    double *cv_acor_init;
    double *dA;
    int *djA;
    int *diA;
    double *dx;
    double* dtempv;
    int nrows;
    int n_shr_empty;
    int n_cells;
    double *ddiag;
    double *dr0;
    double *dr0h;
    double *dn0;
    double *dp0;
    double *dt;
    double *ds;
    double *dy;
    double *dz;
    double* dftemp;     //Guess_helper
    double* dcv_y;
    double* dtempv1;
    double* dtempv2;
    int *flagCells;
    int n_per_cell_state_var;
    double* cv_acor;     //cudacvNewtonIteration
    double* dzn;
    double* dewt;
    double* dsavedJ;    //Auxiliar variables
    ModelDataVariable *mdv; //device
    ModelDataVariable *sCells;
    double init_time_step;     //Constant during solving
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
    int use_deriv_est; //Used as bool
//ODE stats
#ifdef CAMP_PROFILE_SOLVING
#ifdef CAMP_PROFILE_DEVICE_FUNCTIONS
    int clock_khz;
    ModelDataVariable *mdvo; //out device
#endif
#endif
} ModelDataGPU; //CPU and GPU structs

#endif //CAMPGPU_CUDA_STRUCTS_H



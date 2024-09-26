/* Copyright (C) 2021 Barcelona Supercomputing Center and University of
 * Illinois at Urbana-Champaign
 * SPDX-License-Identifier: MIT
 */

#ifndef CAMPGPU_CUDA_STRUCTS_H
#define CAMPGPU_CUDA_STRUCTS_H

/* Time derivative for solver species */
typedef struct {
  double *production_rates;  // Production rates for all species
  double *loss_rates;        // Loss rates for all species
} TimeDerivativeGPU;

#ifndef DEF_JAC_MAP
#define DEF_JAC_MAP
/* Jacobian map */
typedef struct {
  int solver_id;  // solver Jacobian id
  int rxn_id;     // reaction Jacobian id
  int param_id;   // sub model Jacobian id
} JacMap;
#endif

/* Jacobian for solver species */
typedef struct {
  int *num_elem;  // Number of potentially non-zero Jacobian elements
  double
      *production_partials;  // Data array for productions rate partial derivs
  double *loss_partials;     // Data array for loss rate partial derivs
} JacobianGPU;

// Structure representing variables for each cell in the model.
typedef struct {
  // Variables extracted from CVODE library
  double cv_saved_tq5;
  double cv_hu;             // last successful h value used
  int cv_jcur;              // is Jacobian info. for lin. solver current?
  int cv_nstlp;             // step number of last setup call
  int cv_L;                 // L = q + 1
  double cv_acnrm;          // | acor | wrms
  int cv_qwait;             // number of internal steps to wait before
                            // considering a change in q
  double cv_crate;          // estimated corrector convergence rate
  double cv_gamrat;         // gamma / gammap
  double cv_gammap;         // gamma at the last
  double cv_gamma;          // gamma = h * rl1
  double cv_rl1;            // the scalar 1/l[1]
  double cv_eta;            // eta = hprime / h
  int cv_q;                 // current order
  int cv_qprime;            // order to be used on the next step
                            //  = q-1, q, or q+1
  double cv_h;              // current step size
  double cv_next_h;         // step size to be used on the next step
  double cv_hscale;         // value of h used in zn
  double cv_hprime;         // step size to be used on the next step
  double cv_hmin;           // |h| >= hmin
  double cv_tn;             // current internal value of t
  double cv_etamax;         // eta <= etamax
  int cv_nst;               // number of internal steps taken
  int nstlj;                // nstlj = nst at last Jacobian eval.
  double *grid_cell_state;  // Pointer to the current grid cell being solved
                            // on the total_state array
#ifdef CAMP_PROFILE_DEVICE_FUNCTIONS
  // Metrics inside the kernel execution, useful to for example take the
  // percentage of time spent on each part of the code and find the most time
  // consuming functions
  int countercvStep;  // Number of steps of the ODE solver (most external loop
                      // of the solver)
  int counterBCGInternal;      // Number of internal steps of the BCG solver
  int counterBCG;              // Number of calls to the BCG solver
  double timeNewtonIteration;  // Time spent in the Newton iteration
  double timeJac;              // Time spent in the Jacobian calculation
  double timelinsolsetup;      // Time spent in the linear solver setup
  double timecalc_Jac;         // Time spent in the calculation of the Jacobian
  double timef;                // Time spent in the function evaluation
  double timeguess_helper;     // Time spent in the guess helper
  double timeBCG;              // Time spent in the BCG solver
  double timeDeviceCVode;      // Time spent in the ODE solver
#endif
} ModelDataVariable;  // variables for each cell

/*
 * Auxiliary struct for transferring cell data
 * between the CPU and GPU.
 *
 * WARNING: Moving this structure  to the struct "sd" can cause errors
 */
typedef struct {
  ModelDataVariable mdvCPU;
} ModelDataCPU;

typedef struct {
  int *map_state_deriv;       //
  double *J_solver;           //
  double *J_state;            //
  double *J_deriv;            //
  int *rxn_int;               //
  double *rxn_double;         //
  double *state;              //
  double *rxn_env_data;       //
  int *rxn_env_idx;           //
  double *production_rates;   //
  double *loss_rates;         //
  int *rxn_int_indices;       //
  int *rxn_float_indices;     //
  double *grid_cell_state;    //
  int n_rxn;                  //
  int n_rxn_env_data;         //
  int *n_mapped_values;       //
  JacMap *jac_map;            //
  JacobianGPU jac;            //
  double *yout;               //
  double *cv_Vabstol;         //
  double *cv_l;               //
  double *cv_tau;             //
  double *cv_tq;              //
  double *cv_last_yn;         //
  double *cv_acor_init;       //
  double *dA;                 //
  int *djA;                   //
  int *diA;                   //
  double *dx;                 //
  double *dtempv;             //
  int n_shr_empty;            //
  double *ddiag;              //
  double *dr0;                //
  double *dr0h;               //
  double *dn0;                //
  double *dp0;                //
  double *dt;                 //
  double *ds;                 //
  double *dy;                 //
  double *dftemp;             //
  double *dcv_y;              //
  double *dtempv1;            //
  int n_per_cell_state_var;   //
  double *cv_acor;            //
  double **dzn;               //
  double *dewt;               //
  double *dsavedJ;            //
  ModelDataVariable *sCells;  //
  double init_time_step;      //
  double tout;                //
  double cv_reltol;           //
#ifdef DEBUG_SOLVER_FAILURES
  int *flags;  // Error failures on solving
#endif
#ifdef PROFILE_SOLVING
#ifdef CAMP_PROFILE_DEVICE_FUNCTIONS
  int clock_khz;            // Clock frequency
  ModelDataVariable *mdvo;  // out device for time measurement
#endif
#endif
} ModelDataGPU;

#endif  // CAMPGPU_CUDA_STRUCTS_H
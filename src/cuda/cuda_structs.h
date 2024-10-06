/* Copyright (C) 2021 Barcelona Supercomputing Center and University of
 * Illinois at Urbana-Champaign
 * SPDX-License-Identifier: MIT
 */

#ifndef CAMPGPU_CUDA_STRUCTS_H
#define CAMPGPU_CUDA_STRUCTS_H

/* Time derivative of solver species for each cell */
typedef struct {
  double *production_rates;  // Production rates
  double *loss_rates;        // Loss rates
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

// Variables for each cell in the model.
typedef struct {
  // Variables extracted from CVODE library
  double cv_saved_tq5;  // saved value of tq[5]
  double cv_acnrm;      // | acor | wrms
  double cv_eta;        // eta = hprime / h
  double cv_hmin;       // |h| >= hmin
  double cv_hu;         // last successful h value used
  int cv_jcur;   // flag indicating if Jacobian info. for lin. solver is current
  int cv_nstlp;  // step number of last setup call
  int cv_L;      // L = q + 1
  int cv_qwait;  // number of internal steps to wait before
                 // considering a change in q
  double cv_crate;   // estimated corrector convergence rate
  double cv_gamrat;  // gamma / gammap
  double cv_gammap;  // gamma at the last
  double cv_gamma;   // gamma = h * rl1
  double cv_rl1;     // the scalar 1/l[1]
  int cv_q;          // current order
  int cv_qprime;     // order to be used on the next step
                     //  = q-1, q, or q+1
  double cv_h;       // current step size
  double cv_next_h;  // step size to be used on the next step
  double cv_hscale;  // value of h used in zn
  double cv_hprime;  // step size to be used on the next step
  double cv_tn;      // current internal value of t
  double cv_etamax;  // eta <= etamax
  int cv_nst;        // number of internal steps taken
  int nstlj;         // nstlj = nst at last Jacobian eval.
  // Variables from CAMP chemical model
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
} ModelDataVariable;  // Variables for each cell

/*
 * Auxiliary struct for transferring cell data
 * between the CPU and GPU.
 *
 * WARNING: Moving this structure  to the struct "sd" can cause errors
 */
typedef struct {
  ModelDataVariable mdvCPU;
} ModelDataCPU;

// TODO: Group type variables together
typedef struct {
  // Parameters from CAMP chemical model
  double
      *state;  // Concentrations of species, including constant and non-constant
               // values. Used as input and output of the ODE solver
  // Derivatives = Concentrations of non-constant species. The ODE solver works
  // around this array, using auxiliary arrays of the same size.
  int *map_state_deriv;  // Map of state variables to derivatives
  double *dA;            // Jacobian values
  int *djA;  // Jacobian indices for equivalent dense matrix (for CSR are
             // columns, for CSC are rows)
  int *diA;  // Jacobian ranged indices (for CSR and CSC are used to mark the
             // end of elements in a row and column, respectively)
  double *J_solver;  // Auxiliar Jacobian used in derivative calculation, stored
                     // from Jacobian calculation
  double *J_state;   // Last state used to calculate the Jacobian
  double *J_deriv;   // Last derivative used to calculate the Jacobian
  JacobianGPU jac;   // Auxiliar structure to store positive and negative
                     // contributions to the Jacobian
  JacMap *jac_map;   // Mapping JacobianGPU to the solver Jacobian
  int *rxn_int;      // Pointer to the reaction integer parameters
  int *rxn_int_indices;    // Array of indices of integer data
  double *rxn_double;      // Pointer to the reaction floating-point parameters
  int *rxn_float_indices;  // Array of indices of float data
  double *
      rxn_env_data;  // Reaction environment-dependent parameters, i.e. Reaction
                     // parameters that are affected by environemntal variables
  int *rxn_env_idx;  // Mapping of the environment-dependent data and reaction
                     // types
  double *production_rates;  // Production rates of species
  double *loss_rates;        // Loss rates of species
  int n_per_cell_state_var;  // number of state variables per cell
  int n_rxn_env_data;        // Number of reaction environmental parameters
  int n_rxn;                 // Number of reactions
  double init_time_step;     // Initial time step (s)
  // Parameters for the ODE solver, extracted from CVODE library
  double cv_reltol;    // Relative tolerance
  double *cv_Vabstol;  // Vector absolute tolerance
  double *cv_l;        // L = q + 1 (q is the order)
  double *cv_tau;  // array of previous q+1 successful step sizes indexed from 1
                   // to q+1
  double *cv_tq;   // array of test quantities
  double *cv_last_yn;  // last solved value for y_n
  double *cv_acor;  // In the context of the solution of the nonlinear equation,
                    // acor = y_n(m) - y_n(0). On return, this vector is scaled
                    // to give the est. local err.
  double *cv_acor_init;  // Initial guess for acor
  double *yout;          // Solution vector, yout=y(tout)
  double *dcv_y;         // Temporary storage vector
  double *dtempv;        // Temporary storage vector
  double *dtempv1;       // Temporary storage vector
  double *dftemp;        // Temporary storage vector
  double *dewt;          // Error weight vector
  double *dsavedJ;       // Jacobian from previous step to avoid calculating the
                         // jacobian again
  double tout;           // Time (from y'(t)) reached by the solver at the end.
  double **dzn;  // Nordsieck array, of size N x (q+1). zn[j] is a vector of
                 // length N (j=0,...,q) zn[j] = [1/factorial(j)] * h^j * (jth
                 // derivative of the interpolating polynomial
  // Parameters for the BCG solver
  double *dx;     // Auxiliar vector of concentrations
  double *ddiag;  // Auxiliar vector
  double *dr0;    // Auxiliar vector
  double *dr0h;   // Auxiliar vector
  double *dn0;    // Auxiliar vector
  double *dp0;    // Auxiliar vector
  double *dt;     // Auxiliar vector
  double *ds;     // Auxiliar vector
  double *dy;     // Auxiliar vector
  // GPU parameters
  int n_shr_empty;  // Number of empty shared memory slots, used on "reduce"
                    // type operations to treat the input as a power of two
  ModelDataVariable *sCells;  // Variables for each cell in the model.
#ifndef DEBUG_SOLVER_FAILURES
  int *flags;  // Error failures on solving
#endif
#ifdef PROFILE_SOLVING
#ifdef CAMP_PROFILE_DEVICE_FUNCTIONS
  int clock_khz;  // Clock frequency
  ModelDataVariable
      *mdvo;  // Out device for time measurement. Warning: Long time without
              // using, it may fail or exist a better implementation
#endif
#endif
} ModelDataGPU;

#endif  // CAMPGPU_CUDA_STRUCTS_H
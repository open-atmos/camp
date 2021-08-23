

#ifndef CAMPGPU_CUDA_STRUCTS_H
#define CAMPGPU_CUDA_STRUCTS_H

//Move to proper classes instead of englobing all in a single file
typedef struct
{
  //Init variables ("public")
  int use_multicells;
  int threads,blocks;
  int maxIt;
  int mattype;
  int nrows;
  int nnz;
  int n_cells;
  double tolmax;
  double* ddiag;

  // Intermediate variables ("private")
  double * dr0;
  double * dr0h;
  double * dn0;
  double * dp0;
  double * dt;
  double * ds;
  double * dAx2;
  double * dy;
  double * dz;

  // Matrix data ("input")
  double* A;
  int*    jA;
  int*    iA;

  //GPU pointers ("input")
  double* dA;
  int*    djA;
  int*    diA;
  double* dB;
  int*    djB;
  int*    diB;
  double* dx;
  double* aux;
  double* daux;

  // ODE solver variables
  int flag;
  int callSetup;//todo remove
  int convfail;
  int *dflag;
  int *dlast_flag;
  int *cv_jcur;
  int *nje;
  int *nstlj;
  int *cv_nst;
  int *jok;
  int *cv_nsetups;
  int *cv_nfe;

  double *dgammap;
  double *dcv_tq;
  double* dewt;
  double* dacor;
  double* dacor_init;
  double* dtempv;
  double* dtempv1;
  double* dtempv2;
  double* dftemp;
  double* dzn;
  double* dcv_y;

//#ifdef DEBUG_CudaDeviceguess_helper
  double* cv_zn;
  double* cv_last_yn;
  double* cv_ftemp;
  double* cv_tempv;
  double* cv_acor_init;
  double* total_state;
//#endif


    //Intermediate variables
  double* dsavedJ;
  int*    djsavedJ;
  int*    disavedJ;


  //ODE stats

#ifdef PMC_DEBUG_GPU
  int counterSendInit;
  int counterMatScaleAddI;
  int counterMatScaleAddISendA;
  int counterMatCopy;
  int counterprecvStep;
  int counterNewtonIt;
  int counterLinSolSetup;
  int counterLinSolSolve;
  int countercvStep;
  int counterDerivNewton;
  int counterBiConjGrad;
  int counterBiConjGradInternal;
  int *counterBiConjGradInternalGPU;
  int counterDerivSolve;
  int counterJac;
  int countersolveCVODEGPU;

  double timeNewtonSendInit;
  double timeMatScaleAddI;
  double timeMatScaleAddISendA;
  double timeMatCopy;
  double timeprecvStep;
  double timeNewtonIt;
  double timeLinSolSetup;
  double timeLinSolSolve;
  double timecvStep;
  double timeDerivNewton;
  double timeBiConjGrad;
  double timeBiConjGradMemcpy;
  double timeDerivSolve;
  double timeJac;
  double timesolveCVODEGPU;
#ifdef cudaGlobalSolveODE_timers_max_blocks
  double *dtBCG;
  double *dtPreBCG;
  double *dtPostBCG;
#else
  double dtBCG;
  double dtPreBCG;
  double dtPostBCG;
#endif

  cudaEvent_t startDerivNewton;
  cudaEvent_t startDerivSolve;
  cudaEvent_t startLinSolSetup;
  cudaEvent_t startLinSolSolve;
  cudaEvent_t startNewtonIt;
  cudaEvent_t startcvStep;
  cudaEvent_t startBCG;
  cudaEvent_t startBCGMemcpy;
  cudaEvent_t startJac;
  cudaEvent_t startsolveCVODEGPU;

  cudaEvent_t stopDerivNewton;
  cudaEvent_t stopDerivSolve;
  cudaEvent_t stopLinSolSetup;
  cudaEvent_t stopLinSolSolve;
  cudaEvent_t stopNewtonIt;
  cudaEvent_t stopcvStep;
  cudaEvent_t stopBCGMemcpy;
  cudaEvent_t stopBCG;
  cudaEvent_t stopsolveCVODEGPU;

  cudaEvent_t stopJac;

#endif

} itsolver;

#endif //CAMPGPU_CUDA_STRUCTS_H

//Time derivative for solver species
typedef struct {
    unsigned int num_spec;          // Number of species in the derivative
    // long double is treated as double in GPU
    double *production_rates;  // Production rates for all species
    double *loss_rates;        // Loss rates for all species
#ifdef PMC_DEBUG
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

/* Registered elements for a column in the Jacobian */
/*typedef struct {
    unsigned int array_size;  // Size of the array of flagged elements
    unsigned int
            number_of_elements;  // Number of registered elements in the column
    unsigned int
            *row_ids;  // Array of row ids for each registered element in the column
} JacobianColumnElements;*/

/* Jacobian for solver species */
typedef struct {
#ifdef __CUDA_ARCH__
#else
#endif
#ifdef DEV_JACOBIANGPUNUMSPEC
    int num_spec;   // Number of species
#endif
    //unsigned int num_elem;   // Number of potentially non-zero Jacobian elements
    int *num_elem;   // Number of potentially non-zero Jacobian elements
    unsigned int *col_ptrs;  // Index of start/end of each column in data array
    //unsigned int *row_ids;   // Row id of each Jacobian element in data array
    double *production_partials;    // Data array for productions rate partial derivs
    double *loss_partials;  // Data array for loss rate partial derivs
    //JacobianColumnElements *elements;  // Jacobian elements flagged for inclusion
} JacobianGPU;

typedef struct {
    //double *deriv_data_gpu;

    //Allocated from CPU (used during CPU / need some cudamemcpy)
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

    int n_aero_phase;
    int n_added_aero_phases;
    int *aero_phase_int_indices;
    int *aero_phase_float_indices;
    int *aero_phase_int_data;
    double *aero_phase_float_data;

    int n_aero_rep;
    int n_added_aero_reps;
    int n_aero_rep_env_data;
    int *aero_rep_int_indices;
    int *aero_rep_float_indices;
    int *aero_rep_env_idx;
    int *aero_rep_int_data;
    double *aero_rep_float_data;
    double *aero_rep_env_data;

    int *n_mapped_values;
    double *J_rxn;
    //JacMap jac_map;
    JacMap *jac_map;
    JacobianGPU jac;
    //int n_per_cell_solver_jac_elem;

    //Allocated in GPU only
    int i_cell;
    int i_rxn;
    int i_aero_rep;

    double *grid_cell_state;
    double *grid_cell_env;
    double *grid_cell_aero_rep_env_data;

    //CVODE variables only GPU
    //double ;
    double cv_gamrat;
    double cv_crate;
    double cv_gamma;
    double cv_gammap;
    double cv_nstlp;
    double cv_rl1;
    int cv_mnewt;
    double cv_maxcor;
    double cv_acnrm;
    double *cv_last_yn;
    double *cv_acor_init;
    int nflag;
    int cv_jcur;
    double min;
    int convfail;
    int callSetup;

//ODE stats
#ifdef PMC_DEBUG_GPU
    int clock_khz;
    double *tguessNewton;
    double *dtNewtonIteration;
    double *dtJac;
    double *dtlinsolsetup;
    double *dtcalc_Jac;
    double *dtRXNJac;
    double *dtf;
    double *dtguess_helper;
#endif

} ModelDataGPU;



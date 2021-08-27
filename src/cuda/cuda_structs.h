

#ifndef CAMPGPU_CUDA_STRUCTS_H
#define CAMPGPU_CUDA_STRUCTS_H

//Move to proper classes instead of englobing all in a single file
typedef struct
{
  //Init variables ("public")
  int use_multicells;

  double* A;
  int*    jA;
  int*    iA;
  double* aux;

#ifdef PMC_DEBUG_GPU
  int counterNewtonIt;
  int counterLinSolSetup;
  int counterLinSolSolve;
  int countercvStep;
  int counterDerivNewton;
  int counterBiConjGrad;
  int counterDerivSolve;
  int counterJac;
  int countersolveCVODEGPU;

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

    //int *djA; //seems works fine using device ptr

    int flag;
    int nflag;//auxiliar flag (maybe delete after completing cudacvstep to device)
    int kflag;
    int eflag;

    int cv_jcur;
    int cv_mnewt;
    int cv_maxcor;
    int cv_nstlp;
    int cv_qmax;
    int cv_L;
    int cv_maxnef;
    int cv_netf;
    double cv_acnrm;
    double dsm;
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

#ifdef PMC_DEBUG_GPU
    int counterDerivGPU;
    int counterBCGInternal;
    double dtBCG;
    double dtPreBCG;
    double dtPostBCG;
#endif
}ModelDataVariable; //things to pass between gpu and cpu

typedef struct {
    //double *deriv_data_gpu;

    //Allocated from CPU (used during CPU / need some cudamemcpy)
    int threads,blocks;
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

    int nnz;

    //Allocated in GPU only
    int i_cell;
    int i_rxn;
    int i_aero_rep;

    double *grid_cell_state;
    double *grid_cell_env;
    double *grid_cell_aero_rep_env_data;

    double *cv_l;
    double *cv_tau;
    double *cv_tq;//NUM_TESTS+1

    //CVODE variables only GPU
    //double ;
    //double cv_nstlp;//wrong, should be int
    //double cv_maxcor;//wrong, should be int
    double saved_t;
    double *cv_last_yn;
    double *cv_acor_init;
    int nflag;
    double min;
    //int convfail;
    //int callSetup;
    int ncf;

    //LS (BCG)
    double *dA;
    int *djA;
    int *diA;
    double *dx;
    double* dtempv;
    int nrows;
    int n_shr_empty;
    int maxIt;
    int mattype;
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
    double *daux;

    //swapCSC_CSR_BCG
    double* dB;
    int*    djB;
    int*    diB;

    //Guess_helper
    //double t_n;
    double* dftemp;
    double* dcv_y;
    double* dtempv1;
    double* dtempv2;
    double cv_reltol;

    //update_state
    double threshhold;
    double replacement_value;
    int *flag;
    int *flagCells;

    //f_gpu
    double time_step;
    int deriv_length_cell;
    int state_size_cell;
    int i_kernel;
    int threads_block;

    //cudacvNewtonIteration
    double* dacor;
    double* dzn;
    double* dewt;
    int *cv_nsetups;
    int *nstlj;
    int *nje;

    //Auxiliar variables
    double* dsavedJ;
    int*    djsavedJ;
    int*    disavedJ;

    ModelDataVariable *mdv;
    ModelDataVariable *mdvo;

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
    double *dtBCG;
    double *dtPreBCG;
    double *dtPostBCG;
#endif

} ModelDataGPU;



//
// Created by Christian on 01/04/2020.
//

#ifndef CAMPGPU_CUDA_STRUCTS_H
#define CAMPGPU_CUDA_STRUCTS_H

//Move to proper classes instead of englobing all in a single file
typedef struct
{
  //Init variables ("public")
  int cells_method;
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
  double* dx;
  double* aux;
  double* daux;

  // ODE solver variables
  double* dewt;
  double* dacor;
  double* dacor_init;
  double* dtempv;
  double* dftemp;
  double* dzn;
  double* dcv_y;

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

  cudaEvent_t startDerivNewton;
  cudaEvent_t startDerivSolve;
  cudaEvent_t startLinSolSetup;
  cudaEvent_t startLinSolSolve;
  cudaEvent_t startNewtonIt;
  cudaEvent_t startcvStep;
  cudaEvent_t startBiConjGrad;
  cudaEvent_t startBiConjGradMemcpy;
  cudaEvent_t startJac;

  cudaEvent_t stopDerivNewton;
  cudaEvent_t stopDerivSolve;
  cudaEvent_t stopLinSolSetup;
  cudaEvent_t stopLinSolSolve;
  cudaEvent_t stopNewtonIt;
  cudaEvent_t stopcvStep;
  cudaEvent_t stopBiConjGradMemcpy;
  cudaEvent_t stopBiConjGrad;

  cudaEvent_t stopJac;

#endif

} itsolver;

#endif //CAMPGPU_CUDA_STRUCTS_H

//todo use default TimeDerivative class (long must change to double to match GPU case)
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

    //Allocated in GPU only
    int i_cell;
    int i_rxn;
    int i_aero_rep;

    double *grid_cell_state;
    double *grid_cell_env;
    double *grid_cell_aero_rep_env_data;


} ModelDataGPU;



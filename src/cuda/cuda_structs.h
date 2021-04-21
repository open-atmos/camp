//
// Created by Christian on 01/04/2020.
//

#ifndef CAMPGPU_CUDA_STRUCTS_H
#define CAMPGPU_CUDA_STRUCTS_H

typedef struct
{
  //Init variables ("public")
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
  double timeDerivSolve;
  double timeJac;

  cudaEvent_t startDerivNewton;
  cudaEvent_t startDerivSolve;
  cudaEvent_t startLinSolSetup;
  cudaEvent_t startLinSolSolve;
  cudaEvent_t startNewtonIt;
  cudaEvent_t startcvStep;
  cudaEvent_t startBiConjGrad;
  cudaEvent_t startJac;

  cudaEvent_t stopDerivNewton;
  cudaEvent_t stopDerivSolve;
  cudaEvent_t stopLinSolSetup;
  cudaEvent_t stopLinSolSolve;
  cudaEvent_t stopNewtonIt;
  cudaEvent_t stopcvStep;
  cudaEvent_t stopBiConjGrad;
  cudaEvent_t stopJac;

#endif

} itsolver;

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
    //double *deriv_gpu_data;

    //Allocated from CPU (used during CPU / need some cudamemcpy)
    int *map_state_deriv_gpu;
    double *deriv_gpu_data;
    double *J_gpu;
    double *J_solver_gpu;
    int *jJ_solver_gpu;
    int *iJ_solver_gpu;
    double *J_state_gpu;
    double *J_deriv_gpu;
    double *J_tmp_gpu;
    double *J_tmp2_gpu;
    int *indexvals_gpu;
    int *indexptrs_gpu;
    int *int_pointer_gpu;
    double *double_pointer_gpu;
    double *state_gpu;
    double *env_gpu;
    double *rxn_env_data_gpu;
    int *rxn_env_data_idx_gpu;
    double *prod_rates;
    double *loss_rates;

    int n_rxn_env_data;

    //Allocated in GPU (only on gpu)
    double *grid_cell_state;
    double *grid_cell_env;
    int n_rxn;

    int n_aero_phase;
    int n_added_aero_phases;
    int *aero_phase_int_data;
    int *aero_phase_int_indices;
    int *aero_phase_float_indices;
    double *aero_phase_float_data;

    int n_aero_rep;
    int n_added_aero_reps;
    int n_aero_rep_env_data;
    int *aero_rep_int_data;
    int *aero_rep_int_indices;
    int *aero_rep_float_indices;
    int *aero_rep_env_idx;
    double *aero_rep_float_data;
    double *grid_cell_aero_rep_env_data;
    double *aero_rep_env_data;

} ModelDataGPU;

#endif //CAMPGPU_CUDA_STRUCTS_H

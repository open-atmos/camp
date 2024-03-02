/* Copyright (C) 2021 Barcelona Supercomputing Center and University of
 * Illinois at Urbana-Champaign
 * SPDX-License-Identifier: MIT
 */

#include "itsolver_gpu.h"

extern "C" {
#include "camp_gpu_solver.h"
#include "rxns_gpu.h"
#include "aeros/aero_rep_gpu_solver.h"
#include "time_derivative_gpu.h"
#include "Jacobian_gpu.h"
}


// Reaction types (Must match parameters defined in camp_rxn_factory)
#define RXN_ARRHENIUS 1
#define RXN_TROE 2
#define RXN_CMAQ_H2O2 3
#define RXN_CMAQ_OH_HNO3 4
#define RXN_PHOTOLYSIS 5
#define RXN_HL_PHASE_TRANSFER 6
#define RXN_AQUEOUS_EQUILIBRIUM 7
#define RXN_SIMPOL_PHASE_TRANSFER 10
#define RXN_CONDENSED_PHASE_ARRHENIUS 11
#define RXN_FIRST_ORDER_LOSS 12
#define RXN_EMISSION 13
#define RXN_WET_DEPOSITION 14

#define STREAM_RXN_ENV_GPU 0
#define STREAM_ENV_GPU 1
#define STREAM_DERIV_GPU 2

// Status codes for calls to camp_solver functions
#define CAMP_SOLVER_SUCCESS 0
#define CAMP_SOLVER_FAIL 1

//GPU async stream related variables to ensure robustness
//int n_solver_objects=0; //Number of solver_new_gpu calls
//cudaStream_t *stream_gpu; //GPU streams to async computation/data movement
//int n_streams = 16;

static void HandleError(cudaError_t err,
                        const char *file,
                        int line) {
  if (err != cudaSuccess) {
    printf("%s in %s at line %d\n", cudaGetErrorString(err),
           file, line);
    exit(EXIT_FAILURE);
  }
}

/** \brief Allocate GPU solver variables
 *
 * \param n_dep_var number of solver variables per grid cell
 * \param n_state_var Number of variables on the state array per grid cell
 * \param n_rxn Number of reactions to include
 * \param n_rxn_int_param Total number of integer reaction parameters
 * \param n_rxn_float_param Total number of floating-point reaction parameters
 * \param n_cells Number of grid cells to solve simultaneously
 */
void solver_new_gpu_cu(SolverData *sd, int n_dep_var,
                       int n_state_var, int n_rxn,
                       int n_rxn_int_param, int n_rxn_float_param, int n_rxn_env_param,
                       int n_cells_total) {
  ModelData *md = &(sd->model_data);
  ModelDataGPU *mGPU;
  sd->mGPUs = (ModelDataGPU *)malloc(sd->nDevices * sizeof(ModelDataGPU));
  int remainder = n_cells_total % sd->nDevices;

  //sd->nDevices = 4;
  int nDevicesMax;
  cudaGetDeviceCount(&nDevicesMax);
  if(sd->nDevices > nDevicesMax){
     printf("ERROR: Not enough GPUs to launch, nDevices %d nDevicesMax %d\n"
           , sd->nDevices, nDevicesMax);
    exit(0);
  }

  for (int iDevice = 0; iDevice < sd->nDevices; iDevice++) {
  cudaSetDevice(iDevice);
  sd->mGPU = &(sd->mGPUs[iDevice]);
  mGPU = sd->mGPU;

  int n_cells = int(n_cells_total / sd->nDevices);
  if (remainder!=0 && iDevice==0 && n_cells_total != 1){
    //printf("WARNING:  PENDING TO CHECK THAT WORKS CASE: sd->nDevicesMODn_cells!=0\n");
    //printf("remainder %d n_cells_total %d nDevices %d n_cells %d\n",remainder,n_cells_total,sd->nDevices,n_cells);
    n_cells+=remainder;
  }

  mGPU->n_cells=n_cells;
  mGPU->state_size = n_state_var * n_cells * sizeof(double);
  mGPU->deriv_size = n_dep_var * n_cells * sizeof(double);
  mGPU->env_size = CAMP_NUM_ENV_PARAM_ * n_cells * sizeof(double); //Temp and pressure
  mGPU->rxn_env_data_size = n_rxn_env_param * n_cells * sizeof(double);
  mGPU->rxn_env_data_idx_size = (n_rxn+1) * sizeof(int);
  mGPU->map_state_deriv_size = n_dep_var * n_cells * sizeof(int);

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, iDevice);
  mGPU->max_n_gpu_thread = prop.maxThreadsPerBlock;
  mGPU->max_n_gpu_blocks = prop.maxGridSize[1];
  int n_blocks = (mGPU->deriv_size + mGPU->max_n_gpu_thread - 1) / mGPU->max_n_gpu_thread;

  if( n_blocks > mGPU->max_n_gpu_blocks){
    printf("\nWarning: More blocks assigned: %d than maximum block numbers: %d",
           n_blocks, mGPU->max_n_gpu_blocks);
  }

  HANDLE_ERROR(cudaMalloc((void **) &mGPU->deriv_data, mGPU->deriv_size));
  mGPU->n_rxn=md->n_rxn;
  mGPU->n_rxn_env_data=md->n_rxn_env_data;
  mGPU->n_aero_phase=md->n_aero_phase;
  mGPU->n_added_aero_phases=md->n_added_aero_phases;
  mGPU->n_aero_rep=md->n_added_aero_reps;
  mGPU->n_aero_rep_env_data=md->n_aero_rep_env_data;

  cudaMalloc((void **) &mGPU->state, mGPU->state_size);
  cudaMalloc((void **) &mGPU->env, mGPU->env_size);
  cudaMalloc((void **) &mGPU->rxn_env_data, mGPU->rxn_env_data_size);
  cudaMalloc((void **) &mGPU->rxn_env_data_idx, mGPU->rxn_env_data_idx_size);
  HANDLE_ERROR(cudaMalloc((void **) &mGPU->map_state_deriv, mGPU->map_state_deriv_size));

  time_derivative_initialize_gpu(sd);

  mGPU->map_state_derivCPU = (int *)malloc(mGPU->map_state_deriv_size);
  int i_dep_var = 0;
  for (int i_cell = 0; i_cell < n_cells; i_cell++) {
    for (int i_spec = 0; i_spec < n_state_var; i_spec++) {
      if (md->var_type[i_spec] == CHEM_SPEC_VARIABLE) {
        mGPU->map_state_derivCPU[i_dep_var] = i_spec + i_cell * n_state_var;
        //printf("%d %d, %d %d %d\n", mGPU->map_state_deriv_size/sizeof(int),
        //       mGPU->map_state_derivCPU[i_dep_var],n_state_var, i_spec, i_cell, i_dep_var);
        i_dep_var++;
      }
    }
  }

  HANDLE_ERROR(cudaMemcpy(mGPU->map_state_deriv, mGPU->map_state_derivCPU,
                          mGPU->map_state_deriv_size, cudaMemcpyHostToDevice));

  //printf("threads_per_block :%d\n", mGPU->max_n_gpu_thread);

  }

  if(n_dep_var<32 && sd->use_cpu==0) {
    printf("CAMP ERROR: TOO FEW SPECIES FOR GPU (Species < 32),"
           " use CPU case instead (More info: https://earth.bsc.es/gitlab/ac/camp/-/issues/49 \n");
    exit(0);
  }

#ifdef CAMP_DEBUG_PRINT_GPU_SPECS
  print_gpu_specs();
#endif

#ifdef CAMP_DEBUG_GPU

  md->timeDerivKernel=0.0;
  cudaEventCreate(&md->startDerivKernel);
  cudaEventCreate(&md->stopDerivKernel);

#endif

}

/** \brief Set reaction data on GPU prepared structure. RXN data is divided
 * into two different matrix, per double and int data respectively. Matrix are
 * reversed to improve memory access on GPU.
 *
 * \param md Pointer to the model data
 */

void set_reverse_int_double_rxn(
  int n_rxn, int rxn_env_data_idx_size,
  int *rxn_int_data, double *rxn_float_data,
  int *rxn_int_indices, int *rxn_float_indices,
  int *rxn_env_idx,
  SolverData *sd
) {

  ModelDataGPU *mGPU = sd->mGPU;

  unsigned int int_max_length = 0;
  unsigned int double_max_length = 0;

  //RXN lengths
  unsigned int int_lengths[n_rxn];
  unsigned int double_lengths[n_rxn];

  //Position on the matrix for each row
  unsigned int rxn_position[n_rxn];

  //Get lengths for int and double arrays
  for (int i_rxn = 0; i_rxn < n_rxn; i_rxn++) {

    //Get RXN lengths
    int_lengths[i_rxn] = rxn_int_indices[i_rxn+1] - rxn_int_indices[i_rxn];
    double_lengths[i_rxn] = rxn_float_indices[i_rxn+1] - rxn_float_indices[i_rxn];

    //Update max size
    if(int_lengths[i_rxn]>int_max_length) int_max_length=int_lengths[i_rxn];
    if(double_lengths[i_rxn]>double_max_length) double_max_length=double_lengths[i_rxn];

    //Set initial position
    rxn_position[i_rxn] = i_rxn;

  }

  //Total lengths of rxn structure
  unsigned int rxn_int_length=n_rxn*int_max_length;
  unsigned int rxn_double_length=n_rxn*double_max_length;

  //Allocate int and double rxn data separately
  //Add -1 to avoid access and have a square matrix
  int *rxn_int = (int *) malloc(rxn_int_length * sizeof(int));
  memset(rxn_int, -1, rxn_int_length * sizeof(int));

  //Add 0 to avoid access and have a square matrix
  double *rxn_double = (double*)calloc(rxn_double_length, sizeof(double));

  int rxn_env_data_idx_aux[n_rxn];

  for (int i_rxn = 0; i_rxn < n_rxn; i_rxn++) {
    int i_pos=rxn_position[i_rxn];//i_rxn;//rxn_position[i_rxn];//for bubblesort
    for (int j = 0; j < int_lengths[i_pos]; j++){
      int *rxn_int_data_aux = &(rxn_int_data[rxn_int_indices[i_pos]]);
      rxn_int[n_rxn*j + i_rxn] = rxn_int_data_aux[j];
    }
    for (int j = 0; j < double_lengths[i_pos]; j++) {
      double *rxn_float_data_aux = &(rxn_float_data[rxn_float_indices[i_pos]]);
      rxn_double[n_rxn*j + i_rxn] = rxn_float_data_aux[j];
    }
    //Reorder the rate indices
    rxn_env_data_idx_aux[i_rxn] = rxn_env_idx[i_pos];
  }

  //GPU allocation
  cudaMalloc((void **) &mGPU->rxn_int, rxn_int_length * sizeof(int));
  cudaMalloc((void **) &mGPU->rxn_double, rxn_double_length * sizeof(double));

  //Save data to GPU
  HANDLE_ERROR(cudaMemcpy(mGPU->rxn_int, rxn_int, rxn_int_length*sizeof(int), cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(mGPU->rxn_double, rxn_double, rxn_double_length*sizeof(double), cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(mGPU->rxn_env_data_idx, rxn_env_data_idx_aux, rxn_env_data_idx_size, cudaMemcpyHostToDevice));

  free(rxn_int);
  free(rxn_double);

}

void set_int_double_rxn(
        int n_rxn, int rxn_env_data_idx_size,
        int *rxn_int_data, double *rxn_float_data,
        int *rxn_int_indices, int *rxn_float_indices,
        int *rxn_env_idx,
        SolverData *sd
) {

  ModelData *md = &(sd->model_data);
  ModelDataGPU *mGPU = sd->mGPU;

  //GPU allocation
  cudaMalloc((void **) &mGPU->rxn_int, (md->n_rxn_int_param + md->n_rxn)*sizeof(int));
  cudaMalloc((void **) &mGPU->rxn_double, md->n_rxn_float_param*sizeof(double));
#ifdef REVERSE_INT_FLOAT_MATRIX
#else
  cudaMalloc((void **) &mGPU->rxn_int_indices, (md->n_rxn+1)*sizeof(int));
  cudaMalloc((void **) &mGPU->rxn_float_indices, (md->n_rxn+1)*sizeof(int));
#endif

  //Save data to GPU
  HANDLE_ERROR(cudaMemcpy(mGPU->rxn_int, rxn_int_data,(md->n_rxn_int_param + md->n_rxn)*sizeof(int), cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(mGPU->rxn_double, rxn_float_data, md->n_rxn_float_param*sizeof(double), cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(mGPU->rxn_env_data_idx, rxn_env_idx, rxn_env_data_idx_size, cudaMemcpyHostToDevice));
#ifdef REVERSE_INT_FLOAT_MATRIX
#else
  HANDLE_ERROR(cudaMemcpy(mGPU->rxn_int_indices, md->rxn_int_indices,(md->n_rxn+1)*sizeof(int), cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(mGPU->rxn_float_indices, md->rxn_float_indices,(md->n_rxn+1)*sizeof(int), cudaMemcpyHostToDevice));
#endif

}

void solver_init_int_double_gpu(SolverData *sd) {

  ModelData *md = &(sd->model_data);
  ModelDataGPU *mGPU;

  for (int iDevice = 0; iDevice < sd->nDevices; iDevice++) {
    cudaSetDevice(iDevice);
    sd->mGPU = &(sd->mGPUs[iDevice]);
    mGPU = sd->mGPU;

#ifdef REVERSE_INT_FLOAT_MATRIX

    set_reverse_int_double_rxn(
            md->n_rxn, mGPU->rxn_env_data_idx_size,
            md->rxn_int_data, md->rxn_float_data,
            md->rxn_int_indices, md->rxn_float_indices, md->rxn_env_idx,
            sd
    );

#else

    set_int_double_rxn(
            md->n_rxn, mGPU->rxn_env_data_idx_size,
            md->rxn_int_data, md->rxn_float_data,
            md->rxn_int_indices, md->rxn_float_indices, md->rxn_env_idx,
            sd
    );

#endif

/*
  set_int_double_aero(
          sd
  );

 */

  }

}

void init_jac_gpu(SolverData *sd, double *J_ptr){

  ModelData *md = &(sd->model_data);
  ModelDataGPU *mGPU;

  int offset_nnz_J_solver = 0;
  int offset_nnz_j_rxn = 0;
  int offset_nrows = 0;
  for (int iDevice = 0; iDevice < sd->nDevices; iDevice++) {
    cudaSetDevice(iDevice);
    sd->mGPU = &(sd->mGPUs[iDevice]);
    mGPU = sd->mGPU;

    mGPU->jac_size = md->n_per_cell_solver_jac_elem * mGPU->n_cells * sizeof(double);
    mGPU->nnz_J_solver = SM_NNZ_S(md->J_solver)/md->n_cells*mGPU->n_cells;
    mGPU->nrows_J_solver = SM_NP_S(md->J_solver)/md->n_cells*mGPU->n_cells;

    //mGPU->n_per_cell_solver_jac_elem = md->n_per_cell_solver_jac_elem;
    cudaMalloc((void **) &mGPU->J, mGPU->jac_size);
    cudaMalloc((void **) &mGPU->J_solver, mGPU->jac_size);
    cudaMalloc((void **) &mGPU->jJ_solver, mGPU->nnz_J_solver * sizeof(int));
    cudaMalloc((void **) &mGPU->iJ_solver, (mGPU->nrows_J_solver + 1) * sizeof(int));
    cudaMalloc((void **) &mGPU->J_state, mGPU->deriv_size);
    cudaMalloc((void **) &mGPU->J_deriv, mGPU->deriv_size);
    cudaMalloc((void **) &mGPU->J_tmp, mGPU->deriv_size);
    cudaMalloc((void **) &mGPU->J_tmp2, mGPU->deriv_size);
    cudaMalloc((void **) &mGPU->jac_map, sizeof(JacMap) * md->n_mapped_values);
    cudaMalloc((void **) &mGPU->J_rxn,sizeof(double) * SM_NNZ_S(md->J_rxn)*mGPU->n_cells);
    cudaMalloc((void **) &mGPU->n_mapped_values, 1 * sizeof(int));

    //Transfer sunindextype to int
    int *jJ_solver = (int *) malloc(sizeof(int) * mGPU->nnz_J_solver);
    int *iJ_solver = (int *) malloc(sizeof(int) * mGPU->nrows_J_solver + 1);
    for (int i = 0; i < mGPU->nnz_J_solver; i++)
      jJ_solver[i] = SM_INDEXVALS_S(md->J_solver)[i];
    //printf("J_solver PTRS:\n");
    for (int i = 0; i <= mGPU->nrows_J_solver; i++){
      iJ_solver[i] = SM_INDEXPTRS_S(md->J_solver)[i];
      //printf("%lld \n",iJ_solver[i]);
    }

    HANDLE_ERROR(cudaMemcpy(mGPU->J, J_ptr+offset_nnz_J_solver, mGPU->jac_size, cudaMemcpyHostToDevice));
    double *J_solver = SM_DATA_S(md->J_solver)+offset_nnz_J_solver;
    cudaMemcpy(mGPU->J_solver, J_solver, mGPU->jac_size, cudaMemcpyHostToDevice);
    cudaMemcpy(mGPU->jJ_solver, jJ_solver, mGPU->nnz_J_solver * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(mGPU->iJ_solver, iJ_solver, (mGPU->nrows_J_solver + 1) * sizeof(int), cudaMemcpyHostToDevice);
    free(jJ_solver);
    free(iJ_solver);

    double *J_state = N_VGetArrayPointer(md->J_state)+offset_nrows;
    double *J_deriv = N_VGetArrayPointer(md->J_deriv)+offset_nrows;
    double *J_tmp = N_VGetArrayPointer(md->J_tmp)+offset_nrows;
    double *J_tmp2 = N_VGetArrayPointer(md->J_tmp2)+offset_nrows;
    HANDLE_ERROR(cudaMemcpy(mGPU->J_state, J_state, mGPU->deriv_size, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(mGPU->J_deriv, J_deriv, mGPU->deriv_size, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(mGPU->J_tmp, J_tmp, mGPU->deriv_size, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemset(mGPU->J_tmp2, 0.0, mGPU->deriv_size));
    HANDLE_ERROR(cudaMemcpy(mGPU->jac_map, md->jac_map, sizeof(JacMap) * md->n_mapped_values, cudaMemcpyHostToDevice));

    double *J_data = SM_DATA_S(md->J_rxn);
    HANDLE_ERROR(cudaMemcpy(mGPU->J_rxn, J_data, sizeof(double) * SM_NNZ_S(md->J_rxn)*mGPU->n_cells,
                            cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(mGPU->n_mapped_values, &md->n_mapped_values, 1 * sizeof(int), cudaMemcpyHostToDevice));

    offset_nnz_J_solver += mGPU->nnz_J_solver;
    offset_nrows += md->n_per_cell_dep_var* mGPU->n_cells;
  }

  jacobian_initialize_gpu(sd);

}

void set_jac_data_gpu(SolverData *sd, double *J){

  ModelData *md = &(sd->model_data);
  ModelDataGPU *mGPU;

  int offset_nnz_J_solver = 0;
  int offset_nrows = 0;
  for (int iDevice = 0; iDevice < sd->nDevices; iDevice++) {
    cudaSetDevice(iDevice);
    sd->mGPU = &(sd->mGPUs[iDevice]);
    mGPU = sd->mGPU;

    double *J_ptr = J+offset_nnz_J_solver;
    double *J_solver = SM_DATA_S(md->J_solver)+offset_nnz_J_solver;
    double *J_state = N_VGetArrayPointer(md->J_state)+offset_nrows;
    double *J_deriv = N_VGetArrayPointer(md->J_deriv)+offset_nrows;
    HANDLE_ERROR(cudaMemcpy(mGPU->J, J_ptr, mGPU->jac_size, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(mGPU->J_solver, J_solver, mGPU->jac_size, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(mGPU->J_state, J_state, mGPU->deriv_size, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(mGPU->J_deriv, J_deriv, mGPU->deriv_size, cudaMemcpyHostToDevice));

    offset_nnz_J_solver += mGPU->nnz_J_solver;
    offset_nrows += md->n_per_cell_dep_var* mGPU->n_cells;

    //HANDLE_ERROR(cudaMemcpy(mGPU->aero_rep_float_data, md->aero_rep_float_data, md->n_aero_rep_float_param*sizeof(double), cudaMemcpyHostToDevice));

    cudaMemcpy(mGPU->djA, mGPU->jA, mGPU->nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(mGPU->diA, mGPU->iA, (mGPU->nrows + 1) * sizeof(int), cudaMemcpyHostToDevice);

  }

}

void rxn_update_env_state_gpu(SolverData *sd) {

  ModelData *md = &(sd->model_data);
  ModelDataGPU *mGPU;
  double *rxn_env_data = md->rxn_env_data;
  double *env = md->total_env;
  double *total_state = md->total_state;

  for (int iDevice = 0; iDevice < sd->nDevices; iDevice++) {
    cudaSetDevice(iDevice);
    sd->mGPU = &(sd->mGPUs[iDevice]);
    mGPU = sd->mGPU;

    HANDLE_ERROR(cudaMemcpy(mGPU->rxn_env_data, rxn_env_data, mGPU->rxn_env_data_size, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(mGPU->env, env, mGPU->env_size, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(mGPU->state, total_state, mGPU->state_size, cudaMemcpyHostToDevice));

    rxn_env_data += mGPU->n_rxn_env_data * mGPU->n_cells;
    env += CAMP_NUM_ENV_PARAM_ * mGPU->n_cells;
    total_state += mGPU->state_size_cell * mGPU->n_cells;

  }

}

__device__
void cudaDevicecamp_solver_check_model_state0(double *state, double *y,
                                        int *map_state_deriv, double threshhold, double replacement_value, int *flag,
                                        int deriv_length_cell, int n_cells)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int active_threads = n_cells*deriv_length_cell;
  //extern __shared__ int flag_shr[];
  __shared__ int flag_shr[1];
  flag_shr[0] = 0;

  if(tid<active_threads) {

    if (y[tid] < threshhold) {

      //*flag = CAMP_SOLVER_FAIL;
      flag_shr[0] = CAMP_SOLVER_FAIL;
#ifdef FAILURE_DETAIL
      printf("\nFailed model state update gpu (Negative value on 'y'):[spec %d] = %le",tid,y[tid]);
#endif

    } else {
      state[map_state_deriv[tid]] =
              y[tid] <= threshhold ?
              replacement_value : y[tid];

      //printf("tid %d map_state_deriv %d\n", tid, map_state_deriv[tid]);

    }

    /*
    if (y[tid] > -SMALL) {
      state_init[map_state_deriv[tid]] =
      y[tid] > threshhold ?
      y[tid] : replacement_value;

      //state_init[map_state_deriv[tid]] = 0.1;
      //printf("tid %d map_state_deriv %d\n", tid, map_state_deriv[tid]);
    } else {
      *status = CAMP_SOLVER_FAIL;
#ifdef FAILURE_DETAIL
      printf("\nFailed model state update gpu (Negative value on 'y'):[spec %d] = %le",tid,y[tid]);
#endif
    }
     */
  }

  __syncthreads();
  *flag = flag_shr[0];
  return;

}

__global__
void camp_solver_check_model_state_cuda(double *state_init, double *y,
        int *map_state_deriv, double threshhold, double replacement_value, int *flag,
        int deriv_length_cell, int n_cells)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int active_threads = n_cells*deriv_length_cell;

  if(tid<active_threads) {

    if (y[tid] < threshhold) {

      *flag = CAMP_SOLVER_FAIL;
#ifdef FAILURE_DETAIL
      printf("\nFailed model state update gpu (Negative value on 'y'):[spec %d] = %le",tid,y[tid]);
#endif

    } else {
      state_init[map_state_deriv[tid]] =
              y[tid] <= threshhold ?
              replacement_value : y[tid];

      //state_init[map_state_deriv[tid]] = 0.1;
      //printf("tid %d map_state_deriv %d\n", tid, map_state_deriv[tid]);

    }

    /*
    if (y[tid] > -SMALL) {
      state_init[map_state_deriv[tid]] =
      y[tid] > threshhold ?
      y[tid] : replacement_value;

      //state_init[map_state_deriv[tid]] = 0.1;
      //printf("tid %d map_state_deriv %d\n", tid, map_state_deriv[tid]);
    } else {
      *status = CAMP_SOLVER_FAIL;
#ifdef FAILURE_DETAIL
      printf("\nFailed model state update gpu (Negative value on 'y'):[spec %d] = %le",tid,y[tid]);
#endif
    }
     */
  }

}

int camp_solver_check_model_state_gpu(N_Vector solver_state, SolverData *sd,
                                      double threshhold0, double replacement_value0)
{
  ModelData *md = &(sd->model_data);
  itsolver *bicg = &(sd->bicg);
  int flag = CAMP_SOLVER_SUCCESS; //0
  int n_cells;
  int n_state_var;
  int n_dep_var;
  int n_threads;
  int n_blocks;
  int *var_type = md->var_type;
  ModelDataGPU *mGPU;

  double replacement_value = TINY;
  double threshhold = -SMALL;

  for (int iDevice = 0; iDevice < sd->nDevices; iDevice++) {
    cudaSetDevice(iDevice);
    sd->mGPU = &(sd->mGPUs[iDevice]);
    mGPU = sd->mGPU;

    n_cells = mGPU->n_cells;
    n_state_var = md->n_per_cell_state_var;
    n_dep_var = md->n_per_cell_dep_var;
    n_threads = n_dep_var*n_cells;
    n_blocks = ((n_threads + mGPU->max_n_gpu_thread - 1) / mGPU->max_n_gpu_thread);

    camp_solver_check_model_state_cuda << < n_blocks, mGPU->max_n_gpu_thread >> >
    (mGPU->state, mGPU->dcv_y, mGPU->map_state_deriv,
            threshhold, replacement_value, &flag, n_dep_var, n_cells);

    HANDLE_ERROR(cudaMemcpy(md->total_state, mGPU->state, mGPU->state_size, cudaMemcpyDeviceToHost));

  }

#ifdef DEBUG_CHECK_MODEL_STATE_CUDA
  for (int i_cell = 0; i_cell < n_cells; i_cell++) {
   for (int i_dep_var = 0; i_dep_var < n_dep_var; i_dep_var++) {

     printf("(%d) %-le \n", i_dep_var+1,
            md->total_state[mGPU->map_state_derivCPU[i_dep_var]]);
   }
}
#endif


  //printf("camp_solver_check_model_state_gpu flag %d\n",flag);

  return flag;
}


void camp_solver_update_model_state_gpu(N_Vector solver_state, SolverData *sd,
                                       double threshhold, double replacement_value)
{
  ModelData *md = &(sd->model_data);
  ModelDataGPU *mGPU;
  double *total_state = md->total_state;

  for (int iDevice = 0; iDevice < sd->nDevices; iDevice++) {
    cudaSetDevice(iDevice);
    sd->mGPU = &(sd->mGPUs[iDevice]);
    mGPU = sd->mGPU;

    HANDLE_ERROR(cudaMemcpy(mGPU->state, total_state, mGPU->state_size, cudaMemcpyHostToDevice));

    total_state += mGPU->state_size_cell * mGPU->n_cells;

  }

}

__device__ void solveRXN0(
#ifdef BASIC_CALC_DERIV
        double *deriv_data,
#else
        TimeDerivativeGPU deriv_data,
#endif
       double time_step,
       ModelDataGPU *md
)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

#ifdef REVERSE_INT_FLOAT_MATRIX

  double *rxn_float_data = &( md->rxn_double[md->i_rxn]);
  int *int_data = &(md->rxn_int[md->i_rxn]);
  int rxn_type = int_data[0];
  int *rxn_int_data = (int *) &(int_data[1*md->n_rxn]);

#else

  double *rxn_float_data = &( md->rxn_double[md->rxn_float_indices[md->i_rxn]]);
  int *int_data = &(md->rxn_int[md->rxn_int_indices[md->i_rxn]]);

  //double *rxn_float_data = &( md->rxn_double[md->i_rxn]);
  //int *int_data = &(md->rxn_int[md->i_rxn]);


  int rxn_type = int_data[0];
  int *rxn_int_data = (int *) &(int_data[1]);

#endif

  //Get indices for rates
  double *rxn_env_data = &(md->rxn_env_data
  [md->n_rxn_env_data*md->i_cell+md->rxn_env_data_idx[md->i_rxn]]);

#ifdef DEBUG_DERIV_GPU
  if(tid==0){
    printf("[DEBUG] GPU solveRXN tid %d, \n", tid);
  }
#endif

  switch (rxn_type) {
    //case RXN_AQUEOUS_EQUILIBRIUM :
    //fix run-time error
    //rxn_gpu_aqueous_equilibrium_calc_deriv_contrib(md, deriv_data, rxn_int_data,
    //                                               rxn_float_data, rxn_env_data,time_step);
    //break;
    case RXN_ARRHENIUS :
      rxn_gpu_arrhenius_calc_deriv_contrib(md, deriv_data, rxn_int_data,
                                           rxn_float_data, rxn_env_data,time_step);
      break;
    case RXN_CMAQ_H2O2 :
      rxn_gpu_CMAQ_H2O2_calc_deriv_contrib(md, deriv_data, rxn_int_data,
                                          rxn_float_data, rxn_env_data,time_step);
      break;
    case RXN_CMAQ_OH_HNO3 :
      rxn_gpu_CMAQ_OH_HNO3_calc_deriv_contrib(md, deriv_data, rxn_int_data,
                                           rxn_float_data, rxn_env_data,time_step);
      break;
    case RXN_CONDENSED_PHASE_ARRHENIUS :
      //rxn_gpu_condensed_phase_arrhenius_calc_deriv_contrib(md, deriv_data, rxn_int_data,
      //                                     rxn_float_data, rxn_env_data,time_step);
      break;
    case RXN_EMISSION :
      //rxn_gpu_emission_calc_deriv_contrib(md, deriv_data, rxn_int_data,
      //                                     rxn_float_data, rxn_env_data,time_step);
      break;
    case RXN_FIRST_ORDER_LOSS :
      //rxn_gpu_first_order_loss_calc_deriv_contrib(md, deriv_data, rxn_int_data,
      //                                     rxn_float_data, rxn_env_data,time_step);
      break;
    case RXN_HL_PHASE_TRANSFER :
      //rxn_gpu_HL_phase_transfer_calc_deriv_contrib(md, deriv_data, rxn_int_data,
      //                                             rxn_float_data, rxn_env_data,time_stepn);
      break;
    case RXN_PHOTOLYSIS :
      rxn_gpu_photolysis_calc_deriv_contrib(md, deriv_data, rxn_int_data,
                                           rxn_float_data, rxn_env_data,time_step);
      break;
    case RXN_SIMPOL_PHASE_TRANSFER :
      //rxn_gpu_SIMPOL_phase_transfer_calc_deriv_contrib(md, deriv_data,
      //        rxn_int_data, rxn_float_data, rxn_env_data, time_step);
      break;
    case RXN_TROE :
#ifdef BASIC_CALC_DERIV
#else
      rxn_gpu_troe_calc_deriv_contrib(md, deriv_data, rxn_int_data,
                                      rxn_float_data, rxn_env_data,time_step);
#endif
      break;
    case RXN_WET_DEPOSITION :
      //rxn_gpu_wet_deposition_calc_deriv_contrib(md, deriv_data, rxn_int_data,
      //                                     rxn_float_data, rxn_env_data,time_step);
      break;
  }
/*
*/

}

__device__ void cudaDevicecalc_deriv0(
#ifdef CAMP_DEBUG_GPU
        int counterDeriv2,
#endif
        //check_model_state
        //double threshhold, double replacement_value, int *flag,
        //f_gpu
        double time_step, int deriv_length_cell, int state_size_cell,
        int n_cells,
        int i_kernel, int threads_block, int n_shr_empty, double *y,
        ModelDataGPU md_object
) //Interface CPU/GPU
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int tid_cell=tid%deriv_length_cell;
  int active_threads = n_cells*deriv_length_cell;
  ModelDataGPU *md = &md_object;

#ifdef DEBUG_DERIV_GPU
  if(tid==0){
    printf("[DEBUG] GPU solveDerivative tid %d, \n", tid);
  }__syncthreads();
#endif

  if(tid<active_threads){

#ifdef DEBUG_solveDerivative_J_DERIV_IN_CPU
#else

    //N_VLinearSum(1.0, y, -1.0, md->J_state, md->J_tmp);
  cudaDevicezaxpby(1.0, y, -1.0, md->J_state, md->J_tmp, active_threads);
  //SUNMatMatvec(md->J_solver, md->J_tmp, md->J_tmp2);
  cudaDeviceSpmvCSC_block(md->J_tmp2, md->J_tmp, active_threads, md->J_solver, md->jJ_solver, md->iJ_solver, 0);
  //N_VLinearSum(1.0, md->J_deriv, 1.0, md->J_tmp2, md->J_tmp);
  cudaDevicezaxpby(1.0, md->J_deriv, 1.0, md->J_tmp2, md->J_tmp, active_threads);
  cudaDevicesetconst(md->J_tmp2, 0.0, active_threads); //Reset for next iter


#endif

    //Debug
    /*
    if(counterDeriv2<=1){
      printf("(%d) y %-le J_state %-le J_solver %-le J_tmp %-le J_tmp2 %-le J_deriv %-le\n",tid+1,
             y[tid], md->J_state[tid], md->J_solver[tid], md->J_tmp[tid], md->J_tmp2[tid], md->J_deriv[tid]);
      //printf("gpu threads %d\n", active_threads);
    }
*/

#ifdef BASIC_CALC_DERIV
    md->i_rxn=tid%n_rxn;
    double *deriv_init = md->deriv_data;
    md->deriv_data = &( md->deriv_init[deriv_length_cell*md->i_cell]);
    if(tid < n_rxn*n_cells){
        solveRXN(deriv_data, time_step, md);
    }
#else
    TimeDerivativeGPU deriv_data;
    deriv_data.num_spec = deriv_length_cell*n_cells;

#ifdef AEROS_CPU
#else
    deriv_data.production_rates = md->production_rates;
    deriv_data.loss_rates = md->loss_rates;
    time_derivative_reset_gpu(deriv_data);
    __syncthreads();
#endif

    int i_cell = tid/deriv_length_cell;
    md->i_cell = i_cell;
    deriv_data.production_rates = &( md->production_rates[deriv_length_cell*i_cell]);
    deriv_data.loss_rates = &( md->loss_rates[deriv_length_cell*i_cell]);

    md->grid_cell_state = &( md->state[state_size_cell*i_cell]);
    md->grid_cell_env = &( md->env[CAMP_NUM_ENV_PARAM_*i_cell]);

    /*
    md->grid_cell_aero_rep_env_data =
    &(md->aero_rep_env_data[md->i_cell*md->n_aero_rep_env_data]);

    //Filter threads for n_aero_rep
    int n_aero_rep = md->n_aero_rep;
    if( tid_cell < n_aero_rep) {
      int n_iters = n_aero_rep / deriv_length_cell;
      //Repeat if there are more reactions than species
      for (int i = 0; i < n_iters; i++) {
        md->i_aero_rep = tid_cell + i*deriv_length_cell;

        aero_rep_gpu_update_state(md);
      }

      //Limit tid to pending rxns to compute
      int residual=n_aero_rep-(deriv_length_cell*n_iters);
      if(tid_cell < residual){
        md->i_aero_rep = tid_cell + deriv_length_cell*n_iters;

        aero_rep_gpu_update_state(md);
      }
    }
     */

    //Filter threads for n_rxn
    int n_rxn = md->n_rxn;
    if( tid_cell < n_rxn) {
      int n_iters = n_rxn / deriv_length_cell;
      //Repeat if there are more reactions than species
      for (int i = 0; i < n_iters; i++) {
        md->i_rxn = tid_cell + i*deriv_length_cell;

        solveRXN0(deriv_data, time_step, md);
      }

      //Limit tid to pending rxns to compute
      int residual=n_rxn-(deriv_length_cell*n_iters);
      if(tid_cell < residual){
        md->i_rxn = tid_cell + deriv_length_cell*n_iters;

        solveRXN0(deriv_data, time_step, md);
      }
    }
    __syncthreads();

    /*if(tid==0){
      printf("tid %d time_deriv.production_rates %-le time_deriv.loss_rates %-le\n",
              tid, deriv_data.production_rates[tid],
             deriv_data.loss_rates[tid]);
    }*/

    deriv_data.production_rates = md->production_rates;
    deriv_data.loss_rates = md->loss_rates;
    __syncthreads();
    time_derivative_output_gpu(deriv_data, md->deriv_data, md->J_tmp,0);
#endif

    /*
    if(tid<deriv_data.num_spec && tid>1022){
      //if(tid<1){
      //deriv_init[tid] = deriv_data.production_rates[tid];
      //deriv_init[tid] = deriv_data.loss_rates[tid];
      printf("tid %d time_deriv.production_rates %-le time_deriv.loss_rates %-le"
             "deriv_init %-le\n",
             tid, deriv_data.production_rates[tid],
             deriv_data.loss_rates[tid],
             //deriv_data.loss_rates[tid]);
             deriv_init[tid]);
    }*/

  }

}

__device__
void cudaDevicef0(
#ifdef CAMP_DEBUG_GPU
        int counterDeriv2,
#endif
        //check_model_state
        double threshhold, double replacement_value, int *flag,
        //f_gpu
        double time_step, int deriv_length_cell, int state_size_cell,
        int n_cells,
        int i_kernel, int threads_block, int n_shr_empty, double *y,
        ModelDataGPU md_object
) //Interface CPU/GPU
{

  ModelDataGPU *md = &md_object;

  cudaDevicecamp_solver_check_model_state0(md->state, y,
                                          md->map_state_deriv, threshhold, replacement_value,
                                          flag, deriv_length_cell, n_cells);

  //__syncthreads();
  //study flag block effect: flag is global for all threads or for only the block?
  if(*flag==CAMP_SOLVER_FAIL)
    return;

  cudaDevicecalc_deriv0(
#ifdef CAMP_DEBUG_GPU
           counterDeriv2,
#endif
          //f_gpu
        time_step, deriv_length_cell, state_size_cell,
           n_cells, i_kernel, threads_block, n_shr_empty, y,
           md_object
          );
}

__global__
void cudaGlobalf(
#ifdef CAMP_DEBUG_GPU
        int counterDeriv2,
#endif
        //check_model_state
        double threshhold, double replacement_value, int *flag,
        //f_gpu
        double time_step, int deriv_length_cell, int state_size_cell,
        int n_cells,
        int i_kernel, int threads_block, int n_shr_empty, double *y,
        ModelDataGPU md_object
) //Interface CPU/GPU
{

  cudaDevicef0(
#ifdef CAMP_DEBUG_GPU
          counterDeriv2,
#endif
          //check_model_state
                threshhold, replacement_value, flag,
                //f_gpu
          time_step, deriv_length_cell, state_size_cell,
          n_cells, i_kernel, threads_block, n_shr_empty, y,
          md_object
  );
}



/** Old routine
 */
__global__ void solveDerivative(
#ifdef CAMP_DEBUG_GPU
        int counterDeriv2,
#endif
        double time_step, int deriv_length_cell, int state_size_cell,
        int n_cells,
        int i_kernel, int threads_block, int n_shr_empty, double *y,
        double threshhold, double replacement_value, ModelDataGPU md_object
) //Interface CPU/GPU
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int tid_cell=tid%deriv_length_cell;
  int active_threads = n_cells*deriv_length_cell;
  ModelDataGPU *md = &md_object;

#ifdef DEBUG_DERIV_GPU
  if(tid==0){
    printf("[DEBUG] GPU solveDerivative tid %d, \n", tid);
  }__syncthreads();
#endif

  if(tid<active_threads){

#ifdef DEBUG_solveDerivative_J_DERIV_IN_CPU
#else

    //N_VLinearSum(1.0, y, -1.0, md->J_state, md->J_tmp);
    cudaDevicezaxpby(1.0, y, -1.0, md->J_state, md->J_tmp, active_threads);
    //SUNMatMatvec(md->J_solver, md->J_tmp, md->J_tmp2);
    cudaDeviceSpmvCSC_block(md->J_tmp2, md->J_tmp, active_threads, md->J_solver, md->jJ_solver, md->iJ_solver, 0);
    //N_VLinearSum(1.0, md->J_deriv, 1.0, md->J_tmp2, md->J_tmp);
    cudaDevicezaxpby(1.0, md->J_deriv, 1.0, md->J_tmp2, md->J_tmp, active_threads);
    cudaDevicesetconst(md->J_tmp2, 0.0, active_threads); //Reset for next iter


#endif

    //Debug
    //printf("HOLA\n");
    /*
    if(counterDeriv2<=1){
      printf("(%d) y %-le J_state %-le J_solver %-le J_tmp %-le J_tmp2 %-le J_deriv %-le\n",tid+1,
             y[tid], md->J_state[tid], md->J_solver[tid], md->J_tmp[tid], md->J_tmp2[tid], md->J_deriv[tid]);
      //printf("gpu threads %d\n", active_threads);
    }
*/

#ifdef BASIC_CALC_DERIV
    md->i_rxn=tid%n_rxn;
    double *deriv_init = md->deriv_data;
    md->deriv_data = &( md->deriv_init[deriv_length_cell*md->i_cell]);
    if(tid < n_rxn*n_cells){
        solveRXN0(deriv_data, time_step, md);
    }
#else
    TimeDerivativeGPU deriv_data;
    deriv_data.num_spec = deriv_length_cell*n_cells;

#ifdef AEROS_CPU
#else
    deriv_data.production_rates = md->production_rates;
    deriv_data.loss_rates = md->loss_rates;
    time_derivative_reset_gpu(deriv_data);
    __syncthreads();
#endif

    int i_cell = tid/deriv_length_cell;
    md->i_cell = i_cell;
    deriv_data.production_rates = &( md->production_rates[deriv_length_cell*i_cell]);
    deriv_data.loss_rates = &( md->loss_rates[deriv_length_cell*i_cell]);

    md->grid_cell_state = &( md->state[state_size_cell*i_cell]);
    md->grid_cell_env = &( md->env[CAMP_NUM_ENV_PARAM_*i_cell]);

    //Filter threads for n_rxn
    int n_rxn = md->n_rxn;
    if( tid_cell < n_rxn) {
      int n_iters = n_rxn / deriv_length_cell;
      //Repeat if there are more reactions than species
      for (int i = 0; i < n_iters; i++) {
        md->i_rxn = tid_cell + i*deriv_length_cell;

        solveRXN0(deriv_data, time_step, md);
      }

      //Limit tid to pending rxns to compute
      int residual=n_rxn-(deriv_length_cell*n_iters);
      if(tid_cell < residual){
        md->i_rxn = tid_cell + deriv_length_cell*n_iters;

        solveRXN0(deriv_data, time_step, md);
      }
    }
    __syncthreads();

    /*if(tid==0){
      printf("tid %d time_deriv.production_rates %-le time_deriv.loss_rates %-le\n",
              tid, deriv_data.production_rates[tid],
             deriv_data.loss_rates[tid]);
    }*/

    deriv_data.production_rates = md->production_rates;
    deriv_data.loss_rates = md->loss_rates;
    __syncthreads();
    time_derivative_output_gpu(deriv_data, md->deriv_data, md->J_tmp,0);
#endif

    /*
    if(tid<deriv_data.num_spec && tid>1022){
      //if(tid<1){
      //deriv_init[tid] = deriv_data.production_rates[tid];
      //deriv_init[tid] = deriv_data.loss_rates[tid];
      printf("tid %d time_deriv.production_rates %-le time_deriv.loss_rates %-le"
             "deriv_init %-le\n",
             tid, deriv_data.production_rates[tid],
             deriv_data.loss_rates[tid],
             //deriv_data.loss_rates[tid]);
             deriv_init[tid]);
    }*/

  }

}


/** \brief Calculate the time derivative \f$f(t,y)\f$ on GPU
 *
 * \param md Pointer to the model data
 * \param deriv NVector to hold the calculated vector
 * \param time_step Current model time step (s)
 */
int rxn_calc_deriv_gpu(SolverData *sd, N_Vector y, N_Vector deriv, double time_step) {

  ModelData *md = &(sd->model_data);
  ModelDataGPU *mGPU;

  double *total_state = md->total_state;
  double *deriv_data = N_VGetArrayPointer(deriv);
  if(sd->use_gpu_cvode==0){

    for (int iDevice = 0; iDevice < sd->nDevices; iDevice++) {
      cudaSetDevice(iDevice);
      sd->mGPU = &(sd->mGPUs[iDevice]);
      mGPU = sd->mGPU;

      HANDLE_ERROR(cudaMemcpy(mGPU->deriv_data, deriv_data, mGPU->deriv_size, cudaMemcpyHostToDevice));
      HANDLE_ERROR(cudaMemcpy(mGPU->state, total_state, mGPU->state_size, cudaMemcpyHostToDevice));

      total_state += mGPU->state_size_cell * mGPU->n_cells;
      deriv_data += mGPU->nrows;
    }

  }else{

    itsolver *bicg = &(sd->bicg);
    ModelDataGPU *mGPU = sd->mGPU;
    int n_cells = md->n_cells;
    int n_kernels = 1; // Divide load into multiple kernel calls
    int total_threads = mGPU->nrows/n_kernels;
    int n_shr_empty = mGPU->max_n_gpu_thread%mGPU->nrows;
    int threads_block = mGPU->max_n_gpu_thread - n_shr_empty; //last multiple of size_cell before max_threads
    int n_blocks = ((total_threads + threads_block - 1) / threads_block);
    double *J_tmp = N_VGetArrayPointer(md->J_tmp);

    //Update state
    double replacement_value = TINY;
    double threshhold = -SMALL;
    int flag = CAMP_SOLVER_SUCCESS; //0

#ifdef DEBUG_rxn_calc_deriv_gpu

    printf("rxn_calc_deriv_gpu start\n");

#endif

    if (camp_solver_check_model_state_gpu(y, sd, -SMALL, TINY) != CAMP_SOLVER_SUCCESS)
      return 1;

   //debug
   /*
    if(sd->counterDerivGPU<=0){
      printf("f_gpu start total_state [(id),conc], n_state_var %d, n_cells %d\n", md->n_per_cell_state_var, n_cells);
      printf("n_deriv %d\n", md->n_per_cell_dep_var);
      for (int i = 0; i < md->n_per_cell_state_var*n_cells; i++) {
        printf("(%d) %-le \n",i+1, md->total_state[i]);
      }
    }
    */

#ifdef BASIC_CALC_DERIV
  //Reset deriv gpu
  //check if cudamemset work fine with doubles
    HANDLE_ERROR(cudaMemset(md->deriv_data_gpu, 0.0, mGPU->deriv_size));
#endif

#ifdef CAMP_DEBUG_GPU
  //timeDerivSend += (clock() - t1);
  //clock_t t2 = clock();

    cudaEventRecord(md->startDerivKernel);

#endif

#ifdef AEROS_CPU

    update_aero_contrib_gpu(sd);

#endif

#ifdef DEBUG_solveDerivative_J_DERIV_IN_CPU

    HANDLE_ERROR(cudaMemcpy(mGPU->J_tmp, J_tmp, mGPU->deriv_size, cudaMemcpyHostToDevice));

#endif

  //Loop to test multiple kernel executions
    for (int i_kernel=0; i_kernel<n_kernels; i_kernel++){
      //cudaDeviceSynchronize();
      //solveDerivative << < (n_blocks), threads_block >> >(
      cudaGlobalf << < (n_blocks), threads_block >> >(
#ifdef CAMP_DEBUG_GPU
     sd->counterDerivGPU,
#endif
      //update_state
      threshhold, replacement_value, &flag,
       //f_gpu
      time_step, md->n_per_cell_dep_var,
       md->n_per_cell_state_var,n_cells,
       i_kernel, threads_block,n_shr_empty, mGPU->dcv_y,
       *sd->mGPU
       );
    }

    if(flag==CAMP_SOLVER_FAIL)
      return flag;

#ifdef CAMP_DEBUG_GPU

    cudaEventRecord(md->stopDerivKernel);
    cudaEventSynchronize(md->stopDerivKernel);
    float msDerivKernel = 0.0;
    cudaEventElapsedTime(&msDerivKernel, md->startDerivKernel, md->stopDerivKernel);
   md->timeDerivKernel+= msDerivKernel;

#endif

      //Async
      //mGPU->deriv_size, cudaMemcpyDeviceToHost, md->stream_gpu[STREAM_DERIV_GPU]));

      //Sync

      HANDLE_ERROR(cudaMemcpy(deriv_data, mGPU->deriv_data, mGPU->deriv_size, cudaMemcpyDeviceToHost));

      //HANDLE_ERROR(cudaMemcpy(mGPU->deriv_data, deriv_data, mGPU->deriv_size, cudaMemcpyHostToDevice));
      //HANDLE_ERROR(cudaMemcpy(mGPU->dcv_y, deriv_data, mGPU->deriv_size, cudaMemcpyHostToDevice));
      //HANDLE_ERROR(cudaMemcpy(mGPU->state, md->total_state, mGPU->state_size, cudaMemcpyHostToDevice));

  }

  return 0;
}

void get_f_from_gpu(SolverData *sd){

  //HANDLE_ERROR(cudaMemcpy(mGPU->state, J, mGPU->jac_size, cudaMemcpyHostToDevice));

}

void get_guess_helper_from_gpu(N_Vector y_n, N_Vector y_n1,
        N_Vector hf, void *solver_data, N_Vector tmp1,
        N_Vector corr){

  //HANDLE_ERROR(cudaMemcpy(mGPU->state, J, mGPU->jac_size, cudaMemcpyHostToDevice));


}


/** \brief Free GPU data structures
 */
void free_gpu_cu(SolverData *sd) {

  ModelData *md = &(sd->model_data);
  ModelDataGPU *mGPU = sd->mGPU;

  free(sd->flagCells);

#ifdef CAMP_DEBUG_GPU

  printf("timeDerivKernel %lf\n", md->timeDerivKernel/1000);

#endif

  //for (int i = 0; i < n_streams; ++i)
  //  HANDLE_ERROR( cudaStreamDestroy(md->stream_gpu[i]) );
/*

  */
  //free(md->jac_aux);
  HANDLE_ERROR(cudaFree(mGPU->rxn_int));
  HANDLE_ERROR(cudaFree(mGPU->rxn_double));
  HANDLE_ERROR(cudaFree(mGPU->deriv_data));
  //HANDLE_ERROR(cudaFree(J_solver_gpu));

  HANDLE_ERROR(cudaFree(mGPU->state));
  HANDLE_ERROR(cudaFree(mGPU->env));
  HANDLE_ERROR(cudaFree(mGPU->rxn_env_data));
  HANDLE_ERROR(cudaFree(mGPU->rxn_env_data_idx));

}

/* Auxiliar functions */

void bubble_sort_gpu(unsigned int *n_zeros, unsigned int *rxn_position, int n_rxn){

  int tmp,s=1,i_rxn=n_rxn;

  while(s){
    s=0;
    for (int i = 1; i < i_rxn; i++) {
      //Few zeros go first
      if (n_zeros[i] < n_zeros[i - 1]) {
        //Swap positions
        tmp = rxn_position[i];
        rxn_position[i] = rxn_position[i - 1];
        rxn_position[i - 1] = tmp;

        tmp = n_zeros[i];
        n_zeros[i] = n_zeros[i - 1];
        n_zeros[i - 1] = tmp;
        s=1;
      }
    }
    i_rxn--;
  }

}

/* Prints */

void print_gpu_specs() {

  printf("GPU specifications \n");

  int nDevices;
  cudaGetDeviceCount(&nDevices);
  for (int i = 0; i < nDevices; i++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    printf("Device Number: %d\n", i);
    printf("  Device name: %s\n", prop.name);
    printf("  Memory Clock Rate (KHz): %d\n",
           prop.memoryClockRate);
    printf("  Peak Memory Bandwidth (GB/s): %f\n",
           2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
    printf("  maxGridSize: %d\n", prop.maxGridSize[1]);
    printf("  maxThreadsPerBlock: %d\n", prop.maxThreadsPerBlock);
    printf("  maxThreadsDim: %d\n", prop.maxThreadsDim[1]);
    printf("  totalGlobalMem: %zu\n", prop.totalGlobalMem);
    printf("  sharedMemPerBlock: %zu\n", prop.sharedMemPerBlock); //bytes
    printf("  multiProcessorCount: %d\n", prop.multiProcessorCount);
  }

}


/* Copyright (C) 2019 Christian Guzman
 * Licensed under the GNU General Public License version 1 or (at your
 * option) any later version. See the file COPYING for details.
 *
 * Interface Host-Device (CPU-GPU) to compute reaction-specific functions on GPU
 *
 */

#include "itsolver_gpu.h"

extern "C" {
#include "camp_gpu_solver.h"
#include "rxns_gpu.h"
#include "time_derivative_gpu.h"

// Reaction types (Must match parameters defined in pmc_rxn_factory)
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

//Gpu hardware info
//int md->max_n_gpu_thread;
//int md->max_n_gpu_blocks;

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
void solver_new_gpu_cu(ModelData *md, int n_dep_var,
                       int n_state_var, int n_rxn,
                       int n_rxn_int_param, int n_rxn_float_param, int n_rxn_env_param,
                       int n_cells) {
  //TODO: Select what % of data we want to compute on GPU simultaneously with CPU remaining %
  //Lengths
  md->state_size = n_state_var * n_cells * sizeof(double);
  md->deriv_size = n_dep_var * n_cells * sizeof(double);
  md->env_size = PMC_NUM_ENV_PARAM_ * n_cells * sizeof(double); //Temp and pressure
  md->rxn_env_data_size = n_rxn_env_param * n_cells * sizeof(double);
  md->rxn_env_data_idx_size = (n_rxn+1) * sizeof(int);
  md->map_state_deriv_size = n_dep_var * n_cells * sizeof(int);
  md->small_data = 0;
  md->implemented_all = true;

  //Allocate streams array and update variables related to streams
  //md->md_id = n_solver_objects;
  //if(n_solver_objects==0){
    //stream_gpu = (cudaStream_t *)malloc(n_streams_limit * sizeof(cudaStream_t));
      //md->stream_gpu = (cudaStream_t *)malloc(n_streams * sizeof(cudaStream_t));
  //}
  //n_solver_objects++;

  //Detect if we are working with few data values
  //todo check if it's worth to maintain this case (we will use small_data?)
  if (n_dep_var*n_cells < DATA_SIZE_LIMIT_OPT){
    md->small_data = 0;//1;
  }

  //Set working GPU: we have 4 gpu available on power9. as default, it should be assign to gpu 0
  int device=0;
  cudaSetDevice(device);

  //Set GPU properties
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, device);

  //Set max threads without triggering too many resources error
  md->max_n_gpu_thread = prop.maxThreadsPerBlock/2;
  md->max_n_gpu_blocks = prop.maxGridSize[1];
  int n_blocks = (n_rxn + md->max_n_gpu_thread - 1) / md->max_n_gpu_thread;

  //GPU allocation
  HANDLE_ERROR(cudaMalloc((void **) &md->deriv_gpu_data, md->deriv_size));
  cudaMalloc((void **) &md->state_gpu, md->state_size);
  cudaMalloc((void **) &md->env_gpu, md->env_size);
  cudaMalloc((void **) &md->rxn_env_data_gpu, md->rxn_env_data_size);
  cudaMalloc((void **) &md->rxn_env_data_idx_gpu, md->rxn_env_data_idx_size);
  HANDLE_ERROR(cudaMalloc((void **) &md->map_state_deriv_gpu, md->map_state_deriv_size));

  time_derivative_initialize_gpu(md,n_dep_var*n_cells);

  //Mapping state-deriv
  md->map_state_deriv = (int *)malloc(md->map_state_deriv_size);
  int i_dep_var = 0;
  for (int i_cell = 0; i_cell < n_cells; i_cell++) {
    for (int i_spec = 0; i_spec < n_state_var; i_spec++) {
      if (md->var_type[i_spec] == CHEM_SPEC_VARIABLE) {
        md->map_state_deriv[i_dep_var] = i_spec + i_cell * n_state_var;
        //printf("%d %d, %d %d %d\n", md->map_state_deriv_size/sizeof(int),
        //       md->map_state_deriv[i_dep_var],n_state_var, i_spec, i_cell, i_dep_var);
        i_dep_var++;
      }
    }
  }

  HANDLE_ERROR(cudaMemcpy(md->map_state_deriv_gpu, md->map_state_deriv,
          md->map_state_deriv_size, cudaMemcpyHostToDevice));

  //HANDLE_ERROR(cudaMemcpy(md->int_pointer_gpu, int_pointer, rxn_int_length*sizeof(int), cudaMemcpyHostToDevice));

  //GPU allocation few data on pinned memory
  if(md->small_data){
    //Notice auxiliar variables are created because we
    // can't pin directly variables initialized before
    cudaMallocHost((void**)&md->deriv_aux, md->deriv_size);
  }
  else{
    md->deriv_aux = (realtype *)malloc(md->deriv_size);
  }

  printf("small_data:%d\n", md->small_data);
  //printf("threads_per_block :%d\n", md->max_n_gpu_thread);

  //GPU create streams
  //for (int i = 0; i < n_streams; ++i)
  //  HANDLE_ERROR( cudaStreamCreate(&md->stream_gpu[i]) );

  // Warning if exceeding GPU limits
  if( n_blocks > md->max_n_gpu_blocks){
    printf("\nWarning: More blocks assigned: %d than maximum block numbers: %d",
           n_blocks, md->max_n_gpu_blocks);
  }

#ifdef PMC_DEBUG_PRINT_GPU_SPECS
  print_gpu_specs();
#endif

#ifdef PMC_DEBUG_GPU

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

#ifdef TEST_HOST_DEVICE

#ifdef __CUDA_ARCH__
__host__ __device__
#endif
void test_host_device(int *a) {

#ifdef __CUDA_ARCH__
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if(i<3)
    a[i]=i;

#else

  for (unsigned int i = 0; i < 3; i++) {
    a[i]=i;
  }
#endif

}

__global__
void test_host_device_global(int *a){

  test_host_device(a);

}

#endif

void solver_set_rxn_data_gpu(SolverData *sd) {


#ifdef TEST_HOST_DEVICE
  int a[3];//a should be declared as host and device var
  //TODO quantify efficiency-readability-work to translate
  // "a" as host&Device var or use "a" and "a_gpu"
  cudaMalloc( (void**)&a, sizeof(int)*3 );

  test_host_device_global << < 1, 512 >> >(a);
  printf("a GPU %d", a[2]);
  test_host_device(a);
  printf("a CPU %d", a[2]);
#endif

  ModelData *md = &(sd->model_data);
  int n_rxn = md->n_rxn;
  unsigned int int_max_length = 0;
  unsigned int double_max_length = 0;

  //RXN lengths
  unsigned int int_lengths[n_rxn];
  unsigned int double_lengths[n_rxn];

  //Number of extra values added to square matrix(zeros and -1's)
  //unsigned int n_zeros[n_rxn];

  //Position on the matrix for each row
  unsigned int rxn_position[n_rxn];

  //Get lengths for int and double arrays
  for (int i_rxn = 0; i_rxn < n_rxn; i_rxn++) {

    // Set a WARNING if the reaction is not implemented yet on GPU
    bool implemented = false;
    int rxn_type = md->rxn_int_data[md->rxn_int_indices[i_rxn]];

    switch (rxn_type) {
      case RXN_AQUEOUS_EQUILIBRIUM :
        implemented = false;
        break;
      case RXN_ARRHENIUS :
        implemented = true;
        break;
      case RXN_CMAQ_H2O2 :
        implemented = true;
        break;
      case RXN_CMAQ_OH_HNO3 :
        implemented = true;
        break;
      case RXN_CONDENSED_PHASE_ARRHENIUS :
        implemented = false;
        break;
      case RXN_EMISSION :
        implemented = false;
        break;
      case RXN_FIRST_ORDER_LOSS :
        implemented = false;
        break;
      case RXN_HL_PHASE_TRANSFER :
        implemented = false;
        break;
      case RXN_PHOTOLYSIS :
        implemented = true;
        break;
      case RXN_SIMPOL_PHASE_TRANSFER :
        implemented = false;
        break;
      case RXN_TROE :
        implemented = true;
        break;
      case RXN_WET_DEPOSITION :
        implemented = false;
        break;
    }
    if(!implemented){
#ifdef FAILURE_DETAIL
      printf("WARNING: Reaction type %d is not fully implemented on GPU\n", rxn_type);
#endif
      md->implemented_all=false;
    }

    //Get RXN lengths
    int_lengths[i_rxn] = md->rxn_int_indices[i_rxn+1] - md->rxn_int_indices[i_rxn];
    double_lengths[i_rxn] = md->rxn_float_indices[i_rxn+1] - md->rxn_float_indices[i_rxn];

    //Update max size
    if(int_lengths[i_rxn]>int_max_length) int_max_length=int_lengths[i_rxn];
    if(double_lengths[i_rxn]>double_max_length) double_max_length=double_lengths[i_rxn];

    //Set initial position
    rxn_position[i_rxn] = i_rxn;

  }

  //Add a for to search the biggest distance int_max_length (ptrs[i] - ptrs[i-1]

  //Total lengths of rxn structure
  unsigned int rxn_int_length=n_rxn*int_max_length;
  unsigned int rxn_double_length=n_rxn*double_max_length;

  //Allocate int and double rxn data separately
  //Add -1 to avoid access and have a square matrix
  int *int_pointer = (int *) malloc(rxn_int_length * sizeof(int));
  memset(int_pointer, -1, rxn_int_length * sizeof(int));

  //Add 0 to avoid access and have a square matrix
  double *double_pointer = (double*)calloc(rxn_double_length, sizeof(double));

  //GPU allocation
  cudaMalloc((void **) &md->int_pointer_gpu, rxn_int_length * sizeof(int));
  cudaMalloc((void **) &md->double_pointer_gpu, rxn_double_length * sizeof(double));

  //Update number of zeros added on each reaction
  /*for (int i_rxn = 0; i_rxn < n_rxn; i_rxn++)
    n_zeros[i_rxn] = (int_max_length - int_lengths[i_rxn]) +
                     (double_max_length - double_lengths[i_rxn]);*/

  //NOTE: no improvement on doing the sorting or not for gpu seems.
  //Sort by lengths
  //BubbleSort RXN by ascendant number of zeros for performance reasons
  //Fix reordered rxn give wrong values
  //bubble_sort_gpu(n_zeros, rxn_position, n_rxn);

  //Copy into gpu rxn data
  //Follows the rxn_position order
  //Rxn matrix is reversed to improve memory access on GPU
  //Matrix order is [int_length][n_rxn]

  int rxn_env_data_idx_aux[n_rxn];

  for (int i_rxn = 0; i_rxn < n_rxn; i_rxn++) {
    int i_pos=rxn_position[i_rxn];//i_rxn;//rxn_position[i_rxn];//for bubblesort
    for (int j = 0; j < int_lengths[i_pos]; j++){
      int *rxn_int_data = &(md->rxn_int_data[md->rxn_int_indices[i_pos]]);
      int_pointer[n_rxn*j + i_rxn] = rxn_int_data[j];
    }
    for (int j = 0; j < double_lengths[i_pos]; j++) {
      double *rxn_float_data = &(md->rxn_float_data[md->rxn_float_indices[i_pos]]);
      double_pointer[n_rxn*j + i_rxn] = rxn_float_data[j];
    }
    //Reorder the rate indices
    //Todo update on main code the rxn_env_data to read consecutively in cpu
    rxn_env_data_idx_aux[i_rxn] = md->rxn_env_idx[i_pos];
  }

  //Save data to GPU
  HANDLE_ERROR(cudaMemcpy(md->int_pointer_gpu, int_pointer, rxn_int_length*sizeof(int), cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(md->double_pointer_gpu, double_pointer, rxn_double_length*sizeof(double), cudaMemcpyHostToDevice));

  //Set rxn_env_data-idx
  HANDLE_ERROR(cudaMemcpy(md->rxn_env_data_idx_gpu, rxn_env_data_idx_aux, md->rxn_env_data_idx_size, cudaMemcpyHostToDevice));

  free(int_pointer);
  free(double_pointer);

}

void init_j_state_deriv_solver_gpu(SolverData *sd, double *J){

  ModelData *md = &(sd->model_data);
/* //cpu case (delete when gpu works fine)
  // Create vectors to store Jacobian state and derivative data
  solver_data->md.J_state = N_VClone(solver_data->y);
  solver_data->md.J_deriv = N_VClone(solver_data->y);
  solver_data->md.J_tmp = N_VClone(solver_data->y);
  solver_data->md.J_tmp2 = N_VClone(solver_data->y);

  // Initialize the Jacobian state and derivative arrays to zero
  // for use before the first call to Jac()
  N_VConst(0.0, solver_data->md.J_state);
  N_VConst(0.0, solver_data->md.J_deriv);
*/

  //todo reduce allocations (use tmp pointers from cvode for j_tmp)
  md->jac_size = md->n_per_cell_solver_jac_elem * md->n_cells * sizeof(double);
  md->nnz_J_solver = SM_NNZ_S(md->J_solver);
  md->nrows_J_solver = SM_NP_S(md->J_solver);

  cudaMalloc((void **) &md->J_gpu, md->jac_size);
  cudaMalloc((void **) &md->J_solver_gpu, md->jac_size);
  cudaMalloc((void **) &md->jJ_solver_gpu, md->nnz_J_solver*sizeof(int));
  cudaMalloc((void **) &md->iJ_solver_gpu, (md->nrows_J_solver+1)*sizeof(int));
  cudaMalloc((void **) &md->J_state_gpu, md->deriv_size);
  cudaMalloc((void **) &md->J_deriv_gpu, md->deriv_size);
  cudaMalloc((void **) &md->J_tmp_gpu, md->deriv_size);
  cudaMalloc((void **) &md->J_tmp2_gpu, md->deriv_size);

  double *J_solver = SM_DATA_S(md->J_solver);
  //Transfer sunindextype to int
  int *jJ_solver=(int*)malloc(sizeof(int)*md->nnz_J_solver);
  int *iJ_solver=(int*)malloc(sizeof(int)*md->nrows_J_solver+1);
  for(int i=0;i<md->nnz_J_solver;i++)
    jJ_solver[i]=SM_INDEXVALS_S(md->J_solver)[i];
  for(int i=0;i<=md->nrows_J_solver;i++)
    iJ_solver[i]=SM_INDEXPTRS_S(md->J_solver)[i];
  double *J_state = N_VGetArrayPointer(md->J_state);
  double *J_deriv = N_VGetArrayPointer(md->J_deriv);
  double *J_tmp = N_VGetArrayPointer(md->J_tmp);
  double *J_tmp2 = N_VGetArrayPointer(md->J_tmp2);

  HANDLE_ERROR(cudaMemcpy(md->J_gpu, J, md->jac_size, cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(md->J_solver_gpu, J_solver, md->jac_size, cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(md->jJ_solver_gpu, jJ_solver, md->nnz_J_solver*sizeof(int), cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(md->iJ_solver_gpu, iJ_solver, (md->nrows_J_solver+1)*sizeof(int), cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(md->J_state_gpu, J_state, md->deriv_size, cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(md->J_deriv_gpu, J_deriv, md->deriv_size, cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(md->J_tmp_gpu, J_tmp, md->deriv_size, cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemset(md->J_tmp2_gpu, 0.0, md->deriv_size));


  if(md->small_data){
    cudaMallocHost((void**)&md->jac_aux, md->jac_size);
  }

}

void update_j_state_deriv_solver_gpu(SolverData *sd, double *J){

  ModelData *md = &(sd->model_data);

  double *J_solver = SM_DATA_S(md->J_solver);
  double *J_state = N_VGetArrayPointer(md->J_state);
  double *J_deriv = N_VGetArrayPointer(md->J_deriv);

  HANDLE_ERROR(cudaMemcpy(md->J_gpu, J, md->jac_size, cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(md->J_solver_gpu, J_solver, md->jac_size, cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(md->J_state_gpu, J_state, md->deriv_size, cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(md->J_deriv_gpu, J_deriv, md->deriv_size, cudaMemcpyHostToDevice));

}


void rxn_update_env_state_gpu(ModelData *md){

  // Get a pointer to the derivative data
  int n_cells = md->n_cells;
  int n_rxn = md->n_rxn;
  int n_threads = n_rxn*n_cells; //Reaction group per number of repetitions/cells
  double *rxn_env_data = md->rxn_env_data;
  double *env = md->total_env;
  int n_blocks = ((n_threads + md->max_n_gpu_thread - 1) / md->max_n_gpu_thread);

  //Faster, use for few values
  if (md->small_data){
    //This method of passing them as a function parameter has a theoric maximum of 4kb of data
    md->rxn_env_data_gpu= rxn_env_data;
    md->env_gpu= env;
  }
  //Slower, use for large values
  else{
    //Async memcpy
    //HANDLE_ERROR(cudaMemcpyAsync(md->rxn_env_data_gpu, rxn_env_data,
    //        md->rxn_env_data_size, cudaMemcpyHostToDevice, md->stream_gpu[STREAM_RXN_ENV_GPU]));
    //HANDLE_ERROR(cudaMemcpyAsync(md->env_gpu, env, md->env_size,
    //        cudaMemcpyHostToDevice, md->stream_gpu[STREAM_ENV_GPU]));

    HANDLE_ERROR(cudaMemcpy(md->rxn_env_data_gpu, rxn_env_data, md->rxn_env_data_size, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(md->env_gpu, env, md->env_size, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(md->state_gpu, md->total_state, md->state_size, cudaMemcpyHostToDevice));
  }
}


__global__
void camp_solver_check_model_state_cuda(double *state_init, double *y,
        int *map_state_deriv, double threshhold, double replacement_value, int *status,
        int deriv_length_cell, int n_cells)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int active_threads = n_cells*deriv_length_cell;

  if(tid<active_threads) {

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
  }

}

int camp_solver_check_model_state_gpu(N_Vector solver_state, SolverData *sd,
                                      double threshhold, double replacement_value)
{
  ModelData *md = &(sd->model_data);
  itsolver *bicg = &(sd->bicg);
  int status = CAMP_SOLVER_SUCCESS; //0
  int n_cells = md->n_cells;
  int n_state_var = md->n_per_cell_state_var;
  int n_dep_var = md->n_per_cell_dep_var;
  int n_threads = n_dep_var*n_cells;
  int n_blocks = ((n_threads + md->max_n_gpu_thread - 1) / md->max_n_gpu_thread);
  int *var_type = md->var_type;
  double *state = md->total_state;
  double *y = NV_DATA_S(solver_state);
  int *map_state_deriv = md->map_state_deriv;

/*
  //HANDLE_ERROR(cudaMemcpy(md->deriv_aux, bicg->dcv_y, md->deriv_size, cudaMemcpyDeviceToHost));
  if(sd->counterDerivCPU<=5){
    printf("counterDeriv2 %d \n", sd->counterDerivCPU);
    for (int i = 0; i < NV_LENGTH_S(solver_state); i++) {
        //printf("(%d) %-le ", i + 1, NV_DATA_S(deriv)[i]);
      if(y[i]!=md->deriv_aux[i]) {
        printf("(%d) dy %-le y %-le\n", i + 1, md->deriv_aux[i], y[i]);
      }
    }
  }
*/

  camp_solver_check_model_state_cuda << < n_blocks, md->max_n_gpu_thread >> >
   (md->state_gpu, bicg->dcv_y, md->map_state_deriv_gpu,
   threshhold, replacement_value, &status, n_dep_var, n_cells);

  HANDLE_ERROR(cudaMemcpy(md->total_state, md->state_gpu, md->state_size, cudaMemcpyDeviceToHost));

/*
#ifdef PMC_DEBUG_ALL
  for (int i_cell = 0; i_cell < n_cells; i_cell++) {
   for (int i_dep_var = 0; i_dep_var < n_dep_var; i_dep_var++) {

     printf("(%d) %-le \n", i_dep_var+1,
            md->total_state[md->map_state_deriv[i_dep_var]]);
   }
}
#endif
*/

  return status;
}

void camp_solver_update_model_state_gpu(N_Vector solver_state, SolverData *sd,
                                       double threshhold, double replacement_value)
{
  ModelData *md = &(sd->model_data);
  HANDLE_ERROR(cudaMemcpy(md->state_gpu, md->total_state, md->state_size, cudaMemcpyHostToDevice));
}

__device__ void solveRXN(int i_rxn,
#ifdef BASIC_CALC_DERIV
        double *deriv_data,
#else
        TimeDerivativeGPU deriv_data,
#endif
       double *state_init, double *deriv_init,double time_step,
       int deriv_length_cell, int state_size_cell,int rxn_env_data_size_cell,
       int n_rxn, int *int_pointer, double *double_pointer,
       double *rxn_env_data_init, int *rxn_env_data_idx, double *env_init,
       int i_kernel, double *prod_rates, double *loss_rates,int threads_block
)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int tid_cell=tid%deriv_length_cell;

#ifdef BASIC_CALC_DERIV
  int i_cell=tid/n_rxn;
  i_rxn=tid%n_rxn;
  deriv_data = &( deriv_init[deriv_length_cell*i_cell]);
#else
  int i_cell = tid/deriv_length_cell;
  deriv_data.production_rates = &( prod_rates[deriv_length_cell*i_cell]);
  deriv_data.loss_rates = &( loss_rates[deriv_length_cell*i_cell]);
#endif

  //Get indices of each reaction
  double *rxn_float_data = (double *) &(((double *) double_pointer)[i_rxn]);
  int *int_data = (int *) &(((int *) int_pointer)[i_rxn]);
  int rxn_type = int_data[0];
  int *rxn_int_data = (int *) &(int_data[1*n_rxn]);

  double *state = &( state_init[state_size_cell*i_cell]);

  //Get indices for rates
  double *rxn_env_data = &(rxn_env_data_init
  [rxn_env_data_size_cell*i_cell+rxn_env_data_idx[i_rxn]]);

  //todo reduce md to allocate less memory
  ModelData md;
  md.grid_cell_state = &( state_init[state_size_cell*i_cell]);
  md.grid_cell_env = &( env_init[PMC_NUM_ENV_PARAM_*i_cell]);
  md.n_rxn = n_rxn;

#ifdef PMC_DEBUG_ALL
  if(tid==0){
    printf("[DEBUG] GPU solveDerivative tid %d, \n", tid);
  }
#endif

  switch (rxn_type) {
    //case RXN_AQUEOUS_EQUILIBRIUM :
    //fix run-time error
    //rxn_gpu_aqueous_equilibrium_calc_deriv_contrib(&md, deriv_data, rxn_int_data,
    //                                               rxn_float_data, rxn_env_data,time_step);
    //break;
    case RXN_ARRHENIUS :
      rxn_gpu_arrhenius_calc_deriv_contrib(&md, deriv_data, rxn_int_data,
                                           rxn_float_data, rxn_env_data,time_step);
      break;
    case RXN_CMAQ_H2O2 :
      rxn_gpu_CMAQ_H2O2_calc_deriv_contrib(&md, deriv_data, rxn_int_data,
                                          rxn_float_data, rxn_env_data,time_step);
      break;
    case RXN_CMAQ_OH_HNO3 :
      rxn_gpu_CMAQ_OH_HNO3_calc_deriv_contrib(&md, deriv_data, rxn_int_data,
                                           rxn_float_data, rxn_env_data,time_step);
      break;
    case RXN_CONDENSED_PHASE_ARRHENIUS :
      //rxn_gpu_condensed_phase_arrhenius_calc_deriv_contrib(&md, deriv_data, rxn_int_data,
      //                                     rxn_float_data, rxn_env_data,time_step);
      break;
    case RXN_EMISSION :
      //rxn_gpu_emission_calc_deriv_contrib(&md, deriv_data, rxn_int_data,
      //                                     rxn_float_data, rxn_env_data,time_step);
      break;
    case RXN_FIRST_ORDER_LOSS :
      //rxn_gpu_first_order_loss_calc_deriv_contrib(&md, deriv_data, rxn_int_data,
      //                                     rxn_float_data, rxn_env_data,time_step);
      break;
    case RXN_HL_PHASE_TRANSFER :
      //rxn_gpu_HL_phase_transfer_calc_deriv_contrib(&md, deriv_data, rxn_int_data,
      //                                             rxn_float_data, rxn_env_data,time_stepn);
      break;
    case RXN_PHOTOLYSIS :
      rxn_gpu_photolysis_calc_deriv_contrib(&md, deriv_data, rxn_int_data,
                                           rxn_float_data, rxn_env_data,time_step);
      break;
    case RXN_SIMPOL_PHASE_TRANSFER :
      rxn_gpu_SIMPOL_phase_transfer_calc_deriv_contrib(&md, deriv_data,
              rxn_int_data, rxn_float_data, rxn_env_data, time_step);
      break;
    case RXN_TROE :
#ifdef BASIC_CALC_DERIV
#else
      rxn_gpu_troe_calc_deriv_contrib(&md, deriv_data, rxn_int_data,
                                      rxn_float_data, rxn_env_data,time_step);
#endif
      break;
    case RXN_WET_DEPOSITION :
      //rxn_gpu_wet_deposition_calc_deriv_contrib(&md, deriv_data, rxn_int_data,
      //                                     rxn_float_data, rxn_env_data,time_step);
      break;
  }
/*
*/

}

__device__ void solveAero(int i_rxn,
#ifdef BASIC_CALC_DERIV
        double *deriv_data,
#else
                         TimeDerivativeGPU deriv_data,
#endif
                         double *state_init, double *deriv_init,double time_step,
                         int deriv_length_cell, int state_size_cell,int rxn_env_data_size_cell,
                         int n_rxn, int *int_pointer, double *double_pointer,
                         double *rxn_env_data_init, int *rxn_env_data_idx, double *env_init,
                         int i_kernel, double *prod_rates, double *loss_rates,int threads_block
)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int tid_cell=tid%deriv_length_cell;

#ifdef BASIC_CALC_DERIV
  int i_cell=tid/n_rxn;
  i_rxn=tid%n_rxn;
  deriv_data = &( deriv_init[deriv_length_cell*i_cell]);
#else
  int i_cell = tid/deriv_length_cell;
  deriv_data.production_rates = &( prod_rates[deriv_length_cell*i_cell]);
  deriv_data.loss_rates = &( loss_rates[deriv_length_cell*i_cell]);
#endif

  //Get indices of each reaction
  double *rxn_float_data = (double *) &(((double *) double_pointer)[i_rxn]);
  int *int_data = (int *) &(((int *) int_pointer)[i_rxn]);
  int rxn_type = int_data[0];
  int *rxn_int_data = (int *) &(int_data[1*n_rxn]);

  double *state = &( state_init[state_size_cell*i_cell]);

  //Get indices for rates
  double *rxn_env_data = &(rxn_env_data_init
  [rxn_env_data_size_cell*i_cell+rxn_env_data_idx[i_rxn]]);

  //todo reduce md to allocate less memory
  ModelData md;
  md.grid_cell_state = &( state_init[state_size_cell*i_cell]);
  md.grid_cell_env = &( env_init[PMC_NUM_ENV_PARAM_*i_cell]);
  md.n_rxn = n_rxn;

#ifdef PMC_DEBUG_ALL
  if(tid==0){
    printf("[DEBUG] GPU solveDerivative tid %d, \n", tid);
  }
#endif

  switch (rxn_type) {
    //case RXN_AQUEOUS_EQUILIBRIUM :
    //fix run-time error
    //rxn_gpu_aqueous_equilibrium_calc_deriv_contrib(&md, deriv_data, rxn_int_data,
    //                                               rxn_float_data, rxn_env_data,time_step);
    //break;
    case RXN_ARRHENIUS :
      rxn_gpu_arrhenius_calc_deriv_contrib(&md, deriv_data, rxn_int_data,
                                           rxn_float_data, rxn_env_data,time_step);
      break;
    case RXN_CMAQ_H2O2 :
      rxn_gpu_CMAQ_H2O2_calc_deriv_contrib(&md, deriv_data, rxn_int_data,
                                           rxn_float_data, rxn_env_data,time_step);
      break;
    case RXN_CMAQ_OH_HNO3 :
      rxn_gpu_CMAQ_OH_HNO3_calc_deriv_contrib(&md, deriv_data, rxn_int_data,
                                              rxn_float_data, rxn_env_data,time_step);
      break;
    case RXN_CONDENSED_PHASE_ARRHENIUS :
      //rxn_gpu_condensed_phase_arrhenius_calc_deriv_contrib(&md, deriv_data, rxn_int_data,
      //                                     rxn_float_data, rxn_env_data,time_step);
      break;
    case RXN_EMISSION :
      //rxn_gpu_emission_calc_deriv_contrib(&md, deriv_data, rxn_int_data,
      //                                     rxn_float_data, rxn_env_data,time_step);
      break;
    case RXN_FIRST_ORDER_LOSS :
      //rxn_gpu_first_order_loss_calc_deriv_contrib(&md, deriv_data, rxn_int_data,
      //                                     rxn_float_data, rxn_env_data,time_step);
      break;
    case RXN_HL_PHASE_TRANSFER :
      //rxn_gpu_HL_phase_transfer_calc_deriv_contrib(&md, deriv_data, rxn_int_data,
      //                                             rxn_float_data, rxn_env_data,time_stepn);
      break;
    case RXN_PHOTOLYSIS :
      rxn_gpu_photolysis_calc_deriv_contrib(&md, deriv_data, rxn_int_data,
                                            rxn_float_data, rxn_env_data,time_step);
      break;
    case RXN_SIMPOL_PHASE_TRANSFER :
      //rxn_gpu_SIMPOL_phase_transfer_calc_deriv_contrib(md, rxn_env_data,
      //        state, deriv_data, rxn_int_data, rxn_float_data, time_step,n_rxn);
      break;
    case RXN_TROE :
      rxn_gpu_troe_calc_deriv_contrib(&md, deriv_data, rxn_int_data,
                                      rxn_float_data, rxn_env_data,time_step);
      break;
    case RXN_WET_DEPOSITION :
      //rxn_gpu_wet_deposition_calc_deriv_contrib(&md, deriv_data, rxn_int_data,
      //                                     rxn_float_data, rxn_env_data,time_step);
      break;
  }
/*
*/

}

/** \brief GPU function: Solve derivative
 *
 * \param state_init Pointer to first value of state array
 * \param deriv_init Pointer to first value of derivative array
 * \param time_step Current time step being computed (s)
 * \param deriv_length_cell Derivative length for one cell
 * \param md->state_size_cell Derivative length for one cell
 * \param n_rxn Number of reactions to include
 * \param n_cells_gpu Number of cells to compute
 * \param md->int_pointer Pointer to integer reaction data
 * \param md->double_pointer Pointer to double reaction data
 * \param rxn_env_data_init Pointer to first value of reaction rates
 */
__global__ void solveDerivative(
#ifdef PMC_DEBUG_GPU
                          int counterDeriv2,
#endif
                          double *state_init, double *deriv_init,
                          double time_step, int deriv_length_cell, int state_size_cell,
                          int rxn_env_data_size_cell, int n_rxn, int n_cells,
                          int *int_pointer, double *double_pointer,
                          double *rxn_env_data_init, int *rxn_env_data_idx,
                          double *env_init, int i_kernel,
                          double *prod_rates, double *loss_rates, int threads_block,
                          double *J_solver, int *jJ_solver, int *iJ_solver,
                          double *J_state, double *J_deriv,
                          double *J_tmp, double *J_tmp2, double *y, int *map_state_deriv,
                          double threshhold, double replacement_value
                          ) //Interface CPU/GPU
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int tid_cell=tid%deriv_length_cell;
  int i_rxn;
  int active_threads = n_cells*deriv_length_cell;

#ifdef PMC_DEBUG_ALL
  if(tid==0){
    printf("[DEBUG] GPU solveDerivative tid %d, \n", tid);
  }__syncthreads();
#endif

  if(tid<active_threads){

  /* Use when all parts that need state are on the GPU (e.g. Jacobian)
    state_init[map_state_deriv[tid]] =
          y[tid] > threshhold ?
          y[tid] : replacement_value;
  */

  //N_VLinearSum(1.0, y, -1.0, md->J_state, md->J_tmp);
  cudaDevicezaxpby(1.0, y, -1.0, J_state, J_tmp, active_threads);
  //SUNMatMatvec(md->J_solver, md->J_tmp, md->J_tmp2);
  cudaDeviceSpmvCSC_block(J_tmp2, J_tmp, active_threads, J_solver, jJ_solver, iJ_solver);
  //N_VLinearSum(1.0, md->J_deriv, 1.0, md->J_tmp2, md->J_tmp);
  cudaDevicezaxpby(1.0, J_deriv, 1.0, J_tmp2, J_tmp, active_threads);
  cudaDevicesetconst(J_tmp2, 0.0, active_threads); //Reset for next iter

  //Debug
/*
  if(counterDeriv2<=10){
       printf("(%d) y %-le J_state %-le J_solver %-le J_tmp %-le J_tmp2 %-le J_deriv %-le\n",tid+1,
              y[tid], J_state[tid], J_solver[tid], J_tmp[tid], J_tmp2[tid], J_deriv[tid]);
       printf("gpu threads %d\n", active_threads);
    }
*/

#ifdef BASIC_CALC_DERIV
    i_rxn=tid%n_rxn;
    if(tid < n_rxn*n_cells){
      solveRXN(i_rxn, deriv_data, state_init, deriv_init, time_step, deriv_length_cell,
              state_size_cell, rxn_env_data_size_cell, n_rxn, int_pointer,
              double_pointer, rxn_env_data_init, rxn_env_data_idx, env_init,
              i_kernel, prod_rates, loss_rates, threads_block
              );
    }
#else
    TimeDerivativeGPU deriv_data;
    deriv_data.production_rates = prod_rates;
    deriv_data.loss_rates = loss_rates;
    deriv_data.num_spec = deriv_length_cell*n_cells;

    time_derivative_reset_gpu(deriv_data);

    __syncthreads();

    if( tid_cell < n_rxn) {
      int n_iters = n_rxn / deriv_length_cell;
      for (int i = 0; i < n_iters; i++) {
        i_rxn = tid_cell + i*deriv_length_cell;

        solveRXN(i_rxn, deriv_data, state_init, deriv_init, time_step, deriv_length_cell,
                 state_size_cell, rxn_env_data_size_cell, n_rxn, int_pointer,
                 double_pointer, rxn_env_data_init, rxn_env_data_idx, env_init,
                 i_kernel, prod_rates, loss_rates, threads_block
        );
      }

      int residual=n_rxn-(deriv_length_cell*n_iters);
      if(tid_cell < residual){
        i_rxn = tid_cell + deriv_length_cell*n_iters;

        solveRXN(i_rxn, deriv_data, state_init, deriv_init, time_step, deriv_length_cell,
                 state_size_cell, rxn_env_data_size_cell, n_rxn, int_pointer,
                 double_pointer, rxn_env_data_init, rxn_env_data_idx, env_init,
                 i_kernel, prod_rates, loss_rates, threads_block
        );
      }
    }
    __syncthreads();

    /*md->grid_cell_aero_rep_env_data =
    &(md->aero_rep_env_data[i_cell * md->n_aero_rep_env_data]);
md->grid_cell_sub_model_env_data =
    &(md->sub_model_env_data[i_cell * md->n_sub_model_env_data]);
*/

/*

    int n_aero_rep=1;

    if( tid_cell < n_aero_rep) {
      int n_iters = n_aero_rep / deriv_length_cell;
      for (int i = 0; i < n_iters; i++) {
        i_rxn = tid_cell + i*deriv_length_cell;

        solveAero(i_rxn, deriv_data, state_init, deriv_init, time_step, deriv_length_cell,
                 state_size_cell, rxn_env_data_size_cell, n_aero_rep, int_pointer,
                 double_pointer, rxn_env_data_init, rxn_env_data_idx, env_init,
                 i_kernel, prod_rates, loss_rates, threads_block
        );
      }

      int residual=n_aero_rep-(deriv_length_cell*n_iters);
      if(tid_cell < residual){
        i_rxn = tid_cell + deriv_length_cell*n_iters;

        solveAero(i_rxn, deriv_data, state_init, deriv_init, time_step, deriv_length_cell,
                 state_size_cell, rxn_env_data_size_cell, n_aero_rep, int_pointer,
                 double_pointer, rxn_env_data_init, rxn_env_data_idx, env_init,
                 i_kernel, prod_rates, loss_rates, threads_block
        );
      }
    }
    */

    /*if(tid==0){
      printf("tid %d time_deriv.production_rates %-le time_deriv.loss_rates %-le\n",
              tid, deriv_data.production_rates[tid],
             deriv_data.loss_rates[tid]);
    }*/

    deriv_data.production_rates = prod_rates;
    deriv_data.loss_rates = loss_rates;
    __syncthreads();
    time_derivative_output_gpu(deriv_data, deriv_init, J_tmp,0);
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
void rxn_calc_deriv_gpu(SolverData *sd, N_Vector deriv, double time_step,
        double threshhold, double replacement_value) {

  ModelData *md = &(sd->model_data);
  itsolver *bicg = &(sd->bicg);
  double *deriv_data = N_VGetArrayPointer(deriv);
  int n_cells = md->n_cells;
  int n_kernels = 1; // Divide load into multiple kernel calls
  //todo n_kernels case division left residual, an extra kernel computes remain residual
  int n_rxn = md->n_rxn;
#ifdef BASIC_CALC_DERIV
  int total_threads = n_rxn*n_cells/n_kernels; //Reaction group per number of repetitions/cells
  int threads_block = md->max_n_gpu_thread;
#else
  int max_threads = md->max_n_gpu_thread;
  int size_cell = md->n_per_cell_dep_var;
  int total_threads = size_cell * n_cells/n_kernels;
  int n_shr_empty = max_threads%size_cell;
  int threads_block = max_threads - n_shr_empty; //last multiple of size_cell before max_threads
#endif
  int n_blocks = ((total_threads + threads_block - 1) / threads_block);
  double *rxn_env_data = md->rxn_env_data;
  double *J_tmp = N_VGetArrayPointer(md->J_tmp);

 //debug
  /*if(sd->counterDerivCPU>=0){
    printf("camp solver_run start [(id),conc], n_state_var %d, n_cells %d\n", md->n_per_cell_state_var, n_cells);
    printf("n_deriv %d\n", md->n_per_cell_dep_var);
    for (int i = 0; i < md->n_per_cell_state_var*n_cells; i++) {
      //printf("(%d) %-le \n",i+1, md->total_state[i]);
    }
  }*/

  //Reset deriv gpu
  HANDLE_ERROR(cudaMemset(md->deriv_gpu_data, 0.0, md->deriv_size));

#ifdef PMC_DEBUG_GPU
  //timeDerivSend += (clock() - t1);
  //clock_t t2 = clock();

  cudaEventRecord(md->startDerivKernel);

#endif

  //Loop to test multiple kernel executions
  for (int i_kernel=0; i_kernel<n_kernels; i_kernel++){
    //cudaDeviceSynchronize();
    solveDerivative << < (n_blocks), threads_block >> >(
#ifdef PMC_DEBUG_GPU
    sd->counterDerivCPU,
#endif
     md->state_gpu, md->deriv_gpu_data, time_step, md->n_per_cell_dep_var,
     md->n_per_cell_state_var, md->n_rxn_env_data,
     n_rxn, n_cells, md->int_pointer_gpu, md->double_pointer_gpu,
     md->rxn_env_data_gpu, md->rxn_env_data_idx_gpu, md->env_gpu,
     i_kernel,md->prod_rates, md->loss_rates,threads_block,md->J_solver_gpu,
     md->jJ_solver_gpu, md->iJ_solver_gpu, md->J_state_gpu,
     md->J_deriv_gpu,md->J_tmp_gpu,md->J_tmp2_gpu,bicg->dcv_y,
     md->map_state_deriv_gpu,threshhold,replacement_value);
  }

#ifdef PMC_DEBUG_GPU
  /*cudaDeviceSynchronize();
  timeDerivKernel += (clock() - t2);
  t3 = clock();*/


  cudaEventRecord(md->stopDerivKernel);
  cudaEventSynchronize(md->stopDerivKernel);
  float msDerivKernel = 0.0;
  cudaEventElapsedTime(&msDerivKernel, md->startDerivKernel, md->stopDerivKernel);
  md->timeDerivKernel+= msDerivKernel;


#endif

  //Use pinned memory for few values
  if (md->small_data){
    HANDLE_ERROR(cudaMemcpy(md->deriv_aux, md->deriv_gpu_data, md->deriv_size, cudaMemcpyDeviceToHost));
    memcpy(deriv_data, md->deriv_aux, md->deriv_size);
  }
  else {
    //Async
    //HANDLE_ERROR(cudaMemcpyAsync(md->deriv_aux, md->deriv_gpu_data,
    //md->deriv_size, cudaMemcpyDeviceToHost, md->stream_gpu[STREAM_DERIV_GPU]));

    //Sync
    //HANDLE_ERROR(cudaMemcpy(md->deriv_aux, md->deriv_gpu_data, md->deriv_size, cudaMemcpyDeviceToHost));
    //HANDLE_ERROR(cudaMemcpy(deriv_data, md->deriv_gpu_data, md->deriv_size, cudaMemcpyDeviceToHost));
  }

  //cudaDeviceSynchronize();

/* //debug
  if(sd->counterDerivCPU>=0){
    n_cells=1;
    int size_j = NV_LENGTH_S(deriv) / n_cells;
    printf("length_deriv %d \n", size_j);
    for (int i = 0; i < n_cells; i++) {
      printf("cell %d \n", i);
      for (int j = 0; j < size_j; j++) {  // NV_LENGTH_S(deriv)
        printf("(%d) %-le ", j + 1, NV_DATA_S(deriv)[j+i*size_j]);
      }
      printf("\n");
    }
  }
*/

#ifdef PMC_DEBUG_GPU
  /*timeDerivReceive += (clock() - t3);
  timeDeriv += (clock() - t1);
  t3 = clock();*/
#endif
}

/** \brief Fusion deriv data calculated from CPU and GPU
 * (Calculations from CPU & GPU or GPU async case)
 *
 * \param md Pointer to the model data
 * \param deriv NVector to hold the calculated vector
 * \param time_step Current model time step (s)
 */
void rxn_fusion_deriv_gpu(ModelData *md, N_Vector deriv) {

  // Get a pointer to the derivative data
  realtype *deriv_data = N_VGetArrayPointer(deriv);

  cudaDeviceSynchronize();
  //HANDLE_ERROR(cudaMemsetAsync(md->deriv_gpu_data, 0.0,
  //        md->deriv_size, md->stream_gpu[STREAM_DERIV_GPU]));

  if (md->small_data){
  }
  else {
    for (int i = 0; i < NV_LENGTH_S(deriv); i++) {  // NV_LENGTH_S(deriv)
      //Add to deriv the auxiliar contributions from gpu
      deriv_data[i] += md->deriv_aux[i];
    }
  }

}

#ifdef PMC_USE_GPU
#else
void rxn_calc_deriv_cpu(ModelData *md, double *deriv_data,
                    double time_step) {

  //clock_t t = clock();

  // Get the number of reactions
  int n_rxn = md->n_rxn;

  // Loop through the reactions advancing the rxn_data pointer each time
  for (int i_rxn = 0; i_rxn < n_rxn; i_rxn++) {
    // Get pointers to the reaction data
    int *rxn_int_data =
        &(md->rxn_int_data[md->rxn_int_indices[i_rxn]]);
    double *rxn_float_data =
        &(md->rxn_float_data[md->rxn_float_indices[i_rxn]]);
    double *rxn_env_data =
        &(md->grid_cell_rxn_env_data[md->rxn_env_idx[i_rxn]]);

    // Get the reaction type
    int rxn_type = *(rxn_int_data++);

    // Call the appropriate function
    switch (rxn_type) {
      case RXN_AQUEOUS_EQUILIBRIUM:
        rxn_gpu_aqueous_equilibrium_calc_deriv_contrib(md, deriv_data,
                                                   rxn_int_data, rxn_float_data,
                                                   rxn_env_data, time_step);
        break;
      case RXN_ARRHENIUS:
        rxn_gpu_arrhenius_calc_deriv_contrib(md, deriv_data, rxn_int_data,
                                         rxn_float_data, rxn_env_data,
                                         time_step);
        break;
      case RXN_CMAQ_H2O2:
        rxn_gpu_CMAQ_H2O2_calc_deriv_contrib(md, deriv_data, rxn_int_data,
                                         rxn_float_data, rxn_env_data,
                                         time_step);
        break;
      case RXN_CMAQ_OH_HNO3:
        rxn_gpu_CMAQ_OH_HNO3_calc_deriv_contrib(md, deriv_data,
                                            rxn_int_data, rxn_float_data,
                                            rxn_env_data, time_step);
        break;
      case RXN_CONDENSED_PHASE_ARRHENIUS:
        rxn_gpu_condensed_phase_arrhenius_calc_deriv_contrib(
            md, deriv_data, rxn_int_data, rxn_float_data, rxn_env_data,
            time_step);
        break;
      case RXN_EMISSION:
        rxn_gpu_emission_calc_deriv_contrib(md, deriv_data, rxn_int_data,
                                        rxn_float_data, rxn_env_data,
                                        time_step);
        break;
      case RXN_FIRST_ORDER_LOSS:
        rxn_gpu_first_order_loss_calc_deriv_contrib(md, deriv_data,
                                                rxn_int_data, rxn_float_data,
                                                rxn_env_data, time_step);
        break;
      case RXN_HL_PHASE_TRANSFER:
        //rxn_gpu_HL_phase_transfer_calc_deriv_contrib(md, deriv_data,
        //                                         rxn_int_data, rxn_float_data,
        //                                         rxn_env_data, time_step);
        break;
      case RXN_PHOTOLYSIS:
        rxn_gpu_photolysis_calc_deriv_contrib(md, deriv_data, rxn_int_data,
                                          rxn_float_data, rxn_env_data,
                                          time_step);
        break;
      case RXN_SIMPOL_PHASE_TRANSFER:
        //rxn_gpu_SIMPOL_phase_transfer_calc_deriv_contrib(
        //   md, deriv_data, rxn_int_data, rxn_float_data, rxn_env_data,
        //    time_step);
        break;
      case RXN_TROE:
        rxn_gpu_troe_calc_deriv_contrib(md, deriv_data, rxn_int_data,
                                    rxn_float_data, rxn_env_data, time_step);
        break;
      case RXN_WET_DEPOSITION:
        rxn_gpu_wet_deposition_calc_deriv_contrib(md, deriv_data,
                                              rxn_int_data, rxn_float_data,
                                              rxn_env_data, time_step);
        break;
    }
  }

  //timeDeriv += (clock()- t);

}
#endif

/** \brief GPU function: Solve jacobian
 *
 * \param state_init Pointer to first value of state array
 * \param jac_init Pointer to first value of jacobian array
 * \param time_step Current time step being computed (s)
 * \param jac_length_cell jacobian length for one cell
 * \param md->state_size_cell jacobian length for one cell
 * \param n_rxn Number of reactions to include
 * \param n_cells_gpu Number of cells to compute
 * \param md->int_pointer Pointer to integer reaction data
 * \param md->double_pointer Pointer to double reaction data
 * \param rxn_env_data_init Pointer to first value of reaction rates
 */
__global__ void solveJacobian(double *state_init, double *jac_init,
                              double time_step, int jac_length_cell, int state_size_cell,
                              int rxn_env_data_size_cell, int n_rxn,
                              int n_cells, int *int_pointer, double *double_pointer,
                              double *rxn_env_data_init, int *rxn_env_data_idx) //Interface CPU/GPU
{
  //Get thread id
  /*int index = blockIdx.x * blockDim.x + threadIdx.x;

  //Maximum number of threads to compute all reactions
  if(index < n_rxn*n_cells){

    //Thread index for jac and state,
    // till we don't finish all reactions of a cell, we stay on same index
    int i_cell=index/n_rxn;
    int i_rxn=index%n_rxn;

    //Get indices of each reaction
    int *int_data = (int *) &(((int *) int_pointer)[i_rxn]); //Same indices for each cell
    double *float_data = (double *) &(((double *) double_pointer)[i_rxn]);
    int rxn_type = int_data[0];
    int *rxn_int_data = (int *) &(int_data[1*n_rxn]);

    //Get indices for concentrations
    double *jac_data = &( jac_init[jac_length_cell*i_cell]);
    double *state = &( state_init[state_size_cell*i_cell]);

    //Get indices for rates
    double *rxn_env_data = &(rxn_env_data_init
    [rxn_env_data_size_cell*i_cell+rxn_env_data_idx[i_rxn]]);

    switch (rxn_type) {
      case RXN_AQUEOUS_EQUILIBRIUM :
        //rxn_gpu_aqueous_equilibrium_calc_jac_contrib(rxn_env_data,
        //        state, jac_data, rxn_int_data, rxn_float_data, time_step, n_rxn);
        break;
      case RXN_ARRHENIUS :
        rxn_gpu_arrhenius_calc_jac_contrib(rxn_env_data,
                                           state, jac_data, rxn_int_data, rxn_float_data, time_step,n_rxn);
        break;
      case RXN_CMAQ_H2O2 :
        rxn_gpu_CMAQ_H2O2_calc_jac_contrib(rxn_env_data,
                                           state, jac_data, rxn_int_data, rxn_float_data, time_step,n_rxn);
        break;
      case RXN_CMAQ_OH_HNO3 :
        rxn_gpu_CMAQ_OH_HNO3_calc_jac_contrib(rxn_env_data,
                                              state, jac_data, rxn_int_data, rxn_float_data, time_step,n_rxn);
        break;
      case RXN_CONDENSED_PHASE_ARRHENIUS :
        //rxn_gpu_condensed_phase_arrhenius_calc_jac_contrib(rxn_env_data,
        //        state, jac_data, rxn_int_data, rxn_float_data, time_step,n_rxn);
        break;
      case RXN_EMISSION :
        rxn_gpu_emission_calc_jac_contrib(rxn_env_data,
                                          state, jac_data, rxn_int_data, rxn_float_data, time_step,n_rxn);
        break;
      case RXN_FIRST_ORDER_LOSS :
        rxn_gpu_first_order_loss_calc_jac_contrib(rxn_env_data,
                                                  state, jac_data, rxn_int_data, rxn_float_data, time_step,n_rxn);
        break;
      case RXN_HL_PHASE_TRANSFER :
        //rxn_gpu_HL_phase_transfer_calc_jac_contrib(rxn_env_data,
        //        state, jac_data, rxn_int_data, rxn_float_data, time_step, n_rxn);
        break;
      case RXN_PHOTOLYSIS :
        rxn_gpu_photolysis_calc_jac_contrib(rxn_env_data,
                                            state, jac_data, rxn_int_data, rxn_float_data, time_step,n_rxn);
        break;
      case RXN_SIMPOL_PHASE_TRANSFER :
        //rxn_gpu_SIMPOL_phase_transfer_calc_jac_contrib(rxn_env_data,
        //        state, jac_data, rxn_int_data, rxn_float_data, time_step,n_rxn);
        break;
      case RXN_TROE :
        rxn_gpu_troe_calc_jac_contrib(rxn_env_data,
                                      state, jac_data, rxn_int_data, rxn_float_data, time_step, n_rxn);
        break;
      case RXN_WET_DEPOSITION :
        rxn_gpu_wet_deposition_calc_jac_contrib(rxn_env_data,
                                                state, jac_data, rxn_int_data, rxn_float_data, time_step,n_rxn);
        break;
    }
    __syncthreads();
  }
   */

}


/** \brief Calculate the Jacobian on GPU
 *
 * \param md Pointer to the model data
 * \param J Jacobian to be calculated
 * \param time_step Current model time step (s)
 */

void rxn_calc_jac_gpu(SolverData *sd, SUNMatrix jac, double time_step) {

  //TODO: Fix jacobian with jac_ids...

  /*

  // Get a pointer to the jacobian data
  ModelData *md = &(sd->model_data);
  double *jac_data = SM_DATA_S(jac);
  int n_cells = md->n_cells;
  int n_rxn = md->n_rxn;
  int n_threads = n_rxn*n_cells; //Reaction group per number of repetitions/cells
  int n_blocks = ((n_threads + md->max_n_gpu_thread - 1) / md->max_n_gpu_thread);
  double *state = md->total_state;
  double *rxn_env_data = md->rxn_env_data;

  //Faster, use for few values
  if (md->small_data){
    //This method of passing them as a function parameter has a theoric maximum of 4kb of data
    md->state_gpu= state;
  }
    //Slower, use for large values
  else{
    HANDLE_ERROR(cudaMemcpy(md->state_gpu, state, md->state_size, cudaMemcpyHostToDevice));
  }

  HANDLE_ERROR(cudaMemset(md->J_solver_gpu, 0, md->jac_size));

  solveJacobian << < (n_blocks), md->max_n_gpu_thread >> >
    (md->state_gpu, md->J_solver_gpu, time_step, md->n_per_cell_rxn_jac_elem,
    md->n_per_cell_state_var, md->n_rxn_env_data,
    n_rxn, n_cells, md->int_pointer_gpu, md->double_pointer_gpu, md->rxn_env_data_gpu, md->rxn_env_data_idx_gpu);

  cudaDeviceSynchronize();// Secure cuda synchronization

  //Use pinned memory for few values
  if (md->small_data){
    HANDLE_ERROR(cudaMemcpy(md->jac_aux, md->J_solver_gpu, md->jac_size, cudaMemcpyDeviceToHost));
    memcpy(jac_data, md->jac_aux, md->jac_size);
  }
  else {
    HANDLE_ERROR(cudaMemcpy(jac_data, md->J_solver_gpu, md->jac_size, cudaMemcpyDeviceToHost));
  }

*/

}

/** \brief Free GPU data structures
 */
void free_gpu_cu(ModelData *md) {

#ifdef PMC_DEBUG_GPU

  printf("timeDerivKernel %lf\n", md->timeDerivKernel/1000);

#endif

  //for (int i = 0; i < n_streams; ++i)
  //  HANDLE_ERROR( cudaStreamDestroy(md->stream_gpu[i]) );
/*

  */
  //free(md->jac_aux);
  HANDLE_ERROR(cudaFree(md->int_pointer_gpu));
  HANDLE_ERROR(cudaFree(md->double_pointer_gpu));
  HANDLE_ERROR(cudaFree(md->deriv_gpu_data));
  //HANDLE_ERROR(cudaFree(J_solver_gpu));

  if(md->small_data){
  }
  else{
    free(md->deriv_aux);
    HANDLE_ERROR(cudaFree(md->state_gpu));
    HANDLE_ERROR(cudaFree(md->env_gpu));
    HANDLE_ERROR(cudaFree(md->rxn_env_data_gpu));
    HANDLE_ERROR(cudaFree(md->rxn_env_data_idx_gpu));

  }

/*
  HANDLE_ERROR(cudaFree(int_pointer_gpu));
  HANDLE_ERROR(cudaFree(double_pointer_gpu));
  HANDLE_ERROR(cudaFree(deriv_gpu_data));
  HANDLE_ERROR(cudaFree(J_solver_gpu));

  if(small_data){
  }
  else{
    HANDLE_ERROR(cudaFree(state_gpu));
    HANDLE_ERROR(cudaFree(rxn_env_data_gpu));
    HANDLE_ERROR(cudaFree(rxn_env_data_idx_gpu));
  }
*/
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

// Old code (Not used now, but could be useful)
/*
 //use this instead of normal update_model_state? is less code
int camp_solver_update_model_state_cpu(N_Vector solver_state, ModelData *md,
                                       realtype threshhold, realtype replacement_value)
{
  int status = CAMP_SOLVER_FAIL;
  int n_cells = md->n_cells;
  int n_state_var = md->n_per_cell_state_var;
  int n_dep_var = md->n_per_cell_dep_var;
  int n_threads = n_state_var*n_cells;
  int n_blocks = ((n_threads + md->max_n_gpu_thread - 1) / md->max_n_gpu_thread);
  int *var_type = md->var_type;
  double *state = md->total_state;
  double *y = NV_DATA_S(solver_state);
  int *map_state_deriv = md->map_state_deriv;

  for(int i_dep_var = 0; i_dep_var < n_dep_var*n_cells; i_dep_var++)
  {
    if (NV_DATA_S(solver_state)[i_dep_var] > -SMALL) {
      md->total_state[map_state_deriv[i_dep_var]] =
              NV_DATA_S(solver_state)[i_dep_var] > threshhold
              ? NV_DATA_S(solver_state)[i_dep_var] : replacement_value;
      status = CAMP_SOLVER_SUCCESS;
    } else { //error
#ifdef FAILURE_DETAIL
      printf("\nFailed model state update: [spec %d] = %le", i_spec,
                 NV_DATA_S(solver_state)[i_dep_var]);
#endif
      status = CAMP_SOLVER_FAIL;
      break;
    }
  }
  return status;
}
*/
}

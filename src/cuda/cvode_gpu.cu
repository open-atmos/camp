/* Copyright (C) 2021 Barcelona Supercomputing Center and University of
 * Illinois at Urbana-Champaign
 * SPDX-License-Identifier: MIT
 */

#include "itsolver_gpu.h"

extern "C" {
#include "cvode_gpu.h"
#include "rxns_gpu.h"
#include "time_derivative_gpu.h"
#include "Jacobian_gpu.h"
}

#ifdef CAMP_USE_MPI
#include <mpi.h>
#endif

#define CV_SUCCESS               0
#define DO_ERROR_TEST    +2
#define PREDICT_AGAIN    +3
#define CONV_FAIL        +4
#define TRY_AGAIN        +5
#define FIRST_CALL       +6
#define PREV_CONV_FAIL   +7
#define PREV_ERR_FAIL    +8
#define RHSFUNC_RECVR    +9
#define NUM_TESTS    5     /* number of error test quantities     */
#define PT1     RCONST(0.1)     /* real 0.1     */
#define POINT2  RCONST(0.2)     /* real 0.2     */
#define FOURTH  RCONST(0.25)    /* real 0.25    */
#define TWO     RCONST(2.0)     /* real 2.0     */
#define THREE   RCONST(3.0)     /* real 3.0     */
#define FOUR    RCONST(4.0)     /* real 4.0     */
#define FIVE    RCONST(5.0)     /* real 5.0     */
#define TWELVE  RCONST(12.0)    /* real 12.0    */
#define HUNDRED RCONST(100.0)   /* real 100.0   */
#define CAMP_TINY RCONST(1.0e-30) /* small number for CAMP */
#define DO_ERROR_TEST    +2
#define PREDICT_AGAIN    +3
#define CONV_FAIL        +4
#define TRY_AGAIN        +5
#define FIRST_CALL       +6
#define PREV_CONV_FAIL   +7
#define PREV_ERR_FAIL    +8
#define RHSFUNC_RECVR    +9
#define RTFOUND          +1
#define CLOSERT          +3
#define CV_NN  0
#define CV_SS  1
#define CV_SV  2
#define CV_WF  3
#define FUZZ_FACTOR RCONST(100.0)
#define HLB_FACTOR RCONST(100.0)
#define HUB_FACTOR RCONST(0.1)
#define H_BIAS     HALF
#define MAX_ITERS  4000
#define CORTES RCONST(0.1)
#define THRESH RCONST(1.5)
#define ETAMX1 RCONST(10000.0)
#define ETAMX2 RCONST(10.0)
#define ETAMX3 RCONST(10.0)
#define ETAMXF RCONST(0.2)
#define ETAMIN RCONST(0.1)
#define ETACF  RCONST(0.25)
#define ADDON  RCONST(0.000001)
#define BIAS1  RCONST(6.0)
#define BIAS2  RCONST(6.0)
#define BIAS3  RCONST(10.0)
#define ONEPSM RCONST(1.000001)
#define SMALL_NST    10
#define MXNCF        10
#define MXNEF         7
#define MXNEF1        3
#define SMALL_NEF     2
#define LONG_WAIT    10
#define NLS_MAXCOR 3
#define CRDOWN RCONST(0.3)
#define DGMAX  RCONST(0.3)
#define RDIV      2.0
#define MSBP       20
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
#define CAMP_SOLVER_SUCCESS 0
#define CAMP_SOLVER_FAIL 1

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

static void HandleError(cudaError_t err,
                        const char *file,
                        int line) {
  if (err != cudaSuccess) {
    printf("%s in %s at line %d\n", cudaGetErrorString(err),
           file, line);
    exit(EXIT_FAILURE);
  }
}

#ifdef DEBUG_CVODE_GPU
__device__
void printmin(ModelDataGPU *md,double* y, const char *s) {

  __syncthreads();
  extern __shared__ double flag_shr2[];
  int i= threadIdx.x + blockDim.x*blockIdx.x;
  __syncthreads();

  double min;
  cudaDevicemin(&min, y[i], flag_shr2, md->n_shr_empty);
  __syncthreads();
  if(i==0)printf("%s min %le\n",s,min);
  __syncthreads();

}
#endif

int nextPowerOfTwoCVODE(int v){

  v--;
  v |= v >> 1;
  v |= v >> 2;
  v |= v >> 4;
  v |= v >> 8;
  v |= v >> 16;
  v++;

  return v;
}

void createLinearSolver_cvode(SolverData *sd)
{
  ModelDataGPU *mGPU = sd->mGPU;
  mGPU->maxIt=1000;
  mGPU->tolmax=1.0e-30;
  int nrows = mGPU->nrows;
  int len_cell = mGPU->nrows/mGPU->n_cells;
  double ** dr0 = &mGPU->dr0;
  double ** dr0h = &mGPU->dr0h;
  double ** dn0 = &mGPU->dn0;
  double ** dp0 = &mGPU->dp0;
  double ** dt = &mGPU->dt;
  double ** ds = &mGPU->ds;
  double ** dAx2 = &mGPU->dAx2;
  double ** dy = &mGPU->dy;
  double ** dz = &mGPU->dz;
  double ** ddiag = &mGPU->ddiag;
  cudaMalloc(dr0,nrows*sizeof(double));
  cudaMalloc(dr0h,nrows*sizeof(double));
  cudaMalloc(dn0,nrows*sizeof(double));
  cudaMalloc(dp0,nrows*sizeof(double));
  cudaMalloc(dt,nrows*sizeof(double));
  cudaMalloc(ds,nrows*sizeof(double));
  cudaMalloc(dAx2,nrows*sizeof(double));
  cudaMalloc(dy,nrows*sizeof(double));
  cudaMalloc(dz,nrows*sizeof(double));
  HANDLE_ERROR(cudaMalloc(ddiag,nrows*sizeof(double)));
  int blocks = mGPU->blocks;
  mGPU->aux=(double*)malloc(sizeof(double)*blocks);

}

#ifdef DEV_JOIN_GPU_INIT_IN_ONE_CALL
__global__
void init_jac_partials_cvode(double* production_partials, double* loss_partials) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  production_partials[tid]=0.0;
  loss_partials[tid]=0.0;
}

int jacobian_initialize_cuda_cvode(SolverData *sd) {
  ModelDataGPU *mGPU = sd->mGPU;
  Jacobian *jac = &sd->jac;

#ifdef DEBUG_jacobian_initialize_gpu
  printf("jacobian_initialize_gpu start \n");
#endif

  int offset_nnz = 0;
  for (int iDevice = sd->startDevice; iDevice < sd->endDevice; iDevice++) {
    cudaSetDevice(iDevice);
    sd->mGPU = &(sd->mGPUs[iDevice]);
    mGPU = sd->mGPU;

    JacobianGPU *jacgpu = &(mGPU->jac);

    cudaMalloc((void **) &jacgpu->num_elem, 1 * sizeof(jacgpu->num_elem));
    cudaMemcpy(jacgpu->num_elem, &jac->num_elem, 1 * sizeof(jacgpu->num_elem), cudaMemcpyHostToDevice);

    int num_elem = jac->num_elem * mGPU->n_cells;
    cudaMalloc((void **) &(jacgpu->production_partials), num_elem * sizeof(jacgpu->production_partials));

    HANDLE_ERROR(cudaMalloc((void **) &(jacgpu->loss_partials), num_elem * sizeof(jacgpu->loss_partials)));

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, iDevice);
    int threads_block = prop.maxThreadsPerBlock;;
    int blocks = (num_elem +threads_block - 1) / threads_block;
    init_jac_partials_cvode <<<blocks,threads_block>>>(jacgpu->production_partials,jacgpu->loss_partials);

    offset_nnz += num_elem;
  }

#ifdef DEBUG_jacobian_initialize_gpu
  printf("jacobian_initialize_gpu end \n");
#endif

  return 1;
}

__global__
void init_J_tmp2_cuda_cvode(double* J_tmp2) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  J_tmp2[tid]=0.0;
}

void init_jac_cuda_cvode(SolverData *sd){

  ModelData *md = &(sd->model_data);
  ModelDataGPU *mGPU;

#ifdef DEBUG_init_jac_cuda

  printf("init_jac_cuda start \n");

#endif

  int offset_nnz_J_solver = 0;
  int offset_nrows = 0;
  for (int iDevice = sd->startDevice; iDevice < sd->endDevice; iDevice++) {
    cudaSetDevice(iDevice);
    sd->mGPU = &(sd->mGPUs[iDevice]);
    mGPU = sd->mGPU;

    mGPU->jac_size = md->n_per_cell_solver_jac_elem * mGPU->n_cells * sizeof(double);
    mGPU->nnz_J_solver = SM_NNZ_S(md->J_solver)/md->n_cells*mGPU->n_cells;
    mGPU->nrows_J_solver = SM_NP_S(md->J_solver)/md->n_cells*mGPU->n_cells;

    //mGPU->n_per_cell_solver_jac_elem = md->n_per_cell_solver_jac_elem;
    cudaMalloc((void **) &mGPU->J, mGPU->jac_size);
    cudaMalloc((void **) &mGPU->J_solver, mGPU->jac_size);
    cudaMalloc((void **) &mGPU->J_state, mGPU->deriv_size);
    cudaMalloc((void **) &mGPU->J_deriv, mGPU->deriv_size);
    cudaMalloc((void **) &mGPU->J_tmp, mGPU->deriv_size);
    cudaMalloc((void **) &mGPU->J_tmp2, mGPU->deriv_size);
    cudaMalloc((void **) &mGPU->jac_map, sizeof(JacMap) * md->n_mapped_values);
    HANDLE_ERROR(cudaMalloc((void **) &mGPU->n_mapped_values, 1 * sizeof(int)));

#ifdef DEBUG_init_jac_cuda
    printf("md->n_per_cell_dep_var %d sd->jac.num_spec %d md->n_per_cell_solver_jac_elem %d "
           "md->n_mapped_values %d %d jac->num_elem %d offset_nnz_J_solver %d  mGPU->nnz_J_solver %d mGPU->nrows_J_solver %d\n",
           md->n_per_cell_dep_var,sd->jac.num_spec,md->n_per_cell_solver_jac_elem, md->n_mapped_values,
           sd->jac.num_elem, offset_nnz_J_solver,mGPU->nnz_J_solver, mGPU->nrows_J_solver);
#endif

    cudaMalloc((void **) &mGPU->jJ_solver, mGPU->nnz_J_solver/mGPU->n_cells * sizeof(int));
    cudaMalloc((void **) &mGPU->iJ_solver, (mGPU->nrows_J_solver/mGPU->n_cells + 1) * sizeof(int));
    int *jJ_solver = (int *) malloc(sizeof(int) * mGPU->nnz_J_solver/mGPU->n_cells);
    int *iJ_solver = (int *) malloc(sizeof(int) * (mGPU->nrows_J_solver/mGPU->n_cells) + 1);
    for (int i = 0; i < mGPU->nnz_J_solver/mGPU->n_cells; i++)
      jJ_solver[i] = SM_INDEXVALS_S(md->J_solver)[i];
    //printf("J_solver PTRS:\n");
    for (int i = 0; i <= mGPU->nrows_J_solver/mGPU->n_cells; i++){
      iJ_solver[i] = SM_INDEXPTRS_S(md->J_solver)[i];
      //printf("%lld \n",iJ_solver[i]);
    }
    cudaMemcpy(mGPU->jJ_solver, jJ_solver, mGPU->nnz_J_solver/mGPU->n_cells * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(mGPU->iJ_solver, iJ_solver, (mGPU->nrows_J_solver/mGPU->n_cells + 1) * sizeof(int), cudaMemcpyHostToDevice);
    free(jJ_solver);
    free(iJ_solver);

    HANDLE_ERROR(cudaMemcpy(mGPU->J, sd->J+offset_nnz_J_solver, mGPU->jac_size, cudaMemcpyHostToDevice));
    double *J_solver = SM_DATA_S(md->J_solver)+offset_nnz_J_solver;
    cudaMemcpy(mGPU->J_solver, J_solver, mGPU->jac_size, cudaMemcpyHostToDevice);

    double *J_state = N_VGetArrayPointer(md->J_state)+offset_nrows;
    HANDLE_ERROR(cudaMemcpy(mGPU->J_state, J_state, mGPU->deriv_size, cudaMemcpyHostToDevice));
    double *J_deriv = N_VGetArrayPointer(md->J_deriv)+offset_nrows;
    double *J_tmp2 = N_VGetArrayPointer(md->J_tmp2)+offset_nrows;
    HANDLE_ERROR(cudaMemcpy(mGPU->J_deriv, J_deriv, mGPU->deriv_size, cudaMemcpyHostToDevice));

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, iDevice);
    int threads_block = prop.maxThreadsPerBlock;;
    int blocks = (mGPU->deriv_size/sizeof(double)+threads_block - 1) / threads_block;
    init_J_tmp2_cuda_cvode <<<blocks,threads_block>>>(mGPU->J_tmp2);
    HANDLE_ERROR(cudaMemcpy(mGPU->jac_map, md->jac_map, sizeof(JacMap) * md->n_mapped_values, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(mGPU->n_mapped_values, &md->n_mapped_values, 1 * sizeof(int), cudaMemcpyHostToDevice));

    offset_nnz_J_solver += mGPU->nnz_J_solver;
    offset_nrows += md->n_per_cell_dep_var* mGPU->n_cells;
  }

  jacobian_initialize_cuda_cvode(sd);

#ifdef DEBUG_init_jac_cuda

  printf("init_jac_cuda end \n");

#endif

}

void set_int_double_cuda_cvode(
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

void solver_init_int_double_cuda_cvode(SolverData *sd) {

  ModelData *md = &(sd->model_data);
  ModelDataGPU *mGPU;

#ifdef DEBUG_solver_init_int_double_gpu
  printf("solver_init_int_double_gpu start \n");
#endif

  for (int iDevice = sd->startDevice; iDevice < sd->endDevice; iDevice++) {
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

    set_int_double_cuda_cvode(
        md->n_rxn, mGPU->rxn_env_data_idx_size,
        md->rxn_int_data, md->rxn_float_data,
        md->rxn_int_indices, md->rxn_float_indices, md->rxn_env_idx,
        sd
    );

#endif

  }

#ifdef DEBUG_solver_init_int_double_gpu
  printf("solver_init_int_double_gpu end \n");
#endif

}

void solver_new_gpu_cu_cvode(SolverData *sd) {
  ModelData *md = &(sd->model_data);
  ModelDataGPU *mGPU;

  int n_dep_var=md->n_per_cell_dep_var;
  int n_state_var=md->n_per_cell_state_var;
  int n_rxn=md->n_rxn;
  int n_rxn_int_param=md->n_rxn_int_param;
  int n_rxn_float_param=md->n_rxn_float_param;
  int n_rxn_env_param=md->n_rxn_env_data;
  int n_cells_total=md->n_cells;

  sd->mGPUs = (ModelDataGPU *)malloc(sd->nDevices * sizeof(ModelDataGPU));
  int remainder = n_cells_total % sd->nDevices;

  int nDevicesMax;
  cudaGetDeviceCount(&nDevicesMax);
  if(sd->nDevices > nDevicesMax){
     printf("ERROR: Not enough GPUs to launch, nDevices %d nDevicesMax %d\n"
           , sd->nDevices, nDevicesMax);
    exit(0);
  }

  for (int iDevice = sd->startDevice; iDevice < sd->endDevice; iDevice++) {
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

  cudaMalloc((void **) &mGPU->state, mGPU->state_size);
  cudaMalloc((void **) &mGPU->env, mGPU->env_size);
  cudaMalloc((void **) &mGPU->rxn_env_data, mGPU->rxn_env_data_size);
  cudaMalloc((void **) &mGPU->rxn_env_data_idx, mGPU->rxn_env_data_idx_size);
  HANDLE_ERROR(cudaMalloc((void **) &mGPU->map_state_deriv, mGPU->map_state_deriv_size));

  int num_spec = md->n_per_cell_dep_var*mGPU->n_cells;
  cudaMalloc((void **) &(mGPU->production_rates),num_spec*sizeof(mGPU->production_rates));
  cudaMalloc((void **) &(mGPU->loss_rates),num_spec*sizeof(mGPU->loss_rates));

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

  }

  if(n_dep_var<32 && sd->use_cpu==0) {
    printf("CAMP ERROR: TOO FEW SPECIES FOR GPU (Species < 32),"
           " use CPU case instead (More info: https://earth.bsc.es/gitlab/ac/camp/-/issues/49 \n");
    exit(0);
  }

#ifdef CAMP_DEBUG_PRINT_GPU_SPECS
  print_gpu_specs();
#endif

}
#endif

void constructor_cvode_gpu(CVodeMem cv_mem, SolverData *sd)
{
  itsolver *bicg = &(sd->bicg);
  ModelData *md = &(sd->model_data);
  CVDlsMem cvdls_mem = (CVDlsMem) cv_mem->cv_lmem;
  SUNMatrix J = cvdls_mem->A;

  sd->flagCells = (int *) malloc((md->n_cells) * sizeof(int));
  ModelDataGPU *mGPU = sd->mGPU;

#ifdef DEV_JOIN_GPU_INIT_IN_ONE_CALL
  solver_new_gpu_cu_cvode(sd);
  cudaSetDevice(sd->startDevice);
  sd->mGPU = &(sd->mGPUs[sd->startDevice]);
  mGPU  = sd->mGPU;
  init_jac_cuda_cvode(sd);
  solver_init_int_double_cuda_cvode(sd);
#endif

  cudaSetDevice(sd->startDevice);
  sd->mGPU = &(sd->mGPUs[sd->startDevice]);
  mGPU = sd->mGPU;

#ifdef DEBUG_constructor_cvode_gpu
  printf("DEBUG_constructor_cvode_gpu start \n");
#endif

#ifdef CAMP_DEBUG_GPU

  bicg->counterNewtonIt=0;
  bicg->counterLinSolSetup=0;
  bicg->counterLinSolSolve=0;
  //bicg->countercvStep=0;
  bicg->counterDerivNewton=0;
  bicg->counterBiConjGrad=0;
  bicg->counterDerivSolve=0;
  bicg->countersolveCVODEGPU=0;

  bicg->timeNewtonIt=CAMP_TINY;
  bicg->timeLinSolSetup=CAMP_TINY;
  bicg->timeLinSolSolve=CAMP_TINY;
  bicg->timecvStep=CAMP_TINY;
  bicg->timeDerivNewton=CAMP_TINY;
  bicg->timeBiConjGrad=CAMP_TINY;
  bicg->timeBiConjGradMemcpy=CAMP_TINY;
  bicg->timeDerivSolve=CAMP_TINY;

  cudaEventCreate(&bicg->startDerivNewton);
  cudaEventCreate(&bicg->startDerivSolve);
  cudaEventCreate(&bicg->startLinSolSetup);
  cudaEventCreate(&bicg->startLinSolSolve);
  cudaEventCreate(&bicg->startNewtonIt);
  cudaEventCreate(&bicg->startcvStep);
  cudaEventCreate(&bicg->startBCG);
  cudaEventCreate(&bicg->startBCGMemcpy);
  cudaEventCreate(&bicg->startJac);

  cudaEventCreate(&bicg->stopDerivNewton);
  cudaEventCreate(&bicg->stopDerivSolve);
  cudaEventCreate(&bicg->stopLinSolSetup);
  cudaEventCreate(&bicg->stopLinSolSolve);
  cudaEventCreate(&bicg->stopNewtonIt);
  cudaEventCreate(&bicg->stopcvStep);
  cudaEventCreate(&bicg->stopBCG);
  cudaEventCreate(&bicg->stopBCGMemcpy);
  cudaEventCreate(&bicg->stopJac);

#endif

  int offset_nnz = 0;
  int offset_nrows = 0;
  for (int iDevice = sd->startDevice; iDevice < sd->endDevice; iDevice++) {
    cudaSetDevice(iDevice);
    sd->mGPU = &(sd->mGPUs[iDevice]);
    mGPU = sd->mGPU;

    mGPU->nnz = SM_NNZ_S(J)/md->n_cells*mGPU->n_cells;
    mGPU->nrows = SM_NP_S(J)/md->n_cells*mGPU->n_cells;

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, iDevice);
    mGPU->threads = prop.maxThreadsPerBlock; //1024
    mGPU->blocks = (mGPU->nrows + mGPU->threads - 1) / mGPU->threads;

    createLinearSolver_cvode(sd);

    mGPU->A = ((double *) SM_DATA_S(J))+offset_nnz;
    //Using int per default as sundindextype give wrong results in CPU, so translate from int64 to int

    if(sd->use_gpu_cvode==1){

      mGPU->jA = (int *) malloc(sizeof(int) *mGPU->nnz/mGPU->n_cells);
      mGPU->iA = (int *) malloc(sizeof(int) * (mGPU->nrows/mGPU->n_cells + 1));
      for (int i = 0; i < mGPU->nnz/mGPU->n_cells; i++)
        mGPU->jA[i] = SM_INDEXVALS_S(J)[i];
      for (int i = 0; i <= mGPU->nrows/mGPU->n_cells; i++)
        mGPU->iA[i] = SM_INDEXPTRS_S(J)[i];

      cudaMalloc((void **) &mGPU->djA, mGPU->nnz/mGPU->n_cells * sizeof(int));
      cudaMalloc((void **) &mGPU->diA, (mGPU->nrows/mGPU->n_cells + 1) * sizeof(int));
      cudaMemcpy(mGPU->djA, mGPU->jA, mGPU->nnz/mGPU->n_cells * sizeof(int), cudaMemcpyHostToDevice);
      cudaMemcpy(mGPU->diA, mGPU->iA, (mGPU->nrows/mGPU->n_cells + 1) * sizeof(int), cudaMemcpyHostToDevice);

    } else{

      mGPU->jA = (int *) malloc(sizeof(int) *mGPU->nnz);
      mGPU->iA = (int *) malloc(sizeof(int) * (mGPU->nrows + 1));
      for (int i = 0; i < mGPU->nnz; i++)
        mGPU->jA[i] = SM_INDEXVALS_S(J)[i];
      for (int i = 0; i <= mGPU->nrows; i++)
        mGPU->iA[i] = SM_INDEXPTRS_S(J)[i];

      cudaMalloc((void **) &mGPU->djA, mGPU->nnz * sizeof(int));
      cudaMalloc((void **) &mGPU->diA, (mGPU->nrows + 1) * sizeof(int));
      cudaMemcpy(mGPU->djA, mGPU->jA, mGPU->nnz * sizeof(int), cudaMemcpyHostToDevice);
      cudaMemcpy(mGPU->diA, mGPU->iA, (mGPU->nrows + 1) * sizeof(int), cudaMemcpyHostToDevice);

    }

    mGPU->dA = mGPU->J;//set itsolver gpu pointer to jac pointer initialized at camp
    mGPU->dftemp = mGPU->deriv_data; //deriv is gpu pointer

    double *ewt = N_VGetArrayPointer(cv_mem->cv_ewt)+offset_nrows;
    double *tempv = N_VGetArrayPointer(cv_mem->cv_tempv)+offset_nrows;
    double *cv_last_yn = N_VGetArrayPointer(cv_mem->cv_last_yn)+offset_nrows;
    double *cv_acor_init = N_VGetArrayPointer(cv_mem->cv_acor_init)+offset_nrows;

#ifdef DEV_DMDV_ARRAY
    cudaMalloc((void **) &mGPU->mdv, sizeof(ModelDataVariable)*mGPU->n_cells);
#else
    cudaMalloc((void **) &mGPU->mdv, sizeof(ModelDataVariable));
#endif
    cudaMalloc((void **) &mGPU->mdvo, sizeof(ModelDataVariable));
    cudaMalloc((void **) &mGPU->flag, 1 * sizeof(int));

    cudaMalloc((void **) &mGPU->flagCells, mGPU->n_cells * sizeof(int));
    cudaMalloc((void **) &mGPU->dsavedJ, mGPU->nnz * sizeof(double));
    cudaMalloc((void **) &mGPU->dewt, mGPU->nrows * sizeof(double));
    cudaMalloc((void **) &mGPU->cv_acor, mGPU->nrows * sizeof(double));
    cudaMalloc((void **) &mGPU->dtempv, mGPU->nrows * sizeof(double));
    cudaMalloc((void **) &mGPU->dtempv1, mGPU->nrows * sizeof(double));
    cudaMalloc((void **) &mGPU->dtempv2, mGPU->nrows * sizeof(double));
    cudaMalloc((void **) &mGPU->dzn, mGPU->nrows * (cv_mem->cv_qmax + 1) * sizeof(double));//L_MAX 6
    cudaMalloc((void **) &mGPU->dcv_y, mGPU->nrows * sizeof(double));
    cudaMalloc((void **) &mGPU->dx, mGPU->nrows * sizeof(double));
    cudaMalloc((void **) &mGPU->cv_last_yn, mGPU->nrows * sizeof(double));
    cudaMalloc((void **) &mGPU->cv_acor_init, mGPU->nrows * sizeof(double));
    cudaMalloc((void **) &mGPU->cv_acor, mGPU->nrows * sizeof(double));
    cudaMalloc((void **) &mGPU->yout, mGPU->nrows * sizeof(double));
    cudaMalloc((void **) &mGPU->cv_l, L_MAX * mGPU->n_cells * sizeof(double));
    cudaMalloc((void **) &mGPU->cv_tau, (L_MAX + 1) * mGPU->n_cells * sizeof(double));
    cudaMalloc((void **) &mGPU->cv_tq, (NUM_TESTS + 1) * mGPU->n_cells * sizeof(double));
    cudaMalloc((void **) &mGPU->cv_Vabstol, mGPU->nrows * sizeof(double));

    HANDLE_ERROR(cudaMemset(mGPU->flagCells, CV_SUCCESS, mGPU->n_cells * sizeof(int)));

    cudaMemcpy(mGPU->dsavedJ, mGPU->A, mGPU->nnz * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(mGPU->dewt, ewt, mGPU->nrows * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(mGPU->cv_acor, ewt, mGPU->nrows * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(mGPU->dftemp, ewt, mGPU->nrows * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(mGPU->dx, tempv, mGPU->nrows * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(mGPU->cv_last_yn, cv_last_yn, mGPU->nrows * sizeof(double), cudaMemcpyHostToDevice);
    HANDLE_ERROR(cudaMemcpy(mGPU->cv_acor_init, cv_acor_init, mGPU->nrows * sizeof(double), cudaMemcpyHostToDevice));

    mGPU->replacement_value = TINY;
    mGPU->threshhold = -SMALL;
    mGPU->deriv_length_cell = mGPU->nrows / mGPU->n_cells;
    mGPU->state_size_cell = md->n_per_cell_state_var;

    int flag = 999; //CAMP_SOLVER_SUCCESS
    cudaMemcpy(mGPU->flag, &flag, 1 * sizeof(int), cudaMemcpyHostToDevice);

    if(md->n_per_cell_dep_var > prop.maxThreadsPerBlock/2){
      printf("ERROR: More species than threads per block availabless\n");
    exit(0);
    }

#ifdef CAMP_DEBUG_GPU
#ifdef CAMP_PROFILE_DEVICE_FUNCTIONS
    int lendt=1;
    //todo changue cudamemset for __global__, since memset seems only works for int
    cudaDeviceGetAttribute(&mGPU->clock_khz, cudaDevAttrClockRate, 0);
    cudaMalloc((void**)&mGPU->tguessNewton,lendt*sizeof(double));
    cudaMemset(mGPU->tguessNewton, 0., lendt*sizeof(double));
    cudaMalloc((void**)&mGPU->dtNewtonIteration,lendt*sizeof(double));
    cudaMemset(mGPU->dtNewtonIteration, 0., lendt*sizeof(double));
    cudaMalloc((void**)&mGPU->dtJac,lendt*sizeof(double));
    cudaMemset(mGPU->dtJac, 0., lendt*sizeof(double));
    cudaMalloc((void**)&mGPU->dtlinsolsetup,lendt*sizeof(double));
    cudaMemset(mGPU->dtlinsolsetup, 0., lendt*sizeof(double));
    cudaMalloc((void**)&mGPU->dtcalc_Jac,lendt*sizeof(double));
    cudaMemset(mGPU->dtcalc_Jac, 0., lendt*sizeof(double));
    cudaMalloc((void**)&mGPU->dtRXNJac,lendt*sizeof(double));
    cudaMemset(mGPU->dtRXNJac, 0., lendt*sizeof(double));
    cudaMalloc((void**)&mGPU->dtf,lendt*sizeof(double));
    cudaMemset(mGPU->dtf, 0., lendt*sizeof(double));
    cudaMalloc((void**)&mGPU->dtguess_helper,lendt*sizeof(double));
    cudaMemset(mGPU->dtguess_helper, 0., lendt*sizeof(double));

    mGPU->mdvCPU.countercvStep=0;
    mGPU->mdvCPU.counterDerivGPU=0;
    mGPU->mdvCPU.counterBCGInternal=0;
    mGPU->mdvCPU.counterBCG=0;
    mGPU->mdvCPU.dtBCG=0.;
    mGPU->mdvCPU.dtcudaDeviceCVode=0.;
    mGPU->mdvCPU.dtPostBCG=0.;
#endif
#endif

#ifdef DEV_DMDV_ARRAY
    for(int i=0; i<mGPU->n_cells;i++){
      cudaMemcpy(&mGPU->mdv[i], &mGPU->mdvCPU, sizeof(ModelDataVariable), cudaMemcpyHostToDevice);
    }
#else
    cudaMemcpy(mGPU->mdv, &mGPU->mdvCPU, sizeof(ModelDataVariable), cudaMemcpyHostToDevice);
#endif

    HANDLE_ERROR(cudaMemcpy(mGPU->mdvo, &mGPU->mdvCPU, sizeof(ModelDataVariable), cudaMemcpyHostToDevice));

    mGPU->mdvCPU.cv_reltol = ((CVodeMem) sd->cvode_mem)->cv_reltol;
    mGPU->mdvCPU.cv_nfe = 0;
    mGPU->mdvCPU.cv_nsetups = 0;
    mGPU->mdvCPU.nje = 0;
    mGPU->mdvCPU.nstlj = 0;

    offset_nnz += mGPU->nnz;
    offset_nrows += mGPU->nrows;
  }

  if(cv_mem->cv_sldeton){
    printf("ERROR: cudaDevicecvBDFStab is pending to implement "
           "(disabled by default on CAMP)\n");
    exit(0);
  }

#ifdef DEBUG_constructor_cvode_gpu
  printf("DEBUG_constructor_cvode_gpu end \n");
#endif

}

__device__
void cudaDeviceJacCopy(int n_row, int* Ap, double* Ax, double* Bx) {
  __syncthreads();
  int nnz=Ap[blockDim.x];
  for(int j=Ap[threadIdx.x]; j<Ap[threadIdx.x+1]; j++){
    Bx[j+nnz*blockIdx.x]=Ax[j+nnz*blockIdx.x];
  }
  __syncthreads();
}

__device__
int cudaDevicecamp_solver_check_model_state(ModelDataGPU *md, ModelDataVariable *dmdv, double *y, int *flag)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  __syncthreads();
  extern __shared__ int flag_shr[];
  flag_shr[0] = 0;

  //printmin(md,md->state,"cudaDevicecamp_solver_check_model_state start state");

  __syncthreads();
  if (y[i] < md->threshhold) {
    flag_shr[0] = CAMP_SOLVER_FAIL;

#ifdef DEBUG_cudaDevicecamp_solver_check_model_state
    printf("Failed model state update gpu:[spec %d] = %le flag_shr %d\n",i,y[i],flag_shr[0]);
#endif

  } else {
    md->state[md->map_state_deriv[i]] =
            y[i] <= md->threshhold ?
            md->replacement_value : y[i];
  }

  __syncthreads();
  *flag = (int)flag_shr[0];
  __syncthreads();
#ifdef DEBUG_printmin
  printmin(md,md->state,"cudaDevicecamp_solver_check_model_state end state");
#endif

  //printmin(md,y,"cudaDevicecamp_solver_check_model_state end y");
  //printmin(md,md->state,"cudaDevicecamp_solver_check_model_state end state");

#ifdef DEBUG_cudaDevicecamp_solver_check_model_state
  __syncthreads();if(i==0)printf("flag %d flag_shr %d\n",*flag,flag_shr2[0]);
#endif

  return *flag;
}

__device__ void solveRXN(
        TimeDerivativeGPU deriv_data,
        double time_step,
        ModelDataGPU *md, ModelDataVariable *dmdv
)
{

#ifdef REVERSE_INT_FLOAT_MATRIX

  double *rxn_float_data = &( md->rxn_double[dmdv->i_rxn]);
  int *int_data = &(md->rxn_int[dmdv->i_rxn]);
  int rxn_type = int_data[0];
  int *rxn_int_data = (int *) &(int_data[1*md->n_rxn]);

#else

  double *rxn_float_data = (double *)&( md->rxn_double[md->rxn_float_indices[dmdv->i_rxn]]);
  int *int_data = (int *)&(md->rxn_int[md->rxn_int_indices[dmdv->i_rxn]]);

  int rxn_type = int_data[0];
  int *rxn_int_data = (int *) &(int_data[1]);

#endif

  //Get indices for rates
  double *rxn_env_data = &(md->rxn_env_data
  [md->n_rxn_env_data*dmdv->i_cell+md->rxn_env_data_idx[dmdv->i_rxn]]);

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
      //printf("RXN_EMISSION");
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
      rxn_gpu_troe_calc_deriv_contrib(md, deriv_data, rxn_int_data,
                                      rxn_float_data, rxn_env_data,time_step);
      break;
    case RXN_WET_DEPOSITION :
      //printf("RXN_WET_DEPOSITION");
      //rxn_gpu_wet_deposition_calc_deriv_contrib(md, deriv_data, rxn_int_data,
      //                                     rxn_float_data, rxn_env_data,time_step);
      break;
  }
}

__device__ void cudaDevicecalc_deriv(
        double time_step, double *y,
        double *yout, ModelDataGPU *md, ModelDataVariable *dmdv
) //Interface CPU/GPU
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int deriv_length_cell = md->deriv_length_cell;
  int tid_cell=tid%deriv_length_cell;
  int state_size_cell = md->state_size_cell;
  int active_threads = md->nrows;

#ifdef DEBUG_DERIV_GPU
  if(tid==0){
    printf("[DEBUG] GPU solveDerivative tid %d, \n", tid);
    printf("md->nrows %d, \n", md->nrows);
    printf("md->deriv_length_cell %d, \n", md->deriv_length_cell);
    printf("blockDim.x %d, \n", blockDim.x);
  }__syncthreads();
#endif

#ifdef DEBUG_printmin

  //__syncthreads();//no effect, but printmin yes
  printmin(md,yout,"cudaDevicecalc_deriv start end yout");
  printmin(md,md->J_tmp,"cudaDevicecalc_deriv start end J_tmp");
  printmin(md,md->J_state,"cudaDevicecalc_deriv start end J_state");
#endif
  md->J_tmp[tid]=y[tid]-md->J_state[tid];
  cudaDeviceSpmvCSC_block(md->J_tmp2, md->J_tmp, md->J_solver, md->jJ_solver, md->iJ_solver, 0);
  md->J_tmp[tid]=md->J_deriv[tid]+md->J_tmp2[tid];

  cudaDevicesetconst(md->J_tmp2, 0.0, active_threads); //Reset for next iter
#ifdef DEBUG_printmin
    printmin(md,md->J_tmp,"cudaDevicecalc_deriv start end J_tmp");
    printmin(md,md->J_state,"cudaDevicecalc_deriv start end J_state");
#endif
    TimeDerivativeGPU deriv_data;
    deriv_data.num_spec = deriv_length_cell*gridDim.x;

#ifdef AEROS_CPU
#else
    deriv_data.production_rates = md->production_rates;
    deriv_data.loss_rates = md->loss_rates;
    time_derivative_reset_gpu(deriv_data);
    __syncthreads();
#endif

    int i_cell = tid/deriv_length_cell;
    dmdv->i_cell = i_cell;
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
        dmdv->i_rxn = tid_cell + i*deriv_length_cell;

        solveRXN(deriv_data, time_step, md, dmdv);
      }

      //Limit tid to pending rxns to compute
      int residual=n_rxn-(deriv_length_cell*n_iters);
      if(tid_cell < residual){
        dmdv->i_rxn = tid_cell + deriv_length_cell*n_iters;

        solveRXN(deriv_data, time_step, md, dmdv);
      }
    }
    __syncthreads();

    deriv_data.production_rates = md->production_rates;
    deriv_data.loss_rates = md->loss_rates;
#ifdef DEBUG_printmin
    printmin(md,yout,"cudaDevicecalc_deriv start end yout");
#endif
    __syncthreads();
    time_derivative_output_gpu(deriv_data, yout, md->J_tmp,0);
#ifdef DEBUG_printmin
    printmin(md,yout,"cudaDevicecalc_deriv start end yout");
#endif

  __syncthreads();

}

__device__
int cudaDevicef(
        double time_step, double *y,
        double *yout, ModelDataGPU *md, ModelDataVariable *dmdv, int *flag
)
{
#ifdef CAMP_DEBUG_GPU
#ifdef CAMP_PROFILE_DEVICE_FUNCTIONS
  int clock_khz=md->clock_khz;
  clock_t start;
  start = clock();
#endif
#endif
#ifdef DEBUG_printmin
  printmin(md,y,"cudaDevicef Start y");
#endif
  time_step = time_step > 0. ? time_step : dmdv->init_time_step;
#ifdef DEBUG_printmin
  printmin(md,md->state,"cudaDevicef start state");
#endif

  int checkflag=cudaDevicecamp_solver_check_model_state(md, dmdv, y, flag);

  __syncthreads();
  if(checkflag==CAMP_SOLVER_FAIL){
    *flag=CAMP_SOLVER_FAIL;

#ifdef DEBUG_printmin
    printmin(md,y,"cudaDevicef End y");
#endif

    __syncthreads();
#ifdef CAMP_DEBUG_GPU
#ifdef CAMP_PROFILE_DEVICE_FUNCTIONS
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i==0) *md->dtf += ((double)(int)(clock() - start))/(clock_khz*1000);
#endif
#endif

#ifdef DEBUG_cudaDevicef
    if(i==0)printf("cudaDevicef CAMP_SOLVER_FAIL %d\n",i);
#endif
    return CAMP_SOLVER_FAIL;
  }
#ifdef DEBUG_printmin
  printmin(md,yout,"cudaDevicef End yout");
#endif
  cudaDevicecalc_deriv(
          //f_cuda
          time_step, y,
          yout, md, dmdv
  );

  //printmin(md,yout,"cudaDevicef End yout");
  //printmin(md,y,"cudaDevicef End y");

#ifdef CAMP_DEBUG_GPU
#ifdef CAMP_PROFILE_DEVICE_FUNCTIONS
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i==0) *md->dtf += ((double)(int)(clock() - start))/(clock_khz*1000);
#endif
#endif

  __syncthreads();
  *flag=0;
  __syncthreads();

  return 0;

}

__device__
int CudaDeviceguess_helper(double cv_tn, double cv_h, double* y_n,
                           double* y_n1, double* hf, double* dtempv1,
                           double* dtempv2, int *flag,
                           ModelDataGPU *md, ModelDataVariable *dmdv
) {

  extern __shared__ double flag_shr2[];
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int tid = threadIdx.x;

  double cv_reltol = dmdv->cv_reltol;
  int n_shr_empty = md->n_shr_empty;

#ifdef DEBUG_CudaDeviceguess_helper
  if(i==0)printf("CudaDeviceguess_helper start gpu\n");
#endif

  __syncthreads();
  double min;
  cudaDevicemin(&min, y_n[i], flag_shr2, n_shr_empty);

#ifdef DEBUG_CudaDeviceguess_helper
  if(i==0)printf("min %le -SMALL %le\n",min, -SMALL);
#endif

  if(min>-SMALL){
#ifdef DEBUG_CudaDeviceguess_helper
    if(i==0)printf("Return 0 %le\n",y_n[i]);
#endif
    return 0;
  }

  __syncthreads();
#ifdef CAMP_DEBUG_GPU
#ifdef CAMP_PROFILE_DEVICE_FUNCTIONS
  int clock_khz=md->clock_khz;
  clock_t start;
  start = clock();
#endif
#endif

  dtempv1[i]=y_n1[i];
  __syncthreads();
  if (cv_h > 0.) {
    dtempv2[i]=(1./cv_h)*hf[i];
  } else {
    dtempv2[i]=hf[i];
  }

  // Advance state interatively
  double t_0 = cv_h > 0. ? cv_tn - cv_h : cv_tn - 1.;
  double t_j = 0.;
  int GUESS_MAX_ITER = 5; //5 //reduce this to improve perf
  __syncthreads();
  for (int iter = 0; iter < GUESS_MAX_ITER && t_0 + t_j < cv_tn; iter++) {
    // Calculate \f$h_j\f$
    //double h_j = cv_tn - (t_0 + t_j);
    //int i_fast = -1;
    __syncthreads();

    double h_j = cv_tn - (t_0 + t_j);
    /*
    for (int i = 0; i < n_elem; i++) {
     realtype t_star = -atmp1[i] / acorr[i];
      if ((t_star > ZERO || (t_star == ZERO && acorr[i] < ZERO)) &&
          t_star < h_j) {
        h_j = t_star;
        i_fast = i;
      }
    }
     */
    __syncthreads();

    double t_star;
    double h_j_init=h_j;

    //if(i==0)printf("*md->h_jPtrInit %le\n",*md->h_jPtr);

    if(dtempv2[i]==0){
      t_star=h_j;
    }else{
      t_star = -dtempv1[i] / dtempv2[i];
    }

    if( !(t_star > 0. || (t_star == 0. && dtempv2[i] < 0.)) ){//&&dtempv2[i]==0.)
      t_star=h_j;
    }

    __syncthreads();
    //(blockIdx.x==0 && iter<=0)printf("i %d t_star %le atmp1 %le acorr %le\n",i,t_star,dtempv1[i],dtempv2[i]);

    flag_shr2[tid]=h_j_init;
    cudaDevicemin(&h_j, t_star, flag_shr2, n_shr_empty);
    flag_shr2[0]=1;
    __syncthreads();

#ifdef DEBUG_CudaDeviceguess_helper
    //if(tid==0 && iter<=5) printf("CudaDeviceguess_helper h_j %le h_j_init %le t_star %le block %d iter %d\n",h_j,h_j_init,t_star,blockIdx.x,iter);
#endif

    // Scale incomplete jumps

    //if (i_fast >= 0 && cv_h > 0.)
    if (cv_h > 0.)
      h_j *= 0.95 + 0.1 * iter / (double)GUESS_MAX_ITER;
    h_j = cv_tn < t_0 + t_j + h_j ? cv_tn - (t_0 + t_j) : h_j;

    __syncthreads();
    // Only make small changes to adjustment vectors used in Newton iteration
    if (cv_h == 0. &&
        cv_tn - (h_j + t_j + t_0) > cv_reltol) {

#ifdef DEBUG_CudaDeviceguess_helper
      if(i==0)printf("CudaDeviceguess_helper small changes \n");
#endif

      __syncthreads();
#ifdef CAMP_DEBUG_GPU
#ifdef CAMP_PROFILE_DEVICE_FUNCTIONS
    if(i==0) *md->dtguess_helper += ((double)(clock() - start))/(clock_khz*1000);
#endif
#endif

    return -1;
    }

    // Advance the state
    //N_VLinearSum(ONE, dtempv1, h_j, dtempv2, dtempv1);
    cudaDevicezaxpby(1., dtempv1, h_j, dtempv2, dtempv1, md->nrows);

    __syncthreads();
    // Advance t_j
    t_j += h_j;

#ifdef DEBUG_CudaDeviceguess_helper
    //  printf("dcorr[%d] %le dhf %le dt_star %le dh_j %le dh_n %le\n",
    //         i,dtempv2[i],hf[i],t_star,h_j,cv_h);
    //if(i==0)
    //  for(int j=0;j<nrows;j++)
    //    printf("dcorr[%d] %le dtmp1 %le dhf %le dt_star %le dh_j %le dh_n %le\n",
    //           j,dtempv2[j],dtempv1[j],hf[j],t_star,h_j,cv_h);
#endif
#ifdef DEBUG_printmin
    printmin(md,md->state,"cudaDevicef start state");
#endif

    int aux_flag=0;
    int fflag=cudaDevicef(
            t_0 + t_j, dtempv1, dtempv2,md,dmdv,&aux_flag
    );
#ifdef DEBUG_printmin
    printmin(md,dtempv1,"cudaDevicef end dtempv1");
#endif
    __syncthreads();

    if (fflag == CAMP_SOLVER_FAIL) {
      //N_VConst(ZERO, dtempv2);
      dtempv2[i] = 0.;

#ifdef DEBUG_CudaDeviceguess_helper
      if(i==0)printf("CudaDeviceguess_helper df(t)\n");
#endif

      __syncthreads();
#ifdef CAMP_DEBUG_GPU
#ifdef CAMP_PROFILE_DEVICE_FUNCTIONS
    if(i==0) *md->dtguess_helper += ((double)(clock() - start))/(clock_khz*1000);
#endif
#endif

     return -1;
    }

    if (iter == GUESS_MAX_ITER - 1 && t_0 + t_j < cv_tn) {
      if (cv_h == 0.){

        __syncthreads();
#ifdef CAMP_DEBUG_GPU
#ifdef CAMP_PROFILE_DEVICE_FUNCTIONS
    if(i==0) *md->dtguess_helper += ((double)(clock() - start))/(clock_khz*1000);
#endif
#endif

        return -1;
      }
    }
    __syncthreads();
  }

  __syncthreads();
#ifdef DEBUG_CudaDeviceguess_helper
   if(i==0)printf("CudaDeviceguess_helper return 1\n");
#endif

  // Set the correction vector
  //N_VLinearSum(ONE, dtempv1, -ONE, y_n, dtempv2);
  cudaDevicezaxpby(1., dtempv1, -1., y_n, dtempv2, md->nrows);


  // Scale the initial corrections
  //if (cv_h > 0.) N_VScale(0.999, dtempv2, dtempv2);
  if (cv_h > 0.) dtempv2[i]=dtempv2[i]*0.999;

  // Update the hf vector
  //N_VLinearSum(ONE, dtempv1, -ONE, y_n1, hf);
  cudaDevicezaxpby(1., dtempv1, -1., y_n1, hf, md->nrows);

  __syncthreads();
#ifdef CAMP_DEBUG_GPU
#ifdef CAMP_PROFILE_DEVICE_FUNCTIONS
    if(i==0) *md->dtguess_helper += ((double)(clock() - start))/(clock_khz*1000);
#endif
#endif

  __syncthreads();
  return 1;
}

__device__ void solveRXNJac(
        JacobianGPU jac,
        double cv_next_h,
        ModelDataGPU *md, ModelDataVariable *dmdv
)
{

#ifdef REVERSE_INT_FLOAT_MATRIX

  double *rxn_float_data = &( md->rxn_double[dmdv->i_rxn]);
  int *int_data = &(md->rxn_int[dmdv->i_rxn]);
  int rxn_type = int_data[0];
  int *rxn_int_data = (int *) &(int_data[1*md->n_rxn]);

#else

  double *rxn_float_data = (double *)&( md->rxn_double[md->rxn_float_indices[dmdv->i_rxn]]);
  int *int_data = (int *)&(md->rxn_int[md->rxn_int_indices[dmdv->i_rxn]]);

  //double *rxn_float_data = &( md->rxn_double[dmdv->i_rxn]);
  //int *int_data = &(md->rxn_int[dmdv->i_rxn]);

  int rxn_type = int_data[0];
  int *rxn_int_data = (int *) &(int_data[1]);

#endif

  //Get indices for rates
  double *rxn_env_data = &(md->rxn_env_data
  [md->n_rxn_env_data*dmdv->i_cell+md->rxn_env_data_idx[dmdv->i_rxn]]);

#ifdef DEBUG_solveRXNJac
  if(tid==0){
    printf("[DEBUG] GPU solveRXN tid %d, \n", tid);
  }
#endif

  switch (rxn_type) {
    //case RXN_AQUEOUS_EQUILIBRIUM :
    //fix run-time error
    //rxn_gpu_aqueous_equilibrium_calc_jac_contrib(md, jac, rxn_int_data,
    //                                               rxn_float_data, rxn_env_data,cv_next_h);
    //break;
    case RXN_ARRHENIUS :
      rxn_gpu_arrhenius_calc_jac_contrib(md, jac, rxn_int_data,
                                         rxn_float_data, rxn_env_data,cv_next_h);
      break;
    case RXN_CMAQ_H2O2 :
      rxn_gpu_CMAQ_H2O2_calc_jac_contrib(md, jac, rxn_int_data,
                                         rxn_float_data, rxn_env_data,cv_next_h);
      break;
    case RXN_CMAQ_OH_HNO3 :
      rxn_gpu_CMAQ_OH_HNO3_calc_jac_contrib(md, jac, rxn_int_data,
                                            rxn_float_data, rxn_env_data,cv_next_h);
      break;
    case RXN_CONDENSED_PHASE_ARRHENIUS :
      //rxn_gpu_condensed_phase_arrhenius_calc_jac_contrib(md, jac, rxn_int_data,
      //                                     rxn_float_data, rxn_env_data,cv_next_h);
      break;
    case RXN_EMISSION :
      //printf("RXN_EMISSION");
      //rxn_gpu_emission_calc_jac_contrib(md, jac, rxn_int_data,
      //                                     rxn_float_data, rxn_env_data,cv_next_h);
      break;
    case RXN_FIRST_ORDER_LOSS :
      //rxn_gpu_first_order_loss_calc_jac_contrib(md, jac, rxn_int_data,
      //                                     rxn_float_data, rxn_env_data,cv_next_h);
      break;
    case RXN_HL_PHASE_TRANSFER :
      //rxn_gpu_HL_phase_transfer_calc_jac_contrib(md, jac, rxn_int_data,
      //                                             rxn_float_data, rxn_env_data,time_stepn);
      break;
    case RXN_PHOTOLYSIS :
      rxn_gpu_photolysis_calc_jac_contrib(md, jac, rxn_int_data,
                                          rxn_float_data, rxn_env_data,cv_next_h);
      break;
    case RXN_SIMPOL_PHASE_TRANSFER :
      //rxn_gpu_SIMPOL_phase_transfer_calc_jac_contrib(md, jac,
      //        rxn_int_data, rxn_float_data, rxn_env_data, cv_next_h);
      break;
    case RXN_TROE :
      rxn_gpu_troe_calc_jac_contrib(md, jac, rxn_int_data,
                                    rxn_float_data, rxn_env_data,cv_next_h);
      break;
    case RXN_WET_DEPOSITION :
      //printf("RXN_WET_DEPOSITION");
      //rxn_gpu_wet_deposition_calc_jac_contrib(md, jac, rxn_int_data,
      //                                     rxn_float_data, rxn_env_data,cv_next_h);
      break;
  }
}

__device__ void cudaDevicecalc_Jac(double *y,
        ModelDataGPU *md, ModelDataVariable *dmdv
)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  double cv_next_h = dmdv->cv_next_h;
  int deriv_length_cell = md->deriv_length_cell;
  int state_size_cell = md->state_size_cell;
  int tid_cell=tid%deriv_length_cell;
  int active_threads = md->n_cells*md->deriv_length_cell;

  __syncthreads();
#ifdef CAMP_DEBUG_GPU
#ifdef CAMP_PROFILE_DEVICE_FUNCTIONS
  int clock_khz=md->clock_khz;
  clock_t start;
  start = clock();
#endif
#endif

#ifdef DEBUG_cudaDeviceJac
  if(tid==0){
    printf("[DEBUG] GPU solveDerivative tid %d, \n", tid);
  }__syncthreads();
#endif

  if(tid<active_threads){
    __syncthreads();
    JacobianGPU *jac = &md->jac;
    JacobianGPU jacBlock;
#ifdef DEV_JACOBIANGPUNUMSPEC
    jac->num_spec = state_size_cell;
    jacBlock.num_spec = state_size_cell;
#endif

    jacBlock.num_elem = jac->num_elem;
    __syncthreads();
    int i_cell = tid/deriv_length_cell;
    dmdv->i_cell = i_cell;
    jacBlock.production_partials = &( jac->production_partials[jacBlock.num_elem[0]*blockIdx.x]);
    jacBlock.loss_partials = &( jac->loss_partials[jacBlock.num_elem[0]*blockIdx.x]);
    __syncthreads();
    md->grid_cell_state = &( md->state[state_size_cell*i_cell]);
    md->grid_cell_env = &( md->env[CAMP_NUM_ENV_PARAM_*i_cell]);
#ifdef DEBUG_cudaDevicecalc_Jac
    if(tid==0)printf("cudaDevicecalc_Jac01\n");

    //if(threadIdx.x==0) {
    //  printf("jac.num_elem %d\n",jacBlock.num_elem);
    //  printf("*md->n_mapped_values %d\n",*md->n_mapped_values);
      //for (int i=0; i<*md->n_mapped_values; i++){
      //  printf("cudaDevicecalc_Jac0 jacBlock [%d]=%le\n",i,jacBlock.production_partials[i]);
      //}
    //}
#endif
    __syncthreads();
    //Filter threads for n_rxn
    int n_rxn = md->n_rxn;
    if( tid_cell < n_rxn) {
      int n_iters = n_rxn / deriv_length_cell;
      //Repeat if there are more reactions than species
      for (int i = 0; i < n_iters; i++) {
        dmdv->i_rxn = tid_cell + i*deriv_length_cell;

        solveRXNJac(jacBlock, cv_next_h, md, dmdv);
      }
      //Limit tid to pending rxns to compute
      int residual=n_rxn-(deriv_length_cell*n_iters);
      if(tid_cell < residual){
        //todo fix this to all threads can enter, because if not the atomicadd can produce problems (maybe assign the leftovers rxn_int to 0?)
        dmdv->i_rxn = tid_cell + deriv_length_cell*n_iters;

        solveRXNJac(jacBlock, cv_next_h, md, dmdv);
      }
    }
    __syncthreads();

  JacMap *jac_map = md->jac_map;
  int nnz = md->n_mapped_values[0];
  int n_iters = nnz / blockDim.x;
  for (int i = 0; i < n_iters; i++) {
    int j = threadIdx.x + i*blockDim.x;
    md->J[jac_map[j].solver_id + nnz * blockIdx.x] =
      jacBlock.production_partials[jac_map[j].rxn_id] - jacBlock.loss_partials[jac_map[j].rxn_id];
    jacBlock.production_partials[jac_map[j].rxn_id] = 0.0;
    jacBlock.loss_partials[jac_map[j].rxn_id] = 0.0;
  }
  int residual=nnz-(blockDim.x*n_iters);
  if(threadIdx.x < residual){
    int j = threadIdx.x + n_iters*blockDim.x;

    md->J[jac_map[j].solver_id + nnz * blockIdx.x] =
      jacBlock.production_partials[jac_map[j].rxn_id] - jacBlock.loss_partials[jac_map[j].rxn_id];
    jacBlock.production_partials[jac_map[j].rxn_id] = 0.0;
    jacBlock.loss_partials[jac_map[j].rxn_id] = 0.0;
  }

    __syncthreads();
#ifdef CAMP_DEBUG_GPU
#ifdef CAMP_PROFILE_DEVICE_FUNCTIONS
    if(tid==0) *md->dtcalc_Jac += ((double)(clock() - start))/(clock_khz*1000);
#endif
#endif

  }
}

__device__
int cudaDeviceJac(int *flag, ModelDataGPU *md, ModelDataVariable *dmdv
) //Interface CPU/GPU
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  double* dftemp = md->dftemp;
  double* dcv_y = md->dcv_y;

  __syncthreads();
#ifdef CAMP_DEBUG_GPU
#ifdef CAMP_PROFILE_DEVICE_FUNCTIONS
  int clock_khz=md->clock_khz;
  clock_t start;
  start = clock();
#endif
#endif

#ifdef DEBUG_printmin
  printmin(md,md->state,"cudaDeviceJac start state");
#endif
  int aux_flag=0;

  //int guessflag=
  int retval=cudaDevicef(
          dmdv->cv_next_h, dcv_y, dftemp,md,dmdv,&aux_flag
  );__syncthreads();
#ifdef DEBUG_cudaDevicef
  printmin(md,dftemp,"cudaDeviceJac dftemp");
#endif

  if(retval==CAMP_SOLVER_FAIL)
    return CAMP_SOLVER_FAIL;

#ifdef DEBUG_printmin
  printmin(md,dcv_y,"cudaDeviceJac dcv_y");
  printmin(md,md->state,"cudaDeviceJac start state");
#endif

  //debug
/*
  int checkflag=cudaDevicecamp_solver_check_model_state(md, dmdv, dcv_y, flag);
  __syncthreads();
  if(checkflag==CAMP_SOLVER_FAIL){
    *flag=CAMP_SOLVER_FAIL;
    //printf("cudaDeviceJac cudaDevicecamp_solver_check_model_state *flag==CAMP_SOLVER_FAIL\n");
    //printmin(md,dcv_y,"cudaDeviceJac end dcv_y");
    return CAMP_SOLVER_FAIL;
  }
*/

#ifdef DEBUG_printmin
  printmin(md,dcv_y,"cudaDeviceJac end dcv_y");
#endif

  //printmin(md,dftemp,"cudaDeviceJac end dftemp");

  cudaDevicecalc_Jac(dcv_y,md, dmdv);
  __syncthreads();
#ifdef DEBUG_printmin
 printmin(md,dftemp,"cudaDevicecalc_Jac end dftemp");
#endif

    __syncthreads();

  int nnz = md->n_mapped_values[0];
  int n_iters = nnz / blockDim.x;
  for (int i = 0; i < n_iters; i++) {
    int j = threadIdx.x + i*blockDim.x;
    md->J_solver[j]=md->J[j];
  }
  int residual=nnz-(blockDim.x*n_iters);
  if(threadIdx.x < residual){
    int j = threadIdx.x + n_iters*blockDim.x;
    md->J_solver[j]=md->J[j];
  }

    __syncthreads();

    md->J_state[tid]=dcv_y[tid];
    md->J_deriv[tid]=dftemp[tid];

  __syncthreads();
#ifdef CAMP_DEBUG_GPU
#ifdef CAMP_PROFILE_DEVICE_FUNCTIONS
    if(tid==0) *md->dtJac += ((double)(clock() - start))/(clock_khz*1000);
#endif
#endif

  __syncthreads();
  *flag = 0;
  __syncthreads();
  return 0;

}

__device__
int cudaDevicelinsolsetup(int *flag,
        ModelDataGPU *md, ModelDataVariable *dmdv,
        int convfail
) {

  extern __shared__ int flag_shr[];

  double* dA = md->dA;
  int* djA = md->djA;
  int* diA = md->diA;
  int nrows = md->nrows;
  double* ddiag = md->ddiag;
  double* dsavedJ = md->dsavedJ;

  double dgamma;
  int jbad, jok;
#ifdef DEBUG_printmin
  printmin(md,dcv_y,"cudaDevicelinsolsetup Start dcv_y");
#endif
  dgamma = fabs((dmdv->cv_gamma / dmdv->cv_gammap) - 1.);//SUNRabs

  jbad = (dmdv->cv_nst == 0) ||
         (dmdv->cv_nst > dmdv->nstlj + CVD_MSBJ) ||
         ((convfail == CV_FAIL_BAD_J) && (dgamma < CVD_DGMAX)) ||
         (convfail == CV_FAIL_OTHER);
  jok = !jbad;
  if (jok==1) {
    __syncthreads();
    dmdv->cv_jcur = 0; //all blocks update this variable
    cudaDeviceJacCopy(nrows, diA, dsavedJ, dA);
    __syncthreads();
  } else {
  __syncthreads();
    dmdv->nje++;
    dmdv->nstlj = dmdv->cv_nst;
    dmdv->cv_jcur = 1;
  __syncthreads();
    int aux_flag=0;
    int guess_flag=cudaDeviceJac(&aux_flag,md,dmdv);
    __syncthreads();
    if (guess_flag < 0) {
      //last_flag = CVDLS_JACFUNC_UNRECVR;
      return -1;
    }
    if (guess_flag > 0) {
      //last_flag = CVDLS_JACFUNC_RECVR;
      return 1;
    }
   cudaDeviceJacCopy(nrows, diA, dA, dsavedJ);
  }
  __syncthreads();
  cudaDevicematScaleAddI(nrows, dA, djA, diA, -dmdv->cv_gamma);
  cudaDevicediagprecond(nrows, dA, djA, diA, ddiag); //Setup linear solver
  *flag=0;
  __syncthreads();
  return 0;
}

__device__
void solveBcgCudaDeviceCVODE(ModelDataGPU *md, ModelDataVariable *dmdv)
{
#ifdef DEBUG_printmin
  printmin(md,dtempv,"solveBcgCudaDeviceCVODEStart dtempv");
#endif

  double* dA = md->dA;
  int* djA = md->djA;
  int* diA = md->diA;
  double* dx = md->dx;
  double* dtempv = md->dtempv;
  int nrows = md->nrows;
  int n_shr_empty = md->n_shr_empty;
  int maxIt = md->maxIt;
  double tolmax = md->tolmax;
  double* ddiag = md->ddiag;
  double* dr0 = md->dr0;
  double* dr0h = md->dr0h;
  double* dn0 = md->dn0;
  double* dp0 = md->dp0;
  double* dt = md->dt;
  double* ds = md->ds;
  double* dAx2 = md->dAx2;
  double* dy = md->dy;
  double* dz = md->dz;

  double alpha,rho0,omega0,beta,rho1,temp1,temp2;
  alpha=rho0=omega0=beta=rho1=temp1=temp2=1.0;
  cudaDevicesetconst(dn0, 0.0, nrows);
  cudaDevicesetconst(dp0, 0.0, nrows);
  __syncthreads();
  cudaDeviceSpmvCSC_block(dr0,dx,dA,djA,diA,n_shr_empty); //y=A*x
  __syncthreads();
  cudaDeviceaxpby(dr0,dtempv,1.0,-1.0,nrows);
  __syncthreads();
  cudaDeviceyequalsx(dr0h,dr0,nrows);
  int it=0;
  do{
    __syncthreads();
    cudaDevicedotxy(dr0, dr0h, &rho1, n_shr_empty);
    __syncthreads();
    beta = (rho1 / rho0) * (alpha / omega0);
    __syncthreads();
    cudaDevicezaxpbypc(dp0, dr0, dn0, beta, -1.0 * omega0 * beta, nrows);   //z = ax + by + c
    __syncthreads();
    cudaDevicemultxy(dy, ddiag, dp0, nrows);
    __syncthreads();
    cudaDeviceSpmvCSC_block(dn0, dy, dA, djA, diA,n_shr_empty);
    __syncthreads();
    cudaDevicedotxy(dr0h, dn0, &temp1, n_shr_empty);
    __syncthreads();
    alpha = rho1 / temp1;
    cudaDevicezaxpby(1.0, dr0, -1.0 * alpha, dn0, ds, nrows);
    __syncthreads();
    cudaDevicemultxy(dz, ddiag, ds, nrows); // precond z=diag*s
    __syncthreads();
    cudaDeviceSpmvCSC_block(dt, dz, dA, djA, diA,n_shr_empty);
    __syncthreads();
    cudaDevicemultxy(dAx2, ddiag, dt, nrows);
    __syncthreads();
    cudaDevicedotxy(dz, dAx2, &temp1, n_shr_empty);
    __syncthreads();
    cudaDevicedotxy(dAx2, dAx2, &temp2, n_shr_empty);
    __syncthreads();
    omega0 = temp1 / temp2;
    cudaDeviceaxpy(dx, dy, alpha, nrows); // x=alpha*y +x
    __syncthreads();
    cudaDeviceaxpy(dx, dz, omega0, nrows);
    __syncthreads();
    cudaDevicezaxpby(1.0, ds, -1.0 * omega0, dt, dr0, nrows);
    __syncthreads();
    cudaDevicesetconst(dt, 0.0, nrows);
    __syncthreads();
    cudaDevicedotxy(dr0, dr0, &temp1, n_shr_empty);
    __syncthreads();
    temp1 = sqrtf(temp1);
    rho0 = rho1;
    __syncthreads();
    it++;
  } while(it<maxIt && temp1>tolmax);//while(it<maxIt && temp1>tolmax);//while(0);
  __syncthreads();
#ifdef CAMP_DEBUG_GPU
#ifdef CAMP_PROFILE_DEVICE_FUNCTIONS
  dmdv->counterBCGInternal += it;
  dmdv->counterBCG++;
#endif
#endif
}

__device__
int cudaDevicecvNewtonIteration(ModelDataGPU *md, ModelDataVariable *dmdv
)
{
  extern __shared__ double flag_shr2[];
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int aux_flag=0;

  double* dx = md->dx;
  double* dtempv = md->dtempv;
  int nrows = md->nrows;
  double cv_tn = dmdv->cv_tn;
  double* dftemp = md->dftemp;
  double* dcv_y = md->dcv_y;
  double* dtempv1 = md->dtempv1;
  double* dtempv2 = md->dtempv2;
  double cv_next_h = dmdv->cv_next_h;
  int n_shr_empty = md->n_shr_empty;
  double* cv_acor = md->cv_acor;
  double* dzn = md->dzn;
  double* dewt = md->dewt;
  double del, delp, dcon, m;
  del = delp = 0.0;
  dmdv->cv_mnewt = m = 0;
  __syncthreads();
#ifdef CAMP_DEBUG_GPU
#ifdef CAMP_PROFILE_DEVICE_FUNCTIONS
  int clock_khz=md->clock_khz;
  clock_t start;
#endif
#endif
#ifdef DEBUG_printmin
  printmin(md,dtempv,"cudaDevicecvNewtonIterationStart dtempv");
#endif

  for(;;) {

#ifdef DEBUG_printmin
    printmin(md,dftemp,"cudaDevicecvNewtonIteration dftemp");
#endif

    __syncthreads();
#ifdef CAMP_DEBUG_GPU
#ifdef CAMP_PROFILE_DEVICE_FUNCTIONS
    start = clock();
#endif
#endif

    cudaDevicezaxpby(dmdv->cv_rl1, (dzn + 1 * nrows), 1.0, cv_acor, dtempv, nrows);
    cudaDevicezaxpby(dmdv->cv_gamma, dftemp, -1.0, dtempv, dtempv, nrows);
    solveBcgCudaDeviceCVODE(md, dmdv);
    __syncthreads();
#ifdef CAMP_DEBUG_GPU
#ifdef CAMP_PROFILE_DEVICE_FUNCTIONS
    //if(threadIdx.x==0)dmdv->dtBCG += ((double)(int)(clock() - start))/(clock_khz*1000);//wrong
    dmdv->dtBCG += ((double)(int)(clock() - start))/(clock_khz*1000);
#endif
#endif

    __syncthreads();
    dtempv[i] = dx[i];
    __syncthreads();
#ifdef DEBUG_printmin
    printmin(md,dcv_y,"cudaDevicecvNewtonIteration dcv_y");
    printmin(md,dtempv,"cudaDevicecvNewtonIteration dtempv");
#endif
    cudaDevicezaxpby(1.0, dcv_y, 1.0, dtempv, dftemp, nrows);
#ifdef DEBUG_cudaDevicecvNewtonIteration
    //if(i==0)printf("cudaDevicecvNewtonIteration dftemp %le dtempv %le dcv_y %le it %d block %d\n",
    //               dftemp[(blockDim.x-1)*0],dtempv[(blockDim.x-1)*0],dcv_y[(blockDim.x-1)*0],it,blockIdx.x);
#endif
#ifdef DEBUG_printmin
    printmin(md,dftemp,"cudaDevicecvNewtonIteration dftemp");
#endif

    __syncthreads();
    int guessflag=CudaDeviceguess_helper(cv_tn, 0., dftemp,
                           dcv_y, dtempv, dtempv1,
                           dtempv2, &aux_flag, md, dmdv
    );
    __syncthreads();
    if (guessflag < 0) {
      if (!(dmdv->cv_jcur)) { //Bool set up during linsolsetup just before Jacobian
        return TRY_AGAIN;
      } else {
        return RHSFUNC_RECVR;
      }
    }
    cudaDevicezaxpby(1., dcv_y, 1., dtempv, dftemp, nrows);
    double min;
    cudaDevicemin(&min, dftemp[i], flag_shr2, md->n_shr_empty);

    if (min < -CAMP_TINY) {
      //if (dftemp[i] < -CAMP_TINY) {
      return CONV_FAIL;
    }
    __syncthreads();
    cudaDevicezaxpby(1., cv_acor, 1., dx, cv_acor, nrows);
    cudaDevicezaxpby(1., dzn, 1., cv_acor, dcv_y, nrows);
    cudaDeviceVWRMS_Norm(dx, dewt, &del, nrows, n_shr_empty);
    if (m > 0) {
      dmdv->cv_crate = SUNMAX(0.3 * dmdv->cv_crate, del / delp);
    }
    dcon = del * SUNMIN(1.0, dmdv->cv_crate) / md->cv_tq[4+blockIdx.x*(NUM_TESTS + 1)];
    flag_shr2[0]=0;
    __syncthreads();
    if (dcon <= 1.0) {
      cudaDeviceVWRMS_Norm(cv_acor, dewt, &dmdv->cv_acnrm, nrows, n_shr_empty);
      __syncthreads();
      dmdv->cv_jcur = 0;
      __syncthreads();
      return CV_SUCCESS;
    }
    dmdv->cv_mnewt = ++m;
    if ((m == dmdv->cv_maxcor) || ((m >= 2) && (del > RDIV * delp))) {
      if (!(dmdv->cv_jcur)) {
        return TRY_AGAIN;
      } else {
        return RHSFUNC_RECVR;
      }
    }
    delp = del;
    __syncthreads();
#ifdef DEBUG_printmin
    printmin(md,md->state,"cudaDevicef start state");
#endif
    int retval=cudaDevicef(
            cv_next_h, dcv_y, dftemp, md, dmdv, &aux_flag
    );
    __syncthreads();
    cudaDevicezaxpby(1., dcv_y, 1., dzn, cv_acor, nrows);
    if (retval < 0) {
      return CV_RHSFUNC_FAIL;
    }
    if (retval > 0) {
      if (!(dmdv->cv_jcur)) {
        return TRY_AGAIN;
      } else {
        return RHSFUNC_RECVR;
      }
    }
    dmdv->cv_nfe=dmdv->cv_nfe+1;
    __syncthreads();
#ifdef CAMP_DEBUG_GPU
#ifdef CAMP_PROFILE_DEVICE_FUNCTIONS
    if(i==0) dmdv->dtPostBCG += ((double)(clock() - start))/(clock_khz*1000);
#endif
#endif
#ifdef DEBUG_cudaDevicecvNewtonIteration
    if(i==0)printf("cudaDevicecvNewtonIteration dzn[(blockDim.x*(blockIdx.x+1)-1)*0] %le it %d block %d\n",dzn[(blockDim.x*(blockIdx.x+1)-1)*0],it,blockIdx.x);
#endif
  }

}

__device__
int cudaDevicecvNlsNewton(int *flag,
        ModelDataGPU *md, ModelDataVariable *dmdv
) {
  extern __shared__ int flag_shr[];
  int flagDevice = 0;
  __syncthreads();*flag = flag_shr[0];__syncthreads();
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  double* dcv_y = md->dcv_y;
  double* cv_acor = md->cv_acor;
  double* dzn = md->dzn;
  double* dftemp = md->dftemp;
  double cv_tn = dmdv->cv_tn;
  double cv_h = dmdv->cv_h;
  double* dtempv = md->dtempv;
  double cv_next_h = dmdv->cv_next_h;
#ifdef DEBUG_printmin
  printmin(md,dtempv,"cudaDevicecvNlsNewtonStart dtempv");
#endif
  __syncthreads();
#ifdef CAMP_DEBUG_GPU
#ifdef CAMP_PROFILE_DEVICE_FUNCTIONS
  int clock_khz=md->clock_khz;
  clock_t start;
#endif
#endif

  int convfail = ((dmdv->nflag == FIRST_CALL) || (dmdv->nflag == PREV_ERR_FAIL)) ?
                 CV_NO_FAILURES : CV_FAIL_OTHER;
  int dgamrat=fabs(dmdv->cv_gamrat - 1.);
  int callSetup = (dmdv->nflag == PREV_CONV_FAIL) || (dmdv->nflag == PREV_ERR_FAIL) ||
                  (dmdv->cv_nst == 0) ||
                  (dmdv->cv_nst >= dmdv->cv_nstlp + MSBP) ||
                  (dgamrat > DGMAX);
  dftemp[i]=dzn[i]+(-md->cv_last_yn[i]);
  __syncthreads();
  int guessflag=CudaDeviceguess_helper(cv_tn, cv_h, dzn,
             md->cv_last_yn, dftemp, dtempv,
             md->cv_acor_init,  &flagDevice,
             md, dmdv
  );
  __syncthreads();
#ifdef DEBUG_printmin
  printmin(md,dtempv,"cudaDevicecvSet after guess_helper dtempv");
#endif
  if(guessflag<0){
    *flag=RHSFUNC_RECVR;
    return RHSFUNC_RECVR;
  }
  for(;;) {
    __syncthreads();
    dcv_y[i] = dzn[i];
#ifdef DEBUG_printmin
    //printmin(md,md->state,"cudaDevicef start state");
#endif
    int aux_flag=0;
    int retval=cudaDevicef(cv_next_h, dcv_y,
            dftemp,md,dmdv,&aux_flag
    );
    if (retval < 0) {
      return CV_RHSFUNC_FAIL;
    }
    if (retval> 0) {
      return RHSFUNC_RECVR;
    }
    __syncthreads();
    //if (i == 0)
    dmdv->cv_nfe++;
    __syncthreads();

    if (callSetup==1) {

      __syncthreads();
#ifdef CAMP_DEBUG_GPU
#ifdef CAMP_PROFILE_DEVICE_FUNCTIONS
      start = clock();
#endif
#endif

      __syncthreads();
      int linflag=cudaDevicelinsolsetup(flag, md, dmdv,convfail);

      __syncthreads();
#ifdef CAMP_DEBUG_GPU
#ifdef CAMP_PROFILE_DEVICE_FUNCTIONS
      if(i==0) *md->dtlinsolsetup += ((double)(clock() - start))/(clock_khz*1000);
#endif
#endif

      dmdv->cv_nsetups++; //needed?
      callSetup = 0;
      dmdv->cv_gamrat = dmdv->cv_crate = 1.0;
      dmdv->cv_gammap = dmdv->cv_gamma;
      dmdv->cv_nstlp = dmdv->cv_nst;

      if (linflag < 0) {
        flag_shr[0] = CV_LSETUP_FAIL;
        break;
      }
      if (linflag > 0) {
        flag_shr[0] = CONV_FAIL;
        break;
      }

    }

    __syncthreads();
    cv_acor[i] = 0.0;

    __syncthreads();
#ifdef CAMP_DEBUG_GPU
#ifdef CAMP_PROFILE_DEVICE_FUNCTIONS
    start = clock();
#endif
#endif

    __syncthreads();
    int nItflag=cudaDevicecvNewtonIteration(md, dmdv);
    __syncthreads();

#ifdef CAMP_DEBUG_GPU
#ifdef CAMP_PROFILE_DEVICE_FUNCTIONS
    if(i==0) *md->dtNewtonIteration += ((double)(clock() - start))/(clock_khz*1000);
#endif
#endif

    if (nItflag != TRY_AGAIN) {
      return nItflag;
    }

    __syncthreads();
    callSetup = 1;
    __syncthreads();
    convfail = CV_FAIL_BAD_J;

    __syncthreads();

  } //for(;;)

  __syncthreads();
  return *flag;

}

__device__
void cudaDevicecvRescale(ModelDataGPU *md, ModelDataVariable *dmdv) {

  extern __shared__ double dzn[];

  int j;
  double factor;

  //if(i==0)printf("cudaDevicecvRescale2 start\n");

  __syncthreads();

  factor = dmdv->cv_eta;
  for (j=1; j <= dmdv->cv_q; j++) {
    //N_VScale(factor, md->dzn[j], md->dzn[j]);

    cudaDevicescaley(&md->dzn[md->nrows*(j)],factor,md->nrows);

    __syncthreads();
    //if(i==0)printf("cudaDevicecvRescale2 factor %le j %d\n",factor,j);
    factor *= dmdv->cv_eta;
    __syncthreads();
  }

  dmdv->cv_h = dmdv->cv_hscale * dmdv->cv_eta;
  dmdv->cv_next_h = dmdv->cv_h;
  dmdv->cv_hscale = dmdv->cv_h;
  dmdv->cv_nscon = 0;

  __syncthreads();

}

__device__
void cudaDevicecvRestore(ModelDataGPU *md, ModelDataVariable *dmdv, double saved_t) {

  extern __shared__ double dzn[];
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  int j, k;

  __syncthreads();
  dmdv->cv_tn=saved_t;

  for (k = 1; k <= dmdv->cv_q; k++){
    for (j = dmdv->cv_q; j >= k; j--) {
      //N_VLinearSum(ONE, cv_mem->cv_zn[j-1], -ONE,
      //             cv_mem->cv_zn[j], cv_mem->cv_zn[j-1]);

    cudaDevicezaxpby(1., &md->dzn[md->nrows*(j-1)], -1.,
            &md->dzn[md->nrows*(j)], &md->dzn[md->nrows*(j-1)], md->nrows);

    }
  }

  //N_VScale(ONE, cv_mem->cv_last_yn, cv_mem->cv_zn[0]);
  md->dzn[i]=md->cv_last_yn[i];

  __syncthreads();

}

__device__
int cudaDevicecvHandleNFlag(ModelDataGPU *md, ModelDataVariable *dmdv, int *nflagPtr, double saved_t,
                             int *ncfPtr) {

  extern __shared__ int flag_shr[];

  //if(i==0)printf("cudaDevicecvHandleNFlag *md->flag %d \n",*md->flag);

  if (*nflagPtr == CV_SUCCESS){
    return(DO_ERROR_TEST);
  }

  // The nonlinear soln. failed; increment ncfn and restore zn
  //if(i==0)
    dmdv->cv_ncfn++;

  cudaDevicecvRestore(md, dmdv, saved_t);
  //__syncthreads();

  if (*nflagPtr == CV_LSETUP_FAIL)  return(CV_LSETUP_FAIL);
  if (*nflagPtr == CV_LSOLVE_FAIL)  return(CV_LSOLVE_FAIL);
  if (*nflagPtr == CV_RHSFUNC_FAIL) return(CV_RHSFUNC_FAIL);


  (*ncfPtr)++;
  dmdv->cv_etamax = 1.;

  // If we had maxncf failures or |h| = hmin,
  //   return CV_CONV_FAILURE or CV_REPTD_RHSFUNC_ERR.

  __syncthreads();

  if ((fabs(dmdv->cv_h) <= dmdv->cv_hmin*ONEPSM) ||
      (*ncfPtr == dmdv->cv_maxncf)) {
    if (*nflagPtr == CONV_FAIL)     return(CV_CONV_FAILURE);
    if (*nflagPtr == RHSFUNC_RECVR) return(CV_REPTD_RHSFUNC_ERR);
  }

  // Reduce step size; return to reattempt the step
  __syncthreads();
  dmdv->cv_eta = SUNMAX(ETACF,
          dmdv->cv_hmin / fabs(dmdv->cv_h));
  __syncthreads();
  *nflagPtr = PREV_CONV_FAIL;
  cudaDevicecvRescale(md, dmdv);
  __syncthreads();

  return (PREDICT_AGAIN);

}

__device__
void cudaDevicecvSetTqBDFt(ModelDataGPU *md, ModelDataVariable *dmdv,
                           double hsum, double alpha0,
                           double alpha0_hat, double xi_inv, double xistar_inv) {

  extern __shared__ int flag_shr[];

  double A1, A2, A3, A4, A5, A6;
  double C, Cpinv, Cppinv;

  __syncthreads();

  A1 = 1. - alpha0_hat + alpha0;
  A2 = 1. + dmdv->cv_q * A1;

  md->cv_tq[2+blockIdx.x*(NUM_TESTS + 1)] = fabs(A1 / (alpha0 * A2));

  md->cv_tq[5+blockIdx.x*(NUM_TESTS + 1)] = fabs(A2 * xistar_inv / (md->cv_l[dmdv->cv_q+blockIdx.x*L_MAX] * xi_inv));
  if (dmdv->cv_qwait == 1) {
    if (dmdv->cv_q > 1) {
      C = xistar_inv / md->cv_l[dmdv->cv_q+blockIdx.x*L_MAX];
      A3 = alpha0 + 1. / dmdv->cv_q;
      A4 = alpha0_hat + xi_inv;
      Cpinv = (1. - A4 + A3) / A3;
      md->cv_tq[1+blockIdx.x*(NUM_TESTS + 1)] = fabs(C * Cpinv);
    }
    else md->cv_tq[1+blockIdx.x*(NUM_TESTS + 1)] = 1.;

    __syncthreads();

    hsum += md->cv_tau[dmdv->cv_q+blockIdx.x*(L_MAX + 1)];
    xi_inv = dmdv->cv_h / hsum;
    A5 = alpha0 - (1. / (dmdv->cv_q+1));
    A6 = alpha0_hat - xi_inv;
    Cppinv = (1. - A6 + A5) / A2;
    md->cv_tq[3+blockIdx.x*(NUM_TESTS + 1)] = fabs(Cppinv / (xi_inv * (dmdv->cv_q+2) * A5));
    __syncthreads();
  }

  md->cv_tq[4+blockIdx.x*(NUM_TESTS + 1)] = dmdv->cv_nlscoef / md->cv_tq[2+blockIdx.x*(NUM_TESTS + 1)];

}

__device__
void cudaDevicecvSetBDF(ModelDataGPU *md, ModelDataVariable *dmdv) {

  extern __shared__ int flag_shr[];

  double alpha0, alpha0_hat, xi_inv, xistar_inv, hsum;
  int z,j;

  __syncthreads();

  md->cv_l[0+blockIdx.x*L_MAX] = md->cv_l[1+blockIdx.x*L_MAX] = xi_inv = xistar_inv = 1.;
  for (z=2; z <= dmdv->cv_q; z++) md->cv_l[z+blockIdx.x*L_MAX] = 0.;
  alpha0 = alpha0_hat = -1.;
  hsum = dmdv->cv_h;
  __syncthreads();
  if (dmdv->cv_q > 1) {
    for (j=2; j < dmdv->cv_q; j++) {
      hsum += md->cv_tau[j-1+blockIdx.x*(L_MAX + 1)];
      xi_inv = dmdv->cv_h / hsum;
      alpha0 -= 1. / j;
      for (z=j; z >= 1; z--) md->cv_l[z+blockIdx.x*L_MAX] += md->cv_l[z-1+blockIdx.x*L_MAX]*xi_inv;
      // The l[z] are coefficients of product(1 to j) (1 + x/xi_i)
    }
    __syncthreads();
    // j = q
    alpha0 -= 1. / dmdv->cv_q;
    xistar_inv = -md->cv_l[1+blockIdx.x*L_MAX] - alpha0;
    hsum += md->cv_tau[dmdv->cv_q-1+blockIdx.x*(L_MAX + 1)];
    xi_inv = dmdv->cv_h / hsum;
    alpha0_hat = -md->cv_l[1+blockIdx.x*L_MAX] - xi_inv;
    for (z=dmdv->cv_q; z >= 1; z--)
      md->cv_l[z+blockIdx.x*L_MAX] += md->cv_l[z-1+blockIdx.x*L_MAX]*xistar_inv;
  }
  __syncthreads();
  cudaDevicecvSetTqBDFt(md, dmdv, hsum, alpha0, alpha0_hat, xi_inv, xistar_inv);

}

__device__
void cudaDevicecvSet(ModelDataGPU *md, ModelDataVariable *dmdv) {

  extern __shared__ int flag_shr[];
#ifdef DEBUG_printmin
  printmin(md,md->dtempv,"cudaDevicecvSet Start dtempv");
#endif
  __syncthreads();
  cudaDevicecvSetBDF(md,dmdv);
  __syncthreads();

  dmdv->cv_rl1 = 1.0 / md->cv_l[1+blockIdx.x*L_MAX];
  dmdv->cv_gamma = dmdv->cv_h * dmdv->cv_rl1;
  __syncthreads();
  if (dmdv->cv_nst == 0){
    //if(threadIdx.x == 0)
      //printf("dmdv->cv_nst == 0\n");
    dmdv->cv_gammap = dmdv->cv_gamma;

  }
  //if(threadIdx.x == 0)printf("cudaDevicecvSet3 dmdv->cv_nst %d dmdv->cv_gammap %le block %d\n", dmdv->cv_nst, dmdv->cv_gammap, blockIdx.x);
  __syncthreads();
  dmdv->cv_gamrat = (dmdv->cv_nst > 0) ?
                    dmdv->cv_gamma / dmdv->cv_gammap : 1.;  // protect x / x != 1.0
  __syncthreads();
}

__device__
void cudaDevicecvPredict(ModelDataGPU *md, ModelDataVariable *dmdv) {

  extern __shared__ double dzn[];
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  int j, k;
#ifdef DEBUG_printmin
  printmin(md,md->dtempv,"cudaDevicecvPredict start dtempv");
#endif
  __syncthreads();
  dmdv->cv_tn += dmdv->cv_h;
  __syncthreads();
  if (dmdv->cv_tstopset) {
    if ((dmdv->cv_tn - dmdv->cv_tstop)*dmdv->cv_h > 0.)
      dmdv->cv_tn = dmdv->cv_tstop;
  }

  //N_VScale(ONE, cv_mem->cv_zn[0], cv_mem->cv_last_yn);
  md->cv_last_yn[i]=md->dzn[i];

  for (k = 1; k <= dmdv->cv_q; k++){
    __syncthreads();
    for (j = dmdv->cv_q; j >= k; j--){
      __syncthreads();
      //N_VLinearSum(ONE, cv_mem->cv_zn[j-1], ONE,
      //             cv_mem->cv_zn[j], cv_mem->cv_zn[j-1]);
      cudaDevicezaxpby(1., &md->dzn[md->nrows*(j-1)], 1.,
                       &md->dzn[md->nrows*(j)], &md->dzn[md->nrows*(j-1)], md->nrows);

    }
    __syncthreads();
  }
  __syncthreads();
}

__device__
void cudaDevicecvDecreaseBDF(ModelDataGPU *md, ModelDataVariable *dmdv) {

  extern __shared__ double dzn[];

  double hsum, xi;
  int z, j;

  for (z=0; z <= dmdv->cv_qmax; z++) md->cv_l[z+blockIdx.x*L_MAX] = 0.;
  md->cv_l[2+blockIdx.x*L_MAX] = 1.;
  hsum = 0.;
  for (j=1; j <= dmdv->cv_q-2; j++) {
    hsum += md->cv_tau[j+blockIdx.x*(L_MAX + 1)];
    xi = hsum /dmdv->cv_hscale;
    for (z=j+2; z >= 2; z--)
      md->cv_l[z+blockIdx.x*L_MAX] = md->cv_l[z+blockIdx.x*L_MAX]*xi + md->cv_l[z-1+blockIdx.x*L_MAX];
  }
  for (j=2; j < dmdv->cv_q; j++){
    cudaDevicezaxpby(-md->cv_l[j+blockIdx.x*L_MAX],
                     &md->dzn[md->nrows*(dmdv->cv_q)],
                     1., &md->dzn[md->nrows*(j)],
                     &md->dzn[md->nrows*(j)], md->nrows);

    }
}

__device__
int cudaDevicecvDoErrorTest(ModelDataGPU *md, ModelDataVariable *dmdv,
                             int *nflagPtr,
                             double saved_t, int *nefPtr, double *dsmPtr) {

  extern __shared__ double dzn[];
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  double dsm;
  double min_val;
  int retval;
  cudaDevicezaxpby(md->cv_l[0+blockIdx.x*L_MAX],
                   md->cv_acor, 1., md->dzn, md->dftemp, md->nrows);
  cudaDevicemin(&min_val, md->dftemp[i], dzn, md->n_shr_empty);

  if (min_val < 0. && min_val > -CAMP_TINY) {
    md->dftemp[i]=fabs(md->dftemp[i]);
    cudaDevicezaxpby(-md->cv_l[0+blockIdx.x*L_MAX],
                     md->cv_acor, 1., md->dftemp, md->dzn, md->nrows);
    min_val = 0.;
  }

  dsm = dmdv->cv_acnrm * md->cv_tq[2+blockIdx.x*(NUM_TESTS + 1)];
  *dsmPtr = dsm;
  if (dsm <= 1. && min_val >= 0.) return(CV_SUCCESS);
  (*nefPtr)++;
  dmdv->cv_netf++;
  *nflagPtr = PREV_ERR_FAIL;
  cudaDevicecvRestore(md, dmdv, saved_t);

  __syncthreads();

  // At maxnef failures or |h| = hmin, return CV_ERR_FAILURE
  if ((fabs(dmdv->cv_h) <= dmdv->cv_hmin*ONEPSM) ||
      (*nefPtr == dmdv->cv_maxnef)) return(CV_ERR_FAILURE);

  // Set etamax = 1 to prevent step size increase at end of this step
  dmdv->cv_etamax = 1.;

  __syncthreads();

  // Set h ratio eta from dsm, rescale, and return for retry of step
  if (*nefPtr <= MXNEF1) {
    //dmdv->cv_eta = 1. / (SUNRpowerR(BIAS2*dsm,ONE/cv_mem->cv_L) + ADDON);
    dmdv->cv_eta = 1. / (pow(BIAS2*dsm,1./dmdv->cv_L) + ADDON);
    __syncthreads();
    dmdv->cv_eta = SUNMAX(ETAMIN, SUNMAX(dmdv->cv_eta,
                           dmdv->cv_hmin / fabs(dmdv->cv_h)));
    __syncthreads();
    if (*nefPtr >= SMALL_NEF)
      dmdv->cv_eta = SUNMIN(dmdv->cv_eta, ETAMXF);
    __syncthreads();

    cudaDevicecvRescale(md, dmdv);
    return(TRY_AGAIN);
  }
  __syncthreads();
  // After MXNEF1 failures, force an order reduction and retry step
  if (dmdv->cv_q > 1) {
    dmdv->cv_eta = SUNMAX(ETAMIN,
    dmdv->cv_hmin / fabs(dmdv->cv_h));
    //never enters?
    //if(i==0)printf("dmdv->cv_q > 1\n");
    cudaDevicecvDecreaseBDF(md, dmdv);

    dmdv->cv_L = dmdv->cv_q;
    dmdv->cv_q--;
    dmdv->cv_qwait = dmdv->cv_L;
    cudaDevicecvRescale(md, dmdv);
    __syncthreads();
    return(TRY_AGAIN);
  }
  __syncthreads();
  dmdv->cv_eta = SUNMAX(ETAMIN, dmdv->cv_hmin / fabs(dmdv->cv_h));
  __syncthreads();
  dmdv->cv_h *= dmdv->cv_eta;
  dmdv->cv_next_h = dmdv->cv_h;
  dmdv->cv_hscale = dmdv->cv_h;
  dmdv->cv_qwait = 10;
  dmdv->cv_nscon = 0;

#ifdef DEBUG_printmin
  printmin(md,md->state,"cudaDevicef start state");
#endif
  int aux_flag=0;
  retval=cudaDevicef(
          dmdv->cv_tn, md->dzn, md->dtempv,md,dmdv, &aux_flag
  );
  dmdv->cv_nfe++;
  if (retval < 0)  return(CV_RHSFUNC_FAIL);
  if (retval > 0)  return(CV_UNREC_RHSFUNC_ERR);
  md->dzn[1*md->nrows+i]=dmdv->cv_h*md->dtempv[i];
  return(TRY_AGAIN);

}

__device__
void cudaDevicecvCompleteStep(ModelDataGPU *md, ModelDataVariable *dmdv) {

  extern __shared__ double dzn[];
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  int z, j;
#ifdef DEBUG_printmin
  printmin(md,md->dtempv,"cudaDevicecvCompleteStep start dtempv");
#endif
  dmdv->cv_nst++;
  dmdv->cv_nscon++;
  dmdv->cv_hu = dmdv->cv_h;
  dmdv->cv_qu = dmdv->cv_q;

  for (z=dmdv->cv_q; z >= 2; z--)  md->cv_tau[z+blockIdx.x*(L_MAX + 1)] = md->cv_tau[z-1+blockIdx.x*(L_MAX + 1)];
  if ((dmdv->cv_q==1) && (dmdv->cv_nst > 1))
    md->cv_tau[2+blockIdx.x*(L_MAX + 1)] = md->cv_tau[1+blockIdx.x*(L_MAX + 1)];
  md->cv_tau[1+blockIdx.x*(L_MAX + 1)] = dmdv->cv_h;
  __syncthreads();
  for (j=0; j <= dmdv->cv_q; j++){
    cudaDevicezaxpby(md->cv_l[j+blockIdx.x*L_MAX],
                     md->cv_acor,
                     1., &md->dzn[md->nrows*(j)],
                     &md->dzn[md->nrows*(j)], md->nrows);

  }
  dmdv->cv_qwait--;
  if ((dmdv->cv_qwait == 1) && (dmdv->cv_q != dmdv->cv_qmax)) {
    md->dzn[md->nrows*(dmdv->cv_qmax)+i]=md->cv_acor[i];
    dmdv->cv_saved_tq5 = md->cv_tq[5+blockIdx.x*(NUM_TESTS + 1)];
    dmdv->cv_indx_acor = dmdv->cv_qmax;
  }

}

__device__
void cudaDevicecvChooseEta(ModelDataGPU *md, ModelDataVariable *dmdv) {

  extern __shared__ double dzn[];
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  double etam;
  etam = SUNMAX(dmdv->cv_etaqm1, SUNMAX(dmdv->cv_etaq, dmdv->cv_etaqp1));
  __syncthreads();
  if (etam < THRESH) {
    dmdv->cv_eta = 1.;
    dmdv->cv_qprime = dmdv->cv_q;
    return;
  }
  __syncthreads();
  if (etam == dmdv->cv_etaq) {
    dmdv->cv_eta = dmdv->cv_etaq;
    dmdv->cv_qprime = dmdv->cv_q;
  } else if (etam == dmdv->cv_etaqm1) {
    dmdv->cv_eta = dmdv->cv_etaqm1;
    dmdv->cv_qprime = dmdv->cv_q - 1;
  } else {
    dmdv->cv_eta = dmdv->cv_etaqp1;
    dmdv->cv_qprime = dmdv->cv_q + 1;
    __syncthreads();
    if (dmdv->cv_lmm == CV_BDF) {
      md->dzn[md->nrows*(dmdv->cv_qmax)+i]=md->cv_acor[i];
    }
  }
  __syncthreads();

}

__device__
void cudaDevicecvSetEta(ModelDataGPU *md, ModelDataVariable *dmdv) {

  extern __shared__ int flag_shr[];

  __syncthreads();
  if (dmdv->cv_eta < THRESH) {
    dmdv->cv_eta = 1.;
    dmdv->cv_hprime = dmdv->cv_h;
  } else {
    __syncthreads();
    dmdv->cv_eta = SUNMIN(dmdv->cv_eta, dmdv->cv_etamax);
    __syncthreads();
    dmdv->cv_eta /= SUNMAX(ONE,
            fabs(dmdv->cv_h)*dmdv->cv_hmax_inv*dmdv->cv_eta);
    __syncthreads();
    dmdv->cv_hprime = dmdv->cv_h * dmdv->cv_eta;
    __syncthreads();
    if (dmdv->cv_qprime < dmdv->cv_q) dmdv->cv_nscon = 0;
  }
  __syncthreads();
}

__device__
int cudaDevicecvPrepareNextStep(ModelDataGPU *md, ModelDataVariable *dmdv, double dsm) {

  extern __shared__ double sdata[];
  __syncthreads();
#ifdef DEBUG_printmin
  printmin(md,md->dtempv,"cudaDevicecvPrepareNextStep start dtempv");
#endif
  if (dmdv->cv_etamax == 1.) {
    dmdv->cv_qwait = SUNMAX(dmdv->cv_qwait, 2);
    dmdv->cv_qprime = dmdv->cv_q;
    dmdv->cv_hprime = dmdv->cv_h;
    dmdv->cv_eta = 1.;
    return 0;
  }

  __syncthreads();
  dmdv->cv_etaq = 1. /(pow(BIAS2*dsm,1./dmdv->cv_L) + ADDON);
  __syncthreads();
  if (dmdv->cv_qwait != 0) {
    dmdv->cv_eta = dmdv->cv_etaq;
    dmdv->cv_qprime = dmdv->cv_q;
    cudaDevicecvSetEta(md, dmdv);
    return 0;
  }
  __syncthreads();
  dmdv->cv_qwait = 2;
  double ddn;
  dmdv->cv_etaqm1 = 0.;
  __syncthreads();
  if (dmdv->cv_q > 1) {
    cudaDeviceVWRMS_Norm(&md->dzn[md->nrows*(dmdv->cv_q)],
                         md->dewt, &ddn, md->nrows, md->n_shr_empty);
    __syncthreads();
    ddn *= md->cv_tq[1+blockIdx.x*(NUM_TESTS + 1)];
    __syncthreads();
    dmdv->cv_etaqm1 = 1./(pow(BIAS1*ddn, 1./dmdv->cv_q) + ADDON);
  }
  double dup, cquot;
  dmdv->cv_etaqp1 = 0.;
  __syncthreads();
  if (dmdv->cv_q != dmdv->cv_qmax && dmdv->cv_saved_tq5 != 0.) {
    cquot = (md->cv_tq[5+blockIdx.x*(NUM_TESTS + 1)] / dmdv->cv_saved_tq5) *
            pow(double(dmdv->cv_h/md->cv_tau[2+blockIdx.x*(L_MAX + 1)]), double(dmdv->cv_L));
    cudaDevicezaxpby(-cquot,
    &md->dzn[md->nrows*(dmdv->cv_qmax)],
    1., md->cv_acor,
    md->dtempv, md->nrows);

    //dup = N_VWrmsNorm(md->dtempv, cv_mem->cv_ewt) * cv_mem->cv_tq[3];
    cudaDeviceVWRMS_Norm(md->dtempv, md->dewt, &dup, md->nrows, md->n_shr_empty);

    __syncthreads();
    dup *= md->cv_tq[3+blockIdx.x*(NUM_TESTS + 1)];
    __syncthreads();
    dmdv->cv_etaqp1 = 1. / (pow(BIAS3*dup, 1./(dmdv->cv_L+1)) + ADDON);
  }
  __syncthreads();
  cudaDevicecvChooseEta(md, dmdv);
  __syncthreads();
  cudaDevicecvSetEta(md, dmdv);
  __syncthreads();
  return CV_SUCCESS;
}

__device__
void cudaDevicecvIncreaseBDF(ModelDataGPU *md, ModelDataVariable *dmdv) {

  extern __shared__ double dzn[];
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int tid = threadIdx.x;

  double alpha0, alpha1, prod, xi, xiold, hsum, A1;
  int z, j;

  for (z=0; z <= dmdv->cv_qmax; z++) md->cv_l[z+blockIdx.x*L_MAX] = 0.;
  md->cv_l[2+blockIdx.x*L_MAX] = alpha1 = prod = xiold = 1.;

  alpha0 = -1.;
  hsum = dmdv->cv_hscale;
  if (dmdv->cv_q > 1) {
    for (j=1; j < dmdv->cv_q; j++) {
      hsum += md->cv_tau[j+1+blockIdx.x*(L_MAX + 1)];
      xi = hsum / dmdv->cv_hscale;
      prod *= xi;
      alpha0 -= 1. / (j+1);
      alpha1 += 1. / xi;
      for (z=j+2; z >= 2; z--)
        md->cv_l[z+blockIdx.x*L_MAX] = md->cv_l[z+blockIdx.x*L_MAX]*xiold + md->cv_l[z-1+blockIdx.x*L_MAX];
      xiold = xi;
    }
  }
  A1 = (-alpha0 - alpha1) / prod;
  dzn[tid]=md->dzn[md->nrows*(dmdv->cv_L)+i];
  dzn[tid]=A1*md->dzn[md->nrows*(dmdv->cv_indx_acor)+i];
  md->dzn[md->nrows*(dmdv->cv_L)+i]=dzn[tid];
  for (j=2; j <= dmdv->cv_q; j++){
    cudaDevicezaxpby(md->cv_l[j+blockIdx.x*L_MAX],
    &md->dzn[md->nrows*(dmdv->cv_L)],
    1., &md->dzn[md->nrows*(j)],
    &md->dzn[md->nrows*(j)], md->nrows);
  }
}

__device__
void cudaDevicecvAdjustParams(ModelDataGPU *md, ModelDataVariable *dmdv) {

  if (dmdv->cv_qprime != dmdv->cv_q) {

    int deltaq = dmdv->cv_qprime-dmdv->cv_q;
    switch(deltaq) {
      case 1:
        cudaDevicecvIncreaseBDF(md, dmdv);
        break;
      case -1:
        cudaDevicecvDecreaseBDF(md, dmdv);
        break;
    }

    dmdv->cv_q = dmdv->cv_qprime;
    dmdv->cv_L = dmdv->cv_q+1;
    dmdv->cv_qwait = dmdv->cv_L;
  }
  cudaDevicecvRescale(md, dmdv);
}

__device__
int cudaDevicecvStep(ModelDataGPU *md, ModelDataVariable *dmdv) {

  extern __shared__ double sdata[];
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  double saved_t = dmdv->cv_tn;
  int ncf = 0;
  int nef = 0;
  dmdv->nflag = FIRST_CALL;
  int nflag=FIRST_CALL;
  double dsm;

  __syncthreads();
  if ((dmdv->cv_nst > 0) && (dmdv->cv_hprime != dmdv->cv_h)){
    cudaDevicecvAdjustParams(md, dmdv);
  }
  __syncthreads();
  for (;;) {
    __syncthreads();
    cudaDevicecvPredict(md, dmdv);
    __syncthreads();
    cudaDevicecvSet(md, dmdv);
    __syncthreads();

    nflag = cudaDevicecvNlsNewton(&nflag,md, dmdv);

    __syncthreads();
    dmdv->nflag = nflag;
    __syncthreads();
#ifdef DEBUG_cudaDevicecvStep
    if(threadIdx.x==0)printf("DEBUG_cudaDevicecvStep nflag %d dmdv->nflag %d block %d\n",dmdv->nflag, dmdv->nflag, blockIdx.x);
#endif
    int kflag = cudaDevicecvHandleNFlag(md, dmdv, &nflag, saved_t, &ncf);

    __syncthreads();
    dmdv->nflag = nflag;//needed?
    dmdv->kflag = kflag;
    __syncthreads();
#ifdef DEBUG_cudaDevicecvStep
    if(threadIdx.x==0)printf("DEBUG_cudaDevicecvStep kflag %d block %d\n",dmdv->kflag, blockIdx.x);
#endif
    if (dmdv->kflag == PREDICT_AGAIN) {
      //if (threadIdx.x == 0)printf("DEBUG_cudaDevicecvStep kflag PREDICT_AGAIN block %d\n", blockIdx.x);
      continue;
    }
    if (dmdv->kflag != DO_ERROR_TEST) {
      //if(threadIdx.x==0)printf("DEBUG_cudaDevicecvStep kflag!=DO_ERROR_TEST block %d\n", blockIdx.x);
      return (dmdv->kflag);
    }
    __syncthreads();
    int eflag=cudaDevicecvDoErrorTest(md,dmdv,&nflag,saved_t,&nef,&dsm);
    __syncthreads();
    dmdv->nflag = nflag;
    dmdv->eflag = eflag;
    __syncthreads();
#ifdef DEBUG_cudaDevicecvStep
    if(threadIdx.x==0)printf("DEBUG_cudaDevicecvStep nflag %d eflag %d block %d\n",dmdv->nflag, dmdv->eflag, blockIdx.x);    //if(i==0)printf("eflag %d\n", eflag);
#endif
    if (dmdv->eflag == TRY_AGAIN){
      continue;
    }
    if (dmdv->eflag != CV_SUCCESS){
      return (dmdv->eflag);
    }
    break;
  }
  __syncthreads();
  cudaDevicecvCompleteStep(md, dmdv);
  __syncthreads();
  cudaDevicecvPrepareNextStep(md, dmdv, dsm);
  __syncthreads();
  dmdv->cv_etamax=10.;
  md->cv_acor[i]*=md->cv_tq[2+blockIdx.x*(NUM_TESTS + 1)];
  __syncthreads();
  return(CV_SUCCESS);

  }

__device__
int cudaDeviceCVodeGetDky(ModelDataGPU *md, ModelDataVariable *dmdv,
                           double t, int k, double *dky) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  double s, c, r;
  double tfuzz, tp, tn1;
  int z, j;
  __syncthreads();
   tfuzz = FUZZ_FACTOR * dmdv->cv_uround * (fabs(dmdv->cv_tn) + fabs(dmdv->cv_hu));
   if (dmdv->cv_hu < 0.) tfuzz = -tfuzz;
   tp = dmdv->cv_tn - dmdv->cv_hu - tfuzz;
   tn1 = dmdv->cv_tn + tfuzz;
   if ((t-tp)*(t-tn1) > 0.) {
     return(CV_BAD_T);
   }
  __syncthreads();
   s = (t - dmdv->cv_tn) / dmdv->cv_h;
   for (j=dmdv->cv_q; j >= k; j--) {
     c = 1.;
     for (z=j; z >= j-k+1; z--) c *= z;
     if (j == dmdv->cv_q) {
       dky[i]=c*md->dzn[md->nrows*(j)+i];
     } else {
       cudaDevicezaxpby(c,
        &md->dzn[md->nrows*(j)],
        s, dky,
        dky, md->nrows);
     }
   }
  __syncthreads();
   if (k == 0) return(CV_SUCCESS);
  __syncthreads();
   r = pow(double(dmdv->cv_h),double(-k));
  __syncthreads();
   dky[i]=dky[i]*r;

   return(CV_SUCCESS);
}

__device__
int cudaDevicecvEwtSetSV(ModelDataGPU *md, ModelDataVariable *dmdv,
                         double *dzn, double *weight) {

  extern __shared__ double flag_shr2[];
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  md->dtempv[i]=fabs(dzn[i]);
 cudaDevicezaxpby(dmdv->cv_reltol, md->dtempv, 1.,
        md->cv_Vabstol, md->dtempv, md->nrows);
  double min;
  cudaDevicemin(&min, md->dtempv[i], flag_shr2, md->n_shr_empty);
  __syncthreads();
  if (min <= 0.) return(-1);
  weight[i]= 1./md->dtempv[i];
  return(0);
}

__device__
int cudaDeviceCVode(ModelDataGPU *md, ModelDataVariable *dmdv) {

  extern __shared__ int flag_shr[];
  int i = blockIdx.x * blockDim.x + threadIdx.x;

#ifdef DEBUG_printmin
  printmin(md,md->state,"cudaDeviceCVode start state");
#endif
  for(;;) {
    __syncthreads();
#ifdef CAMP_DEBUG_GPU
#ifdef CAMP_PROFILE_DEVICE_FUNCTIONS
    dmdv->countercvStep++;
#endif
#endif
    flag_shr[0] = 0;
    dmdv->flag = 0;
    __syncthreads();
    dmdv->cv_next_h = dmdv->cv_h;
    dmdv->cv_next_q = dmdv->cv_q;
    int ewtsetOK = 0;
    if (dmdv->cv_nst > 0) {
      ewtsetOK = cudaDevicecvEwtSetSV(md, dmdv, md->dzn, md->dewt);
      if (ewtsetOK != 0) {
        dmdv->cv_tretlast = dmdv->tret = dmdv->cv_tn;
        md->yout[i] = md->dzn[i];
        if(i==0) printf("ERROR: ewtsetOK istate %d\n",dmdv->istate);
        return CV_ILL_INPUT;
      }
    }
    if ((dmdv->cv_mxstep > 0) && (dmdv->nstloc >= dmdv->cv_mxstep)) {
      dmdv->cv_tretlast = dmdv->tret = dmdv->cv_tn;
      md->yout[i] = md->dzn[i];
      if(i==0) printf("ERROR: cv_mxstep istate %d\n",dmdv->istate);
      return CV_TOO_MUCH_WORK;
    }
    double nrm;
    cudaDeviceVWRMS_Norm(md->dzn,
                         md->dewt, &nrm, md->nrows, md->n_shr_empty);
    dmdv->cv_tolsf = dmdv->cv_uround * nrm;
    if (dmdv->cv_tolsf > 1.) {
      dmdv->cv_tretlast = dmdv->tret = dmdv->cv_tn;
      md->yout[i] = md->dzn[i];
      dmdv->cv_tolsf *= 2.;
      if(i==0) printf("ERROR: cv_tolsf istate %d\n",dmdv->istate);
      __syncthreads();
      return CV_TOO_MUCH_ACC;
    } else {
      dmdv->cv_tolsf = 1.;
    }

#ifdef ODE_WARNING
    // Check for h below roundoff level in tn
    if (dmdv->cv_tn + dmdv->cv_h == dmdv->cv_tn) {
      dmdv->cv_nhnil++;
      //if (dmdv->cv_nhnil <= dmdv->cv_mxhnil)
      //  cvProcessError(dmdv, CV_WARNING, "CVODE", "CVode",
      //                 MSGCV_HNIL, dmdv->cv_tn, dmdv->cv_h);
      //if (dmdv->cv_nhnil == dmdv->cv_mxhnil)
      //  cvProcessError(dmdv, CV_WARNING, "CVODE", "CVode", MSGCV_HNIL_DONE);
      if ((dmdv->cv_nhnil <= dmdv->cv_mxhnil) ||
              (dmdv->cv_nhnil == dmdv->cv_mxhnil))
        if(i==0)printf("WARNING: h below roundoff level in tn");
    }
#endif

    int kflag2 = cudaDevicecvStep(md, dmdv);
    __syncthreads();
    dmdv->kflag2=kflag2;
    __syncthreads();
#ifdef DEBUG_cudaDeviceCVode
    if(i==0){
      printf("DEBUG_cudaDeviceCVode%d thread %d\n", i);
      printf("dmdv->cv_tn %le dmdv->tout %le dmdv->cv_h %le dmdv->cv_hprime %le\n",
             dmdv->cv_tn,dmdv->tout,dmdv->cv_h,dmdv->cv_hprime);
    }
#endif
    if (dmdv->kflag2 != CV_SUCCESS) {
      dmdv->cv_tretlast = dmdv->tret = dmdv->cv_tn;
      md->yout[i] = md->dzn[i];
      if(i==0) printf("ERROR: dmdv->kflag != CV_SUCCESS istate %d\n",dmdv->istate);
      return dmdv->kflag2;
    }
    dmdv->nstloc++;
    if ((dmdv->cv_tn - dmdv->tout) * dmdv->cv_h >= 0.) {
      dmdv->istate = CV_SUCCESS;
      dmdv->cv_tretlast = dmdv->tret = dmdv->tout;
      cudaDeviceCVodeGetDky(md, dmdv, dmdv->tout, 0, md->yout);
      return CV_SUCCESS;
    }
    if (dmdv->cv_tstopset) {//needed?
      double troundoff = FUZZ_FACTOR * dmdv->cv_uround * (fabs(dmdv->cv_tn) + fabs(dmdv->cv_h));
      if (fabs(dmdv->cv_tn - dmdv->cv_tstop) <= troundoff) {
        //(void) CVodeGetDky(dmdv, dmdv->cv_tstop, 0, md->yout);
        cudaDeviceCVodeGetDky(md, dmdv, dmdv->cv_tstop, 0, md->yout);
        dmdv->cv_tretlast = dmdv->tret = dmdv->cv_tstop;
        dmdv->cv_tstopset = SUNFALSE;
        dmdv->istate = CV_TSTOP_RETURN;
        if(i==0) printf("ERROR: cv_tstopset istate %d\n",dmdv->istate);
        __syncthreads();
        return CV_TSTOP_RETURN;
      }
      if ((dmdv->cv_tn + dmdv->cv_hprime - dmdv->cv_tstop) * dmdv->cv_h > 0.) {
        dmdv->cv_hprime = (dmdv->cv_tstop - dmdv->cv_tn) * (1.0 - 4.0 * dmdv->cv_uround);
        if(i==0) printf("ERROR: dmdv->cv_tn + dmdv->cv_hprime - dmdv->cv_tstop istate %d\n",dmdv->istate);
        dmdv->cv_eta = dmdv->cv_hprime / dmdv->cv_h;
      }
    }
  }
}

__global__
void cudaGlobalCVode(ModelDataGPU md_object) {

  extern __shared__ int flag_shr[];
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  ModelDataGPU *md = &md_object;
#ifdef DEV_DMDV_ARRAY
  ModelDataVariable *dmdv = &md_object.mdv[blockIdx.x];
#else
  ModelDataVariable dmdv_object = *md_object.mdv;
  ModelDataVariable *dmdv = &dmdv_object;
#endif
  int active_threads = md->nrows;

  __syncthreads();
  int istate;
  if(tid<active_threads){

    __syncthreads();
#ifdef CAMP_DEBUG_GPU
#ifdef CAMP_PROFILE_DEVICE_FUNCTIONS
    int clock_khz=md->clock_khz;
    clock_t start;
    start = clock();
    __syncthreads();
#endif
#endif
    istate=cudaDeviceCVode(md,dmdv);
    __syncthreads();
#ifdef CAMP_DEBUG_GPU
#ifdef CAMP_PROFILE_DEVICE_FUNCTIONS
    __syncthreads();
     dmdv->dtcudaDeviceCVode += ((double)(int)(clock() - start))/(clock_khz*1000);
#endif
#endif

  }
  __syncthreads();
  dmdv->istate=istate;
  __syncthreads();
  if(threadIdx.x==0)md->flagCells[blockIdx.x]=dmdv->istate;
  ModelDataVariable *mdvo = md->mdvo;
#ifdef DEV_DMDV_ARRAY
  if(threadIdx.x==0) *mdvo = dmdv[blockIdx.x];
#else
  *mdvo = *dmdv;
#endif
}

void solveCVODEGPU_thr(int blocks, int threads_block, int n_shr_memory, int n_shr_empty, int offset_cells,
                       SolverData *sd, CVodeMem cv_mem)
{

#ifdef DEBUG_SOLVEBCGCUDA
  itsolver *bicg = &(sd->bicg);
  ModelDataGPU *mGPU = sd->mGPU;
  if(bicg->counterBiConjGrad==0) {
    printf("solveCVODEGPU_thr n_cells %d len_cell %d nrows %d nnz %d max_threads_block %d blocks %d threads_block %d n_shr_empty %d offset_cells %d\n",
           mGPU->n_cells,len_cell,mGPU->nrows,mGPU->nnz,n_shr_memory,blocks,threads_block,n_shr_empty,offset_cells);

  }
#endif

  cudaGlobalCVode <<<blocks,threads_block,n_shr_memory*sizeof(double)>>>
                                             (*sd->mGPU);

}

void solveCVODEGPU(SolverData *sd, CVodeMem cv_mem)
{
  ModelDataGPU *mGPU = sd->mGPU;

#ifdef DEBUG_SOLVEBCGCUDA
  if(bicg->counterBiConjGrad==0) {
    printf("solveCVODEGPU\n");
  }
#endif

  int len_cell = mGPU->nrows/mGPU->n_cells;
  int max_threads_block=mGPU->threads;

  int offset_cells=0;

  int threads_block = len_cell;
  //int blocks = mGPU->blocks = mGPU->n_cells;
  int blocks = mGPU->n_cells;
  int n_shr = max_threads_block = nextPowerOfTwoCVODE(len_cell);
  int n_shr_empty = mGPU->n_shr_empty= n_shr-threads_block;

  solveCVODEGPU_thr(blocks, threads_block, max_threads_block, n_shr_empty, offset_cells,
                    sd,cv_mem);

#ifdef DEBUG_SOLVEBCGCUDA
  if(bicg->counterBiConjGrad<2) {
    printf("solveGPUBlock end\n");
  }
#endif

#ifdef DEBUG_SOLVEBCGCUDA
  if(bicg->counterBiConjGrad<2) {
    printf("solveCVODEGPU end\n");
  }
#endif

}

int cudaCVode(void *cvode_mem, realtype tout, N_Vector yout,
               realtype *tret, int itask, SolverData *sd)
{
  CVodeMem cv_mem;
  long int nstloc;
  int retval, hflag, kflag, istate, ier, irfndp;
  realtype troundoff, tout_hin, rh;

  itsolver *bicg = &(sd->bicg);
  ModelDataGPU *mGPU;
  ModelData *md = &(sd->model_data);
  //double *youtArray = N_VGetArrayPointer(yout);

  //printf("cudaCVode start \n");

   // 1. Check and process inputs
  /* Check if cvode_mem exists */
  if (cvode_mem == NULL) {
    cvProcessError(NULL, CV_MEM_NULL, "CVODE", "CVode", MSGCV_NO_MEM);
    return(CV_MEM_NULL);
  }
  cv_mem = (CVodeMem) cvode_mem;

  /* Check if cvode_mem was allocated */
  if (cv_mem->cv_MallocDone == SUNFALSE) {
    cvProcessError(cv_mem, CV_NO_MALLOC, "CVODE", "CVode", MSGCV_NO_MALLOC);
    return(CV_NO_MALLOC);
  }

  /* Check for yout != NULL */
  if ((cv_mem->cv_y = yout) == NULL) {
    cvProcessError(cv_mem, CV_ILL_INPUT, "CVODE", "CVode", MSGCV_YOUT_NULL);
    return(CV_ILL_INPUT);
  }

  /* Check for tret != NULL */
  if (tret == NULL) {
    cvProcessError(cv_mem, CV_ILL_INPUT, "CVODE", "CVode", MSGCV_TRET_NULL);
    return(CV_ILL_INPUT);
  }

  /* Check for valid itask */
  if ( (itask != CV_NORMAL) && (itask != CV_ONE_STEP) ) {
    cvProcessError(cv_mem, CV_ILL_INPUT, "CVODE", "CVode", MSGCV_BAD_ITASK);
    return(CV_ILL_INPUT);
  }

  if (itask == CV_NORMAL) cv_mem->cv_toutc = tout;
  cv_mem->cv_taskc = itask;

  //2. Initializations performed only at
  if (cv_mem->cv_nst == 0) {

    cv_mem->cv_tretlast = *tret = cv_mem->cv_tn;

    ier = cvInitialSetup_gpu(cv_mem);
    if (ier!= CV_SUCCESS) return(ier);

    /* Call f at (t0,y0), set zn[1] = y'(t0),
       set initial h (from H0 or cvHin), and scale zn[1] by h.
       Also check for zeros of root function g at and near t0.    */
    //retval = cv_mem->cv_f(cv_mem->cv_tn, cv_mem->cv_zn[0],
    //                      cv_mem->cv_zn[1], cv_mem->cv_user_data);
    retval = f(cv_mem->cv_tn, cv_mem->cv_zn[0], cv_mem->cv_zn[1], cv_mem->cv_user_data);

    N_VScale(ONE, cv_mem->cv_zn[0], yout);

    cv_mem->cv_nfe++;
    if (retval < 0) {
      cvProcessError(cv_mem, CV_RHSFUNC_FAIL, "CVODE", "CVode",
                     MSGCV_RHSFUNC_FAILED, cv_mem->cv_tn);
      return(CV_RHSFUNC_FAIL);
    }
    if (retval > 0) {
      cvProcessError(cv_mem, CV_FIRST_RHSFUNC_ERR, "CVODE", "CVode",
                     MSGCV_RHSFUNC_FIRST);
      return(CV_FIRST_RHSFUNC_ERR);
    }
    if (cv_mem->cv_tstopset) {
      if ( (cv_mem->cv_tstop - cv_mem->cv_tn)*(tout - cv_mem->cv_tn) <= ZERO ) {
        cvProcessError(cv_mem, CV_ILL_INPUT, "CVODE", "CVode",
                       MSGCV_BAD_TSTOP, cv_mem->cv_tstop, cv_mem->cv_tn);
        return(CV_ILL_INPUT);
      }
    }
    cv_mem->cv_h = cv_mem->cv_hin;
    if ( (cv_mem->cv_h != ZERO) && ((tout-cv_mem->cv_tn)*cv_mem->cv_h < ZERO) ) {
      cvProcessError(cv_mem, CV_ILL_INPUT, "CVODE", "CVode", MSGCV_BAD_H0);
      return(CV_ILL_INPUT);
    }
    if (cv_mem->cv_h == ZERO) {
      tout_hin = tout;
      if ( cv_mem->cv_tstopset && (tout-cv_mem->cv_tn)*(tout-cv_mem->cv_tstop) > ZERO )
        tout_hin = cv_mem->cv_tstop;
      hflag = cvHin_gpu(cv_mem, tout_hin); //set cv_y
      if (hflag != CV_SUCCESS) {
        istate = cvHandleFailure_gpu(cv_mem, hflag);
        return(istate);
      }
    }
    rh = SUNRabs(cv_mem->cv_h)*cv_mem->cv_hmax_inv;
    if (rh > ONE) cv_mem->cv_h /= rh;
    if (SUNRabs(cv_mem->cv_h) < cv_mem->cv_hmin)
      cv_mem->cv_h *= cv_mem->cv_hmin/SUNRabs(cv_mem->cv_h);

    /* Check for approach to tstop */

    if (cv_mem->cv_tstopset) {
      if ( (cv_mem->cv_tn + cv_mem->cv_h - cv_mem->cv_tstop)*cv_mem->cv_h > ZERO )
        cv_mem->cv_h = (cv_mem->cv_tstop - cv_mem->cv_tn)*(ONE-FOUR*cv_mem->cv_uround);
    }
    cv_mem->cv_hscale = cv_mem->cv_h;
    cv_mem->cv_h0u    = cv_mem->cv_h;
    cv_mem->cv_hprime = cv_mem->cv_h;
    N_VScale(cv_mem->cv_h, cv_mem->cv_zn[1], cv_mem->cv_zn[1]);
    if (cv_mem->cv_ghfun) {

      N_VLinearSum(ONE, cv_mem->cv_zn[0], ONE, cv_mem->cv_zn[1], cv_mem->cv_tempv1);
      cv_mem->cv_ghfun(cv_mem->cv_tn + cv_mem->cv_h, cv_mem->cv_h, cv_mem->cv_tempv1,
                       cv_mem->cv_zn[0], cv_mem->cv_zn[1], cv_mem->cv_user_data,
                       cv_mem->cv_tempv2, cv_mem->cv_acor_init);
    }
    if (cv_mem->cv_nrtfn > 0) {

      retval = cvRcheck1_gpu(cv_mem);

      if (retval == CV_RTFUNC_FAIL) {
        cvProcessError(cv_mem, CV_RTFUNC_FAIL, "CVODE", "cvRcheck1",
                       MSGCV_RTFUNC_FAILED, cv_mem->cv_tn);
        return(CV_RTFUNC_FAIL);
      }
    }
  } /* end of first call block */


   //3. At following steps, perform stop tests:
  if (cv_mem->cv_nst > 0) {

    /* Estimate an infinitesimal time interval to be used as
       a roundoff for time quantities (based on current time
       and step size) */
    troundoff = FUZZ_FACTOR*cv_mem->cv_uround*(SUNRabs(cv_mem->cv_tn) + SUNRabs(cv_mem->cv_h));

    /* First, check for a root in the last step taken, other than the
       last root found, if any.  If itask = CV_ONE_STEP and y(tn) was not
       returned because of an intervening root, return y(tn) now.     */
    if (cv_mem->cv_nrtfn > 0) {

      irfndp = cv_mem->cv_irfnd;

      retval = cvRcheck2_gpu(cv_mem);

      if (retval == CLOSERT) {
        cvProcessError(cv_mem, CV_ILL_INPUT, "CVODE", "cvRcheck2",
                       MSGCV_CLOSE_ROOTS, cv_mem->cv_tlo);
        return(CV_ILL_INPUT);
      } else if (retval == CV_RTFUNC_FAIL) {
        cvProcessError(cv_mem, CV_RTFUNC_FAIL, "CVODE", "cvRcheck2",
                       MSGCV_RTFUNC_FAILED, cv_mem->cv_tlo);
        return(CV_RTFUNC_FAIL);
      } else if (retval == RTFOUND) {
        cv_mem->cv_tretlast = *tret = cv_mem->cv_tlo;
        return(CV_ROOT_RETURN);
      }

      /* If tn is distinct from tretlast (within roundoff),
         check remaining interval for roots */
      if ( SUNRabs(cv_mem->cv_tn - cv_mem->cv_tretlast) > troundoff ) {

        retval = cvRcheck3_gpu(cv_mem);

        if (retval == CV_SUCCESS) {     /* no root found */
          cv_mem->cv_irfnd = 0;
          if ((irfndp == 1) && (itask == CV_ONE_STEP)) {
            cv_mem->cv_tretlast = *tret = cv_mem->cv_tn;
            N_VScale(ONE, cv_mem->cv_zn[0], yout);
            return(CV_SUCCESS);
          }
        } else if (retval == RTFOUND) {  /* a new root was found */
          cv_mem->cv_irfnd = 1;
          cv_mem->cv_tretlast = *tret = cv_mem->cv_tlo;
          return(CV_ROOT_RETURN);
        } else if (retval == CV_RTFUNC_FAIL) {  /* g failed */
          cvProcessError(cv_mem, CV_RTFUNC_FAIL, "CVODE", "cvRcheck3",
                         MSGCV_RTFUNC_FAILED, cv_mem->cv_tlo);
          return(CV_RTFUNC_FAIL);
        }

      }

    } /* end of root stop check */
    if ( (itask == CV_NORMAL) && ((cv_mem->cv_tn-tout)*cv_mem->cv_h >= ZERO) ) {
      cv_mem->cv_tretlast = *tret = tout;
      ier =  CVodeGetDky(cv_mem, tout, 0, yout);
      if (ier != CV_SUCCESS) {
        cvProcessError(cv_mem, CV_ILL_INPUT, "CVODE", "CVode",
                       MSGCV_BAD_TOUT, tout);
        return(CV_ILL_INPUT);
      }
      return(CV_SUCCESS);
    }
    if ( itask == CV_ONE_STEP &&
         SUNRabs(cv_mem->cv_tn - cv_mem->cv_tretlast) > troundoff ) {
      cv_mem->cv_tretlast = *tret = cv_mem->cv_tn;
      N_VScale(ONE, cv_mem->cv_zn[0], yout);
      return(CV_SUCCESS);
    }
    if ( cv_mem->cv_tstopset ) {

      if ( SUNRabs(cv_mem->cv_tn - cv_mem->cv_tstop) <= troundoff) {
        ier =  CVodeGetDky(cv_mem, cv_mem->cv_tstop, 0, yout);
        if (ier != CV_SUCCESS) {
          cvProcessError(cv_mem, CV_ILL_INPUT, "CVODE", "CVode",
                         MSGCV_BAD_TSTOP, cv_mem->cv_tstop, cv_mem->cv_tn);
          return(CV_ILL_INPUT);
        }
        cv_mem->cv_tretlast = *tret = cv_mem->cv_tstop;
        cv_mem->cv_tstopset = SUNFALSE;
        return(CV_TSTOP_RETURN);
      }

      /* If next step would overtake tstop, adjust stepsize */
      if ( (cv_mem->cv_tn + cv_mem->cv_hprime - cv_mem->cv_tstop)*cv_mem->cv_h > ZERO ) {
        cv_mem->cv_hprime = (cv_mem->cv_tstop - cv_mem->cv_tn)*(ONE-FOUR*cv_mem->cv_uround);
        cv_mem->cv_eta = cv_mem->cv_hprime/cv_mem->cv_h;
      }

    }

  } /* end stopping tests block */

   //4. Looping point for internal steps

  if (cv_mem->cv_y == NULL) {
    cvProcessError(cv_mem, CV_BAD_DKY, "CVODE", "CVodeGetDky", MSGCV_NULL_DKY);
    return(CV_BAD_DKY);
  }

  istate = 99;
  kflag = 99;
  nstloc = 0;
  int flag;
  for (int i = 0; i < md->n_cells; i++)
    sd->flagCells[i] = 99;

  int offset_state = 0;
  int offset_ncells = 0;
  int offset_nrows = 0;
#ifdef CAMP_DEBUG_GPU
  cudaSetDevice(sd->startDevice);
  cudaEventRecord(bicg->startcvStep);
#endif
  for (int iDevice = sd->startDevice; iDevice < sd->endDevice; iDevice++) {
    cudaSetDevice(iDevice);
    sd->mGPU = &(sd->mGPUs[iDevice]);
    mGPU = sd->mGPU;

    double *ewt = NV_DATA_S(cv_mem->cv_ewt)+offset_nrows;
    double *acor = NV_DATA_S(cv_mem->cv_acor)+offset_nrows;
    double *tempv = NV_DATA_S(cv_mem->cv_tempv)+offset_nrows;
    double *ftemp = NV_DATA_S(cv_mem->cv_ftemp)+offset_nrows;
    double *cv_last_yn = N_VGetArrayPointer(cv_mem->cv_last_yn)+offset_nrows;
    double *cv_acor_init = N_VGetArrayPointer(cv_mem->cv_acor_init)+offset_nrows;
    double *youtArray = N_VGetArrayPointer(yout)+offset_nrows;
    double *cv_Vabstol = N_VGetArrayPointer(cv_mem->cv_Vabstol)+offset_nrows;

    cudaMemcpyAsync(mGPU->dewt, ewt, mGPU->nrows * sizeof(double), cudaMemcpyHostToDevice, 0);
    cudaMemcpyAsync(mGPU->cv_acor, acor, mGPU->nrows * sizeof(double), cudaMemcpyHostToDevice, 0);
    cudaMemcpyAsync(mGPU->dtempv, tempv, mGPU->nrows * sizeof(double), cudaMemcpyHostToDevice, 0);
    cudaMemcpyAsync(mGPU->dftemp, ftemp, mGPU->nrows * sizeof(double), cudaMemcpyHostToDevice, 0);
    cudaMemcpyAsync(mGPU->cv_last_yn, cv_last_yn, mGPU->nrows * sizeof(double), cudaMemcpyHostToDevice, 0);
    cudaMemcpyAsync(mGPU->cv_acor_init, cv_acor_init, mGPU->nrows * sizeof(double), cudaMemcpyHostToDevice, 0);
    cudaMemcpyAsync(mGPU->yout, youtArray, mGPU->nrows * sizeof(double), cudaMemcpyHostToDevice, 0);
    cudaMemcpyAsync(mGPU->cv_Vabstol, cv_Vabstol, mGPU->nrows * sizeof(double), cudaMemcpyHostToDevice, 0);

    for (int i = 0; i < mGPU->n_cells; i++) {
      cudaMemcpyAsync(mGPU->cv_l + i * L_MAX, cv_mem->cv_l, L_MAX * sizeof(double), cudaMemcpyHostToDevice, 0);
      cudaMemcpyAsync(mGPU->cv_tau + i * (L_MAX + 1), cv_mem->cv_tau, (L_MAX + 1) * sizeof(double),
                      cudaMemcpyHostToDevice, 0);
      cudaMemcpyAsync(mGPU->cv_tq + i * (NUM_TESTS + 1), cv_mem->cv_tq, (NUM_TESTS + 1) * sizeof(double),
                      cudaMemcpyHostToDevice, 0);
    }

    for (int i = 0; i <= cv_mem->cv_qmax; i++) {//cv_qmax+1 (6)?
      double *zn = NV_DATA_S(cv_mem->cv_zn[i])+offset_nrows;
      cudaMemcpyAsync((i * mGPU->nrows + mGPU->dzn), zn, mGPU->nrows * sizeof(double), cudaMemcpyHostToDevice, 0);
    }

    cudaMemcpyAsync(mGPU->flagCells, sd->flagCells+offset_ncells, mGPU->n_cells * sizeof(int), cudaMemcpyHostToDevice, 0);
    HANDLE_ERROR(cudaMemcpyAsync(mGPU->state, md->total_state+offset_state, mGPU->state_size, cudaMemcpyHostToDevice, 0));

    mGPU->mdvCPU.init_time_step = sd->init_time_step;
    mGPU->mdvCPU.cv_mxstep = cv_mem->cv_mxstep;
    mGPU->mdvCPU.cv_taskc = cv_mem->cv_taskc;
    mGPU->mdvCPU.cv_uround = cv_mem->cv_uround;
    mGPU->mdvCPU.cv_nrtfn = cv_mem->cv_nrtfn;
    mGPU->mdvCPU.cv_tretlast = cv_mem->cv_tretlast;
    mGPU->mdvCPU.cv_hmax_inv = cv_mem->cv_hmax_inv;
    mGPU->mdvCPU.cv_lmm = cv_mem->cv_lmm;
    mGPU->mdvCPU.cv_iter = cv_mem->cv_iter;
    mGPU->mdvCPU.cv_itol = cv_mem->cv_itol;
    mGPU->mdvCPU.cv_reltol = cv_mem->cv_reltol;
    mGPU->mdvCPU.cv_nhnil = cv_mem->cv_nhnil;
    mGPU->mdvCPU.cv_etaqm1 = cv_mem->cv_etaqm1;
    mGPU->mdvCPU.cv_etaq = cv_mem->cv_etaq;
    mGPU->mdvCPU.cv_etaqp1 = cv_mem->cv_etaqp1;
    mGPU->mdvCPU.cv_lrw1 = cv_mem->cv_lrw1;
    mGPU->mdvCPU.cv_liw1 = cv_mem->cv_liw1;
    mGPU->mdvCPU.cv_lrw = (int) cv_mem->cv_lrw;
    mGPU->mdvCPU.cv_liw = (int) cv_mem->cv_liw;
    mGPU->mdvCPU.cv_saved_tq5 = cv_mem->cv_saved_tq5;
    mGPU->mdvCPU.cv_tolsf = cv_mem->cv_tolsf;
    mGPU->mdvCPU.cv_qmax_alloc = cv_mem->cv_qmax_alloc;
    mGPU->mdvCPU.cv_indx_acor = cv_mem->cv_indx_acor;
    mGPU->mdvCPU.cv_qu = cv_mem->cv_qu;
    mGPU->mdvCPU.cv_h0u = cv_mem->cv_h0u;
    mGPU->mdvCPU.cv_hu = cv_mem->cv_hu;
    mGPU->mdvCPU.cv_jcur = cv_mem->cv_jcur;
    mGPU->mdvCPU.cv_mnewt = cv_mem->cv_mnewt;
    mGPU->mdvCPU.cv_maxcor = cv_mem->cv_maxcor;
    mGPU->mdvCPU.cv_nstlp = (int) cv_mem->cv_nstlp;
    mGPU->mdvCPU.cv_qmax = cv_mem->cv_qmax;
    mGPU->mdvCPU.cv_L = cv_mem->cv_L;
    mGPU->mdvCPU.cv_maxnef = cv_mem->cv_maxnef;
    mGPU->mdvCPU.cv_netf = (int) cv_mem->cv_netf;
    mGPU->mdvCPU.cv_acnrm = cv_mem->cv_acnrm;
    mGPU->mdvCPU.cv_tstop = cv_mem->cv_tstop;
    mGPU->mdvCPU.cv_tstopset = cv_mem->cv_tstopset;
    mGPU->mdvCPU.cv_nlscoef = cv_mem->cv_nlscoef;
    mGPU->mdvCPU.cv_qwait = cv_mem->cv_qwait;
    mGPU->mdvCPU.cv_crate = cv_mem->cv_crate;
    mGPU->mdvCPU.cv_gamrat = cv_mem->cv_gamrat;
    mGPU->mdvCPU.cv_gammap = cv_mem->cv_gammap;
    mGPU->mdvCPU.cv_nst = cv_mem->cv_nst;
    mGPU->mdvCPU.cv_gamma = cv_mem->cv_gamma;
    mGPU->mdvCPU.cv_rl1 = cv_mem->cv_rl1;
    mGPU->mdvCPU.cv_eta = cv_mem->cv_eta;
    mGPU->mdvCPU.cv_q = cv_mem->cv_q;
    mGPU->mdvCPU.cv_qprime = cv_mem->cv_qprime;
    mGPU->mdvCPU.cv_h = cv_mem->cv_h;
    mGPU->mdvCPU.cv_next_h = cv_mem->cv_next_h;//needed?
    mGPU->mdvCPU.cv_hscale = cv_mem->cv_hscale;
    mGPU->mdvCPU.cv_nscon = cv_mem->cv_nscon;
    mGPU->mdvCPU.cv_hprime = cv_mem->cv_hprime;
    mGPU->mdvCPU.cv_hmin = cv_mem->cv_hmin;
    mGPU->mdvCPU.cv_tn = cv_mem->cv_tn;
    mGPU->mdvCPU.cv_etamax = cv_mem->cv_etamax;
    mGPU->mdvCPU.cv_maxncf = cv_mem->cv_maxncf;

    mGPU->mdvCPU.tout = tout;
    mGPU->mdvCPU.tret = *tret;
    mGPU->mdvCPU.istate = istate;
    mGPU->mdvCPU.kflag = kflag;
    mGPU->mdvCPU.kflag2 = 99;

#ifdef DEV_DMDV_ARRAY
    for(int i=0; i<mGPU->n_cells;i++){
      cudaMemcpyAsync(&mGPU->mdv[i], &mGPU->mdvCPU, sizeof(ModelDataVariable), cudaMemcpyHostToDevice, 0);
    }
#else
    cudaMemcpyAsync(mGPU->mdv, &mGPU->mdvCPU, sizeof(ModelDataVariable), cudaMemcpyHostToDevice, 0);
#endif

    solveCVODEGPU(sd, cv_mem);

    cudaMemcpyAsync(&mGPU->mdvCPU, mGPU->mdvo, sizeof(ModelDataVariable), cudaMemcpyDeviceToHost, 0);

    cudaMemcpyAsync(ewt, mGPU->dewt, mGPU->nrows * sizeof(double), cudaMemcpyDeviceToHost, 0);
    cudaMemcpyAsync(acor, mGPU->cv_acor, mGPU->nrows * sizeof(double), cudaMemcpyDeviceToHost, 0);
    cudaMemcpyAsync(tempv, mGPU->dtempv, mGPU->nrows * sizeof(double), cudaMemcpyDeviceToHost, 0);
    cudaMemcpyAsync(ftemp, mGPU->dftemp, mGPU->nrows * sizeof(double), cudaMemcpyDeviceToHost, 0);
    cudaMemcpyAsync(cv_last_yn, mGPU->cv_last_yn, mGPU->nrows * sizeof(double), cudaMemcpyDeviceToHost, 0);
    cudaMemcpyAsync(cv_acor_init, mGPU->cv_acor_init, mGPU->nrows * sizeof(double), cudaMemcpyDeviceToHost, 0);
    cudaMemcpyAsync(youtArray, mGPU->yout, mGPU->nrows * sizeof(double), cudaMemcpyDeviceToHost, 0);
    cudaMemcpyAsync(cv_Vabstol, mGPU->cv_Vabstol, mGPU->nrows * sizeof(double), cudaMemcpyDeviceToHost, 0);

    for (int i = 0; i <= cv_mem->cv_qmax; i++) {//cv_qmax+1 (6)?
      double *zn = NV_DATA_S(cv_mem->cv_zn[i])+offset_nrows;
      cudaMemcpyAsync(zn, (i * mGPU->nrows + mGPU->dzn), mGPU->nrows * sizeof(double), cudaMemcpyDeviceToHost, 0);
    }

    HANDLE_ERROR(cudaMemcpyAsync(sd->flagCells+offset_ncells, mGPU->flagCells, mGPU->n_cells * sizeof(int), cudaMemcpyDeviceToHost, 0));

    offset_state += mGPU->state_size_cell * mGPU->n_cells;
    offset_ncells += mGPU->n_cells;
    offset_nrows += mGPU->nrows;
  }

  for (int iDevice = sd->startDevice+1; iDevice < sd->endDevice; iDevice++) {
    cudaSetDevice(iDevice);
    cudaDeviceSynchronize();
  }

  cudaSetDevice(sd->startDevice);
  sd->mGPU = &(sd->mGPUs[sd->startDevice]);
  mGPU = sd->mGPU;

#ifdef CAMP_DEBUG_GPU
    cudaEventRecord(bicg->stopcvStep);
    cudaEventSynchronize(bicg->stopcvStep);
    float mscvStep = 0.0;
    cudaEventElapsedTime(&mscvStep, bicg->startcvStep, bicg->stopcvStep);
    bicg->timecvStep+= mscvStep/1000;

#ifdef CAMP_PROFILE_DEVICE_FUNCTIONS
    bicg->timeBiConjGrad=bicg->timecvStep*mGPU->mdvCPU.dtBCG/mGPU->mdvCPU.dtcudaDeviceCVode;
    bicg->counterBiConjGrad+= mGPU->mdvCPU.counterBCG;
#else
    bicg->timeBiConjGrad=0.;
    bicg->counterBiConjGrad+=0;
#endif

#endif

  flag = CV_SUCCESS;
    for (int i = 0; i < mGPU->n_cells; i++) {
      if (sd->flagCells[i] != flag) {
        flag = sd->flagCells[i];
        break;
      }
    }
    istate=flag;

    //printf("cudaCVode flag %d kflag %d\n",flag, mGPU->mdvCPU.flag);

    // In NORMAL mode, check if tout reached
    //if ( (cv_mem->cv_tn-tout)*cv_mem->cv_h >= ZERO ) {
    if ( istate==CV_SUCCESS ) {

      //istate = CV_SUCCESS;
      //printf("istate==CV_SUCCESS\n";

      cv_mem->cv_tretlast = mGPU->mdvCPU.cv_tretlast;
      cv_mem->cv_next_q = mGPU->mdvCPU.cv_qprime;
      cv_mem->cv_next_h = mGPU->mdvCPU.cv_hprime;

    }else{

      if (kflag != CV_SUCCESS) {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        printf("cudaCVode2 kflag %d rank %d\n",kflag,rank);
        istate = cvHandleFailure_gpu(cv_mem, kflag);
        //cv_mem->cv_tretlast = *tret = cv_mem->cv_tn;
        //N_VScale(ONE, cv_mem->cv_zn[0], yout);

        cv_mem->cv_next_h = mGPU->mdvCPU.cv_next_h;

      }

      // Reset and check ewt
      if (istate==CV_ILL_INPUT) {
        if (cv_mem->cv_itol == CV_WF)
          cvProcessError(cv_mem, CV_ILL_INPUT, "CVODE", "CVode",
                         MSGCV_EWT_NOW_FAIL, cv_mem->cv_tn);
        else
          cvProcessError(cv_mem, CV_ILL_INPUT, "CVODE", "CVode",
                         MSGCV_EWT_NOW_BAD, cv_mem->cv_tn);
        //Remove break after removing for(;;) in cpu
      }

      // Check for too many steps
      if ( (cv_mem->cv_mxstep>0) && (nstloc >= cv_mem->cv_mxstep) ) {
        cvProcessError(cv_mem, CV_TOO_MUCH_WORK, "CVODE", "CVode",
                       MSGCV_MAX_STEPS, cv_mem->cv_tn);
        istate = CV_TOO_MUCH_WORK;
        //cv_mem->cv_tretlast = *tret = cv_mem->cv_tn;
        //N_VScale(ONE, cv_mem->cv_zn[0], yout);
      }

      // Check for too much accuracy requested
      //nrm = N_VWrmsNorm(cv_mem->cv_zn[0], cv_mem->cv_ewt);
      //cv_mem->cv_tolsf = cv_mem->cv_uround * nrm;
      if (cv_mem->cv_tolsf > ONE) {
        cvProcessError(cv_mem, CV_TOO_MUCH_ACC, "CVODE", "CVode",
                       MSGCV_TOO_MUCH_ACC, cv_mem->cv_tn);
        istate = CV_TOO_MUCH_ACC;
        //cv_mem->cv_tretlast = *tret = cv_mem->cv_tn;
        //N_VScale(ONE, cv_mem->cv_zn[0], yout);
        //cv_mem->cv_tolsf *= TWO;
      }

      //Check for h below roundoff level in tn
      if (cv_mem->cv_tn + cv_mem->cv_h == cv_mem->cv_tn) {
          //cv_mem->cv_nhnil++;
          if (cv_mem->cv_nhnil <= cv_mem->cv_mxhnil)
            cvProcessError(cv_mem, CV_WARNING, "CVODE", "CVode",
                           MSGCV_HNIL, cv_mem->cv_tn, cv_mem->cv_h);
          if (cv_mem->cv_nhnil == cv_mem->cv_mxhnil)
            cvProcessError(cv_mem, CV_WARNING, "CVODE", "CVode", MSGCV_HNIL_DONE);
          }

    }

  return(istate);
}

void solver_get_statistics_gpu(SolverData *sd){

  cudaSetDevice(sd->startDevice);
  sd->mGPU = &(sd->mGPUs[sd->startDevice]);
  ModelDataGPU *mGPU = sd->mGPU;

  cudaMemcpy(&mGPU->mdvCPU,mGPU->mdvo,sizeof(ModelDataVariable),cudaMemcpyDeviceToHost);
  ModelDataGPU *mGPU_max = sd->mGPU;

  //printf("solver_get_statistics_gpu\n");

  for (int iDevice = sd->startDevice+1; iDevice < sd->endDevice; iDevice++) {
    cudaSetDevice(iDevice);
    sd->mGPU = &(sd->mGPUs[iDevice]);
    mGPU = sd->mGPU;

    cudaMemcpy(&mGPU->mdvCPU, mGPU->mdvo, sizeof(ModelDataVariable), cudaMemcpyDeviceToHost);
#ifdef CAMP_PROFILE_DEVICE_FUNCTIONS
    if (mGPU->mdvCPU.dtcudaDeviceCVode>mGPU_max->mdvCPU.dtcudaDeviceCVode){
      cudaSetDevice(iDevice);
      mGPU_max = mGPU;
    }
#endif
  }

  sd->mGPU = mGPU_max;
  mGPU = mGPU_max;

#ifdef CAMP_DEBUG_GPU

#ifdef CAMP_PROFILE_DEVICE_FUNCTIONS

  cudaMemcpy(&sd->tguessNewton,mGPU->tguessNewton,sizeof(double),cudaMemcpyDeviceToHost);
  cudaMemcpy(&sd->timeNewtonIteration,mGPU->dtNewtonIteration,sizeof(double),cudaMemcpyDeviceToHost);
  cudaMemcpy(&sd->timeJac,mGPU->dtJac,sizeof(double),cudaMemcpyDeviceToHost);
  cudaMemcpy(&sd->timelinsolsetup,mGPU->dtlinsolsetup,sizeof(double),cudaMemcpyDeviceToHost);
  cudaMemcpy(&sd->timecalc_Jac,mGPU->dtcalc_Jac,sizeof(double),cudaMemcpyDeviceToHost);
  cudaMemcpy(&sd->timeRXNJac,mGPU->dtRXNJac,sizeof(double),cudaMemcpyDeviceToHost);
  cudaMemcpy(&sd->timef,mGPU->dtf,sizeof(double),cudaMemcpyDeviceToHost);
  cudaMemcpy(&sd->timeguess_helper,mGPU->dtguess_helper,sizeof(double),cudaMemcpyDeviceToHost);

#endif

#endif

}

void solver_reset_statistics_gpu(SolverData *sd){

  ModelDataGPU *mGPU = sd->mGPU;

  //printf("solver_reset_statistics_gpu\n");

  for (int iDevice = sd->startDevice; iDevice < sd->endDevice; iDevice++) {
    cudaSetDevice(iDevice);
    sd->mGPU = &(sd->mGPUs[iDevice]);
    mGPU = sd->mGPU;

#ifdef CAMP_DEBUG_GPU
#ifdef CAMP_PROFILE_DEVICE_FUNCTIONS
    cudaMemset(mGPU->tguessNewton, 0., sizeof(double));
    cudaMemset(mGPU->dtNewtonIteration, 0., sizeof(double));
    cudaMemset(mGPU->dtJac, 0., sizeof(double));
    cudaMemset(mGPU->dtlinsolsetup, 0., sizeof(double));
    cudaMemset(mGPU->dtcalc_Jac, 0., sizeof(double));
    cudaMemset(mGPU->dtRXNJac, 0., sizeof(double));
    cudaMemset(mGPU->dtf, 0., sizeof(double));
    cudaMemset(mGPU->dtguess_helper, 0., sizeof(double));
#endif
#ifdef DEV_DMDV_ARRAY
    for(int i=0; i<mGPU->n_cells;i++){
      cudaMemcpy(&mGPU->mdv[i], &mGPU->mdvCPU, sizeof(ModelDataVariable), cudaMemcpyHostToDevice);
    }
#else
    cudaMemcpy(mGPU->mdv, &mGPU->mdvCPU, sizeof(ModelDataVariable), cudaMemcpyHostToDevice);
#endif
#endif
  }

}

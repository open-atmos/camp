/* Copyright (C) 2021 Barcelona Supercomputing Center and University of
 * Illinois at Urbana-Champaign
 * SPDX-License-Identifier: MIT
 */

#include "itsolver_gpu.h"
#include "cvode_cuda.h"

extern "C" {
#include "cvode_gpu.h"
}

#ifdef CAMP_USE_MPI
#include <mpi.h>
#endif

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

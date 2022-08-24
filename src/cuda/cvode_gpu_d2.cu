/* Copyright (C) 2021 Barcelona Supercomputing Center and University of
 * Illinois at Urbana-Champaign
 * SPDX-License-Identifier: MIT
 */

#include "itsolver_gpu.h"
#include "cvode_cuda_d2.h"
extern "C" {
#include "cvode_gpu_d2.h"
}

__global__
void init_jac_partials_d2(double* production_partials, double* loss_partials) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  production_partials[tid]=0.0;
  loss_partials[tid]=0.0;
}

int jacobian_initialize_cuda_d2(SolverData *sd) {
  ModelDataGPU_d2 *mGPU = sd->mGPU_d2;
  Jacobian *jac = &sd->jac;

#ifdef DEBUG_jacobian_initialize_gpu
  printf("jacobian_initialize_gpu start \n");
#endif

  int offset_nnz = 0;
  for (int iDevice = sd->startDevice; iDevice < sd->endDevice; iDevice++) {
    cudaSetDevice(iDevice);
    sd->mGPU_d2 = &(sd->mGPUs_d2[iDevice]);
    mGPU = sd->mGPU_d2;

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
    init_jac_partials_d2 <<<blocks,threads_block>>>(jacgpu->production_partials,jacgpu->loss_partials);

    offset_nnz += num_elem;
  }

#ifdef DEBUG_jacobian_initialize_gpu
  printf("jacobian_initialize_gpu end \n");
#endif

  return 1;
}

__global__
void init_J_tmp2_cuda_d2(double* J_tmp2) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  J_tmp2[tid]=0.0;
}

void init_jac_cuda_d2(SolverData *sd){

  ModelData *md = &(sd->model_data);
  ModelDataGPU_d2 *mGPU;

#ifdef DEBUG_init_jac_cuda_d2

  printf("init_jac_cuda_d2 start \n");

#endif

  int offset_nnz_J_solver = 0;
  int offset_nrows = 0;
  for (int iDevice = sd->startDevice; iDevice < sd->endDevice; iDevice++) {
    cudaSetDevice(iDevice);
    sd->mGPU_d2 = &(sd->mGPUs_d2[iDevice]);
    mGPU = sd->mGPU_d2;

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

#ifdef DEBUG_init_jac_cuda_d2
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
    init_J_tmp2_cuda_d2 <<<blocks,threads_block>>>(mGPU->J_tmp2);
    HANDLE_ERROR(cudaMemcpy(mGPU->jac_map, md->jac_map, sizeof(JacMap) * md->n_mapped_values, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(mGPU->n_mapped_values, &md->n_mapped_values, 1 * sizeof(int), cudaMemcpyHostToDevice));

    offset_nnz_J_solver += mGPU->nnz_J_solver;
    offset_nrows += md->n_per_cell_dep_var* mGPU->n_cells;
  }

  jacobian_initialize_cuda_d2(sd);

#ifdef DEBUG_init_jac_cuda_d2

  printf("init_jac_cuda_d2 end \n");

#endif

}

void set_int_double_cuda_d2(
    int n_rxn, int rxn_env_data_idx_size,
    int *rxn_int_data, double *rxn_float_data,
    int *rxn_int_indices, int *rxn_float_indices,
    int *rxn_env_idx,
    SolverData *sd
) {

  ModelData *md = &(sd->model_data);
  ModelDataGPU_d2 *mGPU = sd->mGPU_d2;

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

void solver_init_int_double_cuda_d2(SolverData *sd) {

  ModelData *md = &(sd->model_data);
  ModelDataGPU_d2 *mGPU;

#ifdef DEBUG_solver_init_int_double_gpu
  printf("solver_init_int_double_gpu start \n");
#endif

  for (int iDevice = sd->startDevice; iDevice < sd->endDevice; iDevice++) {
    cudaSetDevice(iDevice);
    sd->mGPU_d2 = &(sd->mGPUs_d2[iDevice]);
    mGPU = sd->mGPU_d2;

#ifdef REVERSE_INT_FLOAT_MATRIX

    set_reverse_int_double_rxn(
            md->n_rxn, mGPU->rxn_env_data_idx_size,
            md->rxn_int_data, md->rxn_float_data,
            md->rxn_int_indices, md->rxn_float_indices, md->rxn_env_idx,
            sd
    );

#else

    set_int_double_cuda_d2(
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

void solver_new_gpu_cu_d2(SolverData *sd) {
  ModelData *md = &(sd->model_data);
  ModelDataGPU_d2 *mGPU;

  int n_dep_var=md->n_per_cell_dep_var;
  int n_state_var=md->n_per_cell_state_var;
  int n_rxn=md->n_rxn;
  int n_rxn_int_param=md->n_rxn_int_param;
  int n_rxn_float_param=md->n_rxn_float_param;
  int n_rxn_env_param=md->n_rxn_env_data;
  int n_cells_total=md->n_cells;

  sd->mGPUs_d2 = (ModelDataGPU_d2 *)malloc(sd->nDevices * sizeof(ModelDataGPU_d2));
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
  sd->mGPU_d2 = &(sd->mGPUs_d2[iDevice]);
  mGPU = sd->mGPU_d2;

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

void constructor_cvode_cuda_d2(CVodeMem cv_mem, SolverData *sd)
{
  ModelData *md = &(sd->model_data);
  CVDlsMem cvdls_mem = (CVDlsMem) cv_mem->cv_lmem;
  SUNMatrix J = cvdls_mem->A;

  sd->flagCells = (int *) malloc((md->n_cells) * sizeof(int));
  cudaSetDevice(sd->startDevice);
  sd->mGPU_d2 = &(sd->mGPUs_d2[sd->startDevice]);
  ModelDataGPU_d2 *mGPU  = sd->mGPU_d2;

  solver_new_gpu_cu_d2(sd);

  cudaSetDevice(sd->startDevice);
  sd->mGPU_d2 = &(sd->mGPUs_d2[sd->startDevice]);
  mGPU  = sd->mGPU_d2;

  init_jac_cuda_d2(sd);
  //solver_init_int_double_cuda_d2(sd);

ModelDataCPU_d2 *mCPU  = &sd->mCPU;
#ifdef CAMP_DEBUG_GPU
  mCPU->countercvStep=0;
  mCPU->timecvStep=CAMP_TINY;
  cudaEventCreate(&mCPU->startcvStep);
  cudaEventCreate(&mCPU->stopcvStep);
#endif

  int offset_nnz = 0;
  int offset_nrows = 0;
  for (int iDevice = sd->startDevice; iDevice < sd->endDevice; iDevice++) {
    cudaSetDevice(iDevice);
    sd->mGPU_d2 = &(sd->mGPUs_d2[iDevice]);
    mGPU = sd->mGPU_d2;

    mGPU->nnz = SM_NNZ_S(J)/md->n_cells*mGPU->n_cells;
    mGPU->nrows = SM_NP_S(J)/md->n_cells*mGPU->n_cells;

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, iDevice);
    mGPU->threads = prop.maxThreadsPerBlock; //1024
    mGPU->blocks = (mGPU->nrows + mGPU->threads - 1) / mGPU->threads;

    mGPU->maxIt=1000;
    mGPU->tolmax=1.0e-30;
    int nrows = mGPU->nrows;
    int len_cell = mGPU->nrows/mGPU->n_cells;
    if(len_cell>mGPU->threads){
    printf("ERROR: Size cell greater than available threads per block");
    exit(0);
    }
printf("constructor_cvode_cuda_d2  \n");

    //Auxiliary vectors
    double ** dr0 = &mGPU->dr0;
    double ** dr0h = &mGPU->dr0h;
    double ** dn0 = &mGPU->dn0;
    double ** dp0 = &mGPU->dp0;
    double ** dt = &mGPU->dt;
    double ** ds = &mGPU->ds;
    double ** dy = &mGPU->dy;
    double ** ddiag = &mGPU->ddiag;
double ** dz = &mGPU->dz;
double ** dAx2 = &mGPU->dAx2;
    cudaMalloc(dr0,nrows*sizeof(double));
    cudaMalloc(dr0h,nrows*sizeof(double));
    cudaMalloc(dn0,nrows*sizeof(double));
    cudaMalloc(dp0,nrows*sizeof(double));
    cudaMalloc(dt,nrows*sizeof(double));
    cudaMalloc(ds,nrows*sizeof(double));
    cudaMalloc(dy,nrows*sizeof(double));
cudaMalloc(dz,nrows*sizeof(double));
cudaMalloc(dAx2,nrows*sizeof(double));
    HANDLE_ERROR(cudaMalloc(ddiag,nrows*sizeof(double)));
    int blocks = mGPU->blocks;
    mGPU->aux=(double*)malloc(sizeof(double)*blocks);

#ifdef DEV_swapCSC_CSR_cuda_d2
    mGPU->A = ((double *) SM_DATA_S(J))+offset_nnz;
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
#else

    mGPU->A = ((double *) SM_DATA_S(J))+offset_nnz;
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

#endif

    mGPU->dA = mGPU->J;//set itsolver gpu pointer to jac pointer initialized at camp
    mGPU->dftemp = mGPU->deriv_data; //deriv is gpu pointer
    double *ewt = N_VGetArrayPointer(cv_mem->cv_ewt)+offset_nrows;

    //todo why this? whatever it should be reset i guess
    cudaMemcpy(mGPU->dftemp, ewt, mGPU->nrows * sizeof(double), cudaMemcpyHostToDevice);

    cudaMalloc((void **) &mGPU->dtempv2, mGPU->nrows * sizeof(double));
    cudaMalloc((void **) &mGPU->dcv_y, mGPU->nrows * sizeof(double));
    cudaMalloc((void **) &mGPU->cv_acor, mGPU->nrows * sizeof(double));
    cudaMemcpy(mGPU->cv_acor, ewt, mGPU->nrows * sizeof(double), cudaMemcpyHostToDevice);
    cudaMalloc((void **) &mGPU->dzn, mGPU->nrows * (cv_mem->cv_qmax + 1) * sizeof(double));//L_MAX 6
    cudaMalloc((void **) &mGPU->dewt, mGPU->nrows * sizeof(double));
    cudaMemcpy(mGPU->dewt, ewt, mGPU->nrows * sizeof(double), cudaMemcpyHostToDevice);
    cudaMalloc((void **) &mGPU->dtempv, mGPU->nrows * sizeof(double));
    cudaMalloc((void **) &mGPU->dx, mGPU->nrows * sizeof(double));
    double *tempv = N_VGetArrayPointer(cv_mem->cv_tempv)+offset_nrows;
    cudaMemcpy(mGPU->dx, tempv, mGPU->nrows * sizeof(double), cudaMemcpyHostToDevice);
    cudaMalloc((void **) &mGPU->cv_l, L_MAX * mGPU->n_cells * sizeof(double));
    cudaMalloc((void **) &mGPU->cv_tau, (L_MAX + 1) * mGPU->n_cells * sizeof(double));
    cudaMalloc((void **) &mGPU->cv_tq, (NUM_TESTS + 1) * mGPU->n_cells * sizeof(double));

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

int cvHandleFailure_cuda_d2(CVodeMem cv_mem, int flag)
{
  switch (flag) {
    case CV_ERR_FAILURE:
      cvProcessError(cv_mem, CV_ERR_FAILURE, "CVODE", "CVode", MSGCV_ERR_FAILS,
                     cv_mem->cv_tn, cv_mem->cv_h);
      break;
    case CV_CONV_FAILURE:
      cvProcessError(cv_mem, CV_CONV_FAILURE, "CVODE", "CVode", MSGCV_CONV_FAILS,
                     cv_mem->cv_tn, cv_mem->cv_h);
      break;
    case CV_LSETUP_FAIL:
      cvProcessError(cv_mem, CV_LSETUP_FAIL, "CVODE", "CVode", MSGCV_SETUP_FAILED,
                     cv_mem->cv_tn);
      break;
    case CV_LSOLVE_FAIL:
      cvProcessError(cv_mem, CV_LSOLVE_FAIL, "CVODE", "CVode", MSGCV_SOLVE_FAILED,
                     cv_mem->cv_tn);
      break;
    case CV_RHSFUNC_FAIL:
      cvProcessError(cv_mem, CV_RHSFUNC_FAIL, "CVODE", "CVode", MSGCV_RHSFUNC_FAILED,
                     cv_mem->cv_tn);
      break;
    case CV_UNREC_RHSFUNC_ERR:
      cvProcessError(cv_mem, CV_UNREC_RHSFUNC_ERR, "CVODE", "CVode", MSGCV_RHSFUNC_UNREC,
                     cv_mem->cv_tn);
      break;
    case CV_REPTD_RHSFUNC_ERR:
      cvProcessError(cv_mem, CV_REPTD_RHSFUNC_ERR, "CVODE", "CVode", MSGCV_RHSFUNC_REPTD,
                     cv_mem->cv_tn);
      break;
    case CV_RTFUNC_FAIL:
      cvProcessError(cv_mem, CV_RTFUNC_FAIL, "CVODE", "CVode", MSGCV_RTFUNC_FAILED,
                     cv_mem->cv_tn);
      break;
    case CV_TOO_CLOSE:
      cvProcessError(cv_mem, CV_TOO_CLOSE, "CVODE", "CVode", MSGCV_TOO_CLOSE);
      break;
    default:
      return(CV_SUCCESS);
  }
  return(flag);
}

void cvRestore_cuda_d2(CVodeMem cv_mem, realtype saved_t)
{
  int j, k;
  cv_mem->cv_tn = saved_t;
  for (k = 1; k <= cv_mem->cv_q; k++)
    for (j = cv_mem->cv_q; j >= k; j--)
      N_VLinearSum(ONE, cv_mem->cv_zn[j-1], -ONE,
                   cv_mem->cv_zn[j], cv_mem->cv_zn[j-1]);
  N_VScale(ONE, cv_mem->cv_last_yn, cv_mem->cv_zn[0]);
}

void cvSetEta_cuda_d2(CVodeMem cv_mem)
{
  if (cv_mem->cv_eta < THRESH) {
    cv_mem->cv_eta = ONE;
    cv_mem->cv_hprime = cv_mem->cv_h;
  } else {
    cv_mem->cv_eta = SUNMIN(cv_mem->cv_eta, cv_mem->cv_etamax);
    cv_mem->cv_eta /= SUNMAX(ONE, SUNRabs(cv_mem->cv_h)*cv_mem->cv_hmax_inv*cv_mem->cv_eta);
    cv_mem->cv_hprime = cv_mem->cv_h * cv_mem->cv_eta;
    if (cv_mem->cv_qprime < cv_mem->cv_q) cv_mem->cv_nscon = 0;
  }
}

int cvYddNorm_cuda_d2(CVodeMem cv_mem, realtype hg, realtype *yddnrm)
{
  int retval;

  N_VLinearSum(hg, cv_mem->cv_zn[1], ONE, cv_mem->cv_zn[0], cv_mem->cv_y);
  //retval = cv_mem->cv_f(cv_mem->cv_tn+hg, cv_mem->cv_y,
  //                      cv_mem->cv_tempv, cv_mem->cv_user_data);
  retval = f(cv_mem->cv_tn+hg, cv_mem->cv_y, cv_mem->cv_tempv, cv_mem->cv_user_data);
  cv_mem->cv_nfe++;
  if (retval < 0) return(CV_RHSFUNC_FAIL);
  if (retval > 0) return(RHSFUNC_RECVR);

  N_VLinearSum(ONE, cv_mem->cv_tempv, -ONE, cv_mem->cv_zn[1], cv_mem->cv_tempv);
  N_VScale(ONE/hg, cv_mem->cv_tempv, cv_mem->cv_tempv);

  *yddnrm = N_VWrmsNorm(cv_mem->cv_tempv, cv_mem->cv_ewt);

  return(CV_SUCCESS);
}

int cvInitialSetup_cuda_d2(CVodeMem cv_mem)
{
  int ier;

  /* Did the user specify tolerances? */
  if (cv_mem->cv_itol == CV_NN) {
    cvProcessError(cv_mem, CV_ILL_INPUT, "CVODE", "cvInitialSetup", MSGCV_NO_TOLS);
    return(CV_ILL_INPUT);
  }

  /* Set data for efun */
  if (cv_mem->cv_user_efun) cv_mem->cv_e_data = cv_mem->cv_user_data;
  else                      cv_mem->cv_e_data = cv_mem;

  /* Load initial error weights */
  ier = cv_mem->cv_efun(cv_mem->cv_zn[0], cv_mem->cv_ewt, cv_mem->cv_e_data);
  if (ier != 0) {
    if (cv_mem->cv_itol == CV_WF)
      cvProcessError(cv_mem, CV_ILL_INPUT, "CVODE", "cvInitialSetup", MSGCV_EWT_FAIL);
    else
      cvProcessError(cv_mem, CV_ILL_INPUT, "CVODE", "cvInitialSetup", MSGCV_BAD_EWT);
    return(CV_ILL_INPUT);
  }

  /* Check if lsolve function exists (if needed) and call linit function (if it exists) */
  if (cv_mem->cv_iter == CV_NEWTON) {
    if (cv_mem->cv_lsolve == NULL) {
      cvProcessError(cv_mem, CV_ILL_INPUT, "CVODE", "cvInitialSetup", MSGCV_LSOLVE_NULL);
      return(CV_ILL_INPUT);
    }
    if (cv_mem->cv_linit != NULL) {
      ier = cv_mem->cv_linit(cv_mem);
      if (ier != 0) {
        cvProcessError(cv_mem, CV_LINIT_FAIL, "CVODE", "cvInitialSetup", MSGCV_LINIT_FAIL);
        return(CV_LINIT_FAIL);
      }
    }
  }

  return(CV_SUCCESS);
}

realtype cvUpperBoundH0_cuda_d2(CVodeMem cv_mem, realtype tdist)
{
  realtype hub_inv, hub;
  N_Vector temp1, temp2;

  /*
   * Bound based on |y0|/|y0'| -- allow at most an increase of
   * HUB_FACTOR in y0 (based on a forward Euler step). The weight
   * factor is used as a safeguard against zero components in y0.
   */

  temp1 = cv_mem->cv_tempv;
  temp2 = cv_mem->cv_acor;

  N_VAbs(cv_mem->cv_zn[0], temp2);
  cv_mem->cv_efun(cv_mem->cv_zn[0], temp1, cv_mem->cv_e_data);
  N_VInv(temp1, temp1);
  N_VLinearSum(HUB_FACTOR, temp2, ONE, temp1, temp1);

  N_VAbs(cv_mem->cv_zn[1], temp2);

  N_VDiv(temp2, temp1, temp1);
  hub_inv = N_VMaxNorm(temp1);

  /*
   * bound based on tdist -- allow at most a step of magnitude
   * HUB_FACTOR * tdist
   */

  hub = HUB_FACTOR*tdist;

  /* Use the smaller of the two */

  if (hub*hub_inv > ONE) hub = ONE/hub_inv;

  return(hub);
}

int cvHin_cuda_d2(CVodeMem cv_mem, realtype tout)
{
  int retval, sign, count1, count2;
  realtype tdiff, tdist, tround, hlb, hub;
  realtype hg, hgs, hs, hnew, hrat, h0, yddnrm;
  booleantype hgOK, hnewOK;

  /* If tout is too close to tn, give up */

  if ((tdiff = tout-cv_mem->cv_tn) == ZERO) return(CV_TOO_CLOSE);

  sign = (tdiff > ZERO) ? 1 : -1;
  tdist = SUNRabs(tdiff);
  tround = cv_mem->cv_uround * SUNMAX(SUNRabs(cv_mem->cv_tn), SUNRabs(tout));

  if (tdist < TWO*tround) return(CV_TOO_CLOSE);

  /*
     Set lower and upper bounds on h0, and take geometric mean
     as first trial value.
     Exit with this value if the bounds cross each other.
  */

  hlb = HLB_FACTOR * tround;
  hub = cvUpperBoundH0_cuda_d2(cv_mem, tdist);

  hg  = SUNRsqrt(hlb*hub);

  if (hub < hlb) {
    if (sign == -1) cv_mem->cv_h = -hg;
    else            cv_mem->cv_h =  hg;
    return(CV_SUCCESS);
  }

  /* Outer loop */

  hnewOK = SUNFALSE;
  hs = hg;         /* safeguard against 'uninitialized variable' warning */

  for(count1 = 1; count1 <= MAX_ITERS; count1++) {

    /* Attempts to estimate ydd */

    hgOK = SUNFALSE;

    for (count2 = 1; count2 <= MAX_ITERS; count2++) {
      hgs = hg*sign;
      retval = cvYddNorm_cuda_d2(cv_mem, hgs, &yddnrm);
      /* If f() failed unrecoverably, give up */
      if (retval < 0) return(CV_RHSFUNC_FAIL);
      /* If successful, we can use ydd */
      if (retval == CV_SUCCESS) {hgOK = SUNTRUE; break;}
      /* f() failed recoverably; cut step size and test it again */
      hg *= POINT2;
    }

    /* If f() failed recoverably MAX_ITERS times */

    if (!hgOK) {
      /* Exit if this is the first or second pass. No recovery possible */
      if (count1 <= 2) return(CV_REPTD_RHSFUNC_ERR);
      /* We have a fall-back option. The value hs is a previous hnew which
         passed through f(). Use it and break */
      hnew = hs;
      break;
    }

    /* The proposed step size is feasible. Save it. */
    hs = hg;

    /* If the stopping criteria was met, or if this is the last pass, stop */
    if ( (hnewOK) || (count1 == MAX_ITERS))  {hnew = hg; break;}

    /* Propose new step size */
    hnew = (yddnrm*hub*hub > TWO) ? SUNRsqrt(TWO/yddnrm) : SUNRsqrt(hg*hub);
    hrat = hnew/hg;

    /* Accept hnew if it does not differ from hg by more than a factor of 2 */
    if ((hrat > HALF) && (hrat < TWO)) {
      hnewOK = SUNTRUE;
    }

    /* After one pass, if ydd seems to be bad, use fall-back value. */
    if ((count1 > 1) && (hrat > TWO)) {
      hnew = hg;
      hnewOK = SUNTRUE;
    }

    /* Send this value back through f() */
    hg = hnew;

  }

  /* Apply bounds, bias factor, and attach sign */

  h0 = H_BIAS*hnew;
  if (h0 < hlb) h0 = hlb;
  if (h0 > hub) h0 = hub;
  if (sign == -1) h0 = -h0;
  cv_mem->cv_h = h0;

  return(CV_SUCCESS);
}

int cvRootfind_cuda_d2(CVodeMem cv_mem)
{
  realtype alph, tmid, gfrac, maxfrac, fracint, fracsub;
  int i, retval, imax, side, sideprev;
  booleantype zroot, sgnchg;

  imax = 0;

  /* First check for change in sign in ghi or for a zero in ghi. */
  maxfrac = ZERO;
  zroot = SUNFALSE;
  sgnchg = SUNFALSE;
  for (i = 0;  i < cv_mem->cv_nrtfn; i++) {
    if(!cv_mem->cv_gactive[i]) continue;
    if (SUNRabs(cv_mem->cv_ghi[i]) == ZERO) {
      if(cv_mem->cv_rootdir[i]*cv_mem->cv_glo[i] <= ZERO) {
        zroot = SUNTRUE;
      }
    } else {
      if ( (cv_mem->cv_glo[i]*cv_mem->cv_ghi[i] < ZERO) &&
           (cv_mem->cv_rootdir[i]*cv_mem->cv_glo[i] <= ZERO) ) {
        gfrac = SUNRabs(cv_mem->cv_ghi[i]/(cv_mem->cv_ghi[i] - cv_mem->cv_glo[i]));
        if (gfrac > maxfrac) {
          sgnchg = SUNTRUE;
          maxfrac = gfrac;
          imax = i;
        }
      }
    }
  }

  /* If no sign change was found, reset trout and grout.  Then return
     CV_SUCCESS if no zero was found, or set iroots and return RTFOUND.  */
  if (!sgnchg) {
    cv_mem->cv_trout = cv_mem->cv_thi;
    for (i = 0; i < cv_mem->cv_nrtfn; i++) cv_mem->cv_grout[i] = cv_mem->cv_ghi[i];
    if (!zroot) return(CV_SUCCESS);
    for (i = 0; i < cv_mem->cv_nrtfn; i++) {
      cv_mem->cv_iroots[i] = 0;
      if(!cv_mem->cv_gactive[i]) continue;
      if ( (SUNRabs(cv_mem->cv_ghi[i]) == ZERO) &&
           (cv_mem->cv_rootdir[i]*cv_mem->cv_glo[i] <= ZERO) )
        cv_mem->cv_iroots[i] = cv_mem->cv_glo[i] > 0 ? -1 : 1;
    }
    return(RTFOUND);
  }

  /* Initialize alph to avoid compiler warning */
  alph = ONE;

  /* A sign change was found.  Loop to locate nearest root. */

  side = 0;  sideprev = -1;
  for(;;) {                                    /* Looping point */

    /* If interval size is already less than tolerance ttol, break. */
    if (SUNRabs(cv_mem->cv_thi - cv_mem->cv_tlo) <= cv_mem->cv_ttol) break;

    /* Set weight alph.
       On the first two passes, set alph = 1.  Thereafter, reset alph
       according to the side (low vs high) of the subinterval in which
       the sign change was found in the previous two passes.
       If the sides were opposite, set alph = 1.
       If the sides were the same, then double alph (if high side),
       or halve alph (if low side).
       The next guess tmid is the secant method value if alph = 1, but
       is closer to tlo if alph < 1, and closer to thi if alph > 1.    */

    if (sideprev == side) {
      alph = (side == 2) ? alph*TWO : alph*HALF;
    } else {
      alph = ONE;
    }

    /* Set next root approximation tmid and get g(tmid).
       If tmid is too close to tlo or thi, adjust it inward,
       by a fractional distance that is between 0.1 and 0.5.  */
    tmid = cv_mem->cv_thi - (cv_mem->cv_thi - cv_mem->cv_tlo) *
                            cv_mem->cv_ghi[imax] / (cv_mem->cv_ghi[imax] - alph*cv_mem->cv_glo[imax]);
    if (SUNRabs(tmid - cv_mem->cv_tlo) < HALF*cv_mem->cv_ttol) {
      fracint = SUNRabs(cv_mem->cv_thi - cv_mem->cv_tlo)/cv_mem->cv_ttol;
      fracsub = (fracint > FIVE) ? PT1 : HALF/fracint;
      tmid = cv_mem->cv_tlo + fracsub*(cv_mem->cv_thi - cv_mem->cv_tlo);
    }
    if (SUNRabs(cv_mem->cv_thi - tmid) < HALF*cv_mem->cv_ttol) {
      fracint = SUNRabs(cv_mem->cv_thi - cv_mem->cv_tlo)/cv_mem->cv_ttol;
      fracsub = (fracint > FIVE) ? PT1 : HALF/fracint;
      tmid = cv_mem->cv_thi - fracsub*(cv_mem->cv_thi - cv_mem->cv_tlo);
    }

    (void) CVodeGetDky(cv_mem, tmid, 0, cv_mem->cv_y);
    retval = cv_mem->cv_gfun(tmid, cv_mem->cv_y, cv_mem->cv_grout,
                             cv_mem->cv_user_data);
    cv_mem->cv_nge++;
    if (retval != 0) return(CV_RTFUNC_FAIL);

    /* Check to see in which subinterval g changes sign, and reset imax.
       Set side = 1 if sign change is on low side, or 2 if on high side.  */
    maxfrac = ZERO;
    zroot = SUNFALSE;
    sgnchg = SUNFALSE;
    sideprev = side;
    for (i = 0;  i < cv_mem->cv_nrtfn; i++) {
      if(!cv_mem->cv_gactive[i]) continue;
      if (SUNRabs(cv_mem->cv_grout[i]) == ZERO) {
        if(cv_mem->cv_rootdir[i]*cv_mem->cv_glo[i] <= ZERO) zroot = SUNTRUE;
      } else {
        if ( (cv_mem->cv_glo[i]*cv_mem->cv_grout[i] < ZERO) &&
             (cv_mem->cv_rootdir[i]*cv_mem->cv_glo[i] <= ZERO) ) {
          gfrac = SUNRabs(cv_mem->cv_grout[i]/(cv_mem->cv_grout[i] - cv_mem->cv_glo[i]));
          if (gfrac > maxfrac) {
            sgnchg = SUNTRUE;
            maxfrac = gfrac;
            imax = i;
          }
        }
      }
    }
    if (sgnchg) {
      /* Sign change found in (tlo,tmid); replace thi with tmid. */
      cv_mem->cv_thi = tmid;
      for (i = 0; i < cv_mem->cv_nrtfn; i++)
        cv_mem->cv_ghi[i] = cv_mem->cv_grout[i];
      side = 1;
      /* Stop at root thi if converged; otherwise loop. */
      if (SUNRabs(cv_mem->cv_thi - cv_mem->cv_tlo) <= cv_mem->cv_ttol) break;
      continue;  /* Return to looping point. */
    }

    if (zroot) {
      /* No sign change in (tlo,tmid), but g = 0 at tmid; return root tmid. */
      cv_mem->cv_thi = tmid;
      for (i = 0; i < cv_mem->cv_nrtfn; i++)
        cv_mem->cv_ghi[i] = cv_mem->cv_grout[i];
      break;
    }

    /* No sign change in (tlo,tmid), and no zero at tmid.
       Sign change must be in (tmid,thi).  Replace tlo with tmid. */
    cv_mem->cv_tlo = tmid;
    for (i = 0; i < cv_mem->cv_nrtfn; i++)
      cv_mem->cv_glo[i] = cv_mem->cv_grout[i];
    side = 2;
    /* Stop at root thi if converged; otherwise loop back. */
    if (SUNRabs(cv_mem->cv_thi - cv_mem->cv_tlo) <= cv_mem->cv_ttol) break;

  } /* End of root-search loop */

  /* Reset trout and grout, set iroots, and return RTFOUND. */
  cv_mem->cv_trout = cv_mem->cv_thi;
  for (i = 0; i < cv_mem->cv_nrtfn; i++) {
    cv_mem->cv_grout[i] = cv_mem->cv_ghi[i];
    cv_mem->cv_iroots[i] = 0;
    if(!cv_mem->cv_gactive[i]) continue;
    if ( (SUNRabs(cv_mem->cv_ghi[i]) == ZERO) &&
         (cv_mem->cv_rootdir[i]*cv_mem->cv_glo[i] <= ZERO) )
      cv_mem->cv_iroots[i] = cv_mem->cv_glo[i] > 0 ? -1 : 1;
    if ( (cv_mem->cv_glo[i]*cv_mem->cv_ghi[i] < ZERO) &&
         (cv_mem->cv_rootdir[i]*cv_mem->cv_glo[i] <= ZERO) )
      cv_mem->cv_iroots[i] = cv_mem->cv_glo[i] > 0 ? -1 : 1;
  }
  return(RTFOUND);
}

int cvRcheck1_cuda_d2(CVodeMem cv_mem)
{
  int i, retval;
  realtype smallh, hratio, tplus;
  booleantype zroot;

  for (i = 0; i < cv_mem->cv_nrtfn; i++) cv_mem->cv_iroots[i] = 0;
  cv_mem->cv_tlo = cv_mem->cv_tn;
  cv_mem->cv_ttol = (SUNRabs(cv_mem->cv_tn) + SUNRabs(cv_mem->cv_h)) *
                    cv_mem->cv_uround*HUNDRED;

  /* Evaluate g at initial t and check for zero values. */
  retval = cv_mem->cv_gfun(cv_mem->cv_tlo, cv_mem->cv_zn[0],
                           cv_mem->cv_glo, cv_mem->cv_user_data);
  cv_mem->cv_nge = 1;
  if (retval != 0) return(CV_RTFUNC_FAIL);

  zroot = SUNFALSE;
  for (i = 0; i < cv_mem->cv_nrtfn; i++) {
    if (SUNRabs(cv_mem->cv_glo[i]) == ZERO) {
      zroot = SUNTRUE;
      cv_mem->cv_gactive[i] = SUNFALSE;
    }
  }
  if (!zroot) return(CV_SUCCESS);

  /* Some g_i is zero at t0; look at g at t0+(small increment). */
  hratio = SUNMAX(cv_mem->cv_ttol/SUNRabs(cv_mem->cv_h), PT1);
  smallh = hratio*cv_mem->cv_h;
  tplus = cv_mem->cv_tlo + smallh;
  N_VLinearSum(ONE, cv_mem->cv_zn[0], hratio,
               cv_mem->cv_zn[1], cv_mem->cv_y);
  retval = cv_mem->cv_gfun(tplus, cv_mem->cv_y,
                           cv_mem->cv_ghi, cv_mem->cv_user_data);
  cv_mem->cv_nge++;
  if (retval != 0) return(CV_RTFUNC_FAIL);

  /* We check now only the components of g which were exactly 0.0 at t0
   * to see if we can 'activate' them. */
  for (i = 0; i < cv_mem->cv_nrtfn; i++) {
    if (!cv_mem->cv_gactive[i] && SUNRabs(cv_mem->cv_ghi[i]) != ZERO) {
      cv_mem->cv_gactive[i] = SUNTRUE;
      cv_mem->cv_glo[i] = cv_mem->cv_ghi[i];
    }
  }
  return(CV_SUCCESS);
}

int cvRcheck2_cuda_d2(CVodeMem cv_mem)
{
  int i, retval;
  realtype smallh, hratio, tplus;
  booleantype zroot;

  if (cv_mem->cv_irfnd == 0) return(CV_SUCCESS);

  (void) CVodeGetDky(cv_mem, cv_mem->cv_tlo, 0, cv_mem->cv_y);
  retval = cv_mem->cv_gfun(cv_mem->cv_tlo, cv_mem->cv_y,
                           cv_mem->cv_glo, cv_mem->cv_user_data);
  cv_mem->cv_nge++;
  if (retval != 0) return(CV_RTFUNC_FAIL);

  zroot = SUNFALSE;
  for (i = 0; i < cv_mem->cv_nrtfn; i++) cv_mem->cv_iroots[i] = 0;
  for (i = 0; i < cv_mem->cv_nrtfn; i++) {
    if (!cv_mem->cv_gactive[i]) continue;
    if (SUNRabs(cv_mem->cv_glo[i]) == ZERO) {
      zroot = SUNTRUE;
      cv_mem->cv_iroots[i] = 1;
    }
  }
  if (!zroot) return(CV_SUCCESS);

  /* One or more g_i has a zero at tlo.  Check g at tlo+smallh. */
  cv_mem->cv_ttol = (SUNRabs(cv_mem->cv_tn) + SUNRabs(cv_mem->cv_h)) *
                    cv_mem->cv_uround * HUNDRED;
  smallh = (cv_mem->cv_h > ZERO) ? cv_mem->cv_ttol : -cv_mem->cv_ttol;
  tplus = cv_mem->cv_tlo + smallh;
  if ( (tplus - cv_mem->cv_tn)*cv_mem->cv_h >= ZERO) {
    hratio = smallh/cv_mem->cv_h;
    N_VLinearSum(ONE, cv_mem->cv_y, hratio, cv_mem->cv_zn[1], cv_mem->cv_y);
  } else {
    (void) CVodeGetDky(cv_mem, tplus, 0, cv_mem->cv_y);
  }
  retval = cv_mem->cv_gfun(tplus, cv_mem->cv_y,
                           cv_mem->cv_ghi, cv_mem->cv_user_data);
  cv_mem->cv_nge++;
  if (retval != 0) return(CV_RTFUNC_FAIL);

  /* Check for close roots (error return), for a new zero at tlo+smallh,
  and for a g_i that changed from zero to nonzero. */
  zroot = SUNFALSE;
  for (i = 0; i < cv_mem->cv_nrtfn; i++) {
    if (!cv_mem->cv_gactive[i]) continue;
    if (SUNRabs(cv_mem->cv_ghi[i]) == ZERO) {
      if (cv_mem->cv_iroots[i] == 1) return(CLOSERT);
      zroot = SUNTRUE;
      cv_mem->cv_iroots[i] = 1;
    } else {
      if (cv_mem->cv_iroots[i] == 1)
        cv_mem->cv_glo[i] = cv_mem->cv_ghi[i];
    }
  }
  if (zroot) return(RTFOUND);
  return(CV_SUCCESS);
}

int cvRcheck3_cuda_d2(CVodeMem cv_mem)
{
  int i, ier, retval;

  /* Set thi = tn or tout, whichever comes first; set y = y(thi). */
  if (cv_mem->cv_taskc == CV_ONE_STEP) {
    cv_mem->cv_thi = cv_mem->cv_tn;
    N_VScale(ONE, cv_mem->cv_zn[0], cv_mem->cv_y);
  }
  if (cv_mem->cv_taskc == CV_NORMAL) {
    if ( (cv_mem->cv_toutc - cv_mem->cv_tn)*cv_mem->cv_h >= ZERO) {
      cv_mem->cv_thi = cv_mem->cv_tn;
      N_VScale(ONE, cv_mem->cv_zn[0], cv_mem->cv_y);
    } else {
      cv_mem->cv_thi = cv_mem->cv_toutc;
      (void) CVodeGetDky(cv_mem, cv_mem->cv_thi, 0, cv_mem->cv_y);
    }
  }

  /* Set ghi = g(thi) and call cvRootfind to search (tlo,thi) for roots. */
  retval = cv_mem->cv_gfun(cv_mem->cv_thi, cv_mem->cv_y,
                           cv_mem->cv_ghi, cv_mem->cv_user_data);
  cv_mem->cv_nge++;
  if (retval != 0) return(CV_RTFUNC_FAIL);

  cv_mem->cv_ttol = (SUNRabs(cv_mem->cv_tn) + SUNRabs(cv_mem->cv_h)) *
                    cv_mem->cv_uround * HUNDRED;
  ier = cvRootfind_cuda_d2(cv_mem);
  if (ier == CV_RTFUNC_FAIL) return(CV_RTFUNC_FAIL);
  for(i=0; i<cv_mem->cv_nrtfn; i++) {
    if(!cv_mem->cv_gactive[i] && cv_mem->cv_grout[i] != ZERO)
      cv_mem->cv_gactive[i] = SUNTRUE;
  }
  cv_mem->cv_tlo = cv_mem->cv_trout;
  for (i = 0; i < cv_mem->cv_nrtfn; i++)
    cv_mem->cv_glo[i] = cv_mem->cv_grout[i];

  /* If no root found, return CV_SUCCESS. */
  if (ier == CV_SUCCESS) return(CV_SUCCESS);

  /* If a root was found, interpolate to get y(trout) and return.  */
  (void) CVodeGetDky(cv_mem, cv_mem->cv_trout, 0, cv_mem->cv_y);
  return(RTFOUND);
}


void cvRescale_cuda_d2(CVodeMem cv_mem)
{
  int j;
  realtype factor;

  factor = cv_mem->cv_eta;
  for (j=1; j <= cv_mem->cv_q; j++) {
    N_VScale(factor, cv_mem->cv_zn[j], cv_mem->cv_zn[j]);
    factor *= cv_mem->cv_eta;
  }
  cv_mem->cv_h = cv_mem->cv_hscale * cv_mem->cv_eta;
  cv_mem->cv_next_h = cv_mem->cv_h;
  cv_mem->cv_hscale = cv_mem->cv_h;
  cv_mem->cv_nscon = 0;
}

void cvPredict_cuda_d2(CVodeMem cv_mem)
{
  int j, k;

  cv_mem->cv_tn += cv_mem->cv_h;
  if (cv_mem->cv_tstopset) {
    if ((cv_mem->cv_tn - cv_mem->cv_tstop)*cv_mem->cv_h > ZERO)
      cv_mem->cv_tn = cv_mem->cv_tstop;
  }
  N_VScale(ONE, cv_mem->cv_zn[0], cv_mem->cv_last_yn);
  for (k = 1; k <= cv_mem->cv_q; k++)
    for (j = cv_mem->cv_q; j >= k; j--)
      N_VLinearSum(ONE, cv_mem->cv_zn[j-1], ONE,
                   cv_mem->cv_zn[j], cv_mem->cv_zn[j-1]);

}

void cvIncreaseBDf_gpu(CVodeMem cv_mem)
{
  realtype alpha0, alpha1, prod, xi, xiold, hsum, A1;
  int i, j;

  for (i=0; i <= cv_mem->cv_qmax; i++) cv_mem->cv_l[i] = ZERO;
  cv_mem->cv_l[2] = alpha1 = prod = xiold = ONE;
  alpha0 = -ONE;
  hsum = cv_mem->cv_hscale;
  if (cv_mem->cv_q > 1) {
    for (j=1; j < cv_mem->cv_q; j++) {
      hsum += cv_mem->cv_tau[j+1];
      xi = hsum / cv_mem->cv_hscale;
      prod *= xi;
      alpha0 -= ONE / (j+1);
      alpha1 += ONE / xi;
      for (i=j+2; i >= 2; i--)
        cv_mem->cv_l[i] = cv_mem->cv_l[i]*xiold + cv_mem->cv_l[i-1];
      xiold = xi;
    }
  }
  A1 = (-alpha0 - alpha1) / prod;
  N_VScale(A1, cv_mem->cv_zn[cv_mem->cv_indx_acor],
           cv_mem->cv_zn[cv_mem->cv_L]);
  for (j=2; j <= cv_mem->cv_q; j++)
    N_VLinearSum(cv_mem->cv_l[j], cv_mem->cv_zn[cv_mem->cv_L], ONE,
                 cv_mem->cv_zn[j], cv_mem->cv_zn[j]);
}

void cvDecreaseBDf_gpu(CVodeMem cv_mem)
{
  realtype hsum, xi;
  int i, j;

  for (i=0; i <= cv_mem->cv_qmax; i++) cv_mem->cv_l[i] = ZERO;
  cv_mem->cv_l[2] = ONE;
  hsum = ZERO;
  for (j=1; j <= cv_mem->cv_q-2; j++) {
    hsum += cv_mem->cv_tau[j];
    xi = hsum /cv_mem->cv_hscale;
    for (i=j+2; i >= 2; i--)
      cv_mem->cv_l[i] = cv_mem->cv_l[i]*xi + cv_mem->cv_l[i-1];
  }

  for (j=2; j < cv_mem->cv_q; j++)
    N_VLinearSum(-cv_mem->cv_l[j], cv_mem->cv_zn[cv_mem->cv_q],
                 ONE, cv_mem->cv_zn[j], cv_mem->cv_zn[j]);
}

void cvSetTqBDf_gpu(CVodeMem cv_mem, realtype hsum, realtype alpha0,
                       realtype alpha0_hat, realtype xi_inv, realtype xistar_inv)
{
  realtype A1, A2, A3, A4, A5, A6;
  realtype C, Cpinv, Cppinv;

  A1 = ONE - alpha0_hat + alpha0;
  A2 = ONE + cv_mem->cv_q * A1;
  cv_mem->cv_tq[2] = SUNRabs(A1 / (alpha0 * A2));
  cv_mem->cv_tq[5] = SUNRabs(A2 * xistar_inv / (cv_mem->cv_l[cv_mem->cv_q] * xi_inv));
  if (cv_mem->cv_qwait == 1) {
    if (cv_mem->cv_q > 1) {
      C = xistar_inv / cv_mem->cv_l[cv_mem->cv_q];
      A3 = alpha0 + ONE / cv_mem->cv_q;
      A4 = alpha0_hat + xi_inv;
      Cpinv = (ONE - A4 + A3) / A3;
      cv_mem->cv_tq[1] = SUNRabs(C * Cpinv);
    }
    else cv_mem->cv_tq[1] = ONE;
    hsum += cv_mem->cv_tau[cv_mem->cv_q];
    xi_inv = cv_mem->cv_h / hsum;
    A5 = alpha0 - (ONE / (cv_mem->cv_q+1));
    A6 = alpha0_hat - xi_inv;
    Cppinv = (ONE - A6 + A5) / A2;
    cv_mem->cv_tq[3] = SUNRabs(Cppinv / (xi_inv * (cv_mem->cv_q+2) * A5));
  }
  cv_mem->cv_tq[4] = cv_mem->cv_nlscoef / cv_mem->cv_tq[2];
}


void cvSetBDf_gpu(CVodeMem cv_mem)
{
  realtype alpha0, alpha0_hat, xi_inv, xistar_inv, hsum;
  int i,j;

  cv_mem->cv_l[0] = cv_mem->cv_l[1] = xi_inv = xistar_inv = ONE;
  for (i=2; i <= cv_mem->cv_q; i++) cv_mem->cv_l[i] = ZERO;
  alpha0 = alpha0_hat = -ONE;
  hsum = cv_mem->cv_h;
  if (cv_mem->cv_q > 1) {
    for (j=2; j < cv_mem->cv_q; j++) {
      hsum += cv_mem->cv_tau[j-1];
      xi_inv = cv_mem->cv_h / hsum;
      alpha0 -= ONE / j;
      for (i=j; i >= 1; i--) cv_mem->cv_l[i] += cv_mem->cv_l[i-1]*xi_inv;
      /* The l[i] are coefficients of product(1 to j) (1 + x/xi_i) */
    }

    /* j = q */
    alpha0 -= ONE / cv_mem->cv_q;
    xistar_inv = -cv_mem->cv_l[1] - alpha0;
    hsum += cv_mem->cv_tau[cv_mem->cv_q-1];
    xi_inv = cv_mem->cv_h / hsum;
    alpha0_hat = -cv_mem->cv_l[1] - xi_inv;
    for (i=cv_mem->cv_q; i >= 1; i--)
      cv_mem->cv_l[i] += cv_mem->cv_l[i-1]*xistar_inv;
  }

  cvSetTqBDf_gpu(cv_mem, hsum, alpha0, alpha0_hat, xi_inv, xistar_inv);
}

void cvAdjustParams_cuda_d2(CVodeMem cv_mem)
{
  if (cv_mem->cv_qprime != cv_mem->cv_q) {
    //cvAdjustOrder(cv_mem, cv_mem->cv_qprime-cv_mem->cv_q);

    int deltaq = cv_mem->cv_qprime-cv_mem->cv_q;
    switch(deltaq) {
      case 1:
        cvIncreaseBDf_gpu(cv_mem);
        break;
      case -1:
        cvDecreaseBDf_gpu(cv_mem);
        break;
    }

    cv_mem->cv_q = cv_mem->cv_qprime;
    cv_mem->cv_L = cv_mem->cv_q+1;
    cv_mem->cv_qwait = cv_mem->cv_L;
  }
  cvRescale_cuda_d2(cv_mem);
}

void cvSet_cuda_d2(CVodeMem cv_mem)
{

  cvSetBDf_gpu(cv_mem);
  cv_mem->cv_rl1 = ONE / cv_mem->cv_l[1];
  cv_mem->cv_gamma = cv_mem->cv_h * cv_mem->cv_rl1;
  if (cv_mem->cv_nst == 0) cv_mem->cv_gammap = cv_mem->cv_gamma;
  cv_mem->cv_gamrat = (cv_mem->cv_nst > 0) ?
                      cv_mem->cv_gamma / cv_mem->cv_gammap : ONE;  /* protect x / x != 1.0 */
}


int cvHandleNFlag_cuda_d2(CVodeMem cv_mem, int *nflagPtr, realtype saved_t,
                         int *ncfPtr)
{
  int nflag;

  nflag = *nflagPtr;

  if (nflag == CV_SUCCESS) return(DO_ERROR_TEST);

  /* The nonlinear soln. failed; increment ncfn and restore zn */
  cv_mem->cv_ncfn++;
  cvRestore_cuda_d2(cv_mem, saved_t);

  /* Return if lsetup, lsolve, or rhs failed unrecoverably */
  if (nflag == CV_LSETUP_FAIL)  return(CV_LSETUP_FAIL);
  if (nflag == CV_LSOLVE_FAIL)  return(CV_LSOLVE_FAIL);
  if (nflag == CV_RHSFUNC_FAIL) return(CV_RHSFUNC_FAIL);

  /* At this point, nflag = CONV_FAIL or RHSFUNC_RECVR; increment ncf */

  (*ncfPtr)++;
  cv_mem->cv_etamax = ONE;

  /* If we had maxncf failures or |h| = hmin,
     return CV_CONV_FAILURE or CV_REPTD_RHSFUNC_ERR. */

  if ((SUNRabs(cv_mem->cv_h) <= cv_mem->cv_hmin*ONEPSM) ||
      (*ncfPtr == cv_mem->cv_maxncf)) {
    if (nflag == CONV_FAIL)     return(CV_CONV_FAILURE);
    if (nflag == RHSFUNC_RECVR) return(CV_REPTD_RHSFUNC_ERR);
  }

  /* Reduce step size; return to reattempt the step */

  cv_mem->cv_eta = SUNMAX(ETACF, cv_mem->cv_hmin / SUNRabs(cv_mem->cv_h));
  *nflagPtr = PREV_CONV_FAIL;
  cvRescale_cuda_d2(cv_mem);

  return(PREDICT_AGAIN);
}

void cvChooseEta_cuda_d2(CVodeMem cv_mem)
{
  realtype etam;

  etam = SUNMAX(cv_mem->cv_etaqm1, SUNMAX(cv_mem->cv_etaq, cv_mem->cv_etaqp1));

  if (etam < THRESH) {
    cv_mem->cv_eta = ONE;
    cv_mem->cv_qprime = cv_mem->cv_q;
    return;
  }

  if (etam == cv_mem->cv_etaq) {

    cv_mem->cv_eta = cv_mem->cv_etaq;
    cv_mem->cv_qprime = cv_mem->cv_q;

  } else if (etam == cv_mem->cv_etaqm1) {

    cv_mem->cv_eta = cv_mem->cv_etaqm1;
    cv_mem->cv_qprime = cv_mem->cv_q - 1;

  } else {

    cv_mem->cv_eta = cv_mem->cv_etaqp1;
    cv_mem->cv_qprime = cv_mem->cv_q + 1;

    if (cv_mem->cv_lmm == CV_BDF) {
      /*
       * Store Delta_n in zn[qmax] to be used in order increase
       *
       * This happens at the last step of order q before an increase
       * to order q+1, so it represents Delta_n in the ELTE at q+1
       */

      N_VScale(ONE, cv_mem->cv_acor, cv_mem->cv_zn[cv_mem->cv_qmax]);

    }
  }
}

booleantype cvDoErrorTest_cuda_d2(CVodeMem cv_mem, int *nflagPtr,
                                 realtype saved_t, int *nefPtr, realtype *dsmPtr)
{
  realtype dsm;
  realtype min_val;
  int retval;

  /* Find the minimum concentration and if it's small and negative, make it
   * positive */
  N_VLinearSum(cv_mem->cv_l[0], cv_mem->cv_acor, ONE, cv_mem->cv_zn[0],
               cv_mem->cv_ftemp);
  min_val = N_VMin(cv_mem->cv_ftemp);
  if (min_val < ZERO && min_val > -CAMP_TINY) {
    N_VAbs(cv_mem->cv_ftemp, cv_mem->cv_ftemp);
    N_VLinearSum(-cv_mem->cv_l[0], cv_mem->cv_acor, ONE, cv_mem->cv_ftemp,
                 cv_mem->cv_zn[0]);
    min_val = ZERO;
  }

  dsm = cv_mem->cv_acnrm * cv_mem->cv_tq[2];

  /* If est. local error norm dsm passes test and there are no negative values,
   * return CV_SUCCESS */
  *dsmPtr = dsm;
  if (dsm <= ONE && min_val >= ZERO) return(CV_SUCCESS);

  /* Test failed; increment counters, set nflag, and restore zn array */
  (*nefPtr)++;
  cv_mem->cv_netf++;
  *nflagPtr = PREV_ERR_FAIL;
  cvRestore_cuda_d2(cv_mem, saved_t);

  /* At maxnef failures or |h| = hmin, return CV_ERR_FAILURE */
  if ((SUNRabs(cv_mem->cv_h) <= cv_mem->cv_hmin*ONEPSM) ||
      (*nefPtr == cv_mem->cv_maxnef)) return(CV_ERR_FAILURE);

  /* Set etamax = 1 to prevent step size increase at end of this step */
  cv_mem->cv_etamax = ONE;

  /* Set h ratio eta from dsm, rescale, and return for retry of step */
  if (*nefPtr <= MXNEF1) {
    cv_mem->cv_eta = ONE / (SUNRpowerR(BIAS2*dsm,ONE/cv_mem->cv_L) + ADDON);
    cv_mem->cv_eta = SUNMAX(ETAMIN, SUNMAX(cv_mem->cv_eta,
                                           cv_mem->cv_hmin / SUNRabs(cv_mem->cv_h)));
    if (*nefPtr >= SMALL_NEF) cv_mem->cv_eta = SUNMIN(cv_mem->cv_eta, ETAMXF);
    cvRescale_cuda_d2(cv_mem);
    return(TRY_AGAIN);
  }

  /* After MXNEF1 failures, force an order reduction and retry step */
  if (cv_mem->cv_q > 1) {
    cv_mem->cv_eta = SUNMAX(ETAMIN, cv_mem->cv_hmin / SUNRabs(cv_mem->cv_h));

    cvDecreaseBDf_gpu(cv_mem);

    cv_mem->cv_L = cv_mem->cv_q;
    cv_mem->cv_q--;
    cv_mem->cv_qwait = cv_mem->cv_L;
    cvRescale_cuda_d2(cv_mem);
    return(TRY_AGAIN);
  }

  /* If already at order 1, restart: reload zn from scratch */

  cv_mem->cv_eta = SUNMAX(ETAMIN, cv_mem->cv_hmin / SUNRabs(cv_mem->cv_h));
  cv_mem->cv_h *= cv_mem->cv_eta;
  cv_mem->cv_next_h = cv_mem->cv_h;
  cv_mem->cv_hscale = cv_mem->cv_h;
  cv_mem->cv_qwait = LONG_WAIT;
  cv_mem->cv_nscon = 0;


  //retval = cv_mem->cv_f(cv_mem->cv_tn, cv_mem->cv_zn[0],
  //                      cv_mem->cv_tempv, cv_mem->cv_user_data);
  retval = f(cv_mem->cv_tn, cv_mem->cv_zn[0],cv_mem->cv_tempv, cv_mem->cv_user_data);

  cv_mem->cv_nfe++;
  if (retval < 0)  return(CV_RHSFUNC_FAIL);
  if (retval > 0)  return(CV_UNREC_RHSFUNC_ERR);

  N_VScale(cv_mem->cv_h, cv_mem->cv_tempv, cv_mem->cv_zn[1]);

  return(TRY_AGAIN);
}

void cvCompleteStep_cuda_d2(CVodeMem cv_mem)
{
  int i, j;

  cv_mem->cv_nst++;
  cv_mem->cv_nscon++;
  cv_mem->cv_hu = cv_mem->cv_h;
  cv_mem->cv_qu = cv_mem->cv_q;

  for (i=cv_mem->cv_q; i >= 2; i--)  cv_mem->cv_tau[i] = cv_mem->cv_tau[i-1];
  if ((cv_mem->cv_q==1) && (cv_mem->cv_nst > 1))
    cv_mem->cv_tau[2] = cv_mem->cv_tau[1];
  cv_mem->cv_tau[1] = cv_mem->cv_h;

  /* Apply correction to column j of zn: l_j * Delta_n */
  for (j=0; j <= cv_mem->cv_q; j++)
    N_VLinearSum(cv_mem->cv_l[j], cv_mem->cv_acor, ONE,
                 cv_mem->cv_zn[j], cv_mem->cv_zn[j]);
  cv_mem->cv_qwait--;
  if ((cv_mem->cv_qwait == 1) && (cv_mem->cv_q != cv_mem->cv_qmax)) {
    N_VScale(ONE, cv_mem->cv_acor, cv_mem->cv_zn[cv_mem->cv_qmax]);
    cv_mem->cv_saved_tq5 = cv_mem->cv_tq[5];
    cv_mem->cv_indx_acor = cv_mem->cv_qmax;
  }
}

void cvPrepareNextStep_cuda_d2(CVodeMem cv_mem, realtype dsm)
{
  /* If etamax = 1, defer step size or order changes */
  if (cv_mem->cv_etamax == ONE) {
    cv_mem->cv_qwait = SUNMAX(cv_mem->cv_qwait, 2);
    cv_mem->cv_qprime = cv_mem->cv_q;
    cv_mem->cv_hprime = cv_mem->cv_h;
    cv_mem->cv_eta = ONE;
    return;
  }

  /* etaq is the ratio of new to old h at the current order */
  cv_mem->cv_etaq = ONE /(SUNRpowerR(BIAS2*dsm,ONE/cv_mem->cv_L) + ADDON);

  /* If no order change, adjust eta and acor in cvSetEta and return */
  if (cv_mem->cv_qwait != 0) {
    cv_mem->cv_eta = cv_mem->cv_etaq;
    cv_mem->cv_qprime = cv_mem->cv_q;
    cvSetEta_cuda_d2(cv_mem);
    return;
  }

  /* If qwait = 0, consider an order change.   etaqm1 and etaqp1 are
     the ratios of new to old h at orders q-1 and q+1, respectively.
     cvChooseEta selects the largest; cvSetEta adjusts eta and acor */
  cv_mem->cv_qwait = 2;

  //cv_mem->cv_etaqm1 = cvComputeEtaqm1_cuda_d2(cv_mem);
  //compute cv_etaqm1
  realtype ddn;
  cv_mem->cv_etaqm1 = ZERO;
  if (cv_mem->cv_q > 1) {
    ddn = N_VWrmsNorm(cv_mem->cv_zn[cv_mem->cv_q], cv_mem->cv_ewt) * cv_mem->cv_tq[1];
    cv_mem->cv_etaqm1 = ONE/(SUNRpowerR(BIAS1*ddn, ONE/cv_mem->cv_q) + ADDON);
  }

  //cv_mem->cv_etaqp1 = cvComputeEtaqp1_cuda_d2(cv_mem);
  //compute cv_etaqp1
  realtype dup, cquot;
  cv_mem->cv_etaqp1 = ZERO;
  if (cv_mem->cv_q != cv_mem->cv_qmax && cv_mem->cv_saved_tq5 != ZERO) {
    //if (cv_mem->cv_saved_tq5 != ZERO) return(cv_mem->cv_etaqp1);
    cquot = (cv_mem->cv_tq[5] / cv_mem->cv_saved_tq5) *
            SUNRpowerI(cv_mem->cv_h/cv_mem->cv_tau[2], cv_mem->cv_L);
    N_VLinearSum(-cquot, cv_mem->cv_zn[cv_mem->cv_qmax], ONE,
                 cv_mem->cv_acor, cv_mem->cv_tempv);
    dup = N_VWrmsNorm(cv_mem->cv_tempv, cv_mem->cv_ewt) * cv_mem->cv_tq[3];
    cv_mem->cv_etaqp1 = ONE / (SUNRpowerR(BIAS3*dup, ONE/(cv_mem->cv_L+1)) + ADDON);
  }

  cvChooseEta_cuda_d2(cv_mem);
  cvSetEta_cuda_d2(cv_mem);
}

int cvSLdet_cuda_d2(CVodeMem cv_mem)
{
  int i, k, j, it, kmin = 0, kflag = 0;
  realtype rat[5][4], rav[4], qkr[4], sigsq[4], smax[4], ssmax[4];
  realtype drr[4], rrc[4],sqmx[4], qjk[4][4], vrat[5], qc[6][4], qco[6][4];
  realtype rr, rrcut, vrrtol, vrrt2, sqtol, rrtol;
  realtype smink, smaxk, sumrat, sumrsq, vmin, vmax, drrmax, adrr;
  realtype tem, sqmax, saqk, qp, s, sqmaxk, saqj, sqmin;
  realtype rsa, rsb, rsc, rsd, rd1a, rd1b, rd1c;
  realtype rd2a, rd2b, rd3a, cest1, corr1;
  realtype ratp, ratm, qfac1, qfac2, bb, rrb;
  rrcut  = RCONST(0.98);
  vrrtol = RCONST(1.0e-4);
  vrrt2  = RCONST(5.0e-4);
  sqtol  = RCONST(1.0e-3);
  rrtol  = RCONST(1.0e-2);
  rr = ZERO;
  for (k=1; k<=3; k++) {
    smink = cv_mem->cv_ssdat[1][k];
    smaxk = ZERO;
    for (i=1; i<=5; i++) {
      smink = SUNMIN(smink,cv_mem->cv_ssdat[i][k]);
      smaxk = SUNMAX(smaxk,cv_mem->cv_ssdat[i][k]);
    }
    if (smink < TINY*smaxk) {
      kflag = -1;
      return(kflag);
    }
    smax[k] = smaxk;
    ssmax[k] = smaxk*smaxk;
    sumrat = ZERO;
    sumrsq = ZERO;
    for (i=1; i<=4; i++) {
      rat[i][k] = cv_mem->cv_ssdat[i][k]/cv_mem->cv_ssdat[i+1][k];
      sumrat = sumrat + rat[i][k];
      sumrsq = sumrsq + rat[i][k]*rat[i][k];
    }
    rav[k] = FOURTH*sumrat;
    vrat[k] = SUNRabs(FOURTH*sumrsq - rav[k]*rav[k]);
    qc[5][k] = cv_mem->cv_ssdat[1][k] * cv_mem->cv_ssdat[3][k] -
               cv_mem->cv_ssdat[2][k] * cv_mem->cv_ssdat[2][k];
    qc[4][k] = cv_mem->cv_ssdat[2][k] * cv_mem->cv_ssdat[3][k] -
               cv_mem->cv_ssdat[1][k] * cv_mem->cv_ssdat[4][k];
    qc[3][k] = ZERO;
    qc[2][k] = cv_mem->cv_ssdat[2][k] * cv_mem->cv_ssdat[5][k] -
               cv_mem->cv_ssdat[3][k] * cv_mem->cv_ssdat[4][k];
    qc[1][k] = cv_mem->cv_ssdat[4][k] * cv_mem->cv_ssdat[4][k] -
               cv_mem->cv_ssdat[3][k] * cv_mem->cv_ssdat[5][k];
    for (i=1; i<=5; i++)
      qco[i][k] = qc[i][k];
  }
  vmin = SUNMIN(vrat[1],SUNMIN(vrat[2],vrat[3]));
  vmax = SUNMAX(vrat[1],SUNMAX(vrat[2],vrat[3]));
  if (vmin < vrrtol*vrrtol) {
    if (vmax > vrrt2*vrrt2) {
      kflag = -2;
      return(kflag);
    } else {
      rr = (rav[1] + rav[2] + rav[3])/THREE;
      drrmax = ZERO;
      for (k = 1;k<=3;k++) {
        adrr = SUNRabs(rav[k] - rr);
        drrmax = SUNMAX(drrmax, adrr);
      }
      if (drrmax > vrrt2) {kflag = -3; return(kflag);}
      kflag = 1;
    }
  } else {
    if (SUNRabs(qco[1][1]) < TINY*ssmax[1]) {
      kflag = -4;
      return(kflag);
    }
    tem = qco[1][2]/qco[1][1];
    for (i=2; i<=5; i++) {
      qco[i][2] = qco[i][2] - tem*qco[i][1];
    }
    qco[1][2] = ZERO;
    tem = qco[1][3]/qco[1][1];
    for (i=2; i<=5; i++) {
      qco[i][3] = qco[i][3] - tem*qco[i][1];
    }
    qco[1][3] = ZERO;
    if (SUNRabs(qco[2][2]) < TINY*ssmax[2]) {
      kflag = -4;
      return(kflag);
    }
    tem = qco[2][3]/qco[2][2];
    for (i=3; i<=5; i++) {
      qco[i][3] = qco[i][3] - tem*qco[i][2];
    }
    if (SUNRabs(qco[4][3]) < TINY*ssmax[3]) {
      kflag = -4;
      return(kflag);
    }
    rr = -qco[5][3]/qco[4][3];
    if (rr < TINY || rr > HUNDRED) {
      kflag = -5;
      return(kflag);
    }
    for (k=1; k<=3; k++)
      qkr[k] = qc[5][k] + rr*(qc[4][k] + rr*rr*(qc[2][k] + rr*qc[1][k]));
    sqmax = ZERO;
    for (k=1; k<=3; k++) {
      saqk = SUNRabs(qkr[k])/ssmax[k];
      if (saqk > sqmax) sqmax = saqk;
    }
    if (sqmax < sqtol) {
      kflag = 2;
    } else {
      for (it=1; it<=3; it++) {
        for (k=1; k<=3; k++) {
          qp = qc[4][k] + rr*rr*(THREE*qc[2][k] + rr*FOUR*qc[1][k]);
          drr[k] = ZERO;
          if (SUNRabs(qp) > TINY*ssmax[k]) drr[k] = -qkr[k]/qp;
          rrc[k] = rr + drr[k];
        }
        for (k=1; k<=3; k++) {
          s = rrc[k];
          sqmaxk = ZERO;
          for (j=1; j<=3; j++) {
            qjk[j][k] = qc[5][j] + s*(qc[4][j] + s*s*(qc[2][j] + s*qc[1][j]));
            saqj = SUNRabs(qjk[j][k])/ssmax[j];
            if (saqj > sqmaxk) sqmaxk = saqj;
          }
          sqmx[k] = sqmaxk;
        }
        sqmin = sqmx[1] + ONE;
        for (k=1; k<=3; k++) {
          if (sqmx[k] < sqmin) {
            kmin = k;
            sqmin = sqmx[k];
          }
        }
        rr = rrc[kmin];
        if (sqmin < sqtol) {
          kflag = 3;
          /*  can compute charactistic root   */
          /*  break out of Newton correction loop and drop to "given rr,etc" */
          break;
        } else {
          for (j=1; j<=3; j++) {
            qkr[j] = qjk[j][kmin];
          }
        }
      } /*  end of Newton correction loop  */

      if (sqmin > sqtol) {
        kflag = -6;
        return(kflag);
      }
    } /*  end of if (sqmax < sqtol) else   */
  } /*  end of if (vmin < vrrtol*vrrtol) else, quartics to get rr. */

  for (k=1; k<=3; k++) {
    rsa = cv_mem->cv_ssdat[1][k];
    rsb = cv_mem->cv_ssdat[2][k]*rr;
    rsc = cv_mem->cv_ssdat[3][k]*rr*rr;
    rsd = cv_mem->cv_ssdat[4][k]*rr*rr*rr;
    rd1a = rsa - rsb;
    rd1b = rsb - rsc;
    rd1c = rsc - rsd;
    rd2a = rd1a - rd1b;
    rd2b = rd1b - rd1c;
    rd3a = rd2a - rd2b;
    if (SUNRabs(rd1b) < TINY*smax[k]) {
      kflag = -7;
      return(kflag);
    }
    cest1 = -rd3a/rd1b;
    if (cest1 < TINY || cest1 > FOUR) {
      kflag = -7;
      return(kflag);
    }
    corr1 = (rd2b/cest1)/(rr*rr);
    sigsq[k] = cv_mem->cv_ssdat[3][k] + corr1;
  }
  if (sigsq[2] < TINY) {
    kflag = -8;
    return(kflag);
  }
  ratp = sigsq[3]/sigsq[2];
  ratm = sigsq[1]/sigsq[2];
  qfac1 = FOURTH*(cv_mem->cv_q*cv_mem->cv_q - ONE);
  qfac2 = TWO/(cv_mem->cv_q - ONE);
  bb = ratp*ratm - ONE - qfac1*ratp;
  tem = ONE - qfac2*bb;
  if (SUNRabs(tem) < TINY) {
    kflag = -8;
    return(kflag);
  }
  rrb = ONE/tem;
  if (SUNRabs(rrb - rr) > rrtol) {
    kflag = -9;
    return(kflag);
  }
  if (rr > rrcut) {
    if (kflag == 1) kflag = 4;
    if (kflag == 2) kflag = 5;
    if (kflag == 3) kflag = 6;
  }
  return(kflag);
}

void cvBDFStab_cuda_d2(CVodeMem cv_mem)
{
  int i,k, ldflag, factorial;
  realtype sq, sqm1, sqm2;
  if (cv_mem->cv_q >= 3) {
    for (k = 1; k <= 3; k++)
      for (i = 5; i >= 2; i--)
        cv_mem->cv_ssdat[i][k] = cv_mem->cv_ssdat[i-1][k];
    factorial = 1;
    for (i = 1; i <= cv_mem->cv_q-1; i++) factorial *= i;
    sq = factorial * cv_mem->cv_q * (cv_mem->cv_q+1) *
         cv_mem->cv_acnrm / SUNMAX(cv_mem->cv_tq[5],TINY);
    sqm1 = factorial * cv_mem->cv_q *
           N_VWrmsNorm(cv_mem->cv_zn[cv_mem->cv_q], cv_mem->cv_ewt);
    sqm2 = factorial * N_VWrmsNorm(cv_mem->cv_zn[cv_mem->cv_q-1], cv_mem->cv_ewt);
    cv_mem->cv_ssdat[1][1] = sqm2*sqm2;
    cv_mem->cv_ssdat[1][2] = sqm1*sqm1;
    cv_mem->cv_ssdat[1][3] = sq*sq;
  }
  if (cv_mem->cv_qprime >= cv_mem->cv_q) {
    if ( (cv_mem->cv_q >= 3) && (cv_mem->cv_nscon >= cv_mem->cv_q+5) ) {
      ldflag = cvSLdet_cuda_d2(cv_mem);
      if (ldflag > 3) {
        /* A stability limit violation is indicated by
           a return flag of 4, 5, or 6.
           Reduce new order.                     */
        cv_mem->cv_qprime = cv_mem->cv_q-1;
        cv_mem->cv_eta = cv_mem->cv_etaqm1;
        cv_mem->cv_eta = SUNMIN(cv_mem->cv_eta,cv_mem->cv_etamax);
        cv_mem->cv_eta = cv_mem->cv_eta /
                         SUNMAX(ONE,SUNRabs(cv_mem->cv_h)*cv_mem->cv_hmax_inv*cv_mem->cv_eta);
        cv_mem->cv_hprime = cv_mem->cv_h*cv_mem->cv_eta;
        cv_mem->cv_nor = cv_mem->cv_nor + 1;
      }
    }
  }
  else {
    cv_mem->cv_nscon = 0;
  }
}

int linsolsetup_cuda_d2(SolverData *sd, CVodeMem cv_mem,int convfail,N_Vector vtemp1,N_Vector vtemp2,N_Vector vtemp3)
{
  ModelDataGPU_d2 *mGPU;
  booleantype jbad, jok;
  realtype dgamma;
  CVDlsMem cvdls_mem = (CVDlsMem) cv_mem->cv_lmem;;
  int retval = 0;

  int offset_nrows = 0;
  //printf("linsolsetup_cuda_d2 start\n");
  dgamma = fabs((cv_mem->cv_gamma/cv_mem->cv_gammap) - ONE); //In GPU is fabs too
  //dgamma = SUNRabs((cv_mem->cv_gamma/cv_mem->cv_gammap) - ONE);
  jbad = (cv_mem->cv_nst == 0) ||
         (cv_mem->cv_nst > cvdls_mem->nstlj + CVD_MSBJ) ||
         ((convfail == CV_FAIL_BAD_J) && (dgamma < CVD_DGMAX)) ||
         (convfail == CV_FAIL_OTHER);
  jok = !jbad;

  if (jok) {
    //if (0) {
    cv_mem->cv_jcur = SUNFALSE;
    retval = SUNMatCopy(cvdls_mem->savedJ, cvdls_mem->A);

    /* If jok = SUNFALSE, reset J, call jac routine for new J value and save a copy */
  } else {
    cvdls_mem->nje++;
    cvdls_mem->nstlj = cv_mem->cv_nst;
    cv_mem->cv_jcur = SUNTRUE;

    double *cv_y = NV_DATA_S(cv_mem->cv_y);
    offset_nrows = 0;
    for (int iDevice = sd->startDevice; iDevice < sd->endDevice; iDevice++) {
      cudaSetDevice(iDevice);
      sd->mGPU_d2 = &(sd->mGPUs_d2[iDevice]);
      mGPU = sd->mGPU_d2;

      //Ensure cv_y is loaded

      cudaMemcpy(cv_y+offset_nrows, mGPU->dcv_y, mGPU->nrows * sizeof(double), cudaMemcpyDeviceToHost);

      offset_nrows += mGPU->nrows;
    }
#ifdef DEBUG_linsolsetup_cuda_d2
    check_isnand(mGPU->A,mGPU->nnz,"prejac");
#endif

    retval = Jac(cv_mem->cv_tn, cv_mem->cv_y,cv_mem->cv_ftemp, cvdls_mem->A,cvdls_mem->J_data, vtemp1, vtemp2, vtemp3);

    //wrong
    //retval = Jac_cuda_d2(cv_mem->cv_tn, cv_mem->cv_y,cv_mem->cv_ftemp, cvdls_mem->A,cvdls_mem->J_data, vtemp1, vtemp2, vtemp3);

#ifdef DEBUG_linsolsetup_cuda_d2
    check_isnand(mGPU->A,mGPU->nnz,"postjac");
#endif

    if (retval < 0) {
      cvProcessError(cv_mem, CVDLS_JACFUNC_UNRECVR, "CVDLS",
                     "cvDlsSetup",  MSGD_JACFUNC_FAILED);
      cvdls_mem->last_flag = CVDLS_JACFUNC_UNRECVR;
      return(-1);
    }
    if (retval > 0) {
      cvdls_mem->last_flag = CVDLS_JACFUNC_RECVR;
      return(1);
    }

    retval = SUNMatCopy(cvdls_mem->A, cvdls_mem->savedJ);

    if (retval) {
      cvProcessError(cv_mem, CVDLS_SUNMAT_FAIL, "CVDLS",
                     "cvDlsSetup",  MSGD_MATCOPY_FAILED);
      cvdls_mem->last_flag = CVDLS_SUNMAT_FAIL;
      return(-1);
    }

  }

#ifdef LINSOLSOLVEGPU_INCLUDE_CUDAMEMCPY
  cudaEventRecord(mGPU->startBCGMemcpy);
#endif

  for (int iDevice = sd->startDevice; iDevice < sd->endDevice; iDevice++) {
    cudaSetDevice(iDevice);
    sd->mGPU_d2 = &(sd->mGPUs_d2[iDevice]);
    mGPU = sd->mGPU_d2;

    cudaMemcpyAsync(mGPU->diA, mGPU->iA, (mGPU->nrows + 1) * sizeof(int), cudaMemcpyHostToDevice, 0);
    cudaMemcpyAsync(mGPU->djA, mGPU->jA, mGPU->nnz * sizeof(int), cudaMemcpyHostToDevice, 0);
    cudaMemcpyAsync(mGPU->dA, mGPU->A, mGPU->nnz * sizeof(double), cudaMemcpyHostToDevice, 0);

  }
  cudaDeviceSynchronize();

#ifdef LINSOLSOLVEGPU_INCLUDE_CUDAMEMCPY
  cudaEventRecord(mGPU->stopBCGMemcpy);
  cudaEventSynchronize(mGPU->stopBCGMemcpy);
  float msBiConjGradMemcpy = 0.0;
  cudaEventElapsedTime(&msBiConjGradMemcpy, mGPU->startBCGMemcpy, mGPU->stopBCGMemcpy);
  mGPU->timeBiConjGradMemcpy+= msBiConjGradMemcpy/1000;
  mGPU->timeBiConjGrad+= msBiConjGradMemcpy/1000;
#endif

  for (int iDevice = sd->startDevice; iDevice < sd->endDevice; iDevice++) {
    cudaSetDevice(iDevice);
    sd->mGPU_d2 = &(sd->mGPUs_d2[iDevice]);
    mGPU = sd->mGPU_d2;

    gpu_matScaleAddI(mGPU->nrows,mGPU->dA,mGPU->djA,mGPU->diA,-cv_mem->cv_gamma,mGPU->blocks,mGPU->threads);
    cudaMemcpy(mGPU->A, mGPU->dA, mGPU->nnz * sizeof(double), cudaMemcpyDeviceToHost);
    gpu_diagprecond(mGPU->nrows,mGPU->dA,mGPU->djA,mGPU->diA,mGPU->ddiag,mGPU->blocks,mGPU->threads); //Setup linear solver

  }
#ifdef DEBUG_linsolsetup_cuda_d2
  cvcheck_input_globald<<<mGPU->blocks,mGPU->threads>>>(mGPU->ddiag,mGPU->nrows,"mGPU->ddiag");
#endif
  //printf("linsolsetup_cuda_d2 end\n");
  return retval;
}

void swapCSC_CSR_d2(int n_row, int n_col, int* Ap, int* Aj, double* Ax, int* Bp, int* Bi, double* Bx){

  int nnz=Ap[n_row];

  memset(Bp, 0, (n_row+1)*sizeof(int));

  for (int n = 0; n < nnz; n++){
    Bp[Aj[n]]++;
  }

  //cumsum the nnz per column to get Bp[]
  for(int col = 0, cumsum = 0; col < n_col; col++){
    int temp  = Bp[col];
    Bp[col] = cumsum;
    cumsum += temp;
  }
  Bp[n_col] = nnz;

  for(int row = 0; row < n_row; row++){
    for(int jj = Ap[row]; jj < Ap[row+1]; jj++){
      int col  = Aj[jj];
      int dest = Bp[col];

      Bi[dest] = row;
      Bx[dest] = Ax[jj];

      Bp[col]++;
    }
  }

  for(int col = 0, last = 0; col <= n_col; col++){
    int temp  = Bp[col];
    Bp[col] = last;
    last    = temp;
  }

}

void swapCSC_CSR_cuda_d2(SolverData *sd){
  ModelDataGPU_d2 *mGPU = sd->mGPU_d2;

  int n_row=mGPU->nrows;
  int n_col=mGPU->nrows;
  int nnz=mGPU->nnz;
  int* Ap=mGPU->iA;
  int* Aj=mGPU->jA;
  double* Ax=mGPU->A;
  int* Bp=(int*)malloc((mGPU->nrows+1)*sizeof(int));
  int* Bi=(int*)malloc(mGPU->nnz*sizeof(int));
  double* Bx=(double*)malloc(nnz*sizeof(double));

#ifdef DEV_swapCSC_CSR_cuda_d2
  printf("swapCSC_CSR_cuda_d2 start\n");
  int* Aj2=(int*)malloc(mGPU->nnz*sizeof(int));
  int* Ap2=(int*)malloc((mGPU->nrows+1)*sizeof(int));
  Ap2[0]=0;
  for(int j = 0; j<mGPU->n_cells; j++){
    for (int i = 0; i < mGPU->nnz/mGPU->n_cells; i++)
      Aj2[i+j*mGPU->nnz/mGPU->n_cells] = Aj[i]+mGPU->nrows*j;
    for (int i = 1; i < mGPU->nrows/mGPU->n_cells; i++)
      Ap2[i+j*mGPU->nrows/mGPU->n_cells] = Ap[i]+mGPU->nnz*j;
    }
  swapCSC_CSR_d2(n_row,n_col,Ap2,Aj2,Ax,Bp,Bi,Bx);
  cudaMemcpyAsync(mGPU->diA,Bp,(mGPU->nrows/mGPU->n_cells+1)*sizeof(int),cudaMemcpyHostToDevice, 0);
  cudaMemcpyAsync(mGPU->djA,Bi,mGPU->nnz/mGPU->n_cells*sizeof(int),cudaMemcpyHostToDevice, 0);
  cudaMemcpyAsync(mGPU->dA,Bx,mGPU->nnz*sizeof(double),cudaMemcpyHostToDevice, 0);
#else
  swapCSC_CSR_d2(n_row,n_col,Ap,Aj,Ax,Bp,Bi,Bx);
  cudaMemcpyAsync(mGPU->diA,Bp,(mGPU->nrows+1)*sizeof(int),cudaMemcpyHostToDevice, 0);
  cudaMemcpyAsync(mGPU->djA,Bi,mGPU->nnz*sizeof(int),cudaMemcpyHostToDevice, 0);
  cudaMemcpyAsync(mGPU->dA,Bx,mGPU->nnz*sizeof(double),cudaMemcpyHostToDevice, 0);
#endif

  free(Bp);
  free(Bi);
  free(Bx);
#ifdef DEV_swapCSC_CSR_cuda_d2
  printf("swapCSC_CSR_cuda_d2 end\n");
  free(Ap2);
  free(Aj2);
#endif
}

__global__
void solveBcgCuda_d2_66regs(
        double *dA, int *djA, int *diA, double *dx, double *dtempv //Input data
        ,int nrows, int blocks, int n_shr_empty, int maxIt
        ,int n_cells, double tolmax, double *ddiag //Init variables
        ,double *dr0, double *dr0h, double *dn0, double *dp0
        ,double *dt, double *ds, double *dy
)
{
 /*
  ModelDataGPU_d2 *md = &md_object;
  double* dA = md->dA;
  int* djA = md->djA;
  int* diA = md->diA;
  double* dx = md->dx;
  double* dtempv = md->dtempv;
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
  double* dy = md->dy;
*/

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int active_threads = nrows;
  if(tid<active_threads){
    double alpha,rho0,omega0,beta,rho1,temp1,temp2;
    alpha=rho0=omega0=beta=rho1=temp1=temp2=1.0;
    dp0[tid]=0.;
#ifndef CSR_SPMV_CPU
    cudaDeviceSpmvCSR(dr0,dx,dA,djA,diA); //y=A*x
#else
    cudaDeviceSpmvCSC_block(dr0,dx,dA,djA,diA,n_shr_empty)); //y=A*x
#endif
    dr0[tid]=dtempv[tid]-dr0[tid];
    dr0h[tid]=dr0[tid];
    int it=0;
    do{
      cudaDevicedotxy(dr0, dr0h, &rho1, n_shr_empty);
      beta = (rho1 / rho0) * (alpha / omega0);
      dp0[tid]=beta*dp0[tid]+dr0[tid]-omega0 * beta * dn0[tid];
      dy[tid]=ddiag[tid]*dp0[tid];
#ifndef CSR_SPMV_CPU
      cudaDeviceSpmvCSR(dn0, dy,dA, djA, diA);
#else
      cudaDeviceSpmvCSC_block(dn0, dy, dA, djA, diA,n_shr_empty);
#endif
      cudaDevicedotxy(dr0h, dn0, &temp1, n_shr_empty);
    alpha = rho1 / temp1;
    ds[tid]=dr0[tid]-alpha*dn0[tid];
    dx[tid]=alpha*dy[tid]+dx[tid];
    dy[tid]=ddiag[tid]*ds[tid];
#ifndef CSR_SPMV_CPU
      cudaDeviceSpmvCSR(dt, dy,dA, djA, diA);
#else
      cudaDeviceSpmvCSC_block(dt, dy, dA, djA, diA,n_shr_empty);
#endif
      dr0[tid]=ddiag[tid]*dt[tid];
      cudaDevicedotxy(dy, dr0, &temp1, n_shr_empty);
      cudaDevicedotxy(dr0, dr0, &temp2, n_shr_empty);
      omega0 = temp1 / temp2;
      dx[tid]=omega0*dy[tid]+dx[tid];
      dr0[tid]=ds[tid]-omega0*dt[tid];
      cudaDevicedotxy(dr0, dr0, &temp1, n_shr_empty);
      temp1 = sqrtf(temp1);
      rho0 = rho1;
      it++;
    } while(it<maxIt && temp1>tolmax);//while(it<maxIt && temp1>tolmax);//while(0);
  }
}

void solveGPU_block_thr_d2_66regs(int blocks, int threads_block, int n_shr_memory, int n_shr_empty, int offset_cells,
        SolverData *sd, int last_blockN)
{
  ModelDataGPU_d2 *mGPU = sd->mGPU_d2;
  //Init variables ("public")
  int nrows = mGPU->nrows;
  int nnz = mGPU->nnz;
  int n_cells = mGPU->n_cells;
  int maxIt = mGPU->maxIt;
  double tolmax = mGPU->tolmax;
  // Auxiliary vectors ("private")
  double *dr0 = mGPU->dr0;
  double *dr0h = mGPU->dr0h;
  double *dn0 = mGPU->dn0;
  double *dp0 = mGPU->dp0;
  double *dt = mGPU->dt;
  double *ds = mGPU->ds;
  double *dy = mGPU->dy;
  //Input variables
  int offset_nrows=(nrows/n_cells)*offset_cells;
  int offset_nnz=(nnz/n_cells)*offset_cells;
  int *djA=mGPU->djA;
  int *diA=mGPU->diA;
  double *dA=mGPU->dA+offset_nnz;
  double *ddiag=mGPU->ddiag+offset_nrows;
  double *dx=mGPU->dx+offset_nrows;
  double *dtempv=mGPU->dtempv+offset_nrows;

  solveBcgCuda_d2_66regs << < blocks, threads_block, n_shr_memory * sizeof(double) >> >
                                           (dA, djA, diA, dx, dtempv, nrows, blocks, n_shr_empty, maxIt, n_cells,
                                                   tolmax, ddiag, dr0, dr0h, dn0, dp0, dt, ds, dy
                                           );
}

int nextPowerOfTwo_d2(int v){
  v--;
  v |= v >> 1;
  v |= v >> 2;
  v |= v >> 4;
  v |= v >> 8;
  v |= v >> 16;
  v++;
  return v;
}

void solveBCGBlocks_d2_66regs(SolverData *sd, double *dA, int *djA, int *diA, double *dx, double *dtempv)
{
  ModelDataGPU_d2 *mGPU = sd->mGPU_d2;

#ifdef DEBUG_SOLVEBCGCUDA
    printf("solveGPUBlock start\n");
#endif

  int len_cell = mGPU->nrows/mGPU->n_cells;
  int max_threads_block=nextPowerOfTwo_d2(len_cell);
  int n_cells_block =  max_threads_block/len_cell;
  int threads_block = n_cells_block*len_cell;
  int n_shr_empty = max_threads_block-threads_block;
  int blocks = (mGPU->nrows+threads_block-1)/threads_block;
  int offset_cells=0;
  int last_blockN=0;

  solveGPU_block_thr_d2_66regs(blocks, threads_block, max_threads_block, n_shr_empty, offset_cells,
           sd, last_blockN);

#ifdef DEBUG_SOLVEBCGCUDA
    printf("solveGPUBlock end\n");
#endif

}

__noinline__ __device__ void cudaDevicezaxpbypc_d2(double* dz, double* dx,double* dy, double a, double b, int nrows)
{
  int row= threadIdx.x + blockDim.x*blockIdx.x;
  dz[row]=a*dz[row]  + dx[row] + b*dy[row];
}


__device__
void solveBcgCuda_d2(
        double *dA, int *djA, int *diA, double *dx, double *dtempv //Input data
        ,int nrows, int blocks, int n_shr_empty, int maxIt
        ,int n_cells, double tolmax, double *ddiag //Init variables
        ,double *dr0, double *dr0h, double *dn0, double *dp0
        ,double *dt, double *ds, double *dAx2, double *dy, double *dz// Auxiliary vectors
)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int active_threads = nrows;
  if(tid<active_threads){
    double alpha,rho0,omega0,beta,rho1,temp1,temp2;
    alpha=rho0=omega0=beta=rho1=temp1=temp2=1.0;
    dn0[tid]=0.;
    dp0[tid]=0.;
#ifndef CSR_SPMV_CPU
    cudaDeviceSpmvCSR(dr0,dx,dA,djA,diA); //y=A*x
#else
    cudaDeviceSpmvCSC_block(dr0,dx,dA,djA,diA,n_shr_empty)); //y=A*x
#endif
    //cudaDeviceaxpby(dr0,dtempv,1.0,-1.0,nrows);
    dr0[tid]=dtempv[tid]-dr0[tid];
    dr0h[tid]=dr0[tid];
    //cudaDeviceyequalsx(dr0h,dr0,nrows);
    int it=0;
    do{
      cudaDevicedotxy(dr0, dr0h, &rho1, n_shr_empty);
      beta = (rho1 / rho0) * (alpha / omega0);


    cudaDevicezaxpbypc(dp0, dr0, dn0, beta, -1.0 * omega0 * beta, nrows);   //z = ax + by + c
    //dp0[tid]=beta*dp0[tid]+dr0[tid]+ (-1.0)*omega0 * beta * dn0[tid];
            //cudaDevicezaxpbypc_d2(dp0, dr0, dn0, beta, -1.0 * omega0 * beta, nrows);   //z = ax + by + c



      cudaDevicemultxy(dy, ddiag, dp0, nrows);
      cudaDevicesetconst(dn0, 0.0, nrows);
#ifndef CSR_SPMV_CPU
      cudaDeviceSpmvCSR(dn0, dy, dA, djA, diA);
#else
      cudaDeviceSpmvCSC_block(dn0, dy, dA, djA, diA,n_shr_empty);
#endif
      cudaDevicedotxy(dr0h, dn0, &temp1, n_shr_empty);
      alpha = rho1 / temp1;
      cudaDevicezaxpby(1.0, dr0, -1.0 * alpha, dn0, ds, nrows);
      cudaDevicemultxy(dz, ddiag, ds, nrows); // precond z=diag*s
#ifndef CSR_SPMV_CPU
      cudaDeviceSpmvCSR(dt, dz, dA, djA, diA);
#else
      cudaDeviceSpmvCSC_block(dt, dz, dA, djA, diA,n_shr_empty);
#endif
      cudaDevicemultxy(dAx2, ddiag, dt, nrows);
      cudaDevicedotxy(dz, dAx2, &temp1, n_shr_empty);
      cudaDevicedotxy(dAx2, dAx2, &temp2, n_shr_empty);
      omega0 = temp1 / temp2;
      cudaDeviceaxpy(dx, dy, alpha, nrows); // x=alpha*y +x
      cudaDeviceaxpy(dx, dz, omega0, nrows);
      cudaDevicezaxpby(1.0, ds, -1.0 * omega0, dt, dr0, nrows);
      cudaDevicesetconst(dt, 0.0, nrows);
      cudaDevicedotxy(dr0, dr0, &temp1, n_shr_empty);
      temp1 = sqrtf(temp1);
      rho0 = rho1;
      it++;
    } while(it<maxIt && temp1>tolmax);//while(it<maxIt && temp1>tolmax);//while(0);
  }
}

__device__
void solveBcgCuda_d2_device(
        double *dA, int *djA, int *diA, double *dx, double *dtempv //Input data
        ,int nrows, int blocks, int n_shr_empty, int maxIt
        ,int n_cells, double tolmax, double *ddiag //Init variables
        ,double *dr0, double *dr0h, double *dn0, double *dp0
        ,double *dt, double *ds, double *dAx2, double *dy, double *dz// Auxiliary vectors
)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
solveBcgCuda_d2(dA, djA, diA, dx, dtempv, nrows, blocks, n_shr_empty, maxIt, n_cells,
tolmax, ddiag, dr0, dr0h, dn0, dp0, dt, ds, dAx2, dy, dz
);

}

__global__
void cudaGlobalCVode(
        double *dA, int *djA, int *diA, double *dx, double *dtempv //Input data
        ,int nrows, int blocks, int n_shr_empty, int maxIt
        ,int n_cells, double tolmax, double *ddiag //Init variables
        ,double *dr0, double *dr0h, double *dn0, double *dp0
        ,double *dt, double *ds, double *dAx2, double *dy, double *dz// Auxiliary vectors
)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
solveBcgCuda_d2_device(dA, djA, diA, dx, dtempv, nrows, blocks, n_shr_empty, maxIt, n_cells,
tolmax, ddiag, dr0, dr0h, dn0, dp0, dt, ds, dAx2, dy, dz
);

}

void solveGPU_block_thr_d2(int blocks, int threads_block, int n_shr_memory, int n_shr_empty, int offset_cells,
        SolverData *sd, int last_blockN)
{
  ModelDataGPU_d2 *mGPU = sd->mGPU_d2;
  //Init variables ("public")
  int nrows = mGPU->nrows;
  int nnz = mGPU->nnz;
  int n_cells = mGPU->n_cells;
  int maxIt = mGPU->maxIt;
  double tolmax = mGPU->tolmax;
  // Auxiliary vectors ("private")
  double *dr0 = mGPU->dr0;
  double *dr0h = mGPU->dr0h;
  double *dn0 = mGPU->dn0;
  double *dp0 = mGPU->dp0;
  double *dt = mGPU->dt;
  double *ds = mGPU->ds;
  double *dAx2 = mGPU->dAx2;
  double *dy = mGPU->dy;
  double *dz = mGPU->dz;

  //Input variables
  int offset_nrows=(nrows/n_cells)*offset_cells;
  int offset_nnz=(nnz/n_cells)*offset_cells;
  int *djA=mGPU->djA;
  int *diA=mGPU->diA;
  double *dA=mGPU->dA+offset_nnz;
  double *ddiag=mGPU->ddiag+offset_nrows;
  double *dx=mGPU->dx+offset_nrows;
  double *dtempv=mGPU->dtempv+offset_nrows;

  cudaGlobalCVode << < blocks, threads_block, n_shr_memory * sizeof(double) >> >
                                           (dA, djA, diA, dx, dtempv, nrows, blocks, n_shr_empty, maxIt, n_cells,
                                                   tolmax, ddiag, dr0, dr0h, dn0, dp0, dt, ds, dAx2, dy, dz
                                           );
}

void solveBCGBlocks_d2(SolverData *sd, double *dA, int *djA, int *diA, double *dx, double *dtempv)
{
  ModelDataGPU_d2 *mGPU = sd->mGPU_d2;

  int len_cell = mGPU->nrows/mGPU->n_cells;
  int max_threads_block=nextPowerOfTwo_d2(len_cell);

  int n_cells_block =  max_threads_block/len_cell;
  int threads_block = n_cells_block*len_cell;
  int n_shr_empty = max_threads_block-threads_block;
  int blocks = (mGPU->nrows+threads_block-1)/threads_block;

  int offset_cells=0;
  int last_blockN=0;

  solveGPU_block_thr_d2(blocks, threads_block, max_threads_block, n_shr_empty, offset_cells,
           sd, last_blockN);

}

int linsolsolve_cuda_d2(SolverData *sd, CVodeMem cv_mem)
{
  ModelDataGPU_d2 *mGPU;
  int offset_nrows = 0;
  int m, retval;
  realtype del, delp, dcon;

  //printf("linsolsolve_cuda_d2 start\n");

  cv_mem->cv_mnewt = m = 0;

  //Delp = del from last iter (reduce iterations)
  del = delp = 0.0;

  double *acor = NV_DATA_S(cv_mem->cv_acor);
  double *cv_y = NV_DATA_S(cv_mem->cv_y);
  double *tempv = NV_DATA_S(cv_mem->cv_tempv);
  double *cv_ftemp = NV_DATA_S(cv_mem->cv_ftemp);
  //CVDlsMem cvdls_mem = (CVDlsMem) cv_mem->cv_lmem;
  //double *x = NV_DATA_S(cvdls_mem->x);

  N_Vector b;
  b=cv_mem->cv_tempv;
  double *b_ptr=NV_DATA_S(b);

  // Looping point for Newton iteration
  for(;;) {

    offset_nrows = 0;
    for (int iDevice = sd->startDevice; iDevice < sd->endDevice; iDevice++) {
      cudaSetDevice(iDevice);
      sd->mGPU_d2 = &(sd->mGPUs_d2[iDevice]);
      mGPU = sd->mGPU_d2;

      // Evaluate the residual of the nonlinear system
      // a*x + b*y = z
      gpu_zaxpby(cv_mem->cv_rl1, (mGPU->dzn + 1 * mGPU->nrows), 1.0, mGPU->cv_acor, mGPU->dtempv, mGPU->nrows,
                 mGPU->blocks, mGPU->threads);
      gpu_zaxpby(cv_mem->cv_gamma, mGPU->dftemp, -1.0, mGPU->dtempv, mGPU->dtempv, mGPU->nrows, mGPU->blocks,
                 mGPU->threads);
      //N_VLinearSum(cv_mem->cv_rl1, cv_mem->cv_zn[1], ONE,
      //             cv_mem->cv_acor, cv_mem->cv_tempv);
      //N_VLinearSum(cv_mem->cv_gamma, cv_mem->cv_ftemp, -ONE,
      //             cv_mem->cv_tempv, cv_mem->cv_tempv);

#ifndef CSR_SPMV_CPU

      swapCSC_CSR_cuda_d2(sd);

#endif

    }

    offset_nrows = 0;
    for (int iDevice = sd->startDevice; iDevice < sd->endDevice; iDevice++) {
      cudaSetDevice(iDevice);
      sd->mGPU_d2 = &(sd->mGPUs_d2[iDevice]);
      mGPU = sd->mGPU_d2;
      solveBCGBlocks_d2(sd, mGPU->dA, mGPU->djA, mGPU->diA, mGPU->dx, mGPU->dtempv);
    }

    offset_nrows = 0;
    for (int iDevice = sd->startDevice; iDevice < sd->endDevice; iDevice++) {
      cudaSetDevice(iDevice);
      sd->mGPU_d2 = &(sd->mGPUs_d2[iDevice]);
      mGPU = sd->mGPU_d2;

#ifndef CSR_SPMV_CPU
      swapCSC_CSR_cuda_d2(sd);
#endif

      // Get WRMS norm of correction
      del = gpu_VWRMS_Norm(mGPU->nrows, mGPU->dx, mGPU->dewt, mGPU->aux, mGPU->dtempv2, (mGPU->blocks + 1) / 2, mGPU->threads);

      HANDLE_ERROR(cudaMemcpy(cv_ftemp+offset_nrows, mGPU->dftemp, mGPU->nrows * sizeof(double), cudaMemcpyDeviceToHost));
      cudaMemcpy(cv_y+offset_nrows, mGPU->dcv_y, mGPU->nrows * sizeof(double), cudaMemcpyDeviceToHost);
      HANDLE_ERROR(cudaMemcpy(b_ptr+offset_nrows, mGPU->dx, mGPU->nrows * sizeof(double), cudaMemcpyDeviceToHost));
      offset_nrows += mGPU->nrows;
    }

    if (cv_mem->cv_ghfun) {
      N_VLinearSum(ONE, cv_mem->cv_y, ONE, b, cv_mem->cv_ftemp);
      retval = cv_mem->cv_ghfun(cv_mem->cv_tn, ZERO, cv_mem->cv_ftemp,
                                cv_mem->cv_y, b, cv_mem->cv_user_data,
                                cv_mem->cv_tempv1, cv_mem->cv_tempv2);
      if (retval==1) {
      } else if (retval<0) {
        if ((!cv_mem->cv_jcur) && (cv_mem->cv_lsetup))
          return(TRY_AGAIN);
        else
          return(RHSFUNC_RECVR);
      }
    }
    // Check for negative concentrations
    N_VLinearSum(ONE, cv_mem->cv_y, ONE, b, cv_mem->cv_ftemp);
    if (N_VMin(cv_mem->cv_ftemp) < -CAMP_TINY) {
      return(CONV_FAIL);
    }
    offset_nrows = 0;
    for (int iDevice = sd->startDevice; iDevice < sd->endDevice; iDevice++) {
      cudaSetDevice(iDevice);
      sd->mGPU_d2 = &(sd->mGPUs_d2[iDevice]);
      mGPU = sd->mGPU_d2;

      cudaMemcpy(mGPU->dftemp, cv_ftemp+offset_nrows, mGPU->nrows * sizeof(double), cudaMemcpyHostToDevice);

      offset_nrows += mGPU->nrows;

    //cudaMemcpy(mGPU->dftemp,cv_mem->cv_tempv2,mGPU->nrows*sizeof(double),cudaMemcpyHostToDevice);

    //add correction to acor and y
    // a*x + b*y = z
    gpu_zaxpby(1.0, mGPU->cv_acor, 1.0, mGPU->dx, mGPU->cv_acor, mGPU->nrows, mGPU->blocks, mGPU->threads);
    gpu_zaxpby(1.0, mGPU->dzn, 1.0, mGPU->cv_acor, mGPU->dcv_y, mGPU->nrows, mGPU->blocks, mGPU->threads);
    }
    if (m > 0) {
      cv_mem->cv_crate = SUNMAX(0.3 * cv_mem->cv_crate, del / delp);
    }
    dcon = del * SUNMIN(1.0, cv_mem->cv_crate) / cv_mem->cv_tq[4];
    if (dcon <= 1.0) {

      offset_nrows = 0;
      for (int iDevice = sd->startDevice; iDevice < sd->endDevice; iDevice++) {
        cudaSetDevice(iDevice);
        sd->mGPU_d2 = &(sd->mGPUs_d2[iDevice]);
        mGPU = sd->mGPU_d2;

        cudaMemcpy(acor+offset_nrows,mGPU->cv_acor,mGPU->nrows*sizeof(double),cudaMemcpyDeviceToHost);
        HANDLE_ERROR(cudaMemcpy(tempv+offset_nrows,mGPU->dx,mGPU->nrows*sizeof(double),cudaMemcpyDeviceToHost));

        offset_nrows += mGPU->nrows;
      }

      cudaSetDevice(sd->startDevice); //todo cv_acnrm of all GPUs
      sd->mGPU_d2 = &(sd->mGPUs_d2[sd->startDevice]);
      mGPU = sd->mGPU_d2;
      //cv_mem->cv_acnrm = N_VWrmsNorm(cv_mem->cv_acor, cv_mem->cv_ewt);
      cv_mem->cv_acnrm = gpu_VWRMS_Norm(mGPU->nrows, mGPU->cv_acor, mGPU->dewt, mGPU->aux,
                                        mGPU->dtempv2, (mGPU->blocks + 1) / 2, mGPU->threads);
      cv_mem->cv_jcur = SUNFALSE;
      return (CV_SUCCESS);
    }
    cv_mem->cv_mnewt = ++m;
    if ((m == cv_mem->cv_maxcor) || ((m >= 2) && (del > RDIV * delp))) {
      offset_nrows = 0;
      for (int iDevice = sd->startDevice; iDevice < sd->endDevice; iDevice++) {
        cudaSetDevice(iDevice);
        sd->mGPU_d2 = &(sd->mGPUs_d2[iDevice]);
        mGPU = sd->mGPU_d2;

        cudaMemcpy(acor+offset_nrows,mGPU->cv_acor,mGPU->nrows*sizeof(double),cudaMemcpyDeviceToHost);
        HANDLE_ERROR(cudaMemcpy(tempv+offset_nrows,mGPU->dx,mGPU->nrows*sizeof(double),cudaMemcpyDeviceToHost));

        offset_nrows += mGPU->nrows;
      }

      if ((!cv_mem->cv_jcur) && (cv_mem->cv_lsetup)) {
        return (TRY_AGAIN);
      } else {
        return (CONV_FAIL);
      }
    }
    delp = del;
    offset_nrows = 0;
    for (int iDevice = sd->startDevice; iDevice < sd->endDevice; iDevice++) {
      cudaSetDevice(iDevice);
      sd->mGPU_d2 = &(sd->mGPUs_d2[iDevice]);
      mGPU = sd->mGPU_d2;

      HANDLE_ERROR(cudaMemcpy(cv_y+offset_nrows, mGPU->dcv_y, mGPU->nrows * sizeof(double), cudaMemcpyDeviceToHost));
      //retval = cv_mem->cv_f(cv_mem->cv_tn, cv_mem->cv_y,
      //                      cv_mem->cv_ftemp, cv_mem->cv_user_data);
      //int f(realtype t, N_Vector y, N_Vector deriv, void *solver_data)

      offset_nrows += mGPU->nrows;
    }

    retval = f(cv_mem->cv_tn, cv_mem->cv_y, cv_mem->cv_ftemp, cv_mem->cv_user_data);
    //retval = f_cuda_d2(cv_mem->cv_tn, cv_mem->cv_y, cv_mem->cv_ftemp, cv_mem->cv_user_data);

    ModelData *md = &(sd->model_data);
    double *total_state = md->total_state;
    double *deriv_data = N_VGetArrayPointer(cv_mem->cv_ftemp);
    for (int iDevice = sd->startDevice; iDevice < sd->endDevice; iDevice++) {
      cudaSetDevice(iDevice);
      sd->mGPU_d2 = &(sd->mGPUs_d2[iDevice]);
      mGPU = sd->mGPU_d2;

      HANDLE_ERROR(cudaMemcpy(mGPU->deriv_data, deriv_data, mGPU->deriv_size, cudaMemcpyHostToDevice));
      HANDLE_ERROR(cudaMemcpy(mGPU->state, total_state, mGPU->state_size, cudaMemcpyHostToDevice));

      total_state += mGPU->state_size_cell * mGPU->n_cells;
      deriv_data += mGPU->nrows;
    }
    for (int iDevice = sd->startDevice; iDevice < sd->endDevice; iDevice++) {
      cudaSetDevice(iDevice);
      sd->mGPU_d2 = &(sd->mGPUs_d2[iDevice]);
      mGPU = sd->mGPU_d2;

      //N_VLinearSum(ONE, cv_mem->cv_y, -ONE, cv_mem->cv_zn[0], cv_mem->cv_acor);
      // a*x + b*y = z
      gpu_zaxpby(1.0, mGPU->dcv_y, -1.0, mGPU->dzn, mGPU->cv_acor, mGPU->nrows, mGPU->blocks, mGPU->threads);
    }
    cv_mem->cv_nfe++;
    if (retval < 0){
      offset_nrows = 0;
      for (int iDevice = sd->startDevice; iDevice < sd->endDevice; iDevice++) {
        cudaSetDevice(iDevice);
        sd->mGPU_d2 = &(sd->mGPUs_d2[iDevice]);
        mGPU = sd->mGPU_d2;
        cudaMemcpy(acor+offset_nrows,mGPU->cv_acor,mGPU->nrows*sizeof(double),cudaMemcpyDeviceToHost);
        HANDLE_ERROR(cudaMemcpy(tempv+offset_nrows,mGPU->dx,mGPU->nrows*sizeof(double),cudaMemcpyDeviceToHost));

        offset_nrows += mGPU->nrows;
      }
      return(CV_RHSFUNC_FAIL);
    }
    if (retval > 0) {
      offset_nrows = 0;
      for (int iDevice = sd->startDevice; iDevice < sd->endDevice; iDevice++) {
        cudaSetDevice(iDevice);
        sd->mGPU_d2 = &(sd->mGPUs_d2[iDevice]);
        mGPU = sd->mGPU_d2;

        cudaMemcpy(acor+offset_nrows,mGPU->cv_acor,mGPU->nrows*sizeof(double),cudaMemcpyDeviceToHost);
        HANDLE_ERROR(cudaMemcpy(tempv+offset_nrows,mGPU->dx,mGPU->nrows*sizeof(double),cudaMemcpyDeviceToHost));

        offset_nrows += mGPU->nrows;
      }
      if ((!cv_mem->cv_jcur) && (cv_mem->cv_lsetup))
        return(TRY_AGAIN);
      else
        return(RHSFUNC_RECVR);
    }
  }
}

int cvNlsNewton_cuda_d2(SolverData *sd, CVodeMem cv_mem, int nflag)
{
  ModelData *md = &(sd->model_data);
  ModelDataGPU_d2 *mGPU;
  N_Vector vtemp1, vtemp2, vtemp3;
  int convfail, retval, ier;
  booleantype callSetup;

  double *acor = NV_DATA_S(cv_mem->cv_acor);
  double *cv_y = NV_DATA_S(cv_mem->cv_y);
  double *tempv = NV_DATA_S(cv_mem->cv_tempv);
  double *ftemp = NV_DATA_S(cv_mem->cv_ftemp);
  double *J_deriv =N_VGetArrayPointer(md->J_deriv);

  //printf("cvNlsNewton_cuda_d2 start \n");

  int offset_nrows = 0;
  for (int iDevice = sd->startDevice; iDevice < sd->endDevice; iDevice++) {
    cudaSetDevice(iDevice);
    sd->mGPU_d2 = &(sd->mGPUs_d2[iDevice]);
    mGPU = sd->mGPU_d2;

    cudaMemcpy(mGPU->cv_tq, cv_mem->cv_tq, 5 * sizeof(double), cudaMemcpyHostToDevice);
    int znUsedOnNewtonIt = 2;//Only used zn[0] and zn[1] //0.01s
    for (int i = 0; i < znUsedOnNewtonIt; i++) {//cv_qmax+1
      double *zn = NV_DATA_S(cv_mem->cv_zn[i])+offset_nrows;
      cudaMemcpy((i * mGPU->nrows + mGPU->dzn), zn, mGPU->nrows * sizeof(double), cudaMemcpyHostToDevice);
    }
    offset_nrows += mGPU->nrows;
  }

  //printf("cvNlsNewton_gpu start\n");

  /* Set flag convfail, input to lsetup for its evaluation decision */
  convfail = ((nflag == FIRST_CALL) || (nflag == PREV_ERR_FAIL)) ?
             CV_NO_FAILURES : CV_FAIL_OTHER;

  /* Decide whether or not to call setup routine (if one exists) */
  if (cv_mem->cv_lsetup) {
    callSetup = (nflag == PREV_CONV_FAIL) || (nflag == PREV_ERR_FAIL) ||
                (cv_mem->cv_nst == 0) ||
                (cv_mem->cv_nst >= cv_mem->cv_nstlp + MSBP) ||
                (SUNRabs(cv_mem->cv_gamrat-ONE) > DGMAX);
  } else {
    cv_mem->cv_crate = ONE;
    callSetup = SUNFALSE;
  }

  /* Call a user-supplied function to improve guesses for zn(0), if one exists */
  //if not, set to zero
  //N_VConst(ZERO, cv_mem->cv_acor_init);

  if (cv_mem->cv_ghfun) {

  //all are cpu pointers and gpu pointers are dftemp etc
  N_VLinearSum(ONE, cv_mem->cv_zn[0], -ONE, cv_mem->cv_last_yn, cv_mem->cv_ftemp);
  retval = cv_mem->cv_ghfun(cv_mem->cv_tn, cv_mem->cv_h, cv_mem->cv_zn[0],
                            cv_mem->cv_last_yn, cv_mem->cv_ftemp, cv_mem->cv_user_data,
                            cv_mem->cv_tempv, cv_mem->cv_acor_init);
  if (retval<0) return(RHSFUNC_RECVR);
  }

  offset_nrows = 0;
  for (int iDevice = sd->startDevice; iDevice < sd->endDevice; iDevice++) {
    cudaSetDevice(iDevice);
    sd->mGPU_d2 = &(sd->mGPUs_d2[iDevice]);
    mGPU = sd->mGPU_d2;

    cudaMemcpy(mGPU->cv_acor, acor+offset_nrows, mGPU->nrows * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(mGPU->dtempv, tempv+offset_nrows, mGPU->nrows * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(mGPU->dftemp, ftemp+offset_nrows, mGPU->nrows * sizeof(double), cudaMemcpyHostToDevice);

    offset_nrows += mGPU->nrows;
  }

  //remove temps, not used in jac
  vtemp1 = cv_mem->cv_acor;  /* rename acor as vtemp1 for readability  */
  vtemp2 = cv_mem->cv_acor;  /* rename y as vtemp2 for readability     */
  vtemp3 = cv_mem->cv_acor;  /* rename tempv as vtemp3 for readability */

  /* Looping point for the solution of the nonlinear system.
     Evaluate f at the predicted y, call lsetup if indicated, and
     call cvNewtonIteration for the Newton iteration itself.      */
  for(;;) {

    offset_nrows = 0;
    for (int iDevice = sd->startDevice; iDevice < sd->endDevice; iDevice++) {
      cudaSetDevice(iDevice);
      sd->mGPU_d2 = &(sd->mGPUs_d2[iDevice]);
      mGPU = sd->mGPU_d2;
      /* Load prediction into y vector */
      //N_VLinearSum(ONE, cv_mem->cv_zn[0], ONE, cv_mem->cv_acor_init, cv_mem->cv_y);
      cudaDeviceSynchronize();
      gpu_yequalsx(mGPU->dcv_y,mGPU->dzn, mGPU->nrows, mGPU->blocks, mGPU->threads);//Consider acor_init=0
      cudaDeviceSynchronize();

      //copy cv_y to enable debug on cpu
      HANDLE_ERROR(cudaMemcpy(cv_y+offset_nrows, mGPU->dcv_y, mGPU->nrows * sizeof(double), cudaMemcpyDeviceToHost));
      offset_nrows += mGPU->nrows;
    }

    retval = f(cv_mem->cv_tn, cv_mem->cv_y, cv_mem->cv_ftemp, cv_mem->cv_user_data);
    //retval = f_cuda_d2(cv_mem->cv_tn, cv_mem->cv_y, cv_mem->cv_ftemp, cv_mem->cv_user_data);

    double *total_state = md->total_state;
    double *deriv_data = N_VGetArrayPointer(cv_mem->cv_ftemp);
    for (int iDevice = sd->startDevice; iDevice < sd->endDevice; iDevice++) {
      cudaSetDevice(iDevice);
      sd->mGPU_d2 = &(sd->mGPUs_d2[iDevice]);
      mGPU = sd->mGPU_d2;

      HANDLE_ERROR(cudaMemcpy(mGPU->deriv_data, deriv_data, mGPU->deriv_size, cudaMemcpyHostToDevice));
      HANDLE_ERROR(cudaMemcpy(mGPU->state, total_state, mGPU->state_size, cudaMemcpyHostToDevice));

      total_state += mGPU->state_size_cell * mGPU->n_cells;
      deriv_data += mGPU->nrows;
    }

    if (retval < 0) return(CV_RHSFUNC_FAIL);
    if (retval > 0) return(RHSFUNC_RECVR);

    cv_mem->cv_nfe++;
    if (callSetup)
    {
      ier = linsolsetup_cuda_d2(sd, cv_mem, convfail, vtemp1, vtemp2, vtemp3);
      cv_mem->cv_nsetups++;
      callSetup = SUNFALSE;
      cv_mem->cv_gamrat = cv_mem->cv_crate = ONE;
      cv_mem->cv_gammap = cv_mem->cv_gamma;
      cv_mem->cv_nstlp = cv_mem->cv_nst;
      // Return if lsetup failed
      if (ier < 0) return(CV_LSETUP_FAIL);
      if (ier > 0) return(CONV_FAIL);
    }

    for (int iDevice = sd->startDevice; iDevice < sd->endDevice; iDevice++) {
      cudaSetDevice(iDevice);
      sd->mGPU_d2 = &(sd->mGPUs_d2[iDevice]);
      mGPU = sd->mGPU_d2;

      cudaMemset(mGPU->cv_acor, 0.0, mGPU->nrows * sizeof(double));
    }
    ier = linsolsolve_cuda_d2(sd, cv_mem);
    //printf("linsolsolve_cuda_d2 end\n");
    if (ier != TRY_AGAIN) return(ier);
    callSetup = SUNTRUE;
    convfail = CV_FAIL_BAD_J;
  }
}

int cvStep_cuda_d2(SolverData *sd, CVodeMem cv_mem)
{
  ModelDataGPU_d2*mGPU = sd->mGPU_d2;
  realtype saved_t, dsm;
  int ncf, nef;
  int nflag, kflag, eflag;
  //printf("cvStep_gpu start\n");
  double *ewt = NV_DATA_S(cv_mem->cv_ewt);
  int offset_nrows = 0;
  for (int iDevice = sd->startDevice; iDevice < sd->endDevice; iDevice++) {
    cudaSetDevice(iDevice);
    sd->mGPU_d2 = &(sd->mGPUs_d2[iDevice]);
    mGPU = sd->mGPU_d2;

    cudaMemcpy(mGPU->dewt, ewt+offset_nrows, mGPU->nrows * sizeof(double), cudaMemcpyHostToDevice);

    offset_nrows += mGPU->nrows;
  }

  saved_t = cv_mem->cv_tn;
  ncf = nef = 0;
  nflag = FIRST_CALL;

  if ((cv_mem->cv_nst > 0) && (cv_mem->cv_hprime != cv_mem->cv_h))
    cvAdjustParams_cuda_d2(cv_mem);

  /* Looping point for attempts to take a step */
  for(;;) {

    cvPredict_cuda_d2(cv_mem);

    cvSet_cuda_d2(cv_mem);

    nflag = cvNlsNewton_cuda_d2(sd, cv_mem, nflag);//f(y)+BCG

    kflag = cvHandleNFlag_cuda_d2(cv_mem, &nflag, saved_t, &ncf);

    /* Go back in loop if we need to predict again (nflag=PREV_CONV_FAIL)*/
    if (kflag == PREDICT_AGAIN) continue;

    /* Return if nonlinear solve failed and recovery not possible. */
    if (kflag != DO_ERROR_TEST) return(kflag);

    /* Perform error test (nflag=CV_SUCCESS) */
    eflag = cvDoErrorTest_cuda_d2(cv_mem, &nflag, saved_t, &nef, &dsm);

    /* Go back in loop if we need to predict again (nflag=PREV_ERR_FAIL) */
    if (eflag == TRY_AGAIN)  continue;

    /* Return if error test failed and recovery not possible. */
    if (eflag != CV_SUCCESS) return(eflag);

    /* Error test passed (eflag=CV_SUCCESS), break from loop */
    break;

  }

  /* Nonlinear system solve and error test were both successful.
     Update data, and consider change of step and/or order.       */

  cvCompleteStep_cuda_d2(cv_mem);

  cvPrepareNextStep_cuda_d2(cv_mem, dsm);//use tq calculated in cvset and tempv calc in cvnewton

  /* If Stablilty Limit Detection is turned on, call stability limit
     detection routine for possible order reduction. */

  if (cv_mem->cv_sldeton) cvBDFStab_cuda_d2(cv_mem);

  cv_mem->cv_etamax = (cv_mem->cv_nst <= SMALL_NST) ? ETAMX2 : ETAMX3;

  /*  Finally, we rescale the acor array to be the
      estimated local error vector. */

  N_VScale(cv_mem->cv_tq[2], cv_mem->cv_acor, cv_mem->cv_acor);
  return(CV_SUCCESS);

}

int cudaCVode_d2(void *cvode_mem, realtype tout, N_Vector yout,
          realtype *tret, int itask, SolverData *sd)
{
  CVodeMem cv_mem;
  long int nstloc;
  int retval, hflag, kflag, istate, ir, ier, irfndp;
  int ewtsetOK;
  realtype troundoff, tout_hin, rh, nrm;
  booleantype inactive_roots;

  ModelDataGPU_d2 *mGPU;

  /*
   * -------------------------------------
   * 1. Check and process inputs
   * -------------------------------------
   */

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

  /*
   * ----------------------------------------
   * 2. Initializations performed only at
   *    the first step (nst=0)
   * ----------------------------------------
   */

  if (cv_mem->cv_nst == 0) {
    cv_mem->cv_tretlast = *tret = cv_mem->cv_tn;
    ier = cvInitialSetup_cuda_d2(cv_mem);
    if (ier!= CV_SUCCESS) return(ier);
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
      hflag = cvHin_cuda_d2(cv_mem, tout_hin); //set cv_y
      if (hflag != CV_SUCCESS) {
        istate = cvHandleFailure_cuda_d2(cv_mem, hflag);
        return(istate);
      }
    }
    rh = SUNRabs(cv_mem->cv_h)*cv_mem->cv_hmax_inv;
    if (rh > ONE) cv_mem->cv_h /= rh;
    if (SUNRabs(cv_mem->cv_h) < cv_mem->cv_hmin)
      cv_mem->cv_h *= cv_mem->cv_hmin/SUNRabs(cv_mem->cv_h);
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
      retval = cvRcheck1_cuda_d2(cv_mem);
      if (retval == CV_RTFUNC_FAIL) {
        cvProcessError(cv_mem, CV_RTFUNC_FAIL, "CVODE", "cvRcheck1",
                       MSGCV_RTFUNC_FAILED, cv_mem->cv_tn);
        return(CV_RTFUNC_FAIL);
      }
    }
  } /* end of first call block */

  /*
   * ------------------------------------------------------
   * 3. At following steps, perform stop tests
   * -------------------------------------------------------
   */

  if (cv_mem->cv_nst > 0) {
    troundoff = FUZZ_FACTOR*cv_mem->cv_uround*(SUNRabs(cv_mem->cv_tn) + SUNRabs(cv_mem->cv_h));
    if (cv_mem->cv_nrtfn > 0) {

      irfndp = cv_mem->cv_irfnd;

      retval = cvRcheck2_cuda_d2(cv_mem);

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

        retval = cvRcheck3_cuda_d2(cv_mem);

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

    /* In CV_NORMAL mode, test if tout was reached */
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

    /* In CV_ONE_STEP mode, test if tn was returned */
    if ( itask == CV_ONE_STEP &&
         SUNRabs(cv_mem->cv_tn - cv_mem->cv_tretlast) > troundoff ) {
      cv_mem->cv_tretlast = *tret = cv_mem->cv_tn;
      N_VScale(ONE, cv_mem->cv_zn[0], yout);
      return(CV_SUCCESS);
    }

    /* Test for tn at tstop or near tstop */
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

  /*
   * --------------------------------------------------
   * 4. Looping point for internal steps
   * --------------------------------------------------
   */

  ModelDataCPU_d2 *mCPU  = &sd->mCPU;
#ifdef CAMP_DEBUG_GPU
  cudaSetDevice(sd->startDevice);
  cudaEventRecord(mCPU->startcvStep);
#endif

  ModelData *md = &(sd->model_data);
  int offset_state = 0;
  int offset_ncells = 0;
  int offset_nrows = 0;
  for (int iDevice = sd->startDevice; iDevice < sd->endDevice; iDevice++) {
    cudaSetDevice(iDevice);
    sd->mGPU_d2 = &(sd->mGPUs_d2[iDevice]);
    mGPU = sd->mGPU_d2;
    HANDLE_ERROR(cudaMemcpyAsync(mGPU->state, md->total_state+offset_state, mGPU->state_size, cudaMemcpyHostToDevice, 0));

    offset_state += mGPU->state_size_cell * mGPU->n_cells;
    offset_ncells += mGPU->n_cells;
    offset_nrows += mGPU->nrows;
  }
  nstloc = 0;
  for(;;) {

#ifdef CAMP_DEBUG_GPU
#ifdef CAMP_PROFILE_DEVICE_FUNCTIONS

    for (int iDevice = sd->startDevice; iDevice < sd->endDevice; iDevice++) {
      cudaSetDevice(iDevice);
      sd->mGPU_d2 = &(sd->mGPUs_d2[iDevice]);
      mGPU = sd->mGPU_d2;

      mGPU->mdvCPU.countercvStep++;
    }
   #endif
#endif

    cv_mem->cv_next_h = cv_mem->cv_h;
    cv_mem->cv_next_q = cv_mem->cv_q;

    /* Reset and check ewt */
    if (cv_mem->cv_nst > 0) {

      ewtsetOK = cv_mem->cv_efun(cv_mem->cv_zn[0], cv_mem->cv_ewt, cv_mem->cv_e_data);
      //set here copy of ewt to gpu

      if (ewtsetOK != 0) {

        if (cv_mem->cv_itol == CV_WF)
          cvProcessError(cv_mem, CV_ILL_INPUT, "CVODE", "CVode",
                         MSGCV_EWT_NOW_FAIL, cv_mem->cv_tn);
        else
          cvProcessError(cv_mem, CV_ILL_INPUT, "CVODE", "CVode",
                         MSGCV_EWT_NOW_BAD, cv_mem->cv_tn);

        istate = CV_ILL_INPUT;
        cv_mem->cv_tretlast = *tret = cv_mem->cv_tn;
        N_VScale(ONE, cv_mem->cv_zn[0], yout);
        break;

      }
    }

    /* Check for too many steps */
    if ( (cv_mem->cv_mxstep>0) && (nstloc >= cv_mem->cv_mxstep) ) {
      cvProcessError(cv_mem, CV_TOO_MUCH_WORK, "CVODE", "CVode",
                     MSGCV_MAX_STEPS, cv_mem->cv_tn);
      istate = CV_TOO_MUCH_WORK;
      cv_mem->cv_tretlast = *tret = cv_mem->cv_tn;
      N_VScale(ONE, cv_mem->cv_zn[0], yout);
      break;
    }

    /* Check for too much accuracy requested */
    nrm = N_VWrmsNorm(cv_mem->cv_zn[0], cv_mem->cv_ewt);
    cv_mem->cv_tolsf = cv_mem->cv_uround * nrm;
    if (cv_mem->cv_tolsf > ONE) {
      cvProcessError(cv_mem, CV_TOO_MUCH_ACC, "CVODE", "CVode",
                     MSGCV_TOO_MUCH_ACC, cv_mem->cv_tn);
      istate = CV_TOO_MUCH_ACC;
      cv_mem->cv_tretlast = *tret = cv_mem->cv_tn;
      N_VScale(ONE, cv_mem->cv_zn[0], yout);
      cv_mem->cv_tolsf *= TWO;
      break;
    } else {
      cv_mem->cv_tolsf = ONE;
    }

    /* Check for h below roundoff level in tn */
    if (cv_mem->cv_tn + cv_mem->cv_h == cv_mem->cv_tn) {
      cv_mem->cv_nhnil++;
      if (cv_mem->cv_nhnil <= cv_mem->cv_mxhnil)
        cvProcessError(cv_mem, CV_WARNING, "CVODE", "CVode",
                       MSGCV_HNIL, cv_mem->cv_tn, cv_mem->cv_h);
      if (cv_mem->cv_nhnil == cv_mem->cv_mxhnil)
        cvProcessError(cv_mem, CV_WARNING, "CVODE", "CVode", MSGCV_HNIL_DONE);
    }

    /* Call cvStep to take a step */
    //kflag = cvStep(cv_mem);
    kflag = cvStep_cuda_d2(sd, cv_mem);

    /* Process failed step cases, and exit loop */
    if (kflag != CV_SUCCESS) {
      istate = cvHandleFailure_cuda_d2(cv_mem, kflag);
      cv_mem->cv_tretlast = *tret = cv_mem->cv_tn;
      N_VScale(ONE, cv_mem->cv_zn[0], yout);
      break;
    }

    nstloc++;

    /* Check for root in last step taken. */
    if (cv_mem->cv_nrtfn > 0) {

      retval = cvRcheck3_cuda_d2(cv_mem);

      if (retval == RTFOUND) {  /* A new root was found */
        cv_mem->cv_irfnd = 1;
        istate = CV_ROOT_RETURN;
        cv_mem->cv_tretlast = *tret = cv_mem->cv_tlo;
        break;
      } else if (retval == CV_RTFUNC_FAIL) { /* g failed */
        cvProcessError(cv_mem, CV_RTFUNC_FAIL, "CVODE", "cvRcheck3",
                       MSGCV_RTFUNC_FAILED, cv_mem->cv_tlo);
        istate = CV_RTFUNC_FAIL;
        break;
      }
      if (cv_mem->cv_nst==1) {
        inactive_roots = SUNFALSE;
        for (ir=0; ir<cv_mem->cv_nrtfn; ir++) {
          if (!cv_mem->cv_gactive[ir]) {
            inactive_roots = SUNTRUE;
            break;
          }
        }
        if ((cv_mem->cv_mxgnull > 0) && inactive_roots) {
          cvProcessError(cv_mem, CV_WARNING, "CVODES", "CVode",
                         MSGCV_INACTIVE_ROOTS);
        }
      }
    }

    if ( (itask == CV_NORMAL) &&  (cv_mem->cv_tn-tout)*cv_mem->cv_h >= ZERO ) {
      istate = CV_SUCCESS;
      cv_mem->cv_tretlast = *tret = tout;
      (void) CVodeGetDky(cv_mem, tout, 0, yout);
      cv_mem->cv_next_q = cv_mem->cv_qprime;
      cv_mem->cv_next_h = cv_mem->cv_hprime;
      break;
    }
    if ( cv_mem->cv_tstopset ) {
      troundoff = FUZZ_FACTOR*cv_mem->cv_uround*(SUNRabs(cv_mem->cv_tn) + SUNRabs(cv_mem->cv_h));
      if ( SUNRabs(cv_mem->cv_tn - cv_mem->cv_tstop) <= troundoff) {
        (void) CVodeGetDky(cv_mem, cv_mem->cv_tstop, 0, yout);
        cv_mem->cv_tretlast = *tret = cv_mem->cv_tstop;
        cv_mem->cv_tstopset = SUNFALSE;
        istate = CV_TSTOP_RETURN;
        break;
      }
      if ( (cv_mem->cv_tn + cv_mem->cv_hprime - cv_mem->cv_tstop)*cv_mem->cv_h > ZERO ) {
        cv_mem->cv_hprime = (cv_mem->cv_tstop - cv_mem->cv_tn)*(ONE-FOUR*cv_mem->cv_uround);
        cv_mem->cv_eta = cv_mem->cv_hprime/cv_mem->cv_h;
      }
    }
    if (itask == CV_ONE_STEP) {
      istate = CV_SUCCESS;
      cv_mem->cv_tretlast = *tret = cv_mem->cv_tn;
      N_VScale(ONE, cv_mem->cv_zn[0], yout);
      cv_mem->cv_next_q = cv_mem->cv_qprime;
      cv_mem->cv_next_h = cv_mem->cv_hprime;
      break;
    }
  } /* end looping for internal steps */

#ifdef CAMP_DEBUG_GPU
  for (int iDevice = sd->startDevice+1; iDevice < sd->endDevice; iDevice++) {
    cudaSetDevice(iDevice);
    cudaDeviceSynchronize();
  }
  cudaSetDevice(sd->startDevice);
  cudaEventRecord(mCPU->stopcvStep);
  cudaEventSynchronize(mCPU->stopcvStep);
  float mscvStep = 0.0;
  cudaEventElapsedTime(&mscvStep, mCPU->startcvStep, mCPU->stopcvStep);
  mCPU->timecvStep+= mscvStep/1000;
#endif

  return(istate);
}

void set_jac_data_cuda_d2(SolverData *sd, double *J){
  ModelData *md = &(sd->model_data);
  ModelDataGPU_d2 *mGPU;
  int offset_nnz_J_solver = 0;
  int offset_nrows = 0;
  for (int iDevice = sd->startDevice; iDevice < sd->endDevice; iDevice++) {
    cudaSetDevice(iDevice);
    sd->mGPU_d2 = &(sd->mGPUs_d2[iDevice]);
    mGPU = sd->mGPU_d2;
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

#ifdef DEV_swapCSC_CSR_cuda_d2
    cudaMemcpy(mGPU->djA, mGPU->jA, mGPU->nnz/mGPU->n_cells * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(mGPU->diA, mGPU->iA, (mGPU->nrows/mGPU->n_cells + 1) * sizeof(int), cudaMemcpyHostToDevice);
#else
    cudaMemcpy(mGPU->djA, mGPU->jA, mGPU->nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(mGPU->diA, mGPU->iA, (mGPU->nrows + 1) * sizeof(int), cudaMemcpyHostToDevice);
#endif
  }
}

void camp_solver_update_model_state_cuda_d2(N_Vector solver_state, SolverData *sd,
                                       double threshhold, double replacement_value)
{
  ModelData *md = &(sd->model_data);
  ModelDataGPU_d2 *mGPU;
  double *total_state = md->total_state;
  //printf("camp_solver_update_model_state_cuda_d2 start \n");
  for (int iDevice = sd->startDevice; iDevice < sd->endDevice; iDevice++) {
    cudaSetDevice(iDevice);
    sd->mGPU_d2 = &(sd->mGPUs_d2[iDevice]);
    mGPU = sd->mGPU_d2;
    double *J_state = N_VGetArrayPointer(md->J_state);
    HANDLE_ERROR(cudaMemcpy(mGPU->state, total_state, mGPU->state_size, cudaMemcpyHostToDevice));
    total_state += mGPU->state_size_cell * mGPU->n_cells;
  }
}

void solver_get_statistics_cuda_d2(SolverData *sd){
  cudaSetDevice(sd->startDevice);
  sd->mGPU_d2 = &(sd->mGPUs_d2[sd->startDevice]);
  ModelDataGPU_d2 *mGPU = sd->mGPU_d2;

/*
  cudaMemcpy(&mGPU->mdvCPU,mGPU->mdvo,sizeof(ModelDataVariable),cudaMemcpyDeviceToHost);
  ModelDataGPU *mGPU_max = sd->mGPU;

  //printf("solver_get_statistics_gpu\n");

  for (int iDevice = sd->startDevice+1; iDevice < sd->endDevice; iDevice++) {
    cudaSetDevice(iDevice);
    sd->mGPU_d2 = &(sd->mGPUs_d2[iDevice]);
    mGPU = sd->mGPU_d2;

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
*/
}

void solver_reset_statistics_cuda_d2(SolverData *sd){
  ModelDataGPU_d2 *mGPU = sd->mGPU_d2;
  //printf("solver_reset_statistics_cuda_d2\n");

/*
for (int iDevice = sd->startDevice; iDevice < sd->endDevice; iDevice++) {
  cudaSetDevice(iDevice);
  sd->mGPU_d2 = &(sd->mGPUs_d2[iDevice]);
  mGPU = sd->mGPU_d2;

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
*/
}

void free_gpu_cu_d2(SolverData *sd) {

  ModelDataGPU_d2 *mGPU = sd->mGPU_d2;

  //printf("free_gpu_cu start\n");

  free(sd->flagCells);

}

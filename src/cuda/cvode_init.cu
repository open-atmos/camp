/* Copyright (C) 2021 Barcelona Supercomputing Center and University of
 * Illinois at Urbana-Champaign
 * SPDX-License-Identifier: MIT
 */

#include "cvode_cuda.h"
extern "C" {
#include "cvode_gpu.h"
}
#ifdef CAMP_USE_MPI
#include <mpi.h>
#endif

int jacobian_initialize_cuda_cvode(SolverData *sd) {
  ModelDataGPU *mGPU = sd->mGPU;
  Jacobian *jac = &sd->jac;
#ifdef DEBUG_jacobian_initialize_gpu
  printf("jacobian_initialize_gpu start \n");
#endif
  mGPU = sd->mGPU;
  JacobianGPU *jacgpu = &(mGPU->jac);
  cudaMalloc((void **) &jacgpu->num_elem, 1 * sizeof(jacgpu->num_elem));
  cudaMemcpy(jacgpu->num_elem, &jac->num_elem, 1 * sizeof(jacgpu->num_elem), cudaMemcpyHostToDevice);
  int num_elem = jac->num_elem * mGPU->n_cells;
  cudaMalloc((void **) &(jacgpu->production_partials), num_elem * sizeof(double));
  HANDLE_ERROR(cudaMalloc((void **) &(jacgpu->loss_partials), num_elem * sizeof(double)));
  double *aux=(double*)malloc(sizeof(double)*num_elem);
  for (int i = 0; i < num_elem; i++) {
    aux[i]=0.;
  }
  HANDLE_ERROR(cudaMemcpy(jacgpu->production_partials, aux, num_elem * sizeof(double), cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(jacgpu->loss_partials, aux, num_elem * sizeof(double), cudaMemcpyHostToDevice));
#ifdef DEBUG_jacobian_initialize_gpu
  printf("jacobian_initialize_gpu end \n");
#endif
  return 1;
}

void init_jac_cuda_cvode(SolverData *sd){
  ModelData *md = &(sd->model_data);
  ModelDataGPU *mGPU;
  ModelDataCPU *mCPU = &(sd->mCPU);
#ifdef DEBUG_init_jac_cuda
  printf("init_jac_cuda start \n");
#endif
  mGPU = sd->mGPU;
  mCPU->jac_size = md->n_per_cell_solver_jac_elem * mGPU->n_cells * sizeof(double);
  mCPU->nnz_J_solver = SM_NNZ_S(md->J_solver);
  cudaMalloc((void **) &mGPU->dA, mCPU->jac_size);
  cudaMalloc((void **) &mGPU->J_solver, mCPU->jac_size);
  cudaMalloc((void **) &mGPU->J_state, mCPU->deriv_size);
  cudaMalloc((void **) &mGPU->J_deriv, mCPU->deriv_size);
  cudaMalloc((void **) &mGPU->J_tmp, mCPU->deriv_size);
  cudaMalloc((void **) &mGPU->J_tmp2, mCPU->deriv_size);
  cudaMalloc((void **) &mGPU->jac_map, sizeof(JacMap) * md->n_mapped_values);
  HANDLE_ERROR(cudaMalloc((void **) &mGPU->n_mapped_values, 1 * sizeof(int)));
#ifdef DEBUG_init_jac_cuda
  printf("md->n_per_cell_dep_var %d sd->jac.num_spec %d md->n_per_cell_solver_jac_elem %d "
         "md->n_mapped_values %d jac->num_elem %d  mCPU->nnz_J_solver %d "
         "mCPU->jac_size/sizeof(double) %d SM_NNZ_S(sd->J) %d\n",
         md->n_per_cell_dep_var,sd->jac.num_spec,md->n_per_cell_solver_jac_elem, md->n_mapped_values,
         sd->jac.num_elem,mCPU->nnz_J_solver,mCPU->jac_size/sizeof(double),
         SM_NNZ_S(sd->J));
#endif
  double *J = SM_DATA_S(sd->J);
  HANDLE_ERROR(cudaMemcpy(mGPU->dA, J, mCPU->jac_size, cudaMemcpyHostToDevice));
  double *J_solver = SM_DATA_S(md->J_solver);
  cudaMemcpy(mGPU->J_solver, J_solver, mCPU->jac_size, cudaMemcpyHostToDevice);
  double *J_state = N_VGetArrayPointer(md->J_state);
  HANDLE_ERROR(cudaMemcpy(mGPU->J_state, J_state, mCPU->deriv_size, cudaMemcpyHostToDevice));
  double *J_deriv = N_VGetArrayPointer(md->J_deriv);
  HANDLE_ERROR(cudaMemcpy(mGPU->J_deriv, J_deriv, mCPU->deriv_size, cudaMemcpyHostToDevice));
  double *J_tmp2 = N_VGetArrayPointer(md->J_tmp2);
  HANDLE_ERROR(cudaMemcpy(mGPU->J_tmp2, J_tmp2, mCPU->deriv_size, cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(mGPU->jac_map, md->jac_map, sizeof(JacMap) * md->n_mapped_values, cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(mGPU->n_mapped_values, &md->n_mapped_values, 1 * sizeof(int), cudaMemcpyHostToDevice));
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
  cudaMalloc((void **) &mGPU->rxn_int, (md->n_rxn_int_param + md->n_rxn)*sizeof(int));
  cudaMalloc((void **) &mGPU->rxn_double, md->n_rxn_float_param*sizeof(double));
#ifdef REVERSE_INT_FLOAT_MATRIX
#else
  cudaMalloc((void **) &mGPU->rxn_int_indices, (md->n_rxn+1)*sizeof(int));
  cudaMalloc((void **) &mGPU->rxn_float_indices, (md->n_rxn+1)*sizeof(int));
#endif
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
  ModelDataCPU *mCPU = &(sd->mCPU);
  set_int_double_cuda_cvode(
      md->n_rxn, mCPU->rxn_env_data_idx_size,
      md->rxn_int_data, md->rxn_float_data,
      md->rxn_int_indices, md->rxn_float_indices, md->rxn_env_idx,
      sd
  );
}

void solver_new_gpu_cu_cvode(SolverData *sd) {
  ModelData *md = &(sd->model_data);
  ModelDataGPU *mGPU;
  ModelDataCPU *mCPU = &(sd->mCPU);
  int n_dep_var = md->n_per_cell_dep_var;
  int n_state_var = md->n_per_cell_state_var;
  int n_rxn = md->n_rxn;
  int n_rxn_env_param = md->n_rxn_env_data;
  int n_cells = md->n_cells;
#ifdef OLD_DEV_CPUGPU
  sd->n_cells_total = md->n_cells;
  n_cells *= sd->nCellsGPUPerc/10.;
  md->n_cells=n_cells;
#endif
  mCPU->state_size = n_state_var * n_cells * sizeof(double);
  mCPU->deriv_size = n_dep_var * n_cells * sizeof(double);
  mCPU->env_size = CAMP_NUM_ENV_PARAM_ * n_cells * sizeof(double); //Temp and pressure
  mCPU->rxn_env_data_size = n_rxn_env_param * n_cells * sizeof(double);
  mCPU->rxn_env_data_idx_size = (n_rxn+1) * sizeof(int);
  mCPU->map_state_deriv_size = n_dep_var * n_cells * sizeof(int);
  int coresPerNode = 40;
  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size > 40 && size % coresPerNode != 0) {
    printf("ERROR: MORE THAN 40 MPI PROCESSES AND NOT MULTIPLE OF 40, WHEN CTE-POWER ONLY HAS 40 CORES PER NODE\n");
    exit(0);
  }

  int nDevicesMax=4;
  cudaGetDeviceCount(&nDevicesMax);
  if (sd->nDevices > nDevicesMax) {
    printf("ERROR: Not enough GPUs to launch, nDevices %d nDevicesMax %d\n", sd->nDevices, nDevicesMax);
    exit(0);
  }
  //int maxCoresPerDevice = maxCoresPerNode / nDevicesMax
  //sd->nDevices= (int((size-1)/maxCoresPerDevice)+1) % nDevicesMax
  if (size > sd->nDevices*(coresPerNode/nDevicesMax)){
    printf("ERROR: size,sd->nDevices,coresPerNode,nDevicesMax %d %d %d %d "
           "MORE MPI PROCESSES THAN DEVICES (FOLLOW PROPORTION, "
           "FOR CTE-POWER IS 10 PROCESSES FOR EACH GPU)\n",size,sd->nDevices,coresPerNode,nDevicesMax);
    exit(0);
  }
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  cudaSetDevice(0);
  for (int i = 0; i < coresPerNode; i++) {
    if (rank < coresPerNode / nDevicesMax * (i + 1) && rank >= coresPerNode / nDevicesMax * i && i<sd->nDevices) {
      cudaSetDevice(i);
      //printf("rank %d, device %d\n", rank, i);
#ifdef ENABLE_GPU_CHECK
      cudaDeviceProp prop;
      cudaGetDeviceProperties(&prop, i);
      mCPU->threads = prop.maxThreadsPerBlock; //1024
      mCPU->blocks = (n_dep_var*n_cells + mCPU->threads - 1) / mCPU->threads;
      if(md->n_per_cell_dep_var > prop.maxThreadsPerBlock/2){
        printf("ERROR: md->n_per_cell_dep_var, prop.maxThreadsPerBlock/2, %d %d More species than threads per block available\n",md->n_per_cell_dep_var, prop.maxThreadsPerBlock/2);
        exit(0);
      }
#else
        mCPU->threads = 1024;
        mCPU->blocks = (n_dep_var*n_cells + mCPU->threads - 1) / mCPU->threads;
#endif
    }
  }
  sd->mGPU = (ModelDataGPU *)malloc(sizeof(ModelDataGPU));
  mGPU = sd->mGPU;
  mGPU->n_cells=n_cells;
  mGPU->n_rxn=md->n_rxn;
  mGPU->n_rxn_env_data=md->n_rxn_env_data;
  cudaMalloc((void **) &mGPU->state, mCPU->state_size);
  cudaMalloc((void **) &mGPU->env, mCPU->env_size);
  cudaMalloc((void **) &mGPU->rxn_env_data, mCPU->rxn_env_data_size);
  cudaMalloc((void **) &mGPU->rxn_env_data_idx, mCPU->rxn_env_data_idx_size);
  cudaMalloc((void **) &mGPU->map_state_deriv, mCPU->map_state_deriv_size);
  int num_spec = md->n_per_cell_dep_var*mGPU->n_cells;
  cudaMalloc((void **) &(mGPU->production_rates),num_spec*sizeof(mGPU->production_rates));
  cudaMalloc((void **) &(mGPU->loss_rates),num_spec*sizeof(mGPU->loss_rates));
  mCPU->map_state_derivCPU = (int *)malloc(mCPU->map_state_deriv_size);
  int i_dep_var = 0;
  for (int i_cell = 0; i_cell < n_cells; i_cell++) {
    for (int i_spec = 0; i_spec < n_state_var; i_spec++) {
      if (md->var_type[i_spec] == CHEM_SPEC_VARIABLE) {
        mCPU->map_state_derivCPU[i_dep_var] = i_spec + i_cell * n_state_var;
        //printf("%d %d, %d %d %d\n", mCPU->map_state_deriv_size/sizeof(int),
        //       mCPU->map_state_derivCPU[i_dep_var],n_state_var, i_spec, i_cell, i_dep_var);
        i_dep_var++;
      }
    }
  }
  HANDLE_ERROR(cudaMemcpy(mGPU->map_state_deriv, mCPU->map_state_derivCPU,
                          mCPU->map_state_deriv_size, cudaMemcpyHostToDevice));
  if(n_dep_var<32 && sd->use_cpu==0) {
    printf("CAMP ERROR: TOO FEW SPECIES FOR GPU (Species < 32),"
           " use CPU case instead (More info: https://earth.bsc.es/gitlab/ac/camp/-/issues/49 \n");
    exit(0);
  }
}

void initLinearSolver_cvode(SolverData *sd){
  ModelDataGPU *mGPU = sd->mGPU;
  int nrows = mGPU->nrows;
  double ** dr0 = &mGPU->dr0;
  double ** dr0h = &mGPU->dr0h;
  double ** dn0 = &mGPU->dn0;
  double ** dp0 = &mGPU->dp0;
  double ** dt = &mGPU->dt;
  double ** ds = &mGPU->ds;
  double ** dy = &mGPU->dy;
  double ** ddiag = &mGPU->ddiag;
  cudaMalloc(dr0,nrows*sizeof(double));
  cudaMalloc(dr0h,nrows*sizeof(double));
  cudaMalloc(dn0,nrows*sizeof(double));
  cudaMalloc(dp0,nrows*sizeof(double));
  cudaMalloc(dt,nrows*sizeof(double));
  cudaMalloc(ds,nrows*sizeof(double));
  cudaMalloc(dy,nrows*sizeof(double));
  HANDLE_ERROR(cudaMalloc(ddiag,nrows*sizeof(double)));
  ModelDataCPU *mCPU = &(sd->mCPU);
  mCPU->aux=(double*)malloc(sizeof(double)*mCPU->blocks);
}

void constructor_cvode_gpu(CVodeMem cv_mem, SolverData *sd){
  ModelDataCPU *mCPU = &(sd->mCPU);
  ModelData *md = &(sd->model_data);
  CVDlsMem cvdls_mem = (CVDlsMem) cv_mem->cv_lmem;
  SUNMatrix J = cvdls_mem->A;
  sd->flagCells = (int *) malloc((md->n_cells) * sizeof(int));
  ModelDataGPU *mGPU = sd->mGPU;
#ifdef DEBUG_constructor_cvode_gpu
  printf("DEBUG_constructor_cvode_gpu start \n");
#endif
  solver_new_gpu_cu_cvode(sd);
  init_jac_cuda_cvode(sd);
  solver_init_int_double_cuda_cvode(sd);
  mGPU = sd->mGPU;
#ifdef CAMP_DEBUG_GPU
  mCPU->counterNewtonIt=0;
  mCPU->counterLinSolSetup=0;
  mCPU->counterLinSolSolve=0;
  mCPU->counterDerivNewton=0;
  mCPU->counterBCG=0;
  mCPU->counterDerivSolve=0;
  mCPU->countersolveCVODEGPU=0;

  mCPU->timeNewtonIt=0.;
  mCPU->timeLinSolSetup=0.;
  mCPU->timeLinSolSolve=0.;
  mCPU->timecvStep=0.;
  mCPU->timeDerivNewton=0.;
  mCPU->timeBiConjGrad=0.;
  mCPU->timeBiConjGradMemcpy=0.;
  mCPU->timeDerivSolve=0.;

  cudaEventCreate(&mCPU->startDerivNewton);
  cudaEventCreate(&mCPU->startDerivSolve);
  cudaEventCreate(&mCPU->startLinSolSetup);
  cudaEventCreate(&mCPU->startLinSolSolve);
  cudaEventCreate(&mCPU->startNewtonIt);
  cudaEventCreate(&mCPU->startcvStep);
  cudaEventCreate(&mCPU->startBCG);
  cudaEventCreate(&mCPU->startBCGMemcpy);
  cudaEventCreate(&mCPU->stopDerivNewton);
  cudaEventCreate(&mCPU->stopDerivSolve);
  cudaEventCreate(&mCPU->stopLinSolSetup);
  cudaEventCreate(&mCPU->stopLinSolSolve);
  cudaEventCreate(&mCPU->stopNewtonIt);
  cudaEventCreate(&mCPU->stopcvStep);
  cudaEventCreate(&mCPU->stopBCG);
  cudaEventCreate(&mCPU->stopBCGMemcpy);
#endif
  mGPU = sd->mGPU;
  mCPU->nnz = SM_NNZ_S(J);
  mGPU->nrows = SM_NP_S(J);
  initLinearSolver_cvode(sd);
  mCPU->A = ((double *) SM_DATA_S(J));
  //Translate from int64 (sunindextype) to int
  if(sd->use_gpu_cvode==1){
    mCPU->jA = (int *) malloc(sizeof(int) *mCPU->nnz/mGPU->n_cells);
    mCPU->iA = (int *) malloc(sizeof(int) * (mGPU->nrows/mGPU->n_cells + 1));
    for (int i = 0; i < mCPU->nnz/mGPU->n_cells; i++)
      mCPU->jA[i] = SM_INDEXVALS_S(J)[i];
    for (int i = 0; i <= mGPU->nrows/mGPU->n_cells; i++)
      mCPU->iA[i] = SM_INDEXPTRS_S(J)[i];
    cudaMalloc((void **) &mGPU->djA, mCPU->nnz/mGPU->n_cells * sizeof(int));
    cudaMalloc((void **) &mGPU->diA, (mGPU->nrows/mGPU->n_cells + 1) * sizeof(int));
    cudaMemcpy(mGPU->djA, mCPU->jA, mCPU->nnz/mGPU->n_cells * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(mGPU->diA, mCPU->iA, (mGPU->nrows/mGPU->n_cells + 1) * sizeof(int), cudaMemcpyHostToDevice);
  }else{
    mCPU->jA = (int *) malloc(sizeof(int) *mCPU->nnz);
    mCPU->iA = (int *) malloc(sizeof(int) * (mGPU->nrows + 1));
    for (int i = 0; i < mCPU->nnz; i++)
      mCPU->jA[i] = SM_INDEXVALS_S(J)[i];
    for (int i = 0; i <= mGPU->nrows; i++)
      mCPU->iA[i] = SM_INDEXPTRS_S(J)[i];
    cudaMalloc((void **) &mGPU->djA, mCPU->nnz * sizeof(int));
    cudaMalloc((void **) &mGPU->diA, (mGPU->nrows + 1) * sizeof(int));
    cudaMemcpy(mGPU->djA, mCPU->jA, mCPU->nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(mGPU->diA, mCPU->iA, (mGPU->nrows + 1) * sizeof(int), cudaMemcpyHostToDevice);
  }
  double *ewt = N_VGetArrayPointer(cv_mem->cv_ewt);
  double *tempv = N_VGetArrayPointer(cv_mem->cv_tempv);
  double *cv_last_yn = N_VGetArrayPointer(cv_mem->cv_last_yn);
  double *cv_acor_init = N_VGetArrayPointer(cv_mem->cv_acor_init);
#ifdef CAMP_PROFILE_DEVICE_FUNCTIONS
  cudaMalloc((void **) &mGPU->mdvo, sizeof(ModelDataVariable));
#endif
  cudaMalloc((void **) &mGPU->dftemp, mCPU->deriv_size);
  cudaMalloc((void **) &mGPU->sCells, sizeof(ModelDataVariable)*mGPU->n_cells);
  cudaMalloc((void **) &mGPU->flag, 1 * sizeof(int));
  cudaMalloc((void **) &mGPU->flagCells, mGPU->n_cells * sizeof(int));
  cudaMalloc((void **) &mGPU->dsavedJ, mCPU->nnz * sizeof(double));
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
  cudaMemcpy(mGPU->dsavedJ, mCPU->A, mCPU->nnz * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(mGPU->dewt, ewt, mGPU->nrows * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(mGPU->cv_acor, ewt, mGPU->nrows * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(mGPU->dftemp, ewt, mGPU->nrows * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(mGPU->dx, tempv, mGPU->nrows * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(mGPU->cv_last_yn, cv_last_yn, mGPU->nrows * sizeof(double), cudaMemcpyHostToDevice);
  HANDLE_ERROR(cudaMemcpy(mGPU->cv_acor_init, cv_acor_init, mGPU->nrows * sizeof(double), cudaMemcpyHostToDevice));
  mGPU->state_size_cell = md->n_per_cell_state_var;
  int flag = 999; //CAMP_SOLVER_SUCCESS
  cudaMemcpy(mGPU->flag, &flag, 1 * sizeof(int), cudaMemcpyHostToDevice);
#ifdef CAMP_DEBUG_GPU
#ifdef CAMP_PROFILE_DEVICE_FUNCTIONS
  cudaDeviceGetAttribute(&mGPU->clock_khz, cudaDevAttrClockRate, 0);
  mCPU->mdvCPU.countercvStep=0;
  mCPU->mdvCPU.counterBCGInternal=0;
  mCPU->mdvCPU.counterBCG=0;
  mCPU->mdvCPU.timeNewtonIteration=0.;
  mCPU->mdvCPU.timeJac=0.;
  mCPU->mdvCPU.timelinsolsetup=0.;
  mCPU->mdvCPU.timecalc_Jac=0.;
  mCPU->mdvCPU.timef=0.;
  mCPU->mdvCPU.timeguess_helper=0.;
  mCPU->mdvCPU.dtBCG=0.;
  mCPU->mdvCPU.dtcudaDeviceCVode=0.;
  mCPU->mdvCPU.dtPostBCG=0.;
#endif
#endif
  for (int i = 0; i < mGPU->n_cells; i++){
    cudaMemcpy(&mGPU->sCells[i], &mCPU->mdvCPU, sizeof(ModelDataVariable), cudaMemcpyHostToDevice);
  }
#ifdef CAMP_PROFILE_DEVICE_FUNCTIONS
  HANDLE_ERROR(cudaMemcpy(mGPU->mdvo, &mCPU->mdvCPU, sizeof(ModelDataVariable), cudaMemcpyHostToDevice));
#endif
  mCPU->mdvCPU.nstlj = 0;
  if(sd->use_gpu_cvode==1) {
    swapCSC_CSR_ODE_if_enabled(sd);
  }
  if(cv_mem->cv_sldeton){
    printf("ERROR: cudaDevicecvBDFStab is pending to implement "
           "(disabled by default on CAMP)\n");
    exit(0); }
}


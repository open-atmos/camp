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

void constructor_cvode_gpu(SolverData *sd){
  CVodeMem cv_mem = (CVodeMem) sd->cvode_mem;
  ModelDataCPU *mCPU = &(sd->mCPU);
  ModelData *md = &(sd->model_data);
  CVDlsMem cvdls_mem = (CVDlsMem) cv_mem->cv_lmem;
  sd->mGPU = (ModelDataGPU *)malloc(sizeof(ModelDataGPU));
  ModelDataGPU *mGPU = sd->mGPU;
  int n_cells = md->n_cells_gpu;
  mGPU->n_cells= n_cells;
  sd->flagCells = (int *) malloc((n_cells) * sizeof(int));
  int n_dep_var = md->n_per_cell_dep_var;
  int nrows = n_dep_var * n_cells;
  mGPU->nrows = nrows;
  int n_state_var = md->n_per_cell_state_var;
  mGPU->n_per_cell_state_var = md->n_per_cell_state_var;
  int n_rxn = md->n_rxn;
  int nGPUs;
  HANDLE_ERROR(cudaGetDeviceCount(&nGPUs));
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  int iDevice = rank % nGPUs;
  cudaSetDevice(iDevice);
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  mGPU->n_rxn=md->n_rxn;
  mGPU->n_rxn_env_data=md->n_rxn_env_data;
  HANDLE_ERROR(cudaMalloc((void **) &mGPU->state, n_state_var * n_cells * sizeof(double)));
  cudaMalloc((void **) &mGPU->env, CAMP_NUM_ENV_PARAM_ * n_cells * sizeof(double));
  cudaMalloc((void **) &mGPU->rxn_env_data, md->n_rxn_env_data * n_cells * sizeof(double));
  int num_spec = md->n_per_cell_dep_var*n_cells;
  cudaMalloc((void **) &(mGPU->production_rates),num_spec*sizeof(mGPU->production_rates));
  cudaMalloc((void **) &(mGPU->loss_rates),num_spec*sizeof(mGPU->loss_rates));
  size_t map_state_deriv_size = n_dep_var * n_cells * sizeof(int);
  cudaMalloc((void **) &mGPU->map_state_deriv, map_state_deriv_size);
  int *map_state_derivCPU = (int *)malloc(map_state_deriv_size);
  int i_dep_var = 0;
  for (int i_cell = 0; i_cell < n_cells; i_cell++) {
    for (int i_spec = 0; i_spec < n_state_var; i_spec++) {
      if (md->var_type[i_spec] == CHEM_SPEC_VARIABLE) {
        map_state_derivCPU[i_dep_var] = i_spec + i_cell * n_state_var;
        i_dep_var++;
      }
    }
  }
  HANDLE_ERROR(cudaMemcpy(mGPU->map_state_deriv, map_state_derivCPU,
                          map_state_deriv_size, cudaMemcpyHostToDevice));
  free(map_state_derivCPU);
  if(n_dep_var<32) {
    printf("CAMP ERROR: TOO FEW SPECIES FOR GPU (Species < 32),"
           " use CPU case instead\n");
    exit(0);
  }
  size_t deriv_size = n_dep_var * n_cells * sizeof(double);
  int nnz = md->n_per_cell_solver_jac_elem * n_cells;
  size_t jac_size = nnz * sizeof(double);
  cudaMalloc((void **) &mGPU->dA, jac_size);
  cudaMalloc((void **) &mGPU->J_solver, jac_size);
  cudaMalloc((void **) &mGPU->J_state, deriv_size);
  cudaMalloc((void **) &mGPU->J_deriv, deriv_size);

  double *J_deriv = N_VGetArrayPointer(md->J_deriv);
  cudaMemcpyAsync(mGPU->J_deriv,J_deriv,mGPU->nrows*sizeof(double),cudaMemcpyHostToDevice,stream);
  double *J_solver = SM_DATA_S(md->J_solver);
  cudaMemcpyAsync(mGPU->J_solver, J_solver, jac_size, cudaMemcpyHostToDevice,stream);


  cudaMalloc((void **) &mGPU->J_tmp, deriv_size);
  cudaMalloc((void **) &mGPU->jac_map, sizeof(JacMap) * md->n_mapped_values);
  HANDLE_ERROR(cudaMalloc((void **) &mGPU->n_mapped_values, 1 * sizeof(int)));
  HANDLE_ERROR(cudaMemcpy(mGPU->jac_map, md->jac_map, sizeof(JacMap) * md->n_mapped_values, cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(mGPU->n_mapped_values, &md->n_mapped_values, 1 * sizeof(int), cudaMemcpyHostToDevice));
  Jacobian *jac = &sd->jac;
  JacobianGPU *jacgpu = &(mGPU->jac);
  cudaMalloc((void **) &jacgpu->num_elem, 1 * sizeof(jacgpu->num_elem));
  HANDLE_ERROR(cudaMemcpy(jacgpu->num_elem, &jac->num_elem, 1 * sizeof(jacgpu->num_elem), cudaMemcpyHostToDevice));
  int num_elem = jac->num_elem * n_cells;
  HANDLE_ERROR(cudaMalloc((void **) &(jacgpu->production_partials), num_elem * sizeof(double)));
  HANDLE_ERROR(cudaMalloc((void **) &(jacgpu->loss_partials), num_elem * sizeof(double)));
  double *aux=(double*)malloc(sizeof(double)*num_elem);
  for (int i = 0; i < num_elem; i++) {
    aux[i]=0.;
  }
  HANDLE_ERROR(cudaMemcpy(jacgpu->production_partials, aux, num_elem * sizeof(double), cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(jacgpu->loss_partials, aux, num_elem * sizeof(double), cudaMemcpyHostToDevice));
  free(aux);
  cudaMalloc((void **) &mGPU->rxn_int, (md->n_rxn_int_param + md->n_rxn)*sizeof(int));
  cudaMalloc((void **) &mGPU->rxn_double, md->n_rxn_float_param*sizeof(double));
  cudaMalloc((void **) &mGPU->rxn_int_indices, (md->n_rxn+1)*sizeof(int));
  cudaMalloc((void **) &mGPU->rxn_float_indices, (md->n_rxn+1)*sizeof(int));
  HANDLE_ERROR(cudaMemcpy(mGPU->rxn_int, md->rxn_int_data,(md->n_rxn_int_param + md->n_rxn)*sizeof(int), cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(mGPU->rxn_double, md->rxn_float_data, md->n_rxn_float_param*sizeof(double), cudaMemcpyHostToDevice));
  size_t rxn_env_data_idx_size = (n_rxn+1) * sizeof(int);
  cudaMalloc((void **) &mGPU->rxn_env_data_idx, rxn_env_data_idx_size);
  HANDLE_ERROR(cudaMemcpy(mGPU->rxn_env_data_idx, md->rxn_env_idx, rxn_env_data_idx_size, cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(mGPU->rxn_int_indices, md->rxn_int_indices,(md->n_rxn+1)*sizeof(int), cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(mGPU->rxn_float_indices, md->rxn_float_indices,(md->n_rxn+1)*sizeof(int), cudaMemcpyHostToDevice));
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
  SUNMatrix J = cvdls_mem->A;
  double *A = ((double *) SM_DATA_S(J));
  cudaMalloc((void **) &mGPU->dsavedJ, nnz * sizeof(double));
  aux=(double*)malloc(jac_size);
  for (int i = 0; i < n_cells; i++) {
    for (int j = 0; j < md->n_per_cell_solver_jac_elem; j++) {
      aux[j+i*md->n_per_cell_solver_jac_elem] = A[j];
    }
  }
  cudaMemcpy(mGPU->dsavedJ, aux, jac_size, cudaMemcpyHostToDevice);
  free(aux);
  //Translate from int64 (sunindextype) to int
  int *jA = (int *) malloc(sizeof(int) * md->n_per_cell_solver_jac_elem);
  int *iA = (int *) malloc(sizeof(int) * (n_dep_var + 1));
  for (int i = 0; i < md->n_per_cell_solver_jac_elem; i++)
    jA[i] = SM_INDEXVALS_S(J)[i];
  for (int i = 0; i <= n_dep_var; i++)
    iA[i] = SM_INDEXPTRS_S(J)[i];
  cudaMalloc((void **) &mGPU->djA, md->n_per_cell_solver_jac_elem * sizeof(int));
  cudaMalloc((void **) &mGPU->diA, (n_dep_var + 1) * sizeof(int));
  cudaMemcpy(mGPU->djA, jA, md->n_per_cell_solver_jac_elem * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(mGPU->diA, iA, (n_dep_var + 1) * sizeof(int), cudaMemcpyHostToDevice);
  double *ewt = N_VGetArrayPointer(cv_mem->cv_ewt);
  double *tempv = N_VGetArrayPointer(cv_mem->cv_tempv);
  double *cv_last_yn = N_VGetArrayPointer(cv_mem->cv_last_yn);
  double *cv_acor_init = N_VGetArrayPointer(cv_mem->cv_acor_init);
  cudaMalloc((void **) &mGPU->dftemp, deriv_size);
  cudaMalloc((void **) &mGPU->sCells, sizeof(ModelDataVariable)*n_cells);
  cudaMalloc((void **) &mGPU->flagCells, n_cells * sizeof(int));
  cudaMalloc((void **) &mGPU->dewt, nrows * sizeof(double));
  cudaMalloc((void **) &mGPU->cv_acor, nrows * sizeof(double));
  cudaMalloc((void **) &mGPU->dtempv, nrows * sizeof(double));
  cudaMalloc((void **) &mGPU->dtempv1, nrows * sizeof(double));
  cudaMalloc((void **) &mGPU->dtempv2, nrows * sizeof(double));
  cudaMalloc((void **) &mGPU->dzn, nrows * (cv_mem->cv_qmax + 1) * sizeof(double));
  cudaMalloc((void **) &mGPU->dcv_y, nrows * sizeof(double));
  cudaMalloc((void **) &mGPU->dx, nrows * sizeof(double));
  cudaMalloc((void **) &mGPU->cv_last_yn, nrows * sizeof(double));
  cudaMalloc((void **) &mGPU->cv_acor_init, nrows * sizeof(double));
  cudaMalloc((void **) &mGPU->cv_acor, nrows * sizeof(double));
  cudaMalloc((void **) &mGPU->yout, nrows * sizeof(double));
  cudaMalloc((void **) &mGPU->cv_l, L_MAX * n_cells * sizeof(double));
  cudaMalloc((void **) &mGPU->cv_tau, (L_MAX + 1) * n_cells * sizeof(double));
  cudaMalloc((void **) &mGPU->cv_tq, (NUM_TESTS + 1) * n_cells * sizeof(double));
  cudaMalloc((void **) &mGPU->cv_Vabstol, nrows * sizeof(double));
  HANDLE_ERROR(cudaMemset(mGPU->flagCells, CV_SUCCESS, n_cells * sizeof(int)));
  aux=(double*)malloc(nrows*sizeof(double));
  double *aux2=(double*)malloc(nrows*sizeof(double));
  for (int i = 0; i < n_cells; i++) {
    for (int j = 0; j < n_dep_var; j++) {
      aux[j+i*n_dep_var] = ewt[j];
      aux2[j+i*n_dep_var] = tempv[j];
    }
  }
  cudaMemcpy(mGPU->dewt, aux, nrows * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(mGPU->cv_acor, aux, nrows * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(mGPU->dftemp, aux, nrows * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(mGPU->cv_last_yn, aux, nrows * sizeof(double), cudaMemcpyHostToDevice);
  HANDLE_ERROR(cudaMemcpy(mGPU->dx, aux2, nrows * sizeof(double), cudaMemcpyHostToDevice));
  free(aux);
  free(aux2);
  mCPU->mdvCPU.nstlj = 0;
#ifdef CAMP_PROFILE_SOLVING
  cudaEventCreate(&sd->startcvStep);
  cudaEventCreate(&sd->stopcvStep);
#ifdef CAMP_PROFILE_DEVICE_FUNCTIONS
  cudaMalloc((void **) &mGPU->mdvo, sizeof(ModelDataVariable));
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
  HANDLE_ERROR(cudaMemcpy(mGPU->mdvo, &mCPU->mdvCPU, sizeof(ModelDataVariable), cudaMemcpyHostToDevice));
#endif
#endif
  for (int i = 0; i < n_cells; i++){
    cudaMemcpyAsync(&mGPU->sCells[i], &mCPU->mdvCPU, sizeof(ModelDataVariable), cudaMemcpyHostToDevice,stream);
  }
#ifdef IS_DEBUG_MODE_CSR_ODE_GPU
  int n_row=nrows/n_cells;
  int* Ap=iA;
  int* Aj=jA;
  double* Ax=A;
  nnz=nnz/n_cells;
  int* Bp=(int*)malloc((n_row+1)*sizeof(int));
  int* Bi=(int*)malloc(nnz*sizeof(int));
  double* Bx=(double*)malloc(nnz*sizeof(double));
  memset(Bp, 0, (n_row+1)*sizeof(int));
  for (int n = 0; n < nnz; n++){
   Bp[Aj[n]]++;
  }
  for(int col = 0, cumsum = 0; col < n_row; col++){
    int temp  = Bp[col];
    Bp[col] = cumsum;
    cumsum += temp;
  }
  Bp[n_row] = nnz;
  int *mapJSPMV= (int *)malloc(nnz * sizeof(int));
  for(int row = 0; row < n_row; row++){
    for(int jj = Ap[row]; jj < Ap[row+1]; jj++){
      int col  = Aj[jj];
      int dest = Bp[col];
      Bi[dest] = row;
      Bx[dest] = Ax[jj];
      mapJSPMV[jj]=dest;
      Bp[col]++;
    }
  }
  for(int col = 0, last = 0; col <= n_row; col++){
    int temp  = Bp[col];
    Bp[col] = last;
    last    = temp;
  }
  nnz=md->n_mapped_values;
  int *aux_solver_id= (int *)malloc(nnz * sizeof(int));
  for (int i = 0; i < nnz; i++){
    aux_solver_id[i]=mapJSPMV[md->jac_map[i].solver_id];
  }
  free(mapJSPMV);
  int *jac_solver_id= (int *)malloc(nnz * sizeof(int));
  for (int i = 0; i < nnz; i++){
    jac_solver_id[i]=aux_solver_id[i];
    aux_solver_id[i]=md->jac_map[i].solver_id;
    md->jac_map[i].solver_id=jac_solver_id[i];
  }
  cudaMemcpy(mGPU->diA, Bp, (n_row + 1) * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(mGPU->djA, Bi, nnz * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(mGPU->dA, Bx, nnz * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(mGPU->jac_map, md->jac_map, sizeof(JacMap) * md->n_mapped_values, cudaMemcpyHostToDevice);
  free(Bp);
  free(Bi);
  free(Bx);
  free(jac_solver_id);
  free(aux_solver_id);
#endif
}

void free_gpu_cu(SolverData *sd) {
  ModelDataGPU *mGPU = sd->mGPU;
  free(sd->flagCells);
  mGPU = sd->mGPU;
  cudaFree(mGPU->map_state_deriv);
  cudaFree(mGPU->J_solver);
  cudaFree(mGPU->J_state);
  cudaFree(mGPU->J_deriv);
  cudaFree(mGPU->J_tmp);
  cudaFree(mGPU->indexvals);
  cudaFree(mGPU->indexptrs);
  cudaFree(mGPU->rxn_int);
  cudaFree(mGPU->rxn_double);
  cudaFree(mGPU->state);
  cudaFree(mGPU->env);
  cudaFree(mGPU->rxn_env_data);
  cudaFree(mGPU->rxn_env_data_idx);
  cudaFree(mGPU->production_rates);
  cudaFree(mGPU->loss_rates);
  cudaFree(mGPU->rxn_int_indices);
  cudaFree(mGPU->rxn_float_indices);
  cudaFree(mGPU->n_mapped_values);
  cudaFree(mGPU->jac_map);
  cudaFree(mGPU->yout);
  cudaFree(mGPU->cv_Vabstol);
  cudaFree(mGPU->cv_l);
  cudaFree(mGPU->cv_tau);
  cudaFree(mGPU->cv_tq);
  cudaFree(mGPU->cv_last_yn);
  cudaFree(mGPU->cv_acor_init);
  cudaFree(mGPU->dA);
  cudaFree(mGPU->djA);
  cudaFree(mGPU->diA);
  cudaFree(mGPU->dx);
  cudaFree(mGPU->dtempv);
  cudaFree(mGPU->ddiag);
  cudaFree(mGPU->dr0);
  cudaFree(mGPU->dr0h);
  cudaFree(mGPU->dn0);
  cudaFree(mGPU->dp0);
  cudaFree(mGPU->dt);
  cudaFree(mGPU->ds);
  cudaFree(mGPU->dy);
  cudaFree(mGPU->dz);
  cudaFree(mGPU->dftemp);
  cudaFree(mGPU->dcv_y);
  cudaFree(mGPU->dtempv1);
  cudaFree(mGPU->dtempv2);
  cudaFree(mGPU->flagCells);
  cudaFree(mGPU->cv_acor);
  cudaFree(mGPU->dzn);
  cudaFree(mGPU->dewt);
  cudaFree(mGPU->dsavedJ);
  cudaFree(mGPU->mdv);
  cudaFree(mGPU->mdvo);
}
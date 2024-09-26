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

void init_solve_gpu(SolverData *sd) {
  ModelData *md = &(sd->model_data);
  int n_dep_var = md->n_per_cell_dep_var;
#ifndef SANITY_CHECK
  if (n_dep_var < 32) {
    printf("CAMP ERROR: TOO FEW SPECIES FOR GPU, use CPU case instead\n");
    exit(0);
  }
  if (n_dep_var > 1024) {
    printf("CAMP ERROR: TOO MUCH SPECIES FOR GPU,use CPU case instead\n");
    exit(0);
  }
  // Check if reaction types are implemented in the GPU
  int n_rxn = md->n_rxn;
  for (int i_rxn = 0; i_rxn < n_rxn; i_rxn++) {
    int *rxn_int_data = &(md->rxn_int_data[md->rxn_int_indices[i_rxn]]);
    int rxn_type = rxn_int_data[0];
    switch (rxn_type) {
      case RXN_ARRHENIUS:
        break;
      case RXN_CMAQ_H2O2:
        break;
      case RXN_CMAQ_OH_HNO3:
        break;
      case RXN_FIRST_ORDER_LOSS:
        break;
      case RXN_PHOTOLYSIS:
        break;
      case RXN_TROE:
        break;
      default:
        printf("CAMP ERROR: Reaction type not implemented in GPU\n");
        exit(0);
    }
  }
#endif
  CVodeMem cv_mem = (CVodeMem)sd->cvode_mem;
  ModelDataCPU *mCPU = &(sd->mCPU);
  sd->mGPU = (ModelDataGPU *)malloc(sizeof(ModelDataGPU));
  ModelDataGPU *mGPU = sd->mGPU;
  md->n_cells_gpu = md->n_cells * sd->load_gpu / 100.;
  int n_cells = md->n_cells;  // Load balance can differ up to n_cells size
  int nrows = n_dep_var * n_cells;
  int n_state_var = md->n_per_cell_state_var;
  mGPU->n_per_cell_state_var = md->n_per_cell_state_var;
  sd->last_load_balance = 0;
  sd->last_load_gpu = 100;
  sd->acc_load_balance = 0;
  sd->iters_load_balance = 0;
  sd->last_short_gpu = 0;
  int nGPUs;
  HANDLE_ERROR(cudaGetDeviceCount(&nGPUs));
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    printf("Cells to GPU: %.lf%%\n", sd->load_gpu);
  }
  int iDevice = rank % nGPUs;
  double startTime = MPI_Wtime();
  cudaSetDevice(iDevice);
  if (rank == 0) printf("Time INIT: %f\n", MPI_Wtime() - startTime);
  mGPU->n_rxn = md->n_rxn;
  mGPU->n_rxn_env_data = md->n_rxn_env_data;
  HANDLE_ERROR(cudaMalloc((void **)&mGPU->state,
                          n_state_var * n_cells * sizeof(double)));
  cudaMalloc((void **)&mGPU->rxn_env_data,
             md->n_rxn_env_data * n_cells * sizeof(double));
  int num_spec = n_dep_var * n_cells;
  cudaMalloc((void **)&(mGPU->production_rates),
             num_spec * sizeof(mGPU->production_rates));
  cudaMalloc((void **)&(mGPU->loss_rates), num_spec * sizeof(mGPU->loss_rates));
  cudaMalloc((void **)&mGPU->map_state_deriv, n_dep_var * sizeof(int));
  int *map_state_derivCPU = (int *)malloc(n_dep_var * sizeof(int));
  int i_dep_var = 0;
  for (int i_spec = 0; i_spec < n_state_var; i_spec++) {
    if (md->var_type[i_spec] == CHEM_SPEC_VARIABLE) {
      map_state_derivCPU[i_dep_var] = i_spec;
      i_dep_var++;
    }
  }
#ifdef PROFILE_SOLVING
  cudaEventCreate(&sd->startGPU);
  cudaEventCreate(&sd->stopGPU);
  cudaEventCreate(&sd->startGPUSync);
  cudaEventCreate(&sd->stopGPUSync);
#endif
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  cudaMemcpyAsync(mGPU->map_state_deriv, map_state_derivCPU,
                  n_dep_var * sizeof(int), cudaMemcpyHostToDevice, stream);
  free(map_state_derivCPU);
  size_t deriv_size = n_dep_var * n_cells * sizeof(double);
  int nnz = md->n_per_cell_solver_jac_elem * n_cells;
  size_t jac_size = nnz * sizeof(double);
  cudaMalloc((void **)&mGPU->dA, jac_size);
  cudaMalloc((void **)&mGPU->J_solver, jac_size);
  cudaMalloc((void **)&mGPU->J_state, deriv_size);
  double *J_state = N_VGetArrayPointer(md->J_state);
  cudaMemset(mGPU->J_state, 0, deriv_size);
  cudaMalloc((void **)&mGPU->J_deriv, deriv_size);
  double *J_deriv = N_VGetArrayPointer(md->J_deriv);
  cudaMemset(mGPU->J_deriv, 0, deriv_size);
  cudaMemset(mGPU->J_solver, 0, jac_size);
  cudaMalloc((void **)&mGPU->jac_map,
             sizeof(JacMap) * md->n_per_cell_solver_jac_elem);
  cudaMemcpyAsync(mGPU->jac_map, md->jac_map,
                  sizeof(JacMap) * md->n_per_cell_solver_jac_elem,
                  cudaMemcpyHostToDevice, stream);
  Jacobian *jac = &sd->jac;
  JacobianGPU *jacgpu = &(mGPU->jac);
  cudaMalloc((void **)&jacgpu->num_elem, 1 * sizeof(jacgpu->num_elem));
  cudaMemcpyAsync(jacgpu->num_elem, &jac->num_elem,
                  1 * sizeof(jacgpu->num_elem), cudaMemcpyHostToDevice, stream);
  int num_elem = jac->num_elem * n_cells;
  cudaMalloc((void **)&(jacgpu->production_partials),
             num_elem * sizeof(double));
  cudaMalloc((void **)&(jacgpu->loss_partials), num_elem * sizeof(double));
  cudaMemset(jacgpu->production_partials, 0, num_elem * sizeof(double));
  cudaMemset(jacgpu->loss_partials, 0, num_elem * sizeof(double));
  cudaMalloc((void **)&mGPU->rxn_int,
             (md->n_rxn_int_param + md->n_rxn) * sizeof(int));
  cudaMalloc((void **)&mGPU->rxn_double,
             md->n_rxn_float_param * sizeof(double));
  cudaMalloc((void **)&mGPU->rxn_env_idx, (md->n_rxn + 1) * sizeof(int));
  cudaMalloc((void **)&mGPU->rxn_int_indices, (md->n_rxn + 1) * sizeof(int));
  cudaMalloc((void **)&mGPU->rxn_float_indices, (md->n_rxn + 1) * sizeof(int));
  cudaMemcpyAsync(mGPU->rxn_int, md->rxn_int_data,
                  (md->n_rxn_int_param + md->n_rxn) * sizeof(int),
                  cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(mGPU->rxn_double, md->rxn_float_data,
                  md->n_rxn_float_param * sizeof(double),
                  cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(mGPU->rxn_env_idx, md->rxn_env_idx,
                  (md->n_rxn + 1) * sizeof(int), cudaMemcpyHostToDevice,
                  stream);
  cudaMemcpyAsync(mGPU->rxn_int_indices, md->rxn_int_indices,
                  (md->n_rxn + 1) * sizeof(int), cudaMemcpyHostToDevice,
                  stream);
  cudaMemcpyAsync(mGPU->rxn_float_indices, md->rxn_float_indices,
                  (md->n_rxn + 1) * sizeof(int), cudaMemcpyHostToDevice,
                  stream);
  // Translate from int64 (sunindextype) to int
  CVDlsMem cvdls_mem = (CVDlsMem)cv_mem->cv_lmem;
  SUNMatrix J = cvdls_mem->A;
  int *jA = (int *)malloc(sizeof(int) * md->n_per_cell_solver_jac_elem);
  int *iA = (int *)malloc(sizeof(int) * (n_dep_var + 1));
  for (int i = 0; i < md->n_per_cell_solver_jac_elem; i++)
    jA[i] = SM_INDEXVALS_S(J)[i];
  for (int i = 0; i <= n_dep_var; i++) iA[i] = SM_INDEXPTRS_S(J)[i];
  cudaMalloc((void **)&mGPU->djA, md->n_per_cell_solver_jac_elem * sizeof(int));
  cudaMalloc((void **)&mGPU->diA, (n_dep_var + 1) * sizeof(int));
  cudaMemcpyAsync(mGPU->djA, jA, md->n_per_cell_solver_jac_elem * sizeof(int),
                  cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(mGPU->diA, iA, (n_dep_var + 1) * sizeof(int),
                  cudaMemcpyHostToDevice, stream);

  // Variables for each cell (struct ModelDataVariable), extracted from the
  // CVODE library
  mCPU->mdvCPU.cv_saved_tq5 = 0.;
  mCPU->mdvCPU.cv_acnrm = 0.;
  mCPU->mdvCPU.cv_eta = 1.;
  mCPU->mdvCPU.cv_hmin = 0;

  cudaMalloc((void **)&mGPU->sCells, sizeof(ModelDataVariable));
#ifdef CAMP_PROFILE_DEVICE_FUNCTIONS
  // Metrics for statistics
  cudaDeviceGetAttribute(&mGPU->clock_khz, cudaDevAttrClockRate, 0);
  mCPU->mdvCPU.countercvStep = 0;
  mCPU->mdvCPU.counterBCGInternal = 0;
  mCPU->mdvCPU.counterBCG = 0;
  mCPU->mdvCPU.timeNewtonIteration = 0.;
  mCPU->mdvCPU.timeJac = 0.;
  mCPU->mdvCPU.timelinsolsetup = 0.;
  mCPU->mdvCPU.timecalc_Jac = 0.;
  mCPU->mdvCPU.timef = 0.;
  mCPU->mdvCPU.timeguess_helper = 0.;
  mCPU->mdvCPU.timeBCG = 0.;
  mCPU->mdvCPU.timeDeviceCVode = 0.;
  cudaMalloc((void **)&mGPU->mdvo, sizeof(ModelDataVariable));
  cudaMemcpyAsync(mGPU->mdvo, &mCPU->mdvCPU, sizeof(ModelDataVariable),
                  cudaMemcpyHostToDevice, stream);
#endif
  cudaMemcpyAsync(&mGPU->sCells, &mCPU->mdvCPU, sizeof(ModelDataVariable),
                  cudaMemcpyHostToDevice, stream);
  // Parameters for the ODE solver, extracted from CVODE library
  mGPU->cv_reltol = cv_mem->cv_reltol;
  cudaMalloc((void **)&mGPU->cv_Vabstol, n_dep_var * sizeof(double));
  cudaMalloc((void **)&mGPU->cv_l, L_MAX * n_cells * sizeof(double));
  cudaMalloc((void **)&mGPU->cv_tau, (L_MAX + 1) * n_cells * sizeof(double));
  cudaMalloc((void **)&mGPU->cv_tq, (NUM_TESTS + 1) * n_cells * sizeof(double));
  cudaMalloc((void **)&mGPU->cv_last_yn, nrows * sizeof(double));
  cudaMalloc((void **)&mGPU->cv_acor, nrows * sizeof(double));
  cudaMalloc((void **)&mGPU->cv_acor_init, nrows * sizeof(double));
  cudaMalloc((void **)&mGPU->yout, nrows * sizeof(double));
  cudaMalloc((void **)&mGPU->dcv_y, nrows * sizeof(double));
  cudaMalloc((void **)&mGPU->dtempv, nrows * sizeof(double));
  cudaMalloc((void **)&mGPU->dtempv1, nrows * sizeof(double));
  cudaMalloc((void **)&mGPU->dftemp, deriv_size);
  cudaMalloc((void **)&mGPU->dewt, nrows * sizeof(double));
  cudaMalloc((void **)&mGPU->dsavedJ, nnz * sizeof(double));
  double **dzn = (double **)malloc((BDF_Q_MAX + 1) * sizeof(double *));
  for (int i = 0; i <= BDF_Q_MAX; i++)
    cudaMalloc(&dzn[i], nrows * sizeof(double));
  cudaMalloc(&mGPU->dzn, (BDF_Q_MAX + 1) * sizeof(double *));

  cudaMemcpyAsync(mGPU->cv_Vabstol, N_VGetArrayPointer(cv_mem->cv_Vabstol),
                  n_dep_var * sizeof(double), cudaMemcpyHostToDevice, stream);
  for (int i = 0; i < n_cells; i++) {
    cudaMemcpyAsync(mGPU->cv_l + i * L_MAX, cv_mem->cv_l,
                    L_MAX * sizeof(double), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(mGPU->cv_tau + i * (L_MAX + 1), cv_mem->cv_tau,
                    (L_MAX + 1) * sizeof(double), cudaMemcpyHostToDevice,
                    stream);
    cudaMemcpyAsync(mGPU->cv_tq + i * (NUM_TESTS + 1), cv_mem->cv_tq,
                    (NUM_TESTS + 1) * sizeof(double), cudaMemcpyHostToDevice,
                    stream);
  }
  for (int i = 2; i <= BDF_Q_MAX; i++)
    cudaMemsetAsync(dzn[i], 0, nrows * sizeof(double), stream);
  cudaMemcpy(
      mGPU->dzn, dzn, (BDF_Q_MAX + 1) * sizeof(double *),
      cudaMemcpyHostToDevice);  // Synchronous because cudaFree is synchronous

  for (int i = 0; i <= BDF_Q_MAX; i++) {
    cudaFree(&dzn[i]);
  }
  free(dzn);

  // Parameters for the BCG solver
  cudaMalloc((void **)&mGPU->dx, nrows * sizeof(double));
  cudaMalloc((void **)&mGPU->ddiag, nrows * sizeof(double));
  cudaMalloc((void **)&mGPU->dr0, nrows * sizeof(double));
  cudaMalloc((void **)&mGPU->dr0h, nrows * sizeof(double));
  cudaMalloc((void **)&mGPU->dn0, nrows * sizeof(double));
  cudaMalloc((void **)&mGPU->dp0, nrows * sizeof(double));
  cudaMalloc((void **)&mGPU->dt, nrows * sizeof(double));
  cudaMalloc((void **)&mGPU->ds, nrows * sizeof(double));
  cudaMalloc((void **)&mGPU->dy, nrows * sizeof(double));

  // Swap Jacobian format from CSC in the CPU to CSR for the GPU
  int n_row = nrows / n_cells;
  int *Ap = iA;
  int *Aj = jA;
  double *Ax = ((double *)SM_DATA_S(J));
  nnz = nnz / n_cells;
  int *Bp = (int *)malloc((n_row + 1) * sizeof(int));
  int *Bi = (int *)malloc(nnz * sizeof(int));
  double *Bx = (double *)malloc(nnz * sizeof(double));
  memset(Bp, 0, (n_row + 1) * sizeof(int));
  for (int n = 0; n < nnz; n++) {
    Bp[Aj[n]]++;
  }
  for (int col = 0, cumsum = 0; col < n_row; col++) {
    int temp = Bp[col];
    Bp[col] = cumsum;
    cumsum += temp;
  }
  Bp[n_row] = nnz;
  int *mapJSPMV = (int *)malloc(nnz * sizeof(int));
  for (int row = 0; row < n_row; row++) {
    for (int jj = Ap[row]; jj < Ap[row + 1]; jj++) {
      int col = Aj[jj];
      int dest = Bp[col];
      Bi[dest] = row;
      Bx[dest] = Ax[jj];
      mapJSPMV[jj] = dest;
      Bp[col]++;
    }
  }
  for (int col = 0, last = 0; col <= n_row; col++) {
    int temp = Bp[col];
    Bp[col] = last;
    last = temp;
  }
  nnz = md->n_per_cell_solver_jac_elem;
  int *aux_solver_id = (int *)malloc(nnz * sizeof(int));
  for (int i = 0; i < nnz; i++) {
    aux_solver_id[i] = mapJSPMV[md->jac_map[i].solver_id];
  }
  free(mapJSPMV);
  int *jac_solver_id = (int *)malloc(nnz * sizeof(int));
  JacMap *jac_map = (JacMap *)malloc(nnz * sizeof(JacMap));
  for (int i = 0; i < nnz; i++) {
    jac_solver_id[i] = aux_solver_id[i];
    aux_solver_id[i] = md->jac_map[i].solver_id;
    jac_map[i].solver_id = jac_solver_id[i];
    jac_map[i].rxn_id = md->jac_map[i].rxn_id;
    jac_map[i].param_id = md->jac_map[i].param_id;
  }
  cudaMemcpyAsync(mGPU->diA, Bp, (n_row + 1) * sizeof(int),
                  cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(mGPU->djA, Bi, nnz * sizeof(int), cudaMemcpyHostToDevice,
                  stream);
  cudaMemcpyAsync(mGPU->jac_map, jac_map, nnz * sizeof(JacMap),
                  cudaMemcpyHostToDevice, stream);
  free(Bp);
  free(Bi);
  free(Bx);
  free(jac_solver_id);
  free(aux_solver_id);
  free(jac_map);
#ifdef DEBUG_SOLVER_FAILURES
  cudaMalloc((void **)&mGPU->flags, n_cells);
  malloc(mCPU->flags, n_cells);
  int *aux_solver_id = (int *)malloc(nnz * sizeof(int));
#endif
}

void free_gpu_cu(SolverData *sd) {
  ModelDataGPU *mGPU = sd->mGPU;
  mGPU = sd->mGPU;
  cudaFree(mGPU->map_state_deriv);
  cudaFree(mGPU->J_solver);
  cudaFree(mGPU->J_state);
  cudaFree(mGPU->J_deriv);
  cudaFree(mGPU->rxn_int);
  cudaFree(mGPU->rxn_double);
  cudaFree(mGPU->state);
  cudaFree(mGPU->rxn_env_data);
  cudaFree(mGPU->rxn_env_idx);
  cudaFree(mGPU->production_rates);
  cudaFree(mGPU->loss_rates);
  cudaFree(mGPU->rxn_int_indices);
  cudaFree(mGPU->rxn_float_indices);
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
  cudaFree(mGPU->dftemp);
  cudaFree(mGPU->dcv_y);
  cudaFree(mGPU->dtempv1);
  cudaFree(mGPU->cv_acor);
  cudaFree(mGPU->dzn);
  cudaFree(mGPU->dewt);
  cudaFree(mGPU->dsavedJ);
#ifdef DEBUG_SOLVER_FAILURES
  cudaFree(mGPU->flags);
#endif
#ifdef CAMP_PROFILE_DEVICE_FUNCTIONS
  cudaFree(mGPU->mdvo);
#endif
}
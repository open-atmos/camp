/* Copyright (C) 2021 Barcelona Supercomputing Center and University of
 * Illinois at Urbana-Champaign
 * SPDX-License-Identifier: MIT
 */

/** \file
 * \brief All of the host functions interfacing between the CPU and GPU
 * as well as the main (__global__) kernel
 */

#include "cvode_dev.h"
#include "cvode_interface.h"

__global__ void cudaGlobalCVode(double t_initial, ModelDataGPU md_object) {
  ModelDataGPU *md = &md_object;
  extern __shared__ int flag_shr[];
  ModelDataVariable sc_object = *md->sCells;
  ModelDataVariable *sc = &sc_object;
  sc->cv_tn = t_initial;  // Set initial value of "t" to each cell data
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  // Update input
  md->dzn[0][i] = md->state[md->map_state_deriv[threadIdx.x] +
                            blockIdx.x * md->n_per_cell_state_var] > TINY
                      ? md->state[md->map_state_deriv[threadIdx.x] +
                                  blockIdx.x * md->n_per_cell_state_var]
                      : TINY;
#ifdef IS_DEBUG_MODE_RESET_JAC
  // Reset jac
  // Resetting or not is residual, only use for debug,
  // such as get exactly equal values
  int nnz = md->diA[blockDim.x];
  int n_iters = nnz / blockDim.x;
  for (int z = 0; z < n_iters; z++) {
    int j = threadIdx.x + z * blockDim.x + nnz * blockIdx.x;
    md->J_solver[j] = 0;
  }
  int residual = nnz % blockDim.x;
  if (threadIdx.x < residual) {
    int j = threadIdx.x + n_iters * blockDim.x + nnz * blockIdx.x;
    md->J_solver[j] = 0;
  }
  md->J_state[i] = 0;
  md->J_deriv[i] = 0;
#endif

#ifdef DEBUG_IS_ANYTHING_GOING_ON_HERE
  // Disable to accelerate execution.
  if (cudaDeviceis_anything_going_on_here(md, sc) == 0) return;
#endif

#ifdef CAMP_PROFILE_DEVICE_FUNCTIONS
  int clock_khz = md->clock_khz;
  clock_t start;
  start = clock();
#endif
// Solve
#ifdef DEBUG_SOLVER_FAILURES
  int flag = cudaDeviceCVode(md, sc);
  if (threadIdx.x == 0) md->flags[blockIdx.x] = flag;
#else
  cudaDeviceCVode(md, sc);
#endif
  // Update output
  md->state[md->map_state_deriv[threadIdx.x] +
            blockIdx.x * md->n_per_cell_state_var] =
      md->yout[i] > 0. ? md->yout[i] : 0.;
#ifndef GET_NUM_STEPS
  if (threadIdx.x == 0) md->num_steps[blockIdx.x] = sc->cv_nst;
  // if (threadIdx.x == 0) printf("sc->cv_nst %d\n", sc->cv_nst);
#endif
#ifdef CAMP_PROFILE_DEVICE_FUNCTIONS
  if (threadIdx.x == 0)
    sc->timeDeviceCVode +=
        ((double)(int)(clock() - start)) / (clock_khz * 1000);
#endif
#ifdef CAMP_PROFILE_DEVICE_FUNCTIONS
  ModelDataVariable *mdvo = md->mdvo;
  *mdvo = *sc;
#endif
}

extern "C" void init_solve_gpu(SolverData *sd, int max_steps, int max_conv_fails) {
  ModelData *md = &(sd->model_data);
  int n_dep_var = md->n_per_cell_dep_var;
#ifndef SANITY_CHECK
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
      case RXN_EMISSION:
        break;

    // Aerosol reactions
    // case RXN_AQUEOUS_EQUILIBRIUM:
    //   break;
    // case RXN_CONDENSED_PHASE_ARRHENIUS:
    //   break;
    // case RXN_CONDENSED_PHASE_PHOTOLYSIS:
    //   break;
    // case RXN_SIMPOL_PHASE_TRANSFER:
    //   break;
    // case RXN_HL_PHASE_TRANSFER:
    //   break;
    // case RXN_SURFACE:
    //   break;
    // case RXN_WET_DEPOSITION:
    //   break;
    // case RXN_RAOULT_PHASE_TRANSFER:
    //   break;
    default:
      printf("CAMP ERROR: Reaction type not implemented in GPU. Reaction ID: "
             "%d\n",
             rxn_type);
      exit(0);
    }
  }
#endif
  // Set GPU device (e.g. a node can have more than one GPU available) for each
  // CPU thread (i.e. MPI rank)
  // e.g. Run with 4 processes and 4 GPus: 0->GPU0, 1->GPU1, 2->GPU2, 3->GPU3
  int nGPUs;
  HANDLE_ERROR(cudaGetDeviceCount(&nGPUs));
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank); // should be a function input!!!
  if (rank == 0) {
    printf("Cells to GPU: %.lf%%\n", sd->load_gpu);
    printf("Load balance: %d\n", sd->is_load_balance);
  }
  int iDevice = rank % nGPUs;
  // double startTime = MPI_Wtime();
  cudaSetDevice(iDevice);
  // High on MN5 with multiple cores (e.g. 4s for 80)
  // if (rank == 0) printf("Time INIT: %f\n", MPI_Wtime() - startTime);
  // Parameters on the CPU related to the GPU solver
  CVodeMem cv_mem = (CVodeMem)sd->cvode_mem; // Variables from CVODE library
  int n_cells = md->n_cells;
  int nrows = n_dep_var * n_cells;  // Number of rows in the Jacobian and
                                    // chemical concentrations to solve
  ModelDataCPU *mCPU = &(sd->mCPU); // Variables from CPU to gpu
  sd->mGPU = (ModelDataGPU *)malloc(sizeof(ModelDataGPU)); // GPU data
  ModelDataGPU *mGPU = sd->mGPU;
  mGPU->n_per_cell_state_var = md->n_per_cell_state_var;
  mGPU->n_rxn_env_data = md->n_rxn_env_data;
  mGPU->n_aero_rep_env_data = md->n_aero_rep_env_data;
  mGPU->n_aero_rep_env_param = md->n_aero_rep_env_param;
  mGPU->n_rxn = n_rxn;
  mGPU->max_steps = max_steps;
  mGPU->max_conv_fails = max_conv_fails;

  cudaStream_t stream; // Stream for asynchronous memory copy
  cudaStreamCreate(&stream);
  md->n_cells_gpu =
      round(md->n_cells * sd->load_gpu / 100.); // Number of cells to solve in the GPU
  sd->last_load_balance = 0;
  sd->last_load_gpu = 100;
  sd->last_short_gpu = 0;
#ifdef PROFILE_SOLVING
  cudaEventCreate(&sd->startGPU);
  cudaEventCreate(&sd->stopGPU);
  cudaEventCreate(&sd->startGPUSync);
  cudaEventCreate(&sd->stopGPUSync);
#endif
#ifdef DEBUG_SOLVER_FAILURES
  cudaMalloc((void **)&mGPU->flags, n_cells * sizeof(int));
  sd->flags = (int *)malloc(n_cells * sizeof(int));
  for (int i = 0; i < md->n_cells; i++) {
    sd->flags[i] = CV_SUCCESS;
  }
#endif

  // Parameters from CAMP chemical model
  int nnz = md->n_per_cell_solver_jac_elem * n_cells;
  Jacobian *jac = &sd->jac;
  JacobianGPU *jacgpu = &(mGPU->jac);

  HANDLE_ERROR(cudaMallocAsync(
      (void **)&mGPU->state,
      md->n_per_cell_state_var * n_cells * sizeof(double), stream));
  cudaMallocAsync((void **)&mGPU->map_state_deriv, n_dep_var * sizeof(int),
                  stream);
  cudaMallocAsync((void **)&mGPU->dA, nnz * sizeof(double), stream);
  cudaMallocAsync((void **)&mGPU->djA,
                  md->n_per_cell_solver_jac_elem * sizeof(int), stream);
  cudaMallocAsync((void **)&mGPU->diA, (n_dep_var + 1) * sizeof(int), stream);
  cudaMallocAsync((void **)&mGPU->J_solver, nnz * sizeof(double), stream);
  cudaMallocAsync((void **)&mGPU->J_state, n_dep_var * n_cells * sizeof(double),
                  stream);
  cudaMallocAsync((void **)&mGPU->J_deriv, n_dep_var * n_cells * sizeof(double),
                  stream);
  cudaMallocAsync((void **)&jacgpu->num_elem, sizeof(jacgpu->num_elem), stream);
  cudaMallocAsync((void **)&(jacgpu->production_partials),
                  jac->num_elem * n_cells * sizeof(double), stream);
  cudaMallocAsync((void **)&(jacgpu->loss_partials),
                  jac->num_elem * n_cells * sizeof(double), stream);
  cudaMallocAsync((void **)&mGPU->jac_map,
                  sizeof(JacMap) * md->n_per_cell_solver_jac_elem, stream);

  // Allocate gas phase arrays
  cudaMallocAsync((void **)&mGPU->rxn_int,
                  (md->n_rxn_int_param + n_rxn) * sizeof(int), stream);
  cudaMallocAsync((void **)&mGPU->rxn_int_indices, (n_rxn + 1) * sizeof(int),
                  stream);
  cudaMallocAsync((void **)&mGPU->rxn_double,
                  md->n_rxn_float_param * sizeof(double), stream);
  cudaMallocAsync((void **)&mGPU->rxn_float_indices, (n_rxn + 1) * sizeof(int),
                  stream);
  cudaMallocAsync((void **)&mGPU->rxn_env_data,
                  md->n_rxn_env_data * n_cells * sizeof(double), stream);
  cudaMallocAsync((void **)&mGPU->rxn_env_idx, (n_rxn + 1) * sizeof(int),
                  stream);

  // Allocate aerosol arrays
  // cudaMallocAsync((void **)&mGPU->aero_phase_int_data,
  //                 md->n_aero_phase_int_param * sizeof(int), stream);
  // cudaMallocAsync((void **)&mGPU->aero_phase_int_indices,
  //                 (md->n_aero_phase + 1) * sizeof(int), stream);
  // cudaMallocAsync((void **)&mGPU->aero_phase_float_data,
  //                 md->n_aero_phase_float_param * sizeof(double), stream);
  // cudaMallocAsync((void **)&mGPU->aero_phase_float_indices,
  //                 (md->n_aero_phase + 1) * sizeof(int), stream);

  // cudaMallocAsync((void **)&mGPU->aero_rep_int_data,
  //                 (md->n_aero_rep_int_param + md->n_aero_rep) * sizeof(int),
  //                 stream);
  // cudaMallocAsync((void **)&mGPU->aero_rep_int_indices,
  //                 (md->n_aero_rep + 1) * sizeof(int), stream);
  // // cudaMallocAsync((void **)&mGPU->aero_rep_float_data,
  // //                 md->n_aero_rep_float_param * sizeof(double), stream);
  // cudaMallocAsync((void **)&mGPU->aero_rep_float_indices,
  //                 (md->n_aero_rep + 1) * sizeof(int), stream);
  // cudaMallocAsync((void **)&mGPU->aero_rep_env_data,
  //                 n_cells * md->n_aero_rep_env_param * sizeof(double), stream);
  // cudaMallocAsync((void **)&mGPU->aero_rep_env_idx,
  //                 (md->n_aero_rep + 1) * sizeof(int), stream);

  cudaMallocAsync((void **)&(mGPU->production_rates),
                  n_dep_var * n_cells * sizeof(mGPU->production_rates), stream);
  cudaMallocAsync((void **)&(mGPU->loss_rates),
                  n_dep_var * n_cells * sizeof(mGPU->loss_rates), stream);

  int *map_state_derivCPU = (int *)malloc(
      n_dep_var * sizeof(int)); // Auxiliary variable to copy to GPU
  int i_dep_var = 0;
  for (int i_spec = 0; i_spec < md->n_per_cell_state_var; i_spec++) {
    if (md->var_type[i_spec] == CHEM_SPEC_VARIABLE) {
      map_state_derivCPU[i_dep_var] = i_spec;
      i_dep_var++;
    }
  }
  cudaStreamSynchronize(stream);
  cudaMemcpy(mGPU->map_state_deriv, map_state_derivCPU, n_dep_var * sizeof(int),
             cudaMemcpyHostToDevice); // Synchronous due to the
                                      // free(map_state_derivCPU)
  free(map_state_derivCPU);
  CVDlsMem cvdls_mem =
      (CVDlsMem)
          cv_mem->cv_lmem; // Auxiliary variable to translate from datatype
                           // of CVODE library (sunindextype int64) to int
  SUNMatrix J = cvdls_mem->A;
  int *jA = (int *)malloc(sizeof(int) * md->n_per_cell_solver_jac_elem);
  int *iA = (int *)malloc(sizeof(int) * (n_dep_var + 1));
  for (int i = 0; i < md->n_per_cell_solver_jac_elem; i++)
    jA[i] = SM_INDEXVALS_S(J)[i]; // int64 to int
  for (int i = 0; i <= n_dep_var; i++)
    iA[i] = SM_INDEXPTRS_S(J)[i]; // int64 to int
  cudaMemcpyAsync(mGPU->djA, jA, md->n_per_cell_solver_jac_elem * sizeof(int),
                  cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(mGPU->diA, iA, (n_dep_var + 1) * sizeof(int),
                  cudaMemcpyHostToDevice, stream);
  cudaMemset(mGPU->J_deriv, 0, n_dep_var * n_cells * sizeof(double));
  cudaMemset(mGPU->J_state, 0, n_dep_var * n_cells * sizeof(double));
  cudaMemset(mGPU->J_solver, 0.0, nnz * sizeof(mGPU->J_solver));
  cudaMemset(jacgpu->production_partials, 0,
             jac->num_elem * n_cells * sizeof(jacgpu->production_partials));
  cudaMemset(jacgpu->loss_partials, 0,
             jac->num_elem * n_cells * sizeof(jacgpu->loss_partials));
  cudaMemcpyAsync(jacgpu->num_elem, &jac->num_elem,
                  1 * sizeof(jacgpu->num_elem), cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(mGPU->jac_map, md->jac_map,
                  sizeof(JacMap) * md->n_per_cell_solver_jac_elem,
                  cudaMemcpyHostToDevice, stream);

  // Copy gas phase arrays
  cudaMemcpyAsync(mGPU->rxn_int, md->rxn_int_data,
                  (md->n_rxn_int_param + n_rxn) * sizeof(int),
                  cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(mGPU->rxn_int_indices, md->rxn_int_indices,
                  (n_rxn + 1) * sizeof(int), cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(mGPU->rxn_double, md->rxn_float_data,
                  md->n_rxn_float_param * sizeof(double),
                  cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(mGPU->rxn_float_indices, md->rxn_float_indices,
                  (n_rxn + 1) * sizeof(int), cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(mGPU->rxn_env_idx, md->rxn_env_idx, (n_rxn + 1) * sizeof(int),
                  cudaMemcpyHostToDevice, stream);

  // Copy aerosol arrays
  // cudaMemcpyAsync(mGPU->aero_phase_int_data, md->aero_phase_int_data,
  //                 (md->n_aero_phase_int_param) * sizeof(int),
  //                 cudaMemcpyHostToDevice, stream);
  // cudaMemcpyAsync(mGPU->aero_phase_int_indices, md->aero_phase_int_indices,
  //                 (md->n_aero_phase + 1) * sizeof(int), cudaMemcpyHostToDevice,
  //                 stream);
  // cudaMemcpyAsync(mGPU->aero_phase_float_data, md->aero_phase_float_data,
  //                 md->n_aero_phase_float_param * sizeof(double),
  //                 cudaMemcpyHostToDevice, stream);
  // cudaMemcpyAsync(mGPU->aero_phase_float_indices, md->aero_phase_float_indices,
  //                 (md->n_aero_phase + 1) * sizeof(int), cudaMemcpyHostToDevice,
  //                 stream);

  // cudaMemcpyAsync(mGPU->aero_rep_int_data, md->aero_rep_int_data,
  //                 (md->n_aero_rep_int_param + md->n_aero_rep) * sizeof(int),
  //                 cudaMemcpyHostToDevice, stream);
  // cudaMemcpyAsync(mGPU->aero_rep_int_indices, md->aero_rep_int_indices,
  //                 (md->n_aero_rep + 1) * sizeof(int), cudaMemcpyHostToDevice,
  //                 stream);
  // // cudaMemcpyAsync(mGPU->aero_rep_float_data, md->aero_rep_float_data,
  // //                 md->n_aero_rep_float_param * sizeof(double),
  // //                 cudaMemcpyHostToDevice, stream);
  // cudaMemcpyAsync(mGPU->aero_rep_float_indices, md->aero_rep_float_indices,
  //                 (md->n_aero_rep + 1) * sizeof(int), cudaMemcpyHostToDevice,
  //                 stream);
  // cudaMemcpyAsync(mGPU->aero_rep_env_idx, md->aero_rep_env_idx,
  //                 (md->n_aero_rep + 1) * sizeof(int), cudaMemcpyHostToDevice,
  //                 stream);

  // Parameters for the ODE solver, extracted from CVODE library
  mGPU->cv_reltol = cv_mem->cv_reltol;
  cudaMallocAsync((void **)&mGPU->cv_Vabstol, n_dep_var * sizeof(double),
                  stream);
  cudaMallocAsync((void **)&mGPU->cv_l, L_MAX * n_cells * sizeof(double),
                  stream);
  cudaMallocAsync((void **)&mGPU->cv_tau,
                  (L_MAX + 1) * n_cells * sizeof(double), stream);
  cudaMallocAsync((void **)&mGPU->cv_tq,
                  (NUM_TESTS + 1) * n_cells * sizeof(double), stream);
  cudaMallocAsync((void **)&mGPU->cv_last_yn, nrows * sizeof(double), stream);
  cudaMallocAsync((void **)&mGPU->cv_acor, nrows * sizeof(double), stream);
  cudaMallocAsync((void **)&mGPU->yout, nrows * sizeof(double), stream);
  cudaMallocAsync((void **)&mGPU->dcv_y, nrows * sizeof(double), stream);
  cudaMallocAsync((void **)&mGPU->dtempv, nrows * sizeof(double), stream);
  cudaMallocAsync((void **)&mGPU->dtempv1, nrows * sizeof(double), stream);
  cudaMallocAsync((void **)&mGPU->dftemp, nrows * sizeof(double), stream);
  cudaMallocAsync((void **)&mGPU->dewt, nrows * sizeof(double), stream);
  cudaMallocAsync((void **)&mGPU->dsavedJ, nnz * sizeof(double), stream);
  cudaMallocAsync((void **)&mGPU->cv_acor_init, nrows * sizeof(double), stream);
  cudaMemsetAsync(mGPU->cv_acor_init, 0, nrows * sizeof(double), stream);
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

  sd->dzn = (double **)malloc((BDF_Q_MAX + 1) * sizeof(double *));
  for (int i = 0; i <= BDF_Q_MAX; i++) {
    cudaMalloc(&sd->dzn[i], nrows * sizeof(double));
  }
  cudaMalloc(&mGPU->dzn, (BDF_Q_MAX + 1) * sizeof(double *));
  for (int i = 2; i <= BDF_Q_MAX; i++)
    cudaMemsetAsync(sd->dzn[i], 0, nrows * sizeof(double), stream);

  cudaStreamSynchronize(stream);
  //  Synchronous because cudaFree is synchronous
  cudaMemcpy(mGPU->dzn, sd->dzn, (BDF_Q_MAX + 1) * sizeof(double *),
             cudaMemcpyHostToDevice);

  // Parameters for the BCG solver
  cudaMallocAsync((void **)&mGPU->dx, nrows * sizeof(double), stream);
  cudaMallocAsync((void **)&mGPU->ddiag, nrows * sizeof(double), stream);
  cudaMallocAsync((void **)&mGPU->dr0, nrows * sizeof(double), stream);
  cudaMallocAsync((void **)&mGPU->dr0h, nrows * sizeof(double), stream);
  cudaMallocAsync((void **)&mGPU->dn0, nrows * sizeof(double), stream);
  cudaMallocAsync((void **)&mGPU->dp0, nrows * sizeof(double), stream);
  cudaMallocAsync((void **)&mGPU->dt, nrows * sizeof(double), stream);
  cudaMallocAsync((void **)&mGPU->ds, nrows * sizeof(double), stream);
  cudaMallocAsync((void **)&mGPU->dy, nrows * sizeof(double), stream);

  // Variables for each cell (struct ModelDataVariable), extracted from the
  // CVODE library
  mCPU->mdvCPU.cv_saved_tq5 = 0.;
  mCPU->mdvCPU.cv_acnrm = 0.;
  mCPU->mdvCPU.cv_eta = 1.;
  mCPU->mdvCPU.cv_hmin = 0;
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
  cudaMallocAsync((void **)&mGPU->sCells, sizeof(ModelDataVariable), stream);
  cudaMemcpyAsync(mGPU->sCells, &mCPU->mdvCPU, sizeof(ModelDataVariable),
                  cudaMemcpyHostToDevice, stream);

  // Swap Jacobian format from CSC in the CPU to CSR for the GPU (for efficiency
  // reasons)
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
#ifndef GET_NUM_STEPS
  cudaMalloc((void **)&mGPU->num_steps, n_cells * sizeof(int));
#endif
  cudaStreamSynchronize(stream);
}

extern "C" void cvodeRun(double t_initial, ModelDataGPU *mGPU, int blocks,
                         int threads_block, cudaStream_t stream) {
  // The * 2 fixes out of bounds access but still gives a race condition
  // * 4 fixes the race condition but the solver still fails
  int n_shr_memory = nextPowerOfTwo(threads_block);
  mGPU->n_shr_empty = n_shr_memory - threads_block;
  cudaGlobalCVode<<<blocks, threads_block, n_shr_memory * sizeof(double),
                    stream>>>(t_initial, *mGPU); // Call to GPU
}

extern "C" int cudaCVode(double t_final, SolverData *sd, double t_initial,
              int is_get_solver_stats, int *status_code, int *solver_flag,
              int *num_steps) {
  ModelDataGPU *mGPU = sd->mGPU;
  ModelData *md = &(sd->model_data);
  int n_cells = md->n_cells_gpu;
  cudaStream_t stream;  // Variable for asynchronous execution of the GPU
  cudaStreamCreate(&stream);
#ifdef LOAD_BALANCE
  if (sd->is_load_balance == 1) {
    cudaEventRecord(sd->startGPU, stream);  // Start GPU timer
  }
#endif
  // Transfer data to GPU
  cudaMemcpyAsync(mGPU->rxn_env_data, md->rxn_env_data,
                  md->n_rxn_env_data * n_cells * sizeof(double),
                  cudaMemcpyHostToDevice, stream);
  // cudaMemcpyAsync(mGPU->aero_rep_env_data, md->aero_rep_env_data,
  //                 n_cells * md->n_aero_rep_env_param * sizeof(double),
  //                 cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(mGPU->state, md->total_state,
                  md->n_per_cell_state_var * n_cells * sizeof(double),
                  cudaMemcpyHostToDevice, stream);
  mGPU->init_time_step = sd->init_time_step;
  mGPU->tout = t_final;

  // Solve
  if (n_cells > 0)
    cvodeRun(t_initial, mGPU, n_cells, md->n_per_cell_dep_var,
             stream); // Asynchronous
#ifdef LOAD_BALANCE
  if (sd->is_load_balance == 1) {
    cudaEventRecord(sd->stopGPU, stream); // End GPU timer
  }
#endif
  // CPU solver, equivalent to the CPU solver for the option CPU-Only
#ifdef TRACE_CPUGPU
  // Start of profiling trace using a tag to display on the trace
  nvtxRangePushA("CPU Code");
#endif
#ifdef LOAD_BALANCE
  double startTime;
  if (sd->is_load_balance == 1) {
    startTime = MPI_Wtime();
  }
#endif
  // Set data
  n_cells = md->n_cells;
  int flag = CV_SUCCESS;
  int n_state_var = md->n_per_cell_state_var;
  // Get pointers to the first value of the arrays
  double *state = md->total_state;
  double *env = md->total_env;
  double *rxn_env_data = md->rxn_env_data;
  double *aero_rep_env_data = md->aero_rep_env_data;
  // Set pointers of arrays to the next cell after the GPU cells
  md->total_state += n_state_var * md->n_cells_gpu;
  md->total_env += CAMP_NUM_ENV_PARAM_ * md->n_cells_gpu;
  md->rxn_env_data += md->n_rxn_env_data * md->n_cells_gpu;
  md->aero_rep_env_data += md->n_aero_rep_env_data * md->n_cells_gpu;
  for (int i_cell = md->n_cells_gpu; i_cell < n_cells; i_cell++) {
    int i_dep_var = 0;
    // Update input
    for (int i_spec = 0; i_spec < n_state_var; i_spec++) {
      if (sd->model_data.var_type[i_spec] == CHEM_SPEC_VARIABLE) {
        NV_Ith_S(sd->y, i_dep_var++) = md->total_state[i_spec] > TINY
                                           ? (realtype)md->total_state[i_spec]
                                           : TINY;
      }
    }
#ifdef IS_DEBUG_MODE_RESET_JAC
    // Reset Jacobian
    N_VConst(0.0, md->J_state);
    N_VConst(0.0, md->J_deriv);
    SM_NNZ_S(md->J_solver) = SM_NNZ_S(md->J_init);
    for (int i = 0; i < SM_NNZ_S(md->J_solver); i++) {
      (SM_DATA_S(md->J_solver))[i] = 0.0;
    }
#endif
    // Reset solver
    flag = CVodeReInit(sd->cvode_mem, t_initial, sd->y);
    flag = SUNKLUReInit(sd->ls, sd->J, SM_NNZ_S(sd->J), SUNKLU_REINIT_PARTIAL);
    flag = CVodeSetInitStep(sd->cvode_mem, sd->init_time_step);
    realtype t_rt = (realtype)t_initial;
    // Solve
    flag = CVode(sd->cvode_mem, t_final, sd->y, &t_rt, CV_NORMAL);
    if (is_get_solver_stats) {
      solver_flag[i_cell] = flag;
      long int nst;
      CVodeGetNumSteps(sd->cvode_mem, &nst);
      num_steps[i_cell] = (int)nst;
    }
    if (flag < 0) {
      flag = CAMP_SOLVER_FAIL;
      if (is_get_solver_stats) status_code[i_cell] = CAMP_SOLVER_FAIL;
      continue;
    } else {
      if (is_get_solver_stats) status_code[i_cell] = CAMP_SOLVER_SUCCESS;
    }
    // Get output
    i_dep_var = 0;
    for (int i_spec = 0; i_spec < n_state_var; i_spec++) {
      if (md->var_type[i_spec] == CHEM_SPEC_VARIABLE) {
        md->total_state[i_spec] = (double)(NV_Ith_S(sd->y, i_dep_var) > 0.0
                                               ? NV_Ith_S(sd->y, i_dep_var)
                                               : 0.0);
        i_dep_var++;
      }
    }
    // Update pointers for next iteration
    md->total_state += n_state_var;
    md->total_env += CAMP_NUM_ENV_PARAM_;
    md->rxn_env_data += md->n_rxn_env_data;
    md->aero_rep_env_data += md->n_aero_rep_env_data;
  }
  // Reset pointers
  md->total_state = state;
  md->total_env = env;
  md->rxn_env_data = rxn_env_data;
  md->aero_rep_env_data = aero_rep_env_data;
#ifdef LOAD_BALANCE
  double timeCPU;
  if (sd->is_load_balance == 1) {
    timeCPU = (MPI_Wtime() - startTime);
  }
#endif
#ifdef TRACE_CPUGPU
  nvtxRangePop();  // End of profiling trace
#endif
#ifdef LOAD_BALANCE
  if (sd->is_load_balance == 1) {
    // Start synchronization timer between CPU and GPU
    cudaEventRecord(sd->startGPUSync, stream);
  }
#endif
  // Transfer data back to CPU. This is located after the CPU solver and not
  // before because it blocks CPU execution until finish the GPU kernel
  cudaMemcpyAsync(md->total_state, mGPU->state,
                  md->n_per_cell_state_var * md->n_cells_gpu * sizeof(double),
                  cudaMemcpyDeviceToHost, stream);
#ifndef GET_NUM_STEPS
  if (is_get_solver_stats) {
    cudaMemcpyAsync(num_steps, mGPU->num_steps, md->n_cells_gpu * sizeof(int),
                    cudaMemcpyDeviceToHost, stream);
  }
#endif
#ifdef DEBUG_SOLVER_FAILURES
  cudaMemcpyAsync(sd->flags, mGPU->flags, md->n_cells_gpu * sizeof(int),
                  cudaMemcpyDeviceToHost, stream);
#endif
  // Ensure synchronization
  cudaStreamSynchronize(stream);
  cudaDeviceSynchronize();
#ifdef LOAD_BALANCE
  // Balance load between CPU and GPU, changing the number of cells solved on
  // both architectures. Method explained on C. Guzman PhD Thesis - Chapter 6
  if (sd->is_load_balance == 1) {
    cudaEventRecord(sd->stopGPUSync, stream);  // End synchronization timer
    cudaEventSynchronize(sd->stopGPUSync);
    cudaEventSynchronize(sd->stopGPU);
    float msDevice = 0.0;
    cudaEventElapsedTime(&msDevice, sd->startGPU, sd->stopGPU);
    double timeGPU = msDevice / 1000;
    cudaEventElapsedTime(&msDevice, sd->startGPUSync, sd->stopGPUSync);
    timeGPU += msDevice / 1000;
    double load_balance = 100;
    double min = fmin(timeGPU, timeCPU);
    double max = fmax(timeGPU, timeCPU);
    load_balance = 100 * min / max;
    /* Set if GPU time is less than CPU */
    int short_gpu = 0;
    if (timeGPU < timeCPU) short_gpu = 1;
    double increase_in_load_gpu = sd->load_gpu - sd->last_load_gpu;
    double last_short_gpu = sd->last_short_gpu;
    /* Set how much the load balance has increased */
    double diff_load_balance = load_balance - sd->last_load_balance;
    if (short_gpu != last_short_gpu) {
      diff_load_balance = 100 - sd->last_load_balance + 100 - load_balance;
      /* Change the increase sign because we surpass the limit of 100%,
      swapping from one architecture being short in time to the other */
      increase_in_load_gpu *= -1;
    }
    /* Set the remaining load balance to reach the ideal case of 100% */
    double remaining_load_balance = 100 - load_balance;
    /* Set the Increase in Load GPU */
    if (remaining_load_balance > diff_load_balance)
      increase_in_load_gpu *= 1.5;
    else
      increase_in_load_gpu /= 2;
    /* Update values for next iteration */
    sd->last_short_gpu = short_gpu;
    sd->last_load_balance = load_balance;
    sd->last_load_gpu = sd->load_gpu;
    if (load_balance != 100) sd->load_gpu += increase_in_load_gpu;
    /* Avoid the GPU percentage reaching or exceeding 100\% and 0\% */
    if (sd->load_gpu > 99) sd->load_gpu = 99;
    if (sd->load_gpu < 1) sd->load_gpu = 1;
    sd->acc_load_balance += load_balance;
    sd->iters_load_balance++;
    // Update values for next iteration from automatic load balance
    md->n_cells_gpu = md->n_cells * sd->load_gpu / 100;
#ifdef DEBUG_LOAD_BALANCE
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0)
      printf(
          "load_gpu: %.2lf%% Load balance: %.2lf%% short_gpu %d \
          increase_in_load_gpu %.2lf\n",
          sd->last_load_gpu, load_balance, sd->last_short_gpu,
          increase_in_load_gpu);

#endif
  }
#ifdef CAMP_PROFILE_DEVICE_FUNCTIONS
  cudaMemcpy(&sd->mCPU.mdvCPU, mGPU->mdvo, sizeof(ModelDataVariable),
             cudaMemcpyDeviceToHost);
#endif
#endif
  cudaStreamDestroy(stream);  // reset stream for next iteration
#ifdef DEBUG_SOLVER_FAILURES
  int solver_failures = 0;
  for (int i = 0; i < md->n_cells_gpu; i++) {
    if (sd->flags[i] != CV_SUCCESS) {     // Check if there was a failure
      cudacvHandleFailure(sd->flags[i]);  // Print error message.
      // WARNING: Many prints
      solver_failures++;
      flag = CAMP_SOLVER_FAIL;  // Update return flag
    }
    printf("Solver failures: %d\n", solver_failures);
    sd->flags[i] = CV_SUCCESS;  // Reset flags for next iteration
  }
#endif
  return CAMP_SOLVER_SUCCESS;
}

extern "C" void solver_get_profile_gpu(SolverData *sd) {
#ifdef CAMP_PROFILE_DEVICE_FUNCTIONS
  ModelDataGPU *mGPU = sd->mGPU;
  ModelDataCPU *mCPU = &(sd->mCPU);
  cudaMemcpy(&mCPU->mdvCPU, mGPU->mdvo, sizeof(ModelDataVariable),
             cudaMemcpyDeviceToHost);
#endif
}

extern "C" void free_gpu_cu(SolverData *sd) {
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

  // Free aerosol data
  // cudaFree(mGPU->aero_phase_int_data);
  // cudaFree(mGPU->aero_phase_int_indices);
  // cudaFree(mGPU->aero_phase_float_data);
  // cudaFree(mGPU->aero_phase_float_indices);
  // cudaFree(mGPU->aero_rep_int_data);
  // cudaFree(mGPU->aero_rep_int_indices);
  // cudaFree(mGPU->aero_rep_float_data);
  // cudaFree(mGPU->aero_rep_float_indices);
  // cudaFree(mGPU->aero_rep_env_data);
  // cudaFree(mGPU->aero_rep_env_idx);

  JacobianGPU *jacgpu = &(mGPU->jac);
  cudaFree(jacgpu->num_elem);
  cudaFree(jacgpu->production_partials);
  cudaFree(jacgpu->loss_partials);
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
  cudaFree(mGPU->sCells);
  for (int i = 0; i <= BDF_Q_MAX; i++) {
    cudaFree(sd->dzn[i]);
  }
  free(sd->dzn);
#ifdef CAMP_PROFILE_DEVICE_FUNCTIONS
  cudaFree(mGPU->mdvo);
#endif
#ifdef DEBUG_SOLVER_FAILURES
  cudaFree(mGPU->flags);
#endif
#ifndef GET_NUM_STEPS
  cudaFree(mGPU->num_steps);
#endif
  cudaDeviceReset();
  cudaDeviceSynchronize();
}
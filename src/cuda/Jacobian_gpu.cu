/* Copyright (C) 2021 Barcelona Supercomputing Center and University of
 * Illinois at Urbana-Champaign
 * SPDX-License-Identifier: MIT
 */

extern "C" {
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "Jacobian_gpu.h"

#define BUFFER_SIZE 10
#define SMALL_NUMBER 1e-90

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

__global__
void init_jac_partials(double* production_partials, double* loss_partials) {

  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  production_partials[tid]=0.0;
  loss_partials[tid]=0.0;

}

int jacobian_initialize_gpu(SolverData *sd) {
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

    int threads_block = jac->num_elem;
    int blocks = mGPU->n_cells;

    init_jac_partials <<<blocks,threads_block>>>(jacgpu->production_partials,jacgpu->loss_partials);

    /*
    int num_spec = jac->num_spec;
     for (int i = 0; i < num_elem; i++) {
      //printf("%lf ",jac->production_partials[i+offset_nnz]);
      //printf("%lf ",jac->loss_partials[i+offset_nnz]);
      jac->production_partials[i+offset_nnz]=0.0;
      jac->loss_partials[i+offset_nnz]=0.0;
    }


    cudaMemcpy(jacgpu->production_partials, jac->production_partials+offset_nnz, num_elem * sizeof(jacgpu->production_partials),cudaMemcpyHostToDevice);
    cudaMemcpy(jacgpu->loss_partials, jac->loss_partials+offset_nnz, num_elem * sizeof(jacgpu->loss_partials),cudaMemcpyHostToDevice);
*/

    offset_nnz += num_elem;
  }

  //print_int(jac->col_ptrs,jac->num_spec+1,"jac->col_ptrs");
  //printf("jac->num_elem %d\n",jac->num_elem);
  //printf("jac->num_spec %d\n",num_spec);

  /*

  for (int i = 0; i < jac->num_elem; ++i) {
    printf(" jac.production_partials[%d]=%d\n",i,jac->production_partials[i]);
  }

  for (unsigned int i_col = 0; i_col < jac->num_spec; ++i_col) {
    printf(" jac.col_ptrs[%d]=%d\n",i_col,jac->col_ptrs[i_col]);
    for (int i_elem = jac->col_ptrs[i_col];
         i_elem < jac->col_ptrs[i_col + 1]; ++i_elem) {
      printf(" jac.production_partials[%d]=%d\n",i_elem,jac->production_partials[i_elem]);
    }
  }
  */

  /*
  for (int i=0; i<jac->num_spec+1; i++){
    printf("jac->col_ptrs[%d]=%d\n",i,jac->col_ptrs[i]);
  }

  for (unsigned int i_col = 0; i_col < jac.num_spec; ++i_col){
    printf("\n");

  }
*/

#ifdef DEBUG_jacobian_initialize_gpu
  printf("jacobian_initialize_gpu end \n");
#endif

  return 1;
}

#ifdef __CUDA_ARCH__
__host__ __device__
#endif
void jacobian_output_gpu(JacobianGPU jac, double *dest_array) {

#ifdef __CUDA_ARCH__

  __syncthreads();

  int nnz = jac.num_elem[0];
  int n_iters = nnz / blockDim.x;
  for (int i = 0; i < n_iters; i++) {
    int j = threadIdx.x + i*blockDim.x;
    dest_array[j] = jac.production_partials[j] - jac.loss_partials[j];
    jac.production_partials[j] = 0.0;
    jac.loss_partials[j] = 0.0;
  }
  int residual=nnz-(blockDim.x*n_iters);
  if(threadIdx.x < residual){
    int j = threadIdx.x + n_iters*blockDim.x;
    dest_array[j] = jac.production_partials[j] - jac.loss_partials[j];
    jac.production_partials[j] = 0.0;
    jac.loss_partials[j] = 0.0;
  }

__syncthreads();
#endif

}

#ifdef __CUDA_ARCH__
__host__ __device__
#endif
void jacobian_add_value_gpu(JacobianGPU jac, unsigned int elem_id,
                            unsigned int prod_or_loss,
                            double jac_contribution) {
#ifdef __CUDA_ARCH__
  if (prod_or_loss == JACOBIAN_PRODUCTION)
    atomicAdd_block(&(jac.production_partials[elem_id]),jac_contribution);
  if (prod_or_loss == JACOBIAN_LOSS)
    atomicAdd_block(&(jac.loss_partials[elem_id]),jac_contribution);
#else
  if (prod_or_loss == JACOBIAN_PRODUCTION)
    jac.production_partials[elem_id] += jac_contribution;
  if (prod_or_loss == JACOBIAN_LOSS)
    jac.loss_partials[elem_id] += jac_contribution;
#endif

  //check_isnanld(&jac_contribution,1,"post jacobian_add_value jac_contribution");
}

}

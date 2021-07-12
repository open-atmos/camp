/* Copyright (C) 2019 Matthew Dawson
 * Licensed under the GNU General Public License version 2 or (at your
 * option) any later version. See the file COPYING for details.
 *
 * JacobianGPU functions
 *
 */
/** \file
 * \brief JacobianGPU functions
 */

extern "C" {
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "Jacobian_gpu.h"
//#include "time_derivative.h"

#define BUFFER_SIZE 10
#define SMALL_NUMBER 1e-90

/*
int jacobian_initialize_empty_gpu(SolverData *sd) {

  jac->num_spec = num_spec;
  jac->num_elem = 0;
  jac->elements = (JacobianColumnElements *)malloc(
      num_spec * sizeof(JacobianColumnElements));
  if (!jac->elements) {
    jacobian_free(jac);
    return 0;
  }
  for (unsigned int i_col = 0; i_col < num_spec; ++i_col) {
    jac->elements[i_col].array_size = BUFFER_SIZE;
    jac->elements[i_col].number_of_elements = 0;
    jac->elements[i_col].row_ids =
        (unsigned int *)malloc(BUFFER_SIZE * sizeof(unsigned int));
    if (!jac->elements[i_col].row_ids) {
      jacobian_free(jac);
      return 0;
    }
  }
  jac->col_ptrs = NULL;
  jac->row_ids = NULL;
  jac->production_partials = NULL;
  jac->loss_partials = NULL;


  cudaMalloc((void **) &(mGPU->production_rates),num_spec*sizeof(mGPU->production_rates));


  return 1;
}
*/

int jacobian_initialize_gpu(SolverData *sd) {

  ModelData *md = &(sd->model_data);
  ModelDataGPU *mGPU = &sd->mGPU;

  JacobianGPU *jacgpu = &(sd->mGPU.jac);//&mGPU->jac;
  Jacobian *jac = &sd->jac;

#ifdef __CUDA_ARCH__
  int num_spec = jac->num_spec;
#else
  //int num_spec = jacgpu->num_spec = jac->num_spec;
  int num_spec = jac->num_spec;
#endif

  //int num_elem = jacgpu->num_elem = jac->num_elem;
  //int num_elem = jac->num_elem;
  int num_elem = jac->num_elem * md->n_cells;
  cudaMalloc((void **) &jacgpu->num_elem, 1*sizeof(jacgpu->num_elem));//*md->n_mapped_values should be the same
  //cudaMalloc((void **) &(mGPU->jac.num_elem), 1*sizeof(mGPU->jac.num_elem));//*md->n_mapped_values should be the same

  //printf("jac->num_elem %d\n",jac->num_elem);

  cudaMalloc((void **) &(jacgpu->production_partials), num_elem * sizeof(jacgpu->production_partials));
  cudaMalloc((void **) &(jacgpu->loss_partials), num_elem * sizeof(jacgpu->loss_partials));
  cudaMalloc((void **) &(jacgpu->col_ptrs), (num_spec + 1) * sizeof(jacgpu->col_ptrs));

  cudaMemcpy(jacgpu->production_partials, jac->production_partials, num_elem * sizeof(jacgpu->production_partials),
             cudaMemcpyHostToDevice);
  cudaMemcpy(jacgpu->loss_partials, jac->loss_partials, num_elem * sizeof(jacgpu->loss_partials),
             cudaMemcpyHostToDevice);
  cudaMemcpy(jacgpu->col_ptrs, jac->col_ptrs, (num_spec + 1) * sizeof(jacgpu->col_ptrs), cudaMemcpyHostToDevice);

  cudaMemcpy(jacgpu->num_elem, &jac->num_elem, 1*sizeof(jacgpu->num_elem), cudaMemcpyHostToDevice);
  //cudaMemcpy(mGPU->jac.num_elem, &num_elem, 1*sizeof(mGPU->jac.num_elem), cudaMemcpyHostToDevice);


  return 1;
}

#ifdef __CUDA_ARCH__
__host__ __device__
#endif
void jacobian_reset_gpu(JacobianGPU jac) {


#ifdef __CUDA_ARCH__

#ifdef DEV_REMOVE_threadIdx0

  printf("TODO DEV_REMOVE_threadIdx0");
  /*int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i<jac.num_elem[0]){
    jac.production_partials[i] = 0.0;
    jac.loss_partials[i] = 0.0;
  }*/
#else

  /*
  __syncthreads();
  if(threadIdx.x==0){
    int nnz = jac.num_elem[0];
    for (int n = (nnz/gridDim.x)*blockIdx.x; n < (nnz/gridDim.x)*(blockIdx.x+1); n++) {
    //for (int n = 0; n < nnz; n++) {
      jac.production_partials[n] = 0.0;
      jac.loss_partials[n] = 0.0;
    }
*/


    __syncthreads();
  if(threadIdx.x==0){
    //int nnz = jac.num_elem[0];
    int nnz = jac.num_elem[0];
    //for (int n = (nnz/gridDim.x)*blockIdx.x; n < (nnz/gridDim.x)*(blockIdx.x+1); n++) {
    for (int n = 0; n < nnz; n++) {
      jac.production_partials[n] = 0.0;//1.E-100;//0.0;
      jac.loss_partials[n] = 0.0;
    }



    /*
    //dont needed, always jac.num_elem=jacBlock.num_elem*n_cells
     if(blockIdx.x==gridDim.x-1){
      int nnz_left = nnz = jac.num_elem[0]*gridDim.x-((nnz/gridDim.x)*gridDim.x);
      for (int n = (nnz/gridDim.x)*blockIdx.x; n < (nnz/gridDim.x)*(blockIdx.x+1); n++) {
      //for (int n = 0; n < nnz; n++) {
        jac.production_partials[n] = 0.0;
        jac.loss_partials[n] = 0.0;
      }
    }*/



  }__syncthreads();





#endif

#else

  /*for (unsigned int i_elem = 0; i_elem < jac.num_elem[0]; ++i_elem) {
    jac.production_partials[i_elem] = 0.0;
    jac.loss_partials[i_elem] = 0.0;
  }*/
#endif
}

#ifdef __CUDA_ARCH__
__host__ __device__
#endif
void jacobian_output_gpu(JacobianGPU jac, double *dest_array) {

#ifdef __CUDA_ARCH__

#ifdef DEV_REMOVE_threadIdx0

  //todo use col_ptrs*i_cell and num_spec to better memory access than nnz_left


#else


  //todo adapt to multi-cells gpu (not only one-cell per block)
  __syncthreads();
  //todo if this works, delete col_ptrs since it's not used during calc jac
  if(threadIdx.x==0){
    int i_col = blockIdx.x * blockDim.x + threadIdx.x;
    int nnz = jac.num_elem[0];
    //for (int n = (nnz/gridDim.x)*blockIdx.x; n < (nnz/gridDim.x)*(blockIdx.x+1); n++) {
    for (int n = 0; n < nnz; n++) {
         double drf_dy = jac.production_partials[n];
         double drr_dy = jac.loss_partials[n];

        //check_isnanld(&drf_dy,1,"post jacobian_output drf_dy");
        //check_isnanld(&drr_dy,1,"post jacobian_output drr_dy");

        dest_array[n] = drf_dy - drr_dy;
    }
  }__syncthreads();

#endif

  /*
  if(threadIdx.x==0){
    int i_col = blockIdx.x * blockDim.x + threadIdx.x;
    for (unsigned int i_col = blockDim.x/gridDim.x*blockIdx.x;
    i_col < blockDim.x/gridDim.x*(blockIdx.x+1); ++i_col) {
      for (unsigned int i_elem = jac.col_ptrs[i_col];
           i_elem < jac.col_ptrs[i_col + 1]; ++i_elem) {
         double drf_dy = jac.production_partials[i_elem];
         double drr_dy = jac.loss_partials[i_elem];

        //check_isnanld(&drf_dy,1,"post jacobian_output drf_dy");
        //check_isnanld(&drr_dy,1,"post jacobian_output drr_dy");

        dest_array[i_elem] = drf_dy - drr_dy;
      }
    }
  }
   */

  //todo check if this works:
  //crashes
  /*
    int i_col = blockIdx.x * blockDim.x + threadIdx.x;
    for (unsigned int i_elem = jac.col_ptrs[i_col];
         i_elem < jac.col_ptrs[i_col + 1]; ++i_elem) {
       double drf_dy = jac.production_partials[i_elem];
       double drr_dy = jac.loss_partials[i_elem];

      //check_isnanld(&drf_dy,1,"post jacobian_output drf_dy");
      //check_isnanld(&drr_dy,1,"post jacobian_output drr_dy");

      dest_array[i_elem] = drf_dy - drr_dy;
  }
   */

#else
  /*
   for (unsigned int i_col = 0; i_col < jac.num_spec; ++i_col) {
     for (unsigned int i_elem = jac.col_ptrs[i_col];
          i_elem < jac.col_ptrs[i_col + 1]; ++i_elem) {
       double drf_dy = jac.production_partials[i_elem];
       double drr_dy = jac.loss_partials[i_elem];

       //check_isnanld(&drf_dy,1,"post jacobian_output drf_dy");
       //check_isnanld(&drr_dy,1,"post jacobian_output drr_dy");

       dest_array[i_elem] = drf_dy - drr_dy;
     }
   }
   */
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

//
// Created by cguzman on 05/05/23.
//

#ifndef GPUPARTMC_CUDA_MATH_H
#define GPUPARTMC_CUDA_MATH_H

#include <cuda.h>
__device__ void cudaDevicezaxpby2(double a, double* dx, double b, double* dy, double* dz, int nrows)
{
  int row= threadIdx.x + blockDim.x*blockIdx.x;
  dz[row]=a*dx[row] + b*dy[row];
}
#endif //GPUPARTMC_CUDA_MATH_H

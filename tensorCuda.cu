#include "tensorCuda.h"

__global__ void sliceTensorKernel(float *dst, float *src, int ddim, int sdim, int start, int block_size)
{
     int di = blockIdx.x * block_size + threadIdx.x;
     int si = (blockIdx.x / ddim * sdim + blockIdx.x % ddim + start) * block_size + threadIdx.x;
     dst[di] = src[si];
}

__global__ void reduceArgMaxKernel(float *src, float *dst, float *arg, int dim_size, int block_size)
{
     int di = blockIdx.x * block_size + threadIdx.x;
     int si = di * dim_size;
     float now = src[si], max = now;
     for (int i = 1; i < dim_size; i++) {
          now = src[si+i];
          max = now > max ? now : max;
     }
     dst[di] = max;
     arg[di] = i - 1;
}

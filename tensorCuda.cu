#include "tensorCuda.h"

__global__ void sliceTensorKernel(float *dst, float *src, int ddim, int sdim, int start, int block_size)
{
     int di = blockIdx.x * block_size + threadIdx.x;
     int si = (blockIdx.x / ddim * sdim + blockIdx.x % ddim + start) * block_size + threadIdx.x;
     dst[di] = src[si];
}

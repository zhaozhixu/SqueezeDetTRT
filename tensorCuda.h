#include <cuda_runtime.h>
#include <iostream>

#define MAX_THREADS_PER_BLOCK 1024
#define BLOCK_SIZE MAX_THREADS_PER_BLOCK

__global__ void sliceTensorKernel(float *dst, float *src, int ddim, int sdim, int start, int block_size);
__global__ void reduceArgMaxKernel(float *src, float *dst, float *arg, int dim_size, int block_size)

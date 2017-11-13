#include <cuda_runtime.h>
#include <iostream>

#define MAX_THREADS_PER_BLOCK 1024
#define BLOCK_SIZE MAX_THREADS_PER_BLOCK

__global__ void sliceTensorKernel(float *dst, float *src, int ddim, int sdim, int start, int block_size);

#include <cuda_runtime.h>
#include <iostream>

#define MAX_THREADS_PER_BLOCK 1024
#define BLOCK_SIZE MAX_THREADS_PER_BLOCK

__global__ void sliceTensorKernel(float *src, float *dst, int sdim, int ddim, int start, int block_size);
__global__ void reduceArgMaxKernel(float *src, float *dst, float *arg, int dim_size, int block_size);
__global__ void multiplyElementKernel(float *src1, float *src2, float *dst, int block_size);
__global__ void transformBboxSQDKernel(float *delta, float *anchor, float *res, int block_size);

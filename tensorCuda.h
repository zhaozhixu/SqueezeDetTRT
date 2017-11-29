#ifndef _TENSOR_CUDA_H_
#define _TENSOR_CUDA_H_

#define MAX_THREADS_PER_BLOCK 1024
#define BLOCK_SIZE MAX_THREADS_PER_BLOCK

__global__ void sliceTensorKernel(float *src, float *dst, int sdim, int ddim, int start, int block_size);
__global__ void reduceArgMaxKernel(float *src, float *dst, float *arg, int dim_size, int block_size, int total);
__global__ void multiplyElementKernel(float *src1, float *src2, float *dst, int block_size, int total);
__global__ void transformBboxSQDKernel(float *delta, float *anchor, float *res, float width, float height, float *x_scales, float *y_scales, int batch_vol, int block_size, int total);
__global__ void pickElementsKernel(float *src, float *dst, int *idx, int len, int stride, int block_size);

#endif  /* _TENSOR_CUDA_H_ */

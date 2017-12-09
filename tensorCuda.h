#ifndef _TENSOR_CUDA_H_
#define _TENSOR_CUDA_H_

#define MAX_THREADS_PER_BLOCK 1024
#define BLOCK_SIZE MAX_THREADS_PER_BLOCK

__global__ void sliceTensorKernel(float *src, float *dst, int start, int s_vol, int d_vol, int vol, int block_size, int total);
__global__ void reduceArgMaxKernel(float *src, float *dst, float *arg, int dim_size, int reduce_vol, int batch_vol, int block_size, int total);
__global__ void multiplyElementKernel(float *src1, float *src2, float *dst, int block_size, int total);
__global__ void transposeTensorKernel(float *src, float *dst, int ndim, int *s_dims, int *d_dims, int *s_ids, int *d_ids, int *axes, int block_size, int total);
__global__ void transformBboxSQDKernel(float *delta, float *anchor, float *res, float width, float height, float *x_scales, float *y_scales, int block_size, int total);
__global__ void pickElementsKernel(float *src, float *dst, int *idx, int stride, int block_size, int total);

#endif  /* _TENSOR_CUDA_H_ */

#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>

#define max(a, b) ((a) > (b) ? (a) : (b))
#define min(a, b) ((a) < (b) ? (a) : (b))

static __device__ float E = 2.718281828;

static __device__ int getIndex(int *ids, int ndim, int *dims)
{
     int i, id;
     for (i = 0, id = ids[0]; i < ndim-1; i++)
          id = dims[i+1] * id + ids[i+1];
     return id;
}

static __device__ void getIndexes(int id, int *ids, int ndim, int *dims)
{
     for (int i = ndim-1; i >=0; i--) {
          ids[i] = id % dims[i];
          id = id / dims[i];
     }
}

/* __global__ void sliceTensorKernel(float *src, float *dst, int sdim, int ddim, int start, int block_size) */
/* { */
/*      int di = blockIdx.x * block_size + threadIdx.x; */
/*      /\* si is the index of src elements to be copied. */
/*         The "block index" of src[si] is (blockIdx.x / ddim * sdim + blockIdx.x % ddim + start) *\/ */
/*      int si = (blockIdx.x / ddim * sdim + blockIdx.x % ddim + start) * block_size + threadIdx.x; */
/*      dst[di] = src[si]; */
/* } */

__global__ void sliceTensorKernel(float *src, float *dst, int start, int s_vol, int d_vol, int vol, int block_size, int total)
{
     int di = blockIdx.x * block_size + threadIdx.x;
     if (di >= total)
          return;
     int si = di / d_vol * s_vol + di % d_vol + start * vol;
     dst[di] = src[si];
}

__global__ void reduceArgMaxKernel(float *src, float *dst, float *arg, int dim_size, int reduce_vol, int batch_vol, int block_size, int total)
{
     int di = blockIdx.x * block_size + threadIdx.x;
     if (di >= total)
          return;

     /* src[si] is the first element in this thread to be compared, then
        si = batch_vol * batch + (di - reduce_vol * batch),
        where batch = di / reduce_vol,
        which is the same as the following code: */
     int si = (batch_vol - reduce_vol) * (di / reduce_vol) + di;
     float now = src[si], max = now;
     int maxi = 0;
     for (int i = 1; i < dim_size; i++) {
          now = src[si+i*reduce_vol];
          if (now > max) {
               max = now;
               maxi = i;
          }
     }
     dst[di] = max;
     arg[di] = maxi;
}

__global__ void multiplyElementKernel(float *src1, float *src2, float *dst, int block_size, int total)
{
     int di = blockIdx.x * block_size + threadIdx.x;
     if (di >= total)
          return;
     dst[di] = src1[di] * src2[di];
}

__global__ void transposeTensorKernel(float *src, float *dst, int ndim, int *s_dims, int *d_dims, int *s_ids, int *d_ids, int *axes, int block_size, int total)
{
     int di = blockIdx.x * block_size + threadIdx.x;
     if (di >= total)
          return;

     int *t_s_ids = s_ids + di * ndim;
     int *t_d_ids = d_ids + di * ndim;
     getIndexes(di, t_d_ids, ndim, d_dims);
     for (int i = 0; i < ndim; i++)
          t_s_ids[axes[i]] = t_d_ids[i];
     int si = getIndex(t_s_ids, ndim, s_dims);

     dst[di] = src[si];
}

__global__ void transformBboxSQDKernel(float *delta, float *anchor, float *res, float width, float height, float *x_scales, float *y_scales, int anchor_num, int block_size, int total)
{
     int di = blockIdx.x * block_size + threadIdx.x;
     if (di >= total)
          return;

     int batch_idx = di / anchor_num;
     float x_scale = x_scales[batch_idx];
     float y_scale = y_scales[batch_idx];
     float img_width = width / x_scale;
     float img_height = height / y_scale;

     /* si is the index of the first elements to be computed in the thread, then
        si = 4 * anchor_num * batch_idx + (di - anchor_num * batch_idx),
        which is the same as the following code: */
     int si = 3 * anchor_num * batch_idx  + di;
     /* take 4 elements from each of delta and anchor */
     float d[4] = {delta[si], delta[si+anchor_num], delta[si+2*anchor_num], delta[si+3*anchor_num]};
     float a[4] = {anchor[si], anchor[si+anchor_num], anchor[si+2*anchor_num], anchor[si+3*anchor_num]};
     /* compute and put 4 result elements to res, according to SqueezeDet's source code */
     float cx = a[0] + d[0] * a[2] / x_scale;
     float cy = a[1] + d[1] * a[3] / y_scale;
     float w = a[2] * (d[2] < 1 ? expf(d[2]) : d[2] * E) / x_scale;
     float h = a[3] * (d[3] < 1 ? expf(d[3]) : d[3] * E) / y_scale;
     res[di] = min(max(cx - w * 0.5, 0), img_width - 1);
     res[di+anchor_num] = min(max(cy - h * 0.5, 0), img_height - 1);
     res[di+2*anchor_num] = max(min(cx + w * 0.5, img_width - 1), 0);
     res[di+3*anchor_num] = max(min(cy + h * 0.5, img_height - 1), 0);
}

__global__ void pickElementsKernel(float *src, float *dst, int *idx, int stride, int block_size, int total)
{
     int di = blockIdx.x * block_size + threadIdx.x;
     if (di >= total)
          return;
     int si = idx[di];
     for (int i = 0; i < stride; i++)
          dst[di*stride+i] = src[si*stride+i];
}

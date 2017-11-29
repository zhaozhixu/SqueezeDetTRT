#include <cuda_runtime.h>

#define max(a, b) ((a) > (b) ? (a) : (b))
#define min(a, b) ((a) < (b) ? (a) : (b))

static __device__ float E = 2.718281828;

__global__ void sliceTensorKernel(float *src, float *dst, int sdim, int ddim, int start, int block_size)
{
     int di = blockIdx.x * block_size + threadIdx.x;
     int si = (blockIdx.x / ddim * sdim + blockIdx.x % ddim + start) * block_size + threadIdx.x;
     dst[di] = src[si];
}

__global__ void reduceArgMaxKernel(float *src, float *dst, float *arg, int dim_size, int block_size, int total)
{
     int di = blockIdx.x * block_size + threadIdx.x;
     if (di >= total)
          return;
     int si = di * dim_size;
     float now = src[si], max = now;
     int maxi = 0;
     for (int i = 1; i < dim_size; i++) {
          now = src[si+i];
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

__global__ void transformBboxSQDKernel(float *delta, float *anchor, float *res, float img_width, float img_height, float x_scale, float y_scale, int block_size, int total)
{
     int di = (blockIdx.x * block_size + threadIdx.x) * 4;
     if (di >= total)
          return;
     float d[4] = {delta[di], delta[di+1], delta[di+2], delta[di+3]};
     float a[4] = {anchor[di], anchor[di+1], anchor[di+2], anchor[di+3]};
     float cx = a[0] + d[0] * a[2] / x_scale;
     float cy = a[1] + d[1] * a[3] / y_scale;
     float w = a[2] * (d[2] < 1 ? expf(d[2]) : d[2] * E) / x_scale;
     float h = a[3] * (d[3] < 1 ? expf(d[3]) : d[3] * E) / y_scale;
     res[di] = min(max(cx - w * 0.5, 0), img_width - 1);
     res[di+1] = min(max(cy - h * 0.5, 0), img_height - 1);
     res[di+2] = max(min(cx + w * 0.5, img_width - 1), 0);
     res[di+3] = max(min(cy + h * 0.5, img_height - 1), 0);
}

__global__ void pickElementsKernel(float *src, float *dst, int *idx, int len, int stride, int block_size)
{
     int di = blockIdx.x * block_size + threadIdx.x;
     if (di >= len)
          return;
     int si = idx[di];
     for (int i = 0; i < stride; i++)
          dst[di*stride+i] = src[si*stride+i];
}

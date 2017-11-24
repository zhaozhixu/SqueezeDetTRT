#include <cuda_runtime.h>

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

__global__ void transformBboxSQDKernel(float *delta, float *anchor, float *res, int block_size, int total)
{
     int di = (blockIdx.x * block_size + threadIdx.x) * 4;
     if (di >= total)
          return;
     float d[4] = {delta[di], delta[di+1], delta[di+2], delta[di+3]};
     float a[4] = {anchor[di], anchor[di+1], anchor[di+2], anchor[di+3]};
     float cx = a[0] + d[0] * a[2];
     float cy = a[1] + d[1] * a[3];
     float w = a[2] * (d[2] < 1 ? expf(d[2]) : d[2] * E);
     float h = a[3] * (d[3] < 1 ? expf(d[3]) : d[3] * E);
     res[di] = cx - w * 0.5;
     res[di+1] = cy - h * 0.5;
     res[di+2] = cx + w * 0.5;
     res[di+3] = cy + h * 0.5;
}

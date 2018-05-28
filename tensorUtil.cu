#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
/* #include "tensorCuda.h" */
#include "tensorUtil.h"
#include "errorHandle.h"
#include "sdt_alloc.h"

#define MAXDIM 8
#define max(a, b) ((a) > (b) ? (a) : (b))
#define min(a, b) ((a) < (b) ? (a) : (b))
#define MAX_THREADS_PER_BLOCK 1024
#define BLOCK_SIZE MAX_THREADS_PER_BLOCK

static float EPSILON = 1e-16;

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

/* __global__ void sliceTensorKernel(float *src, float *dst, int ndim, int *sdims, int *ddims, int naxis, int *axes, int *n_pick_ids, int **pick_ids, int *sids, int *dids, int block_size, int total) */
/* { */
/*      int di = blockIdx.x * block_size + threadIdx.x; */
/*      if (di >= total) */
/*           return; */

/*      int *t_sids = sids + di * ndim; */
/*      int *t_dids = dids + di * ndim; */
/*      getIndexes(di, t_dids, ndim, ddims); */
/*      int i, j; */
/*      for (i = 0, j = 0; i < ndim; i++) { */
/*           if (i != axes[j]) { */
/*                t_sids[i] = t_dids[i]; */
/*           } else { */
/*                t_sids[i] = pick_ids[j][t_dids[i]]; */
/*                j++; */
/*           } */
/*      } */

/*      int si = getIndex(t_sids, ndim, sdims); */
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

__global__ void transformBboxSQDKernel(float *delta, float *anchor, float *res, float width, float height, float img_width, float img_height, int x_shift, int y_shift, int block_size, int total)
{
     int di = blockIdx.x * block_size + threadIdx.x;
     if (di >= total)
          return;

     /* int batch_idx = di / anchor_num; */
     /* now only support batch_size = 1 */
     float x_scale = 1.0 * img_width / width;
     float y_scale = 1.0 * img_height / height;

     /* (not used) si is the index of the first elements to be computed in the thread, then
        si = 4 * anchor_num * batch_idx + (di - anchor_num * batch_idx),
        which is the same as the following code: */
     /* int si = 3 * anchor_num * batch_idx  + di; */
     /* take 4 elements from each of delta and anchor */
     int si = di * 4;
     float d[4] = {delta[si], delta[si+1], delta[si+2], delta[si+3]};
     float a[4] = {anchor[si], anchor[si+1], anchor[si+2], anchor[si+3]};
     /* compute and put 4 result elements to res, according to SqueezeDet's source code */

     /* TODO: don't know why (maybe the resize), always has some shift compared to groundtruth*/
     float cx = (a[0] + d[0] * a[2]) * x_scale + x_shift;
     float cy = (a[1] + d[1] * a[3]) * y_scale + y_shift;
     float w = (a[2] * (d[2] < 1 ? expf(d[2]) : d[2] * E)) * x_scale;
     float h = (a[3] * (d[3] < 1 ? expf(d[3]) : d[3] * E)) * y_scale;
     res[si] = min(max(cx - w * 0.5, 0), img_width - 1);
     res[si+1] = min(max(cy - h * 0.5, 0), img_height - 1);
     res[si+2] = max(min(cx + w * 0.5, img_width - 1), 0);
     res[si+3] = max(min(cy + h * 0.5, img_height - 1), 0);
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

static void assertTensor(const Tensor *tensor)
{
     assert(tensor && tensor->data);
     assert(tensor->ndim < MAXDIM && tensor->ndim > 0);
     assert(tensor->len == computeLength(tensor->ndim, tensor->dims));
}

int isTensorValid(const Tensor *tensor)
{
     assert(tensor);
     assert(tensor->data);
     assert(tensor->ndim < MAXDIM && tensor->ndim > 0 );
     assert(tensor->len == computeLength(tensor->ndim, tensor->dims));
     return (tensor && tensor->data &&
             tensor->ndim < MAXDIM && tensor->ndim > 0 &&
             tensor->len == computeLength(tensor->ndim, tensor->dims));
}

int isShapeEqual(const Tensor *t1, const Tensor *t2)
{
     assertTensor(t1);
     assertTensor(t2);
     if (t1->ndim == t2->ndim) {
          int ndim = t1->ndim;
          while (--ndim >= 0)
               if (t1->dims[ndim] != t2->dims[ndim])
                    return 0;
          return 1;
     }
     return 0;
}

/* can only identify host memory alloced by cudaMallocHost, etc */
int isHostMem(const void *ptr)
{
     cudaPointerAttributes attributes;
     checkError(cudaPointerGetAttributes(&attributes, ptr));
     return attributes.memoryType == cudaMemoryTypeHost;
}

int isDeviceMem(const void *ptr)
{
     cudaPointerAttributes attributes;
     checkError(cudaPointerGetAttributes(&attributes, ptr));
     return attributes.memoryType == cudaMemoryTypeDevice;
}

void *cloneMem(const void *src, size_t size, CloneKind kind)
{
     assert(src);
     void *p;
     switch (kind) {
     case H2H:
          p = sdt_alloc(size);
          memmove(p, src, size);
          return p;
     case H2D:
          checkError(cudaMalloc(&p, size));
          checkError(cudaMemcpy(p, src, size, cudaMemcpyHostToDevice));
          return p;
     case D2D:
          assert(isDeviceMem(src));
          checkError(cudaMalloc(&p, size));
          checkError(cudaMemcpy(p, src, size, cudaMemcpyDeviceToDevice));
          return p;
     case D2H:
          assert(isDeviceMem(src));
          p = sdt_alloc(size);
          checkError(cudaMemcpy(p, src, size, cudaMemcpyDeviceToHost));
          return p;
     default:
          fprintf(stderr, "unknown CloneKind %d\n", kind);
          return NULL;
     }

}

Tensor *cloneTensor(const Tensor *src, CloneKind kind)
{
     assert(isTensorValid(src));
     float *data = (float *)cloneMem(src->data, src->len * sizeof(float), kind);
     Tensor *dst = createTensor(data, src->ndim, src->dims);
     return dst;
}

void *repeatMem(void *data, size_t size, int times, CloneKind kind)
{
     assert(data && times > 0);
     void *p, *dst;
     int i;
     switch (kind) {
     case H2H:
          dst = p = sdt_alloc(size * times);
          for (i = 0; i < times; i++, p = (char *)p + size * times)
               memmove(p, data, size);
          return dst;
     case H2D:
          checkError(cudaMalloc(&p, size * times));
          dst = p;
          for (i = 0; i < times; i++, p = (char *)p + size * times)
               checkError(cudaMemcpy(p, data, size, cudaMemcpyHostToDevice));
          return dst;
     case D2D:
          assert(isDeviceMem(data));
          checkError(cudaMalloc(&p, size * times));
          dst = p;
          for (i = 0; i < times; i++, p = (char *)p + size * times)
               checkError(cudaMemcpy(p, data, size, cudaMemcpyDeviceToDevice));
          return dst;
     case D2H:
          assert(isDeviceMem(data));
          dst = p = sdt_alloc(size * times);
          for (i = 0; i < times; i++, p = (char *)p + size * times)
               checkError(cudaMemcpy(p, data, size, cudaMemcpyDeviceToHost));
          return dst;
     default:
          fprintf(stderr, "unknown CloneKind %d\n", kind);
          return NULL;
     }
}


int computeLength(int ndim, const int *dims)
{
     if (dims) {
          int i, len = 1;
          for (i = 0; i < ndim; i++)
               len *= dims[i];
          return len;
     }
     fprintf(stderr, "Warning: null dims in computeLength\n");
     return 0;
}

Tensor *createTensor(float *data, int ndim, const int *dims)
{
     Tensor *t = (Tensor *)sdt_alloc(sizeof(Tensor));
     t->data = data;
     t->ndim = ndim;
     t->dims = (int *)sdt_alloc(sizeof(int) * ndim);
     memmove(t->dims, dims, sizeof(int) * ndim);
     t->len = computeLength(ndim, dims);
     return t;
}

Tensor *mallocTensor(int ndim, const int* dims, const MallocKind mkind)
{
     Tensor *t = createTensor(NULL, ndim, dims);
     float *f;

     switch (mkind) {
     case HOST:
          f = (float *)sdt_alloc(t->len * sizeof(float));
          break;
     case DEVICE:
          checkError(cudaMalloc(&f, t->len * sizeof(float)));
          break;
     default:
          fprintf(stderr, "unknown MallocKind %d\n", mkind);
          return NULL;
     }

     t->data = f;
     return t;
}

void freeTensor(Tensor *t, int do_free_data)
{
     assert(isTensorValid(t));
     sdt_free(t->dims);
     if (do_free_data) {
          if (isDeviceMem(t->data))
               checkError(cudaFree(t->data));
          else
               sdt_free(t->data);
     }
     sdt_free(t);
}

void fprintShape(FILE *stream, const Tensor *tensor)
{
     assert(tensor && tensor->dims);

     int i;
     fprintf(stream, "(");
     for (i = 0; i < tensor->ndim; i++) {
          fprintf(stream, "%d", tensor->dims[i]);
          if (i != tensor->ndim-1)
               fprintf(stream, ", ");
     }
     fprintf(stream, ")\n");
}

void fprintTensor(FILE *stream, const Tensor *tensor, const char *fmt)
{
     assertTensor(tensor);
     int dim_sizes[MAXDIM], dim_levels[MAXDIM]; /* dimision size and how deep current chars go */
     int ndim = tensor->ndim, len = tensor->len, *dims = tensor->dims; /* pointer short cut */
     float *data = tensor->data;
     char left_buf[MAXDIM+1], right_buf[MAXDIM+1]; /* buffer for brackets */
     char *lp = left_buf, *rp = right_buf;
     size_t right_len;
     int i, j, k;

     dim_sizes[ndim-1] = tensor->dims[ndim-1];
     dim_levels[ndim-1] = 0;
     for (i = ndim-2; i >= 0; i--) {
          dim_sizes[i] = dims[i] * dim_sizes[i+1];
          dim_levels[i] = 0;
     }
     for (i = 0; i < len; i++) {
          for (j = 0; j < ndim; j++) {
               if (i % dim_sizes[j] == 0)
                    dim_levels[j]++;
               if (dim_levels[j] == 1) {
                    *lp++ = '[';
                    dim_levels[j]++;
               }
               if (dim_levels[j] == 3) {
                    *rp++ = ']';
                    if (j != 0 && dim_levels[j] > dim_levels[j-1]) {
                         *lp++ = '[';
                         dim_levels[j] = 2;
                    } else
                         dim_levels[j] = 0;
               }
          }
          *lp = *rp = '\0';
          fprintf(stream, "%s", right_buf);
          if (*right_buf != '\0') {
               fprintf(stream, "\n");
               right_len = strlen(right_buf);
               for (k = ndim-right_len; k > 0; k--)
                    fprintf(stream, " ");
          }
          fprintf(stream, "%s", left_buf);
          if (*left_buf == '\0')
               fprintf(stream, " ");
          fprintf(stream, fmt, data[i]);
          lp = left_buf, rp = right_buf;
     }
     for (j = 0; j < ndim; j++)
          fprintf(stream, "]");
     fprintf(stream, "\n");
}

void printTensor(const Tensor *tensor, const char *fmt)
{
     fprintTensor(stdout, tensor, fmt);
}

void fprintDeviceTensor(FILE *stream, const Tensor *d_tensor, const char *fmt)
{
     assert(isTensorValid(d_tensor));
     Tensor *h_tensor = cloneTensor(d_tensor, D2H);
     fprintTensor(stream, h_tensor, fmt);
     free(h_tensor->data); /* TODO: free t_tensor */
}

void printDeviceTensor(const Tensor *d_tensor, const char *fmt)
{
     fprintDeviceTensor(stdout, d_tensor, fmt);
}

void saveTensor(const char *file_name, const Tensor *tensor, const char *fmt)
{
     FILE *fp = fopen(file_name, "w");
     fprintTensor(fp, tensor, fmt);
     fclose(fp);
}

void saveDeviceTensor(const char *file_name, const Tensor *d_tensor, const char *fmt)
{
     FILE *fp = fopen(file_name, "w");
     fprintDeviceTensor(fp, d_tensor, fmt);
     fclose(fp);
}

/* Tensor *createSlicedTensor(const Tensor *src, int dim, int start, int len) */
/* { */
/*      assert(isTensorValid(src)); */
/*      assert(dim <= src->ndim && dim >= 0); */
/*      assert(len+start <= src->dims[dim]); */

/*      Tensor *dst = (Tensor *)sdt_alloc(sizeof(Tensor)); /\* new tensor *\/ */
/*      dst->ndim = src->ndim; */
/*      dst->dims = (int *)sdt_alloc(sizeof(int) * dst->ndim); */
/*      memmove(dst->dims, src->dims, sizeof(int) * dst->ndim); */
/*      dst->dims[dim] = len; */
/*      dst->len = src->len / src->dims[dim] * len; */
/*      dst->data = (float *)sdt_alloc(dst->len * sizeof(float)); */
/*      return dst; */
/* } */

/* Tensor *sliceTensor(const Tensor *src, Tensor *dst, int dim, int start, int len) */
/* { */
/*      assert(isTensorValid(src) && isTensorValid(dst)); */
/*      assert(dst->ndim == src->ndim); */
/*      for (int i = 0; i < dst->ndim; i++) */
/*           assert(i == dim ? dst->dims[i] == len : dst->dims[i] == src->dims[i]); */

/*      int i, block_size, block_num; /\* block size and number for copy operation *\/ */
/*      for (i = dim+1, block_size = 1; i < dst->ndim; i++) */
/*           block_size *= dst->dims[i]; */
/*      for (i = 0, block_num = 1; i <= dim; i++) */
/*           block_num *= dst->dims[i]; */

/*      int index; */
/*      float *dp = dst->data, *sp = src->data; */
/*      size_t floats_size = block_size * sizeof(float); */
/*      for (i = 0; i < block_num; i++) { */
/*           index = i / len * src->dims[dim] + i % len + start; */
/*           memmove(dp+i*block_size, sp+index*block_size, floats_size); */
/*      } */

/*      return dst; */
/* } */

Tensor *createSlicedTensor(const Tensor *src, int dim, int start, int len)
{
     assert(isTensorValid(src));
     assert(dim <= MAXDIM);
     assert(len+start <= src->dims[dim]);

     Tensor *dst = (Tensor *)sdt_alloc(sizeof(Tensor)); /* new tensor */
     dst->ndim = src->ndim;
     dst->dims = (int *)sdt_alloc(sizeof(int) * dst->ndim);
     memmove(dst->dims, src->dims, sizeof(int) * dst->ndim);
     dst->dims[dim] = len;
     dst->len = src->len / src->dims[dim] * len;
     checkError(cudaMalloc(&dst->data, sizeof(float) * dst->len));
     return dst;
}

/* Tensor *sliceTensor(const Tensor *src, Tensor *dst, int dim, int start, int len) */
/* { */
/*      assert(isTensorValid(src) && isTensorValid(dst)); */
/*      assert(isDeviceMem(src->data) && isDeviceMem(dst->data)); */
/*      assert(dst->ndim == src->ndim); */
/*      for (int i = 0; i < dst->ndim; i++) */
/*           assert(i == dim ? dst->dims[i] == len : dst->dims[i] == src->dims[i]); */

/*      int i, block_size, block_num; /\* block size and number of cuda threads *\/ */
/*      int ddim = dst->dims[dim], sdim = src->dims[dim]; */
/*      for (i = dim+1, block_size = 1; i < dst->ndim; i++) */
/*           block_size *= dst->dims[i]; */
/*      for (i = 0, block_num = 1; i <= dim; i++) */
/*           block_num *= dst->dims[i]; */

/*      sliceTensorKernel<<<block_num, block_size>>>(src->data, dst->data, sdim, ddim, start, block_size); */
/*      return dst; */
/* } */
/* axes should be in ascending order */
/* Tensor *sliceTensor(const Tensor *src, Tensor *dst, int naxis, int *axes, int *starts, int *ends, int *strides, int **workspace) */
/* { */
/*      assert(isTensorValid(src) && isTensorValid(dst)); */
/*      assert(isDeviceMem(src->data) && isDeviceMem(dst->data)); */
/*      assert(dst->ndim == src->ndim); */
/*      for (int i = 0; i < dst->ndim; i++) */
/*           assert(i == dim ? dst->dims[i] == len : dst->dims[i] == src->dims[i]); */

/*      int i; */
/*      int *sdims, *ddims, *axes_device, *n_pick_ids_host, **pick_ids_host, *n_pick_ids_device, **pick_ids_device, *sids, *dids, *ends_tmp; */
/*      sdims = (int *)cloneMem(src->dims, sizeof(int)*src->ndim, H2D); */
/*      ddims = (int *)cloneMem(dst->dims, sizeof(int)*dst->ndim, H2D); */
/*      axes_device = (int *)cloneMem(axes, sizeof(int)*naxis, H2D); */
/*      n */
/*      ends_tmp = (int *)cloneMem(ends, sizeof(int)*naxis, H2H); */
/*      if (!workspace) { */
/*           checkError(cudaMalloc(&sids, sizeof(int) * dst->ndim * dst->len)); */
/*           checkError(cudaMalloc(&dids, sizeof(int) * dst->ndim * dst->len)); */
/*      } else { */
/*           sids = workspace[0]; */
/*           dids = workspace[1]; */
/*      } */
/*      for (i = 0; i < naxis; i++) { */
/*           if (ends_tmp[i] > src->dims[axes[i]]) */
/*                ends_tmp[i] = src->dims[axes[i]]; */
/*           n_pick_ids[i] = */
/*      } */

/*      int thread_num, block_size, block_num; /\* block size and number of cuda threads *\/ */
/*      thread_num = dst->len; */
/*      block_size = MAX_THREADS_PER_BLOCK; */
/*      block_num = thread_num / block_size + 1; */

/*      sliceTensorKernel<<<block_num, block_size>>>(src->data, dst->data, start, s_vol, d_vol, vol, block_size, thread_num); */

/*      checkError(cudaFree(sdims)); */
/*      checkError(cudaFree(ddims)); */
/*      checkError(cudaFree(axes_device)); */
/*      checkError(cudaFree(n_pick_ids)); */
/*      sdt_free(ends_tmp); */
/*      return dst; */
/* } */

Tensor *sliceTensor(const Tensor *src, Tensor *dst, int dim, int start, int len)
{
     assert(isTensorValid(src));
     assert(isTensorValid(dst));
     assert(isDeviceMem(src->data) && isDeviceMem(dst->data));
     assert(dst->ndim == src->ndim);
     for (int i = 0; i < dst->ndim; i++)
          assert(i == dim ? dst->dims[i] == len : dst->dims[i] == src->dims[i]);

     int i, d_vol, s_vol, vol;
     int thread_num, block_size, block_num; /* block size and number of cuda threads */
     for (i = dim+1, vol = 1; i < dst->ndim; i++)
          vol *= dst->dims[i];
     d_vol = vol * dst->dims[dim];
     s_vol = vol * src->dims[dim];
     thread_num = dst->len;
     block_size = MAX_THREADS_PER_BLOCK;
     block_num = thread_num / block_size + 1;

     sliceTensorKernel<<<block_num, block_size>>>(src->data, dst->data, start, s_vol, d_vol, vol, block_size, thread_num);
     return dst;
}

/* in-place reshape tensor */
Tensor *reshapeTensor(const Tensor *src, int newNdim, const int *newDims)
{
     assert(isTensorValid(src));
     assert(newDims);
     /* fprintf(stderr, "len = %d, compute = %d\n", src->len, computeLength(newNdim, newDims)); */
     assert(src->len == computeLength(newNdim, newDims));
     Tensor *dst = createTensor(src->data, newNdim, newDims); /* new tensor */
     return dst;
}

Tensor *createReducedTensor(const Tensor *src, int dim)
{
     assert(isTensorValid(src));
     assert(dim < src->ndim && dim >= 0);

     Tensor *dst = (Tensor *)sdt_alloc(sizeof(Tensor));
     dst->ndim = src->ndim;
     dst->dims = (int *)sdt_alloc(sizeof(int) * dst->ndim);
     memmove(dst->dims, src->dims, sizeof(int) * dst->ndim);
     dst->dims[dim] = 1;
     dst->len = computeLength(dst->ndim, dst->dims);
     checkError(cudaMalloc(&dst->data, sizeof(float) * dst->len));
     return dst;
}

void *reduceArgMax(const Tensor *src, Tensor *dst, Tensor *arg, int dim)
{
     assert(isTensorValid(src) && isTensorValid(dst) && isTensorValid(arg));
     assert(isDeviceMem(src->data) && isDeviceMem(dst->data) && isDeviceMem(arg->data));
     assert(dim < src->ndim && dim >= 0);
     for (int i = 0; i < dst->ndim; i++)
          assert(i == dim ? dst->dims[i] == 1 : dst->dims[i] == src->dims[i] &&
                 i == dim ? arg->dims[i] == 1 : arg->dims[i] == src->dims[i]);

     /* suppose the shape of src is [N, C, H, W], dim = 1, then thread_num is N x H x W
        reduce_vol is H x W, index_vol is C x H x W */
     int i, thread_num, block_size, block_num, reduce_vol, index_vol;
     for (i = dim+1, thread_num = 1; i < dst->ndim; i++)
          thread_num *= dst->dims[i];
     reduce_vol = thread_num;
     index_vol = thread_num * src->dims[dim];
     for (i = 0; i < dim; i++)
          thread_num *= dst->dims[i];
     block_size = MAX_THREADS_PER_BLOCK;
     block_num = thread_num / block_size + 1;

     reduceArgMaxKernel<<<block_num, block_size>>>(src->data, dst->data, arg->data, src->dims[dim], reduce_vol, index_vol, block_size, thread_num);
     return dst;
}

Tensor *multiplyElement(const Tensor *src1, const Tensor *src2, Tensor *dst)
{
     assert(isShapeEqual(src1, src2));
     assert(isShapeEqual(src1, dst));
     assert(isDeviceMem(src1->data) && isDeviceMem(src2->data) && isDeviceMem(dst->data));

     int thread_num, block_size, block_num;
     thread_num = dst->len;
     block_size = MAX_THREADS_PER_BLOCK;
     block_num = thread_num / block_size + 1;

     multiplyElementKernel<<<block_num, block_size>>>(src1->data, src2->data, dst->data, block_size, dst->len);
     return dst;
}

/* (optional) workspace size equals (sizeof(int) * dst->ndim * dst->len), two of them */
Tensor *transposeTensor(const Tensor *src, Tensor *dst, int *axes, int **workspace)
{
     assert(isTensorValid(src) && isTensorValid(dst));
     assert(src->len == dst->len);
     assert(src->ndim == dst->ndim);

     int *s_ids, *d_ids, *s_dims, *d_dims;
     int thread_num, block_size, block_num;
     thread_num = dst->len;
     block_size = MAX_THREADS_PER_BLOCK;
     block_num = thread_num / block_size + 1;
     s_dims = (int *)cloneMem(src->dims, sizeof(int) * src->ndim, H2D);
     d_dims = (int *)cloneMem(dst->dims, sizeof(int) * dst->ndim, H2D);
     if (!workspace) {
          checkError(cudaMalloc(&s_ids, sizeof(int) * dst->ndim * thread_num));
          checkError(cudaMalloc(&d_ids, sizeof(int) * dst->ndim * thread_num));
     } else {
          s_ids = workspace[0];
          d_ids = workspace[1];
     }

     transposeTensorKernel<<<block_num, block_size>>>(src->data, dst->data, dst->ndim, s_dims, d_dims, s_ids, d_ids, axes, block_size, thread_num);

     if (!workspace) {
          checkError(cudaFree(s_ids));
          checkError(cudaFree(d_ids));
     }
     checkError(cudaFree(s_dims));
     checkError(cudaFree(d_dims));
     return dst;
}

/* transform from bbox delta to bbox coordinates, using hyper param EXP_THRESH = 1.0.
   delta, anchor, res are all of the same shape [..., 4]
   width and height are resized image width and height.
   x_scales and y_scales are (temporary) pointers to width/original_width and height/original_height. */
Tensor *transformBboxSQD(const Tensor *delta, const Tensor *anchor, Tensor *res, float width, float height, float img_width, float img_height, int x_shift, int y_shift)
{
     assert(isShapeEqual(delta, anchor));
     assert(isShapeEqual(delta, res));
     assert(delta->ndim == 5);
     assert(delta->dims[4] == 4);
     assert(isDeviceMem(delta->data) && isDeviceMem(anchor->data) && isDeviceMem(res->data));

     /* take 4 elements from each of delta and anchor,
        and put 4 result elements to res in one thread */
     int i, thread_num, block_size, block_num;
     for (i = 0, thread_num = 1; i < res->ndim-1; i++)
          thread_num *= res->dims[i];
     block_size = MAX_THREADS_PER_BLOCK;
     block_num = thread_num / block_size + 1;

     transformBboxSQDKernel<<<block_num, block_size>>>(delta->data, anchor->data, res->data, width, height, img_width, img_height, x_shift, y_shift, block_size, thread_num);
     return res;
}

void tensorIndexSort(Tensor *src, int *idx)
{
     assert(isTensorValid(src));
     assert(idx);
     assert(isDeviceMem(src->data) && isDeviceMem(idx));

     /* the thrust call below can be unreliable, sometimes produces error */
     /* now it works with compilation flag -arch=sm_35 */
     /* TODO: replace thrust call by our own kernel */
     thrust::sort_by_key(thrust::device, src->data, src->data + src->len, idx, thrust::greater<float>());
}

void pickElements(float *src, float *dst, int stride, int *idx, int len)
{
     assert(src && dst && idx);
     assert(isDeviceMem(src) && isDeviceMem(dst) && isDeviceMem(idx));

     int thread_num, block_size, block_num;
     thread_num = len;
     block_size = MAX_THREADS_PER_BLOCK;
     block_num = thread_num / block_size + 1;

     pickElementsKernel<<<block_num, block_size>>>(src, dst, idx, stride, block_size, thread_num);
}

/* void pickElements(float* src,float* dst,int stride,int* idx,int len) */
/* { */
/*      assert(src && dst && idx); */

/*      for (int i = 0; i < len; i++) { */
/*           for (int j = 0; j < stride; j++) { */
/*                fprintf(stderr, "i: %d j: %d idx[i]: %d src[idx[i]]: %.2f", */
/*                        i, j, idx[i], src[idx[i]]); */
/*                fprintf(stderr, "\n"); */
/*                dst[i*stride+j] = src[idx[i]*stride+j]; */
/*           } */
/*      } */
/* } */

/* compute the iou of two bboxes whose elements are {top_left_x, top_left_y, bottom_right_x, bottom_right_y} */
float computeIou(float *bbox0, float *bbox1)
{
     assert(bbox0 && bbox1);

     float lr, tb;              /* left-right, top-bottom for intersection*/
     float intersection, total;
     lr = min(bbox0[2], bbox1[2]) - max(bbox0[0], bbox1[0]);
     if (lr >= 0) {
          tb = min(bbox0[3], bbox1[3]) - max(bbox0[1], bbox1[1]);
          if (tb >= 0) {
               intersection = tb * lr + EPSILON;
               total = (bbox0[2] - bbox0[0]) * (bbox0[3] - bbox0[1]) +
                    (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1]) - intersection;
               return intersection / (total + EPSILON);
          }
     }
     return 0;
}

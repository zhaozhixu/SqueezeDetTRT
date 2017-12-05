#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include "tensorCuda.h"
#include "tensorUtil.h"
#include "errorHandle.h"

#define MAXDIM 8
#define max(a, b) ((a) > (b) ? (a) : (b))
#define min(a, b) ((a) < (b) ? (a) : (b))

static float EPSILON = 1e-16;

static void assertTensor(const Tensor *tensor)
{
     assert(tensor && tensor->data);
     assert(tensor->ndim < MAXDIM && tensor->ndim > 0);
     assert(tensor->len == computeLength(tensor->ndim, tensor->dims));
}

int isTensorValid(const Tensor *tensor)
{
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
     cudaPointerGetAttributes(&attributes, ptr);
     return attributes.memoryType == cudaMemoryTypeHost;
}

int isDeviceMem(const void *ptr)
{
     cudaPointerAttributes attributes;
     cudaPointerGetAttributes(&attributes, ptr);
     return attributes.memoryType == cudaMemoryTypeDevice;
}

void *cloneMem(const void *src, size_t size, CloneKind kind)
{
     assert(src);
     void *p;
     switch (kind) {
     case H2H:
          p = malloc(size);
          assert(p);
          memmove(p, src, size);
          return p;
     case H2D:
          checkError(cudaMalloc(&p, size));
          assert(p);
          checkError(cudaMemcpy(p, src, size, cudaMemcpyHostToDevice));
          return p;
     case D2D:
          assert(isDeviceMem(src));
          checkError(cudaMalloc(&p, size));
          assert(p);
          checkError(cudaMemcpy(p, src, size, cudaMemcpyDeviceToDevice));
          return p;
     case D2H:
          assert(isDeviceMem(src));
          p = malloc(size);
          assert(p);
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
          dst = p = malloc(size * times);
          assert(p);
          for (i = 0; i < times; i++, p = (char *)p + size * times)
               memmove(p, data, size);
          return dst;
     case H2D:
          cudaMalloc(&p, size * times);
          dst = p;
          assert(p);
          for (i = 0; i < times; i++, p = (char *)p + size * times)
               cudaMemcpy(p, data, size, cudaMemcpyHostToDevice);
          return dst;
     case D2D:
          assert(isDeviceMem(data));
          cudaMalloc(&p, size * times);
          dst = p;
          assert(p);
          for (i = 0; i < times; i++, p = (char *)p + size * times)
               cudaMemcpy(p, data, size, cudaMemcpyDeviceToDevice);
          return dst;
     case D2H:
          assert(isDeviceMem(data));
          dst = p = malloc(size * times);
          assert(p);
          for (i = 0; i < times; i++, p = (char *)p + size * times)
               cudaMemcpy(p, data, size, cudaMemcpyDeviceToHost);
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
     Tensor *t = (Tensor *)malloc(sizeof(Tensor));
     t->data = data;
     t->ndim = ndim;
     t->dims = (int *)malloc(sizeof(int) * ndim);
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
          f = (float *)malloc(t->len * sizeof(float));
          assert(f);
          break;
     case DEVICE:
          cudaMalloc(&f, t->len * sizeof(float));
          assert(f);
          break;
     default:
          fprintf(stderr, "unknown MallocKind %d\n", mkind);
          return NULL;
     }

     t->data = f;
     return t;
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

/*      Tensor *dst = (Tensor *)malloc(sizeof(Tensor)); /\* new tensor *\/ */
/*      dst->ndim = src->ndim; */
/*      dst->dims = (int *)malloc(sizeof(int) * dst->ndim); */
/*      memmove(dst->dims, src->dims, sizeof(int) * dst->ndim); */
/*      dst->dims[dim] = len; */
/*      dst->len = src->len / src->dims[dim] * len; */
/*      dst->data = (float *)malloc(dst->len * sizeof(float)); */
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

     Tensor *dst = (Tensor *)malloc(sizeof(Tensor)); /* new tensor */
     dst->ndim = src->ndim;
     dst->dims = (int *)malloc(sizeof(int) * dst->ndim);
     memmove(dst->dims, src->dims, sizeof(int) * dst->ndim);
     dst->dims[dim] = len;
     dst->len = src->len / src->dims[dim] * len;
     cudaMalloc(&dst->data, sizeof(float) * dst->len);
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

Tensor *sliceTensor(const Tensor *src, Tensor *dst, int dim, int start, int len)
{
     assert(isTensorValid(src) && isTensorValid(dst));
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

     sliceTensorKernel<<<block_num, block_size>>>(src->data, dst->data, start, s_vol, d_vol, vol, block_size);
     return dst;
}

/* in-place reshape tensor */
Tensor *reshapeTensor(const Tensor *src, int newNdim, const int *newDims)
{
     assert(isTensorValid(src));
     assert(newDims);
     assert(src->len == computeLength(newNdim, newDims));
     Tensor *dst = createTensor(src->data, newNdim, newDims); /* new tensor */
     return dst;
}

Tensor *createReducedTensor(const Tensor *src, int dim)
{
     assert(isTensorValid(src));
     assert(dim < src->ndim && dim >= 0);

     Tensor *dst = (Tensor *)malloc(sizeof(Tensor));
     dst->ndim = src->ndim;
     dst->dims = (int *)malloc(sizeof(int) * dst->ndim);
     memmove(dst->dims, src->dims, sizeof(int) * dst->ndim);
     dst->dims[dim] = 1;
     dst->len = computeLength(dst->ndim, dst->dims);
     cudaMalloc(&dst->data, sizeof(float) * dst->len);
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

/* transform from bbox delta to bbox coordinates, using hyper param EXP_THRESH = 1.0.
   delta, anchor, res are all of shape [N, 4, A], where N is batch size and A is anchors number per image.
   width and height are resized image width and height.
   x_scales and y_scales are length of N. */
Tensor *transformBboxSQD(const Tensor *delta, const Tensor *anchor, Tensor *res, float width, float height, float *x_scales, float *y_scales)
{
     assert(isShapeEqual(delta, anchor));
     assert(isShapeEqual(delta, res));
     assert(delta->ndim == 3);
     assert(delta->dims[1] == 4);
     assert(isDeviceMem(delta->data) && isDeviceMem(anchor->data) && isDeviceMem(res->data));
     assert(isDeviceMem(x_scales) && isDeviceMem(y_scales));

     /* take 4 elements from each of delta and anchor,
        and put 4 result elements to res in one thread */
     int thread_num, block_size, block_num, anchor_num;
     anchor_num = res->dims[2];
     thread_num = res->dims[0] * res->dims[2];
     block_size = MAX_THREADS_PER_BLOCK;
     block_num = thread_num / block_size + 1;

     transformBboxSQDKernel<<<block_num, block_size>>>(delta->data, anchor->data, res->data, width, height, x_scales, y_scales, anchor_num, block_size, thread_num);
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

     pickElementsKernel<<<block_num, block_size>>>(src, dst, idx, len, stride, block_size);
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

     float lr, tb;              /* left-right, top-bottom */
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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "tensorCuda.h"
#include "tensorUtil.h"

#define MAXDIM 8

static void assertTensor(const Tensor *tensor)
{
     assert(tensor && tensor->data);
     assert(tensor->ndim < MAXDIM && tensor->ndim > 0);
     assert(tensor->len == computeLength(tensor->ndim, tensor->dims));
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
          cudaMalloc(&p, size);
          assert(p);
          cudaMemcpy(p, src, size, cudaMemcpyHostToDevice);
          return p;
     case D2D:
          cudaMalloc(&p, size);
          assert(p);
          cudaMemcpy(p, src, size, cudaMemcpyDeviceToDevice);
          return p;
     case D2H:
          p = malloc(size);
          assert(p);
          cudaMemcpy(p, src, size, cudaMemcpyDeviceToHost);
          return p;
     default:
          fprintf(stderr, "unknown CloneKind %d\n", kind);
          return NULL;
     }

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
          cudaMalloc(&p, size * times);
          dst = p;
          assert(p);
          for (i = 0; i < times; i++, p = (char *)p + size * times)
               cudaMemcpy(p, data, size, cudaMemcpyDeviceToDevice);
          return dst;
     case D2H:
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
     assert(dims);
     int i, len = 1;
     for (i = 0; i < ndim; i++)
          len *= dims[i];
     return len;
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

void printTensor(const Tensor *tensor, const char *fmt)
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
          printf("%s", right_buf);
          if (*right_buf != '\0') {
               putchar('\n');
               right_len = strlen(right_buf);
               for (k = ndim-right_len; k > 0; k--)
                    putchar(' ');
          }
          printf("%s", left_buf);
          if (*left_buf == '\0')
               putchar(' ');
          printf(fmt, data[i]);
          lp = left_buf, rp = right_buf;
     }
     for (j = 0; j < ndim; j++)
          putchar(']');
     putchar('\n');
}

Tensor *createSlicedTensor(const Tensor *src, int dim, int start, int len)
{
     assertTensor(src);
     assert(dim <= src->ndim && dim >= 0);
     assert(len+start <= src->dims[dim]);

     Tensor *dst = (Tensor *)malloc(sizeof(Tensor)); /* new tensor */
     dst->ndim = src->ndim;
     dst->dims = (int *)malloc(sizeof(int) * dst->ndim);
     memmove(dst->dims, src->dims, sizeof(int) * dst->ndim);
     dst->dims[dim] = len;
     dst->len = src->len / src->dims[dim] * len;
     dst->data = (float *)malloc(dst->len * sizeof(float));
     return dst;
}

Tensor *sliceTensor(const Tensor *src, Tensor *dst, int dim, int start, int len)
{
     assertTensor(src);
     assertTensor(dst);
     assert(dst->ndim == src->ndim);
     for (int i = 0; i < dst->ndim; i++)
          assert(i == dim ? dst->dims[i] == len : dst->dims[i] == src->dims[i]);

     int i, block_size, block_num; /* block size and number for copy operation */
     for (i = dim+1, block_size = 1; i < dst->ndim; i++)
          block_size *= dst->dims[i];
     for (i = 0, block_num = 1; i <= dim; i++)
          block_num *= dst->dims[i];

     int index;
     float *dp = dst->data, *sp = src->data;
     size_t floats_size = block_size * sizeof(float);
     for (i = 0; i < block_num; i++) {
          index = i / len * src->dims[dim] + i % len + start;
          memmove(dp+i*block_size, sp+index*block_size, floats_size);
     }

     return dst;
}

Tensor *creatSlicedTensorCuda(const Tensor *src, int dim, int start, int len)
{
     assertTensor(src);
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

void *sliceTensorCuda(const Tensor *src, Tensor *dst, int dim, int start, int len)
{
     assertTensor(src);
     assertTensor(dst);
     assert(dst->ndim == src->ndim);
     for (int i = 0; i < dst->ndim; i++)
          assert(i == dim ? dst->dims[i] == len : dst->dims[i] == src->dims[i]);

     int i, block_size, block_num; /* block size and number of cuda threads */
     int ddim = dst->dims[dim], sdim = src->dims[dim];
     for (i = dim+1, block_size = 1; i < dst->ndim; i++)
          block_size *= dst->dims[i];
     for (i = 0, block_num = 1; i <= dim; i++)
          block_num *= dst->dims[i];

     sliceTensorKernel<<<block_num, block_size>>>(src->data, dst->data, sdim, ddim, start, block_size);
     return dst;
}

/* in-place reshape tensor */
Tensor *reshapeTensor(const Tensor *src, int newNdim, const int *newDims)
{
     assertTensor(src);
     assert(newDims);
     assert(src->len == computeLength(newNdim, newDims));
     Tensor *dst = createTensor(src->data, newNdim, newDims); /* new tensor */
     return dst;
}

/* current only support dim = src->dims[src->ndim-1] */
Tensor *createReducedTensor(const Tensor *src, int dim)
{
     assertTensor(src);
     assert(dim < src->ndim && dim >= 0);
     assert(dim == src->ndim-1); /* TODO: get rid of this limit */

     Tensor *dst = (Tensor *)malloc(sizeof(Tensor));
     dst->ndim = src->ndim;
     dst->dims = (int *)malloc(sizeof(int) * dst->ndim);
     memmove(dst->dims, src->dims, sizeof(int) * dst->ndim);
     dst->dims[dim] = 1;
     dst->len = computeLength(dst->ndim, dst->dims);
     cudaMalloc(&dst->data, sizeof(float) * dst->len);
     return dst;
}

/* current only support dim = src->dims[src->ndim-1] */
void *reduceArgMax(const Tensor *src, Tensor *dst, Tensor *arg, int dim)
{
     assertTensor(src);
     assertTensor(dst);
     assertTensor(arg);
     assert(dim == src->ndim-1); /* TODO: get rid of this limit */
     for (int i = 0; i < dst->ndim; i++)
          assert(i == dim ? dst->dims[i] == 1 : dst->dims[i] == src->dims[i] &&
                 i == dim ? arg->dims[i] == 1 : arg->dims[i] == src->dims[i]);

     int i, thread_num, block_size, block_num;
     for (i = 0, thread_num = 1; i < dim; i++)
          thread_num *= dst->dims[i];
     block_size = MAX_THREADS_PER_BLOCK;
     block_num = thread_num / block_size + 1;

     reduceArgMaxKernel<<<block_num, block_size>>>(src->data, dst->data, arg->data, src->dims[dim], block_size);
     return dst;
}

Tensor *multiplyElement(const Tensor *src1, const Tensor *src2, Tensor *dst)
{
     assert(isShapeEqual(src1, src2));
     assert(isShapeEqual(src1, dst));

     int thread_num, block_size, block_num;
     thread_num = dst->len;
     block_size = MAX_THREADS_PER_BLOCK;
     block_num = thread_num / block_size + 1;

     multiplyElementKernel<<<block_num, block_size>>>(src1->data, src2->data, dst->data, block_size);
     return dst;
}

/* transform from bbox delta to bbox coordinates, using hyper param EXP_THRESH = 1.0 */
Tensor *transformBboxSQD(const Tensor *delta, const Tensor *anchor, Tensor *res)
{
     assert(isShapeEqual(delta, anchor));
     assert(isShapeEqual(delta, res));
     assert(delta->dims[delta->ndim-1] == 4);

     int i, thread_num, block_size, block_num;
     for (i = 0, thread_num = 1; i < res->ndim-1; i++)
          thread_num *= res->dims[i];
     block_size = MAX_THREADS_PER_BLOCK;
     block_num = thread_num / block_size + 1;

     transformBboxSQDKernel<<<block_num, block_size>>>(delta->data, anchor->data, res->data, block_size);
     return res;
}

#include "tensorUtil.h"
#include "tensorCuda.h"

static void assertTensor(Tensor *tensor)
{
     assert(tensor && tensor->data);
     assert(tensor->ndim < MAXDIM && tensor->ndim > 0);
     assert(tensor->len == computeLength(tensor->ndim, tensor->dims));
}

int computeLength(int ndim, int *dims)
{
     assert(dims);
     int i, len = 1;
     for (i = 0; i < ndim; i++)
          len *= dims[i];
     return len;
}

Tensor *createTensor(float *data, int ndim, int *dims)
{
     Tensor *t = (Tensor *)malloc(sizeof(Tensor));
     t->data = data;
     t->ndim = ndim;
     t->dims = (int *)malloc(sizeof(int) * ndim);
     memmove(t->dims, dims, sizeof(int) * ndim);
     t->len = computeLength(ndim, dims);
     return t;
}

void printTensor(Tensor *tensor, const char *fmt)
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

Tensor *sliceTensor(Tensor *src, int dim, int start, int len)
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
     dst->data = (float *)malloc(dst->len * sizeof(float));

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

Tensor *sliceTensorCuda(Tensor *src, int dim, int start, int len)
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

     int i, block_size, block_num; /* block size and number of cuda threads */
     int ddim = dst->dims[dim], sdim = src->dims[dim];
     for (i = dim+1, block_size = 1; i < dst->ndim; i++)
          block_size *= dst->dims[i];
     for (i = 0, block_num = 1; i <= dim; i++)
          block_num *= dst->dims[i];

     sliceTensorKernel<<<block_num, block_size>>>(dst->data, src->data, ddim, sdim, start, block_size);

     return dst;
}

/* (not use) sliceTensor cuda version using cudaMemcpy, like CPU version. Very slow. */
Tensor *sliceTensorCuda2(Tensor *src, int dim, int start, int len)
{
     assertTensor(src);
     assert(dim <= MAXDIM);
     assert(len+start <= src->dims[dim]);

     Tensor *dst = (Tensor *)malloc(sizeof(Tensor));
     dst->ndim = src->ndim;
     dst->dims = (int *)malloc(sizeof(int) * dst->ndim);
     memmove(dst->dims, src->dims, sizeof(int) * dst->ndim);
     dst->dims[dim] = len;
     dst->len = src->len / src->dims[dim] * len;
     cudaMalloc(&dst->data, sizeof(float) * dst->len);

     int i, block_size, block_num;
     for (i = dim+1, block_size = 1; i < dst->ndim; i++)
          block_size *= dst->dims[i];
     for (i = 0, block_num = 1; i <= dim; i++)
          block_num *= dst->dims[i];

     int index;
     float *dp = dst->data, *sp = src->data;
     size_t floats_size = block_size * sizeof(float);
     for (i = 0; i < block_num; i++) {
          index = i / len * src->dims[dim] + i % len + start;
          /* memmove(dp+i*block_size, sp+index*block_size, floats_size); */
          cudaMemcpy(dp+i*block_size, sp+index*block_size, floats_size, cudaMemcpyDeviceToDevice);
     }

     return dst;
}

/* in-place reshape tensor */
Tensor *reshapeTensor(Tensor *src, int newNdim, int *newDims)
{
     assertTensor(src);
     assert(newDims);
     assert(src->len == computeLength(newNdim, newDims));
     Tensor *dst = createTensor(src->data, newNdim, newDims); /* new tensor */
     return dst;
}

/* current only support dim = src->dims[src->ndim-1] */
void *reduceArgMax(Tensor *src, Tensor *dst, Tensor *arg, int dim)
{
     assertTensor(src);
     assert(dim < src->ndim);
     assert(dim == src->ndim-1); /* TODO: get rid of this limit */

     dst = (Tensor *)malloc(sizeof(Tensor));
     dst->ndim = src->ndim;
     dst->dims = (int *)malloc(sizeof(int) * dst->ndim);
     memmove(dst->dims, src->dims, sizeof(int) * dst->ndim);
     dst->dims[dim] = 1;
     dst->len = computeLength(dst->ndim, dst->dims);
     dst->data = (float *)malloc(sizeof(float) * dst->len);

     arg = (Tensor *)malloc(sizeof(Tensor));
     arg->ndim = dst->ndim;
     arg->dims = (int *)malloc(sizeof(int) * arg->ndim);
     memmove(arg->dims, dst->dims, sizeof(int) * arg->ndim);
     arg->len = dst->len;
     arg->data = (float *)malloc(sizeof(float) * arg->len);

     int i, thread_num, block_size, block_num;
     for (i = 0, thread_num = 1; i < dim; i++)
          thread_num *= dst->dims[i];
     block_size = MAX_THREADS_PER_BLOCK;
     block_num = thread_num / block_size + 1;

     reduceArgMaxKernel<<<block_num, block_size>>>(src->data, dst->data, arg->data, src->dims[dim], block_size);
}

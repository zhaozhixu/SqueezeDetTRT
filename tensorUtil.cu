#include "tensorUtil.h"
#include "tensorCuda.h"

static void assertTensor(Tensor *tensor)
{
     assert(tensor && tensor->data);
     assert(tensor->ndim < MAXDIM && tensor->ndim > 0);
     assert(tensor->len == computeLength(tensor->dims, tensor->ndim));
}

int computeLength(int *dims, int ndim)
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
     t->dims = dims;
     t->len = computeLength(dims, ndim);
     return t;
}

void printTensor(Tensor *tensor, const char *fmt)
{
     assertTensor(tensor);
     int i, j, k, dim_sizes[MAXDIM], dim_levels[MAXDIM];
     int ndim = tensor->ndim, len = tensor->len, *dims = tensor->dims;
     float *data = tensor->data;
     char left_buf[MAXDIM+1], right_buf[MAXDIM+1];
     char *lp = left_buf, *rp = right_buf;
     size_t right_len;

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

     Tensor *dst = (Tensor *)malloc(sizeof(Tensor));
     dst->ndim = src->ndim;
     dst->dims = (int *)malloc(sizeof(int) * dst->ndim);
     memmove(dst->dims, src->dims, sizeof(int) * dst->ndim);
     dst->dims[dim] = len;
     dst->len = src->len / src->dims[dim] * len;
     dst->data = (float *)malloc(dst->len * sizeof(float));

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
          memmove(dp+i*block_size, sp+index*block_size, floats_size);
     }

     return dst;
}

Tensor *sliceTensorCuda(Tensor *src, int dim, int start, int len)
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
     int ddim = dst->dims[dim], sdim = src->dims[dim];
     for (i = dim+1, block_size = 1; i < dst->ndim; i++)
          block_size *= dst->dims[i];
     for (i = 0, block_num = 1; i <= dim; i++)
          block_num *= dst->dims[i];

     sliceTensorKernel<<<block_num, block_size>>>(dst->data, src->data, ddim, sdim, start, block_size);

     return dst;
}

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

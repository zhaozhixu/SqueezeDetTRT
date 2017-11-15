#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "tensorCuda.h"

#define MAXDIM 8

typedef struct {
     int ndim;
     int *dims;
     int len;
     float *data;
} Tensor;

int computeLength(int ndim, int *dims);
Tensor *createTensor(float *data, int ndim, int *dims);
void printTensor(Tensor *tensor, const char *fmt);
Tensor *sliceTensor(Tensor *src, int dim, int start, int len);
Tensor *sliceTensorCuda(Tensor *src, int dim, int start, int len);
Tensor *sliceTensorCuda2(Tensor *src, int dim, int start, int len);
Tensor *reshapeTensor(Tensor *src, int newNdim, int *newDims);

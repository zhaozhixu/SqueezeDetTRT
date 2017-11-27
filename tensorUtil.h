#ifndef _TENSOR_UTIL_H_
#define _TENSOR_UTIL_H_

typedef enum CloneKind {
     H2H, H2D, D2D, D2H
} CloneKind;

typedef struct {
     int ndim;
     int *dims;
     int len;
     float *data;
} Tensor;

int isTensorValid(Tensor *tensor);
int isShapeEqual(const Tensor *t1, const Tensor *t2);
void *cloneMem(const void *src, size_t size, CloneKind kind);
Tensor *cloneTensor(Tensor *src, CloneKind kind);
void *repeatMem(void *data, size_t size, int times, CloneKind kind);
int computeLength(int ndim, const int *dims);
Tensor *createTensor(float *data, int ndim, const int *dims);
void fprintTensor(FILE *stream, const Tensor *tensor, const char *fmt);
void printTensor(const Tensor *tensor, const char *fmt);
Tensor *createSlicedTensor(const Tensor *src, int dim, int start, int len);
Tensor *sliceTensor(const Tensor *src, Tensor *dst, int dim, int start, int len);
Tensor *creatSlicedTensorCuda(const Tensor *src, int dim, int start, int len);
void *sliceTensorCuda(const Tensor *src, Tensor *dst, int dim, int start, int len);
Tensor *reshapeTensor(const Tensor *src, int newNdim, const int *newDims);
Tensor *createReducedTensor(const Tensor *src, int dim);
void *reduceArgMax(const Tensor *src, Tensor *dst, Tensor *arg, int dim);
Tensor *multiplyElement(const Tensor *src1, const Tensor *src2, Tensor *dst);
Tensor *transformBboxSQD(const Tensor *delta, const Tensor *anchor, Tensor *res, float img_width, float img_height);
void tensorIndexSort(Tensor *src, int *idx);
void pickElements(float *src, float *dst, int stride, int *idx, int len);
float computeIou(float *bbox0, float *bbox1);

#endif  /* _TENSOR_UTIL_H_ */

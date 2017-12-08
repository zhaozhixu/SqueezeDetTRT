#include "errorHandle.h"

void checkError(cudaError_t error)
{
     if (error == cudaSuccess)
          return;
     fprintf(stderr, "CUDA_ERROR(%d) %s: %s\n", error, cudaGetErrorName(error), cudaGetErrorString(error));
     abort();
}

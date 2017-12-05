#include "errorHandle.h"

void checkError(cudaError_t error)
{
     if (error == cudaSuccess)
          return;
     fprintf(stderr, "CUDA_ERROR(%s): %s\n", cudaGetErrorName(error), cudaGetErrorString(error));
     abort();
}

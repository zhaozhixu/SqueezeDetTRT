#ifndef _ERROR_HANDLE_H_
#define _ERROR_HANDLE_H_

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

void checkError(cudaError_t error);

#endif

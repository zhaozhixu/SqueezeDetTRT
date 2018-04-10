#ifndef _ERROR_HANDLE_H_
#define _ERROR_HANDLE_H_

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

	void checkError(cudaError_t error);

#ifdef __cplusplus
}
#endif

#endif

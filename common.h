#ifndef _TRT_COMMON_H_
#define _TRT_COMMON_H_
#include "NvInfer.h"
#include <string>
#include <vector>
#include <fstream>
#include <cassert>
#include <iostream>
#define CHECK(status)									\
{														\
	if (status != 0)									\
	{													\
		fprintf(stderr, "CUDA_ERROR(%d) %s: %s\n", status, cudaGetErrorName(status), cudaGetErrorString(status)); \
		abort();										\
	}													\
}

#ifdef __cplusplus
extern "C" {
#endif

    /* std::string locateFile(const std::string& input, const std::vector<std::string> & directories); */
    /* void readPGMFile(const std::string& fileName,  uint8_t *buffer, int inH, int inW); */

#ifdef __cplusplus
}
#endif

#endif // _TRT_COMMON_H_

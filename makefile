CC = g++
CUCC = nvcc -m64 -arch=sm_35
# DEBUG=0
ifeq ($(DEBUG),1)
CC += -g -O0 -DDEBUG
CUCC += -lineinfo -ccbin $(CC)
else
CC += -O3 -DNDEBUG -g
CUCC +=  -ccbin $(CC)
endif
TARGET = sqdtrt

TRIPLE?=x86_64-linux
CUDA_INSTALL_DIR = /usr/local/cuda-8.0
CUDNN_INSTALL_DIR = /usr/local/cuda-8.0
CUDA_LIBDIR = lib
CUDNN_LIBDIR = lib64

INCPATHS    =-I"$(CUDA_INSTALL_DIR)/include" -I "/usr/include" -I"/usr/local/include" -I"../include" -I"../common" -I"$(CUDNN_INSTALL_DIR)/include" -I"../../include" $(TGT_INCLUDES)
LIBPATHS    =-L"$(CUDA_INSTALL_DIR)/targets/$(TRIPLE)/$(CUDA_LIBDIR)" -L "/usr/lib/i386-linux-gnu" -L"/usr/local/lib" -L"../lib" -L"$(CUDA_INSTALL_DIR)/$(CUDA_LIBDIR)" -L"$(CUDNN_INSTALL_DIR)/$(CUDNN_LIBDIR)" -L"../../lib" $(TGT_LIBS)

# COMMON_LIBS = -lcudnn -lcublas -lcudart_static -lnvToolsExt -lcudart
COMMON_LIBS = -lcudnn -lcudart -lcudart_static
# LIBS  =-lnvinfer -lnvparsers -lnvinfer_plugin $(COMMON_LIBS)
# DLIBS =-lnvinfer -lnvparsers -lnvinfer_plugin $(COMMON_LIBS)
LIBS  =-lnvinfer -lnvinfer_plugin $(COMMON_LIBS)
DLIBS =-lnvinfer -lnvinfer_plugin $(COMMON_LIBS)

COMMON_FLAGS += -std=c++11 $(INCPATHS) `pkg-config --cflags --libs opencv`
CFLAGS=$(COMMON_FLAGS)
CFLAGSD=$(COMMON_FLAGS)

COMMON_LD_FLAGS += $(LIBPATHS) -L$(OUTDIR)
LFLAGS=$(COMMON_LD_FLAGS)
LFLAGSD=$(COMMON_LD_FLAGS)

$(TARGET): sqdtrt.o common.o trtUtil.o tensorUtil.o tensorCuda.o errorHandle.o sdt_alloc.o
	$(CC) -Wall sqdtrt.o common.o trtUtil.o tensorUtil.o tensorCuda.o errorHandle.o sdt_alloc.o -o $(TARGET) $(CFLAGSD) $(LIBPATHS) $(LIBS)
sqdtrt.o: sqdtrt.cpp tensorUtil.h tensorCuda.h common.h errorHandle.h sdt_alloc.h
	$(CUCC) -c sqdtrt.cpp $(CFLAGSD) $(LFLAGSD) $(LIBS)
common.o: common.cpp common.h errorHandle.h
	$(CUCC) -c common.cpp $(CFLAGSD) $(LFLAGSD) $(LIBS)
trtUtil.o: trtUtil.cpp trtUtil.h errorHandle.h sdt_alloc.h
	$(CUCC) -c trtUtil.cpp $(CFLAGSD) $(LFLAGSD) $(LIBS)
tensorUtil.o: tensorUtil.cu tensorUtil.h tensorCuda.h errorHandle.h sdt_alloc.h
	$(CUCC) -c tensorUtil.cu $(CFLAGSD) $(LFLAGSD) $(LIBS)
tensorCuda.o: tensorCuda.cu tensorCuda.h errorHandle.h
	$(CUCC) -c tensorCuda.cu $(CFLAGSD) $(LFLAGSD) $(LIBS)
errorHandle.o: errorHandle.cu errorHandle.h
	$(CUCC) -c errorHandle.cu $(INCPATHS) $(LIBPATHS) $(COMMON_LIBS)
sdt_alloc.o: sdt_alloc.c sdt_alloc.h
	$(CC) -c sdt_alloc.c
clean:
	rm -f *.o


OUT = 1_saxpy 2_sgemv 3_sgemm 4_edge_detection

CUDA_PATH := /usr/local/cuda-8.0-cudnn-v5
CXX := $(CUDA_PATH)/bin/nvcc

INCLUDES += -I/usr/local/cuda-8.0-cudnn-v5/include -I/usr/local/cuda-8.0-cudnn-v5/include/include
LIBRARIES += -L/usr/local/cuda-8.0-cudnn-v5/lib64 -lcublas

INCLUDES += $(shell pkg-config --libs opencv)
LIBRARIES += $(shell pkg-config --cflags opencv)

all: $(OUT)

%: %.cu
	$(CXX) $(INCLUDES) $(LIBRARIES) -o $@ $<

clean:
	rm -f $(OUT)

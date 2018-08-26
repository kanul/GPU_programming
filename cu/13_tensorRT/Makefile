
CUDA_PATH := /usr/local/cuda
TENSOR_RT_PATH := /usr/local/TensorRT

CXX := $(CUDA_PATH)/bin/nvcc
CXXFLAGS += -std=c++11
CXXFLAGS += -O3

INCLUDES += -I$(TENSOR_RT_PATH)/include
INCLUDES += -L$(TENSOR_RT_PATH)/lib
LIBRARIES += -lcudnn -lcublas -lcudart_static -lnvToolsExt -lcudart
LIBRARIES += -lnvinfer -lnvparsers
#LIBRARIES += -lrt -ldl -lpthread

NAME=lenet_tensorrt

BIN=${NAME}
OBJ=${NAME}.o
SRC=${NAME}.cu

all: $(BIN)

$(BIN): $(OBJ)
	$(CXX) $(CXXFLAGS) ${INCLUDES} ${LIBRARIES} -o $@ $+

$(OBJ): $(SRC)
	$(CXX) $(CXXFLAGS) ${INCLUDES} ${LIBRARIES} -c -o $@ $+

clean:
	rm -f $(BIN) $(OBJ)


#	nvcaffe_parser
#	nvinfer
#	nvparsers
#	nvinfer
#	nvcaffe_parser


NVCC=/usr/local/cuda/bin/nvcc

lenet: lenet.cu readubyte.cpp
	$(NVCC) -std=c++11 -O3 -lcudnn -lcublas \
			 -gencode arch=compute_61,code=sm_61 \
			 -o $@ $+

clean:
	rm -f lenet


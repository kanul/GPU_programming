all: 03_convolution.o ImageLoadSave.o 
	nvcc -O3 03_convolution.o ImageLoadSave.o -o exe

03_convolution.o: 03_convolution.cu
	nvcc -c $< -O3 -o $@

ImageLoadSave.o: ImageLoadSave.cpp
	nvcc -c $< -O3 -o $@

dir:
	mkdir -p obj

clean:
	rm -rf obj
	rm -f exe
	rm -f output.pgm

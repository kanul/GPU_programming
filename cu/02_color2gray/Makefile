all: 02_color2gray.o ImageLoadSave.o 
	nvcc -O3 02_color2gray.o ImageLoadSave.o -o exe

02_color2gray.o: 02_color2gray.cu
	nvcc -c $< -O3 -o $@

ImageLoadSave.o: ImageLoadSave.cpp
	nvcc -c $< -O3 -o $@

dir:
	mkdir -p obj

clean:
	rm -rf obj
	rm -f exe
	rm -f output.pgm


# GPU_programming

# Environmental setup:
guest08@node08:
- scp -P 54329 output.pgm  gpu:

guest08@node08:
- scp output.pgm  node08:

psftp
- open guest08@210.125.181.20 54329
- get output.pgm

# 07_shuffle:
- Issue: Error for undefined __shfl_down()
- Resolve: nvcc 07_shuffle.cu -arch=sm_61
	
# 08_cublasSaxpy:
- export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
- source ~/.bashrc
- nvcc -lcubas 08_cublasSaxpy.cu -o 1_saxpy
	
# 10_edgeDetection
0. https://sourceforge.net/projects/xming/
1. putty >> connection >> SSH >> X11 >> Enable X11 forwarding
2. Run xming
3. log in server
4. ssh - Y [the name of node]
5. eog [the name of image]
	
# 11_async
- ssh - Y [the name of node]
- nvvp ./a.out ASYNC_V1
	
# 13_TensorRT
0. export LD_LIBRARY_PATH=/usr/local/TensorRT/lib:$LD_LIBRARY_PATH
1. nvcc -std=c++11 -O3 -I/usr/local/TensorRT/include -L/usr/local/TensorRT/lib \
			 -lcudnn -lcublas -lcudart_static -lnvToolsExt -lcudart -lnvinfer -lnvparsers \
			 -o lenet_tensorrt lenet_tensorrt.cu

# 14_cudnn_training
0. nvcc -std=c++11 -O3 -I/usr/local/opencv/include 
-L/usr/local/opencv/lib -lopencv_core -lopencv_imgcodecs 
-lcudnn -lcublas -o lenet lenet.cu
1. cp -r ~/local/pretrained ./
2. cp -r ~/local/image ./
3. cd image
4. rm –f input.pgm
5. ln –s image-4.pgm input.pgm
6. cd ..
7. ./lenet


#include "ImageLoadSave.h"

#define CUDA
#define KSIZE 3

int KERNEL[KSIZE*KSIZE] = { 1, 1, 1,
	1, -8, 1,
	1, 1, 1};

__device__ int cuKERNEL[KSIZE*KSIZE] = { 1, 1, 1,
	1, -8, 1,
	1, 1, 1};
	
__global__ void cuConvolution(unsigned char* dst, unsigned char* src, int width, int height, int ksize);
void convolution(unsigned char* src, unsigned char* dst, int width, int height, int ksize);

int main()
{
#ifdef CUDA
	PGM pgm = read_pgm("testvector/gray.pgm");
    PGM edge;

    edge.width = pgm.width - KSIZE + 1;
    edge.height = pgm.height - KSIZE + 1;

    // output for cpu
    edge.value = (unsigned char *)malloc(sizeof(unsigned char) * edge.width * edge.height);

    // Memory allocation
    unsigned char *d_img, *d_conv;
    cudaMalloc((void **) &d_img, sizeof(unsigned char) * pgm.width * pgm.height);
    cudaMalloc((void **) &d_conv, sizeof(unsigned char) * edge.width * edge.height);

	// Memcpy from cpu to gpu for input data
	cudaMemcpy(d_img, pgm.value, sizeof(unsigned char) * pgm.width * pgm.height, cudaMemcpyHostToDevice);

	int blockDim = 256;
	int gridDim = (edge.width * edge.height) / blockDim + ((edge.width * edge.height) % blockDim == 0 ? 0 : 1);
												    
	// Run cuda version of convolution
	cuConvolution<<<gridDim, blockDim>>>(d_conv, d_img, pgm.width, pgm.height, KSIZE);

	//cudaDevicesSynchronize();
	cudaMemcpy(edge.value, d_conv, sizeof(unsigned char) * edge.width * edge.height, cudaMemcpyDeviceToHost);
	
	write_pgm(edge, "./03_output.pgm");


	cudaFree(d_img);
	cudaFree(d_conv);
	free(pgm.value);
	free(edge.value);
#else 
	PGM pgm = read_pgm("testvector/frame_000550.ppm");
	
	PGM edge;
	edge.width = pgm.width - KSIZE + 1;		
	edge.height = pgm.height - KSIZE + 1;
	
	edge.value = (unsigned char *)malloc(sizeof(unsigned char) * edge.width * edge.height);
	
	convolution(pgm.value, edge.value pgm.width, pgm.height, KSIZE);
	
	write_pgm(edge, "./output_laplacian.pgm");
#endif
	cudaFree(d_img);
	cudaFree(d_conv);
	free(pgm.value);
	free(edge.value);

	return 0;
}

__global__ void cuConvolution(unsigned char* dst, unsigned char* src, int width, int height, int ksize)
{
	int gid = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (gid >= (width - ksize + 1) * (height - ksize + 1)) return;

	int y = gid / width;
	int x = gid % width;
	//src += (y*width + x);
	int sum = 0;
	for(int ky = 0; ky < ksize; ky++) {
		for(int kx = 0; kx < ksize; kx++) {
			sum += src[(y+ky) * width + (x+kx)] * cuKERNEL[ky*ksize + kx];
		}
	}
	dst[gid] = (unsigned char) sum;

}

void convolution(unsigned char* src, unsigned char* dst, int width, int height, int ksize)
{
	for(int y = 0; y < height - ksize + 1; y++) {
		for(int x = 0; x < width - ksize + 1; x++) {
			int sum = 0;
			for(int ky = 0; ky < ksize; ky++) {
				for(int kx = 0; kx < ksize; kx++) {
					sum += src[(y+ky) * width + (x+kx)] *KERNEL[ky*ksize + kx];
				}
			}
			sum /= ksize * ksize;
			dst[y*(width - ksize + 1) + x] = (unsigned char) sum;
		}
	}
}

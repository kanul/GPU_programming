
#include "ImageLoadSave.h" 

__global__ void cuColor2Gray(unsigned char* src, unsigned char* dst, int width, int height);

int main()
{
	PPM ppm = read_ppm("testvector/frame_000550.ppm");

	int img_size = ppm.width * ppm.height;

	//Memory allocation for GPU(device)
	unsigned char *d_ppm, *d_gray;
	//cudaMalloc(Memory_pointer, Size);
	cudaMalloc((void **) &d_ppm, sizeof(unsigned char) * img_size * 3);
	cudaMalloc((void **) &d_gray, sizeof(unsigned char) * img_size);
	
	//H2D memcpy
	cudaMemcpy(d_ppm, ppm.value, sizeof(unsigned char) * img_size * 3, cudaMemcpyHostToDevice);


	int blockDim = 256;
	int gridDim = img_size / blockDim + (img_size % blockDim == 0 ? 0 : 1);

	cuColor2Gray<<<gridDim, blockDim>>>(d_ppm, d_gray, ppm.width, ppm.height);

	//Wait until thread function(gpu) is over
	cudaDeviceSynchronize();

	PGM pgm;
	pgm.width = ppm.width; pgm.height = ppm.height;
	pgm.value = (unsigned char *)malloc(sizeof(unsigned char) * img_size);

	cudaMemcpy(pgm.value, d_gray, sizeof(unsigned char) * img_size, cudaMemcpyDeviceToHost);


	write_pgm(pgm, "./output.pgm");

	cudaFree(d_ppm);
	cudaFree(d_gray);
	free(ppm.value);
	free(pgm.value);
	return 0;
}

__global__ void cuColor2Gray(unsigned char* src, unsigned char* dst, int width, int height) {

	int gid = blockIdx.x * blockDim.x + threadIdx.x;

	if(gid >= width * height) return;

	int y = gid / width;
	int x = gid % width;

	src += 3*(y*width + x);
	uchar result = (uchar)(0.0722f*(float)src[0] + 0.7152*(float)src[1] + 0.2126f*(float)src[2]);
	
	dst[gid] = result;
}

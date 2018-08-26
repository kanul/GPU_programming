
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>

__global__ void additionGPU(float *a, float *b, float *r, int n);

void additionCPU(float *a, float *b, float *r, int n);

int main(int argc, char* argv[])
{
	if(argc < 2) {
		puts("Usage: matmul [N]");
		return 0;
	}

	int N = atoi(argv[1]);
	printf("N: %d\n", N);

	//Total size
	size_t sz = sizeof(float) * N;

	//Struct timeval start, end, timer;
	struct timeval start, end, timer;

	//Memory allocation for cpu(host)
	//vectorC = vectorA + vectorB
	float *h_a = (float *)malloc(sz);
	float *h_b = (float *)malloc(sz);
	float *h_c = (float *)malloc(sz);

	for(int i = 0; i < N; i++) {
		h_a[i] = i;
		h_b[i] = N-i;
		h_c[i] = 0.0;
	}

	//Memory allocation for GPU(device)
	float *d_a, *d_b, *d_c;
	//cudaMalloc(Memory_pointer, Size);
	cudaMalloc((void **) &d_a, sz);
	cudaMalloc((void **) &d_b, sz);
	cudaMalloc((void **) &d_c, sz);
	float *h_result = (float *)malloc(sz);

	//H2D memcpy
	cudaMemcpy(d_a, h_a, sz, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, sz, cudaMemcpyHostToDevice);

	//CPU vector addition
	gettimeofday(&start, NULL);
	additionCPU(h_a, h_b, h_c, N);
	gettimeofday(&end, NULL);
	timersub(&end, &start, &timer);
	printf("CPU elapsed time: %lf\n", (timer.tv_usec / 1000.0 + timer.tv_sec * 1000.0));

	//GPU vector addition
	int threads = 256;
	int grid = (N % threads) ? N/threads+1 : N/threads;

	//Time measure start
	gettimeofday(&start, NULL);
	additionGPU<<< grid, threads >>>(d_a, d_b, d_c, N);
	//Wait until thread function(gpu) is over
	cudaDeviceSynchronize();
	//Time measure end
	gettimeofday(&end, NULL);

	cudaMemcpy(h_result, d_c, sz, cudaMemcpyDeviceToHost);
	timersub(&end, &start, &timer);
	printf("GPU elapsed time: %lf\n", (timer.tv_usec / 1000.0 + timer.tv_sec * 1000.0));

	//Verification
	for(int i=0;i<N;i++)
	{
		if(h_c[i] != h_result[i]) {
			printf("Failed at %d, [CPU]: %f, [GPU]: %f\n", i, h_c[i], h_result[i]);
			break;
		}
	}

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
	free(h_result);
	free(h_a);
	free(h_b);
	free(h_c);
	return 0;
}

__global__ void additionGPU(float *a, float *b, float *c, int n) {
	int gid = blockIdx.x * blockDim.x + threadIdx.x;
	//guarding for what?
	// if don't do this, what will happen?
	if(gid < n) {
		c[gid] = a[gid] + b[gid];
	}
}

void additionCPU(float *a, float *b, float *r, int n)
{
	int i = 0;
	for(i=0;i<n;i++){
		r[i]=a[i]+b[i];
	}
}



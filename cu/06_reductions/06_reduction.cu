
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include <stdio.h>

#define THREADS 512

__global__ void getMax_global(unsigned int *d_data, unsigned int *d_max, int n);
__global__ void getMax_local(unsigned int *d_data, unsigned int *d_max, int n);
__global__ void getMax_binary(unsigned int *d_data, unsigned int *d_max, int n);

int main(int argc, char* argv[]){
	if (argc < 2){
		puts("Usage: ./a.out [N]");
		return 0;
	}

	int N=atoi(argv[1]);
	printf("N: %d\n", N);

	size_t sz = sizeof(int) * N;

	unsigned int *data = (unsigned int*)malloc(sz);
	srand(time(NULL));

	for(int i =0; i<N;i++)
		data[i] = (unsigned int)(rand()%100000000);

	struct timeval start, end, timer;

	unsigned int *d_data;
	cudaMalloc((void **) &d_data, sz);
	
	unsigned int *d_max;
	cudaMalloc((void **) &d_max, sizeof(unsigned int));

	unsigned int max;

	gettimeofday(&start, NULL);

	max = 0;
	
	for(int i=0; i<N; i++){
		if (max < data[i]){
			max = data[i];
		}
	}

	gettimeofday(&end, NULL);
	timersub(&end, &start, &timer);
	printf("CPU, elapsed time: %lf\n", (timer.tv_usec / 1000.0 + timer.tv_sec * 1000.0));
	printf("CPU, max value: %d \n", max);

	int threads = 512;
	int grid = (N % threads) ? N/threads+1 : N/threads;
	max = 0;
	cudaMemcpy(d_data, data, sz, cudaMemcpyHostToDevice);
	cudaMemcpy(d_max, data, sizeof(unsigned int), cudaMemcpyHostToDevice);

	gettimeofday(&start, NULL);
	getMax_global<<< grid, threads >>>(d_data, d_max, N);
	cudaDeviceSynchronize();
	cudaMemcpy(&max, d_max, sizeof(unsigned int), cudaMemcpyDeviceToHost);

	gettimeofday(&end, NULL);
	timersub(&end, &start, &timer);
	printf("Global GPU, elapsed time: %lf\n", (timer.tv_usec / 1000.0 + timer.tv_sec * 1000.0));
	printf("Global GPU, max value : %d\n", max);

    max = 0;
	cudaMemcpy(d_data, data, sz, cudaMemcpyHostToDevice);
	cudaMemcpy(d_max, data, sizeof(unsigned int), cudaMemcpyHostToDevice);

	gettimeofday(&start, NULL);
	getMax_local<<< grid, threads >>>(d_data, d_max, N);
	cudaDeviceSynchronize();
	cudaMemcpy(&max, d_max, sizeof(unsigned int), cudaMemcpyDeviceToHost);

	gettimeofday(&end, NULL);
	timersub(&end, &start, &timer);

	printf("Local GPU, elapsed time: %lf\n", (timer.tv_usec / 1000.0 + timer.tv_sec * 1000.0));
	printf("Local GPU, max value : %d\n", max);
	
	
    max = 0;
	cudaMemcpy(d_data, data, sz, cudaMemcpyHostToDevice);
	cudaMemcpy(d_max, data, sizeof(unsigned int), cudaMemcpyHostToDevice);

	gettimeofday(&start, NULL);
	getMax_binary<<< grid, threads >>>(d_data, d_max, N);
	cudaDeviceSynchronize();
	cudaMemcpy(&max, d_max, sizeof(unsigned int), cudaMemcpyDeviceToHost);

	gettimeofday(&end, NULL);
	timersub(&end, &start, &timer);

	printf("Binary GPU, elapsed time: %lf\n", (timer.tv_usec / 1000.0 + timer.tv_sec * 1000.0));
	printf("Binary GPU, max value : %d\n", max);
	
	
	
	return 0;
}

__global__ void getMax_global(unsigned int * d_data, unsigned int *d_max, int n){
	int gid = blockIdx.x * blockDim.x + threadIdx.x;
	if(gid < n)
		atomicMax(d_max, d_data[gid]);
}

__global__ void getMax_local(unsigned int *d_data, unsigned int *d_max, int n){
	__shared__ unsigned int s_max;
	__shared__ unsigned int s_data[THREADS];

	int tid = threadIdx.x;
	int gid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid == 0)
		s_max = 0;

	atomicMax(&s_max, d_data[gid]);
	__syncthreads();

	if (tid == 0)
		atomicMax(d_max, s_max);
}

__global__ void getMax_binary(unsigned int *d_data, unsigned int *d_max, int n){
	__shared__ unsigned int s_data[THREADS];

	int tid = threadIdx.x;
	int gid = blockIdx.x * blockDim.x + threadIdx.x;

	int stride = THREADS>>1;

	s_data[tid] = d_data[gid];
	__syncthreads();

	for(;;){
		if(stride == 1)
			break;
		
		if(tid < stride){
			if(s_data[tid] < s_data[tid+stride])
				s_data[tid] = s_data[tid+stride];
		}

		stride >>= 1;
		__syncthreads();
	}

	if(tid == 0)
		atomicMax(d_max, s_data[0]);
}

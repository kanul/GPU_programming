
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>

#define TILE_WIDTH 16

void matmulCPU(float *a, float *b, float *r, int n);
__global__ void matmulGPU(float *a, float *b, float *r, int n);
__global__ void MatrixMulKernel(float* Md, float* Nd, float* Pd, int Width);

int main(int argc, char* argv[]){

	if (argc < 2) {
		puts("Usage: matmul [N]");
		return 0;
	}

	int N = atoi(argv[1]);
	printf("N: %d\n", N);

	//Total size
	size_t sz = sizeof(float) * N * N;

	//Struct for time measure
	struct timeval start, end, timer;

	//Memory allocation for cpu(host)
	float *h_a = (float*)malloc(sz);
	float *h_b = (float*)malloc(sz);
	float *h_r = (float*)malloc(sz);

	srand(time(NULL));
	for(int i=0; i<N*N; i++) {
		h_a[i] = (float)(rand()%100);
		h_b[i] = (float)(rand()%100);
		h_r[i] = 0;
	}

	//Memory alocation for gpu(device)
	float *d_a, *d_b, *d_r;
	cudaMalloc((void **) &d_a, sz);
	cudaMalloc((void **) &d_b, sz);
	cudaMalloc((void **) &d_r, sz);

	float *h_result_global = (float*)malloc(sz);

	gettimeofday(&start, NULL);
	matmulCPU(h_a, h_b, h_r, N);
	gettimeofday(&end, NULL);
	timersub(&end, &start, &timer);
	printf("CPU elapsed time: %lf\n", (timer.tv_usec / 1000.0 + timer.tv_sec * 1000.0));

	int threads_width = 16;
	int grid_width = N % threads_width ? N / threads_width + 1 : N / threads_width;
	dim3 dim_threads(threads_width, threads_width);
	dim3 dim_grid(grid_width, grid_width);

	gettimeofday(&start, NULL);

	cudaMemcpy(d_a, h_a, sz, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, sz, cudaMemcpyHostToDevice);

	MatrixMulKernel<<<dim_grid, dim_threads>>>(d_a, d_b, d_r, N);

	cudaDeviceSynchronize();

	cudaMemcpy(h_result_global, d_r, sz, cudaMemcpyDeviceToHost);
	gettimeofday(&end, NULL);

	timersub(&end, &start, &timer);

	printf("GPU elapsed time: %lf\n", (timer.tv_usec / 1000.0 + timer.tv_sec * 1000.0));

	for (int i = 0; i<N*N; i++){
		if(h_r[i] != h_result_global[i]){
			printf("Failed at %d, h_result_global, %f, %f\n", i, h_r[i], h_result_global[i]);
			break;
		}
	}

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_r);

	free(h_result_global);
	free(h_a);
	free(h_b);
	free(h_r);

	return 0;

}

void matmulCPU(float *a, float *b, float *r, int n){
	int i=0,j=0,x=0;
	for(i=0;i<n;i++){
		for(j=0;j<n;j++){
			float sum = 0.0f;
			for(x=0;x<n;x++){
				sum+=a[j*n + x] * b[x * n + i];
			}
			r[j*n + i] = sum;
		}
	}
}

__global__ void matmulGPU(float *a, float *b, float *r, int n){
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if(x>=n || y>=n)
		return;

	float sum = 0;
	for(int i=0;i<n;i++)
		sum+=(a[y*n +i] * b[i*n+x]);

	r[y*n+x] = sum;
}

__global__ void MatrixMulKernel(float*Md, float* Nd, float* Pd, int Width){
	__shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
	__shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

	int bx = blockIdx.x;
	int by = blockIdx.y;

	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int Row = by * TILE_WIDTH + ty;
	int Col = bx * TILE_WIDTH + tx;

	if(Row >= Width || Col >= Width)
		return;

	float Pvalue;
	if(tx == 0 || ty == 0)
		Pvalue = 0;

	int num_tile = Width % TILE_WIDTH == 0 ? Width / TILE_WIDTH : Width / TILE_WIDTH + 1;

	for(int m=0;m<num_tile;++m){
		Mds[ty][tx] = Md[Row * Width + (tx + m * TILE_WIDTH)];
		Nds[ty][tx] = Nd[Col + (m * TILE_WIDTH + ty) * Width];
		__syncthreads();
		
		int var_tile_width = m == (num_tile - 1) ? Width % TILE_WIDTH : TILE_WIDTH;
		for(int k=0; k<var_tile_width;++k){
			Pvalue+=Mds[ty][k]*Nds[k][tx];
		}
		__syncthreads();
	}
	Pd[Row*Width+Col] = Pvalue;
}


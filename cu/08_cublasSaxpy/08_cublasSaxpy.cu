
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#define N 1000000

#define checkCUDA(expression)						\
{													\
	cudaError_t status = (expression);				\
	if (status != cudaSuccess) {					\
		printf("Error on line %d: err code %d\n",	\
				__LINE__, status);					\
		exit(EXIT_FAILURE);							\
	}												\
}

#define checkCUBLAS(expression)						\
{													\
	cublasStatus_t status = (expression);			\
	if (status != CUBLAS_STATUS_SUCCESS) {			\
		printf("Error on line %d: err code %d\n",	\
				__LINE__, status);					\
		exit(EXIT_FAILURE);							\
	}												\
}

#define getMillisecond(start, end) (end.tv_sec-start.tv_sec) * 1000 + (end.tv_usec-start.tv_usec) / 1000.0

int main(void) {
	cublasHandle_t handle;
	float *x, *y;
	float *devPtrX, *devPtrY;
	float alpha = 1.2;
	float incx = 1, incy = 1;

	float ms = 0;
	struct timeval start, end;
	srand(2018);

	x = (float *)malloc (N*sizeof(float));
	y = (float *)malloc (N*sizeof(float));

	float *resultCPU = (float *)malloc(N*sizeof(float));
	float *resultGPU = (float *)malloc(N*sizeof(float));
	
	for (int i = 0; i < N; i++) {
		x[i] = (rand() % 1000000) / 10000.0;
		y[i] = (rand() % 1000000) / 10000.0;
	}

	checkCUDA(cudaMalloc((void**)&devPtrX, N * sizeof(float)));
	checkCUDA(cudaMalloc((void**)&devPtrY, N * sizeof(float)));

	checkCUBLAS(cublasCreate(&handle));

	checkCUBLAS(cublasSetVector (N, sizeof(float), x, 1, devPtrX, 1));
	checkCUBLAS(cublasSetVector (N, sizeof(float), y, 1, devPtrY, 1));

	gettimeofday(&start, NULL);
	checkCUBLAS(cublasSaxpy (handle, N, &alpha, devPtrX, incx, devPtrY, incy));
	gettimeofday(&end, NULL);

	ms = getMillisecond(start, end);
	printf("GPU time: %f (ms)\n", ms);

	checkCUBLAS (cublasGetVector(N, sizeof(float), devPtrY, 1, resultGPU, 1));

	gettimeofday(&start, NULL);
	for(int i = 0; i < N; i++) {
		resultCPU[i] = x[i]*alpha + y[i];
	}

	gettimeofday(&end, NULL);

	ms = getMillisecond(start, end);
	printf("CPU time: %f (ms)\n", ms);

	float error = 0;
	for (int i = 0 ; i < N; i++) {
		error += abs((resultCPU[i] - resultGPU[i]) / resultCPU[i]);
	}

	error = error / N * 100;
	printf("Mean Absolute Percentage Error: %f (%%)\n", error);

	checkCUDA(cudaFree(devPtrX));
	checkCUDA(cudaFree(devPtrY));
	checkCUBLAS(cublasDestroy(handle));
	free(x);
	free(y);
	free(resultCPU);
	free(resultGPU);
	return EXIT_SUCCESS;
}



#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#define M 1000
#define K 10
#define N 1000

#define checkCUDA(expression)                  \
{                                              \
	cudaError_t status = (expression);           \
	if (status != cudaSuccess) {                 \
		printf("Error on line %d: err code %d\n",  \
				__LINE__, status);                     \
		exit(EXIT_FAILURE);                        \
	}                                            \
}

#define checkCUBLAS(expression)                \
{                                              \
	cublasStatus_t status = (expression);        \
	if (status != CUBLAS_STATUS_SUCCESS) {       \
		printf("Error on line %d: err code %d\n",  \
				__LINE__, status);                     \
		exit(EXIT_FAILURE);                        \
	}                                            \
}

#define getMillisecond(start, end) \
	(end.tv_sec-start.tv_sec)*1000 + \
	(end.tv_usec-start.tv_usec)/1000.0


int main (void){
	cublasHandle_t handle;
	float *A, *B, *C, *resultCPU, *resultGPU;
	float *devPtrA, *devPtrB, *devPtrC;
	float alpha = 0.9;
	float beta  = 1.1;

	float ms = 0;
	struct timeval start, end;
	srand(2018);

	// Memory for host
	A = (float *)malloc (M * K * sizeof (float));
	B = (float *)malloc (K * N * sizeof (float));
	C = (float *)malloc (M * N * sizeof (float));
	resultCPU = (float *)malloc (M * N * sizeof (float));
	resultGPU = (float *)malloc (M * N * sizeof (float));

	// Init values
	for (int i = 0; i < K; i++) {
		for (int j = 0; j < M; j++) {
			A[i*M+j] = (rand() % 1000000) / 10000.0;
		}
	}

	for (int i = 0; i < N; i++) {
		for (int j = 0; j < K; j++) {
			B[i*K+j] = (rand() % 1000000) / 10000.0;
		}
	}

	for (int i = 0; i < N; i++) {
		for (int j = 0; j < M; j++) {
			C[i*M+j] = (rand() % 1000000) / 10000.0;
		}
	}

	// Memory for device
	checkCUDA (cudaMalloc ((void**)&devPtrA, M * K * sizeof (float)));
	checkCUDA (cudaMalloc ((void**)&devPtrB, K * N * sizeof (float)));
	checkCUDA (cudaMalloc ((void**)&devPtrC, M * N * sizeof (float)));

	// Init cuBLAS
	checkCUBLAS (cublasCreate (&handle));

	// Memcpy host to device
	checkCUBLAS (cublasSetMatrix (M, K, sizeof (float), A, M, devPtrA, M));
	checkCUBLAS (cublasSetMatrix (K, N, sizeof (float), B, K, devPtrB, K));
	checkCUBLAS (cublasSetMatrix (M, N, sizeof (float), C, M, devPtrC, M));

	// Sgemv with GPU
	gettimeofday(&start, NULL);
	checkCUBLAS (cublasSgemm (handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K,
				&alpha, devPtrA, M, devPtrB, K, &beta, devPtrC, M));
	gettimeofday(&end, NULL);

	// Print duration
	ms = getMillisecond(start, end);
	printf("GPU time: %f (ms)\n", ms);

	// Memcpy device to host
	checkCUBLAS (cublasGetMatrix (M, N, sizeof (float),
				devPtrC, M, resultGPU, M));

	// Sgemv with CPU
	gettimeofday(&start, NULL);
	for (int j = 0; j < M; j++) {
		for (int i = 0; i < N; i++) {
		  resultCPU[i*M+j] = 0;
			for (int k = 0; k < K; k++) {
				resultCPU[i*M+j] += alpha * A[k*M+j] * B[i*K+k];
			}
			resultCPU[i*M+j] += beta * C[i*M+j];
		}
	}
	gettimeofday(&end, NULL);

	// Print duration
	ms = getMillisecond(start, end);
	printf("CPU time: %f (ms)\n", ms);

	// Validate the result
	float error = 0;
	for (int i = 0; i < M * N; i++) {
		error += abs((resultCPU[i] - resultGPU[i]) / resultCPU[i]);
	}
	error = error / (M*N) * 100;
	printf ("Mean Absolute Percentage Error: %f (%%)\n", error);

	// Free
	checkCUDA (cudaFree (devPtrA));
	checkCUDA (cudaFree (devPtrB));
	checkCUDA (cudaFree (devPtrC));
	checkCUBLAS (cublasDestroy (handle));
	free(A);
	free(B);
	free(C);
	free(resultCPU);
	free(resultGPU);
	return EXIT_SUCCESS;
}

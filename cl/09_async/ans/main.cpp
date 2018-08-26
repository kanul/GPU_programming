#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#include <iostream>
#include <fstream>
#include <sstream>
#include <ctime>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define CHECK_ERROR(err)	\
	if( err != CL_SUCCESS ) \
		{ std::cout << __LINE__ << ", OpenCL Error: " << err << std::endl; exit(-1); }

#define KB 1024
#define MB 1048576

#define ASYNC_V1 1
#define ASYNC_V2 2
#define ASYNC_V3 3

#define nstreams 4

using namespace std;

static string *sAsyncMethod = new string[4]
{
	"0 (None, Sequential)",
		"1 (Async V1)",
		"2 (Async V2)",
		"3 (Async V3)",
};

size_t roundUp(int group_size, int global_size)
{
	int r = global_size % group_size;
	if (r == 0) {
		return global_size;
	}
	else {
		return global_size + group_size - r;
	}
}

std::string readKernel(const char* filename)
{
	std::ifstream ifs(filename, std::ios_base::in);
	if (!ifs.is_open()) {
		std::cerr << "Failed to open file" << std::endl;
		return NULL;
	}

	std::ostringstream oss;
	oss << ifs.rdbuf();

	return oss.str();
}

int main(int argc, char* argv[])
{
	if (argc < 2) {
		puts("usage: async [N]");
		return 0;
	}

	const int niterations = 2 * 3 * 4; // number of iterations for the loop outside the kernel
	const int BURNING = 50; // number of iterations for the loop inside the kernel
	int async_mode = ASYNC_V1; // default Async_V1
	float factor = 1.1;
	int N = 6 * MB;
	if (argc > 1) async_mode = atoi(argv[1]);

	printf("N: %d\n", N);
	printf("# BURNING: %d\n", BURNING);
	printf("# iterations: %d\n", niterations);
	printf("ASync method: %s\n", sAsyncMethod[async_mode].c_str());

	//Total size
	size_t sz = sizeof(float) * N;
	if ((sz / sizeof(float)) < BURNING) {
		printf("error: 'sz' must be larger than BURNING\n");
		exit(-1);
	}

	//Struct for time measure
	clock_t start, end;
	double timer;

	//Memory allocation for cpu(host)
	//vectorC = vectorA + vectorB
	float *h_a[niterations], *h_r[niterations];

	for (int i = 0; i < niterations; i++) {
		h_a[i] = (float*)malloc(sz);
		h_r[i] = (float*)malloc(sz);
		memset(h_a[i], 0, sz);
	}

	//-------------------------------------------------------------------------
	// Set up the OpenCL platform using whchever platform is "first"
	//-------------------------------------------------------------------------
	int err;
	int ndim = 1;
	cl_device_id        device_id;
	cl_context          context;
	cl_command_queue    commands[nstreams];
	cl_kernel           kernel[nstreams];
	cl_program          program;
	cl_uint				numPlatforms;
	cl_platform_id		firstPlatformId;

	err = clGetPlatformIDs(1, &firstPlatformId, &numPlatforms);
	CHECK_ERROR(err);

	err = clGetDeviceIDs(firstPlatformId, CL_DEVICE_TYPE_GPU, 1,
		&device_id, NULL);
	CHECK_ERROR(err);

	cl_context_properties context_props[] =
	{
		CL_CONTEXT_PLATFORM, (cl_context_properties)firstPlatformId, 0
	};

	cl_command_queue_properties queue_props = CL_QUEUE_PROFILING_ENABLE;

	context = clCreateContext(context_props, 1, &device_id, NULL, NULL, &err);
	CHECK_ERROR(err);

	for (int i = 0; i < nstreams; i++)
	{
		commands[i] = clCreateCommandQueue(context, device_id, queue_props, &err);
		CHECK_ERROR(err);
	}

	// Print device info
	char cBuffer[1024];
	printf(" ---------------------------------\n");
	err = clGetDeviceInfo(device_id, CL_DEVICE_NAME, sizeof(cBuffer), &cBuffer, NULL);
	printf(" Device: %s\n", cBuffer);
	printf(" ---------------------------------\n");
	CHECK_ERROR(err);

	cl_uint compute_units;
	err = clGetDeviceInfo(device_id, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(compute_units), &compute_units, NULL);
	printf("  CL_DEVICE_MAX_COMPUTE_UNITS:\t\t%u\n", compute_units);
	CHECK_ERROR(err);

	//-------------------------------------------------------------------------
	// Set up the buffers, initialize matrices, and wirte them into
	// global memory
	//-------------------------------------------------------------------------
	//Memory allocation for gpu(device)
	cl_mem d_a[nstreams], d_r[nstreams];
	for (int i = 0; i < nstreams; i++)
	{
		d_a[i] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
			sz, NULL, NULL);
		d_r[i] = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
			sz, NULL, NULL);
	}

	// Create the compute program from the source buffer
	std::string strSource = readKernel("kernel.cl");
	const char* source = strSource.c_str();
	program = clCreateProgramWithSource(context, 1,
		(const char **)&source, NULL, &err);
	CHECK_ERROR(err);

	// Build the program
	char kernel_name[1000];
	char build_option[1000] = { 0, };
	sprintf_s(build_option, "-D BURNING=%d", BURNING);
	err = clBuildProgram(program, 0, NULL, build_option, NULL, NULL);
	CHECK_ERROR(err);

	if (err != CL_SUCCESS) {
		size_t len;
		char buffer[2048];

		printf("Error: Failed to build program executable!\n");
		clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG,
			sizeof(buffer), buffer, &len);
		printf("%s\n", buffer);
		return -1;
	}

	// Create the compute kernel from the program
	strcpy_s(kernel_name, "timeBurningKernel");
	for (int i = 0; i < nstreams; i++) {
		kernel[i] = clCreateKernel(program, kernel_name, &err);
		CHECK_ERROR(err);
	}
	/*************************************************
	Operlapping Data Transfer and Kernel Execution
	*************************************************/
	start = clock();
	size_t local[] = { 256 };							// number of work-items per work-group
	size_t global[] = { (size_t)roundUp(local[0], N) };	// number of total work-items

	if (nstreams == 1) {
		for (int i = 0; i < niterations; i++) {
			err = clEnqueueWriteBuffer(commands[0], d_a[0], CL_TRUE, 0, sz, h_a[i], 0, NULL, NULL);
			CHECK_ERROR(err);

			err = clSetKernelArg(kernel[0], 0, sizeof(cl_mem), (void*)&d_a[0]);
			err |= clSetKernelArg(kernel[0], 1, sizeof(cl_mem), (void*)&d_r[0]);
			err |= clSetKernelArg(kernel[0], 2, sizeof(float), (void*)&factor);
			err |= clSetKernelArg(kernel[0], 3, sizeof(int), (void*)&N);
			CHECK_ERROR(err);

			// Kernel call
			err = clEnqueueNDRangeKernel(commands[0], kernel[0], ndim, NULL,
				global, local, 0, NULL, NULL);
			CHECK_ERROR(err);

			// Read back the results from the compute device
			err = clEnqueueReadBuffer(commands[0], d_r[0], CL_TRUE, 0,
				sz, h_r[i], 0, NULL, NULL);
			CHECK_ERROR(err);
		}
	}
	else {
		if (async_mode == ASYNC_V1) {
			// Async V1
			for (int i = 0; i < niterations; i += nstreams) {
				// H2D memcpy
				for (int j = 0; j < nstreams; j++) {
					err = clEnqueueWriteBuffer(commands[j], d_a[j], CL_FALSE, 0, sz, h_a[i + j], 0, NULL, NULL);
					CHECK_ERROR(err);

					err = clSetKernelArg(kernel[j], 0, sizeof(cl_mem), (void*)&d_a[j]);
					err |= clSetKernelArg(kernel[j], 1, sizeof(cl_mem), (void*)&d_r[j]);
					err |= clSetKernelArg(kernel[j], 2, sizeof(float), (void*)&factor);
					err |= clSetKernelArg(kernel[j], 3, sizeof(int), (void*)&N);
					CHECK_ERROR(err);

					// Kernel call
					err = clEnqueueNDRangeKernel(commands[j], kernel[j], ndim, NULL,
						global, local, 0, NULL, NULL);
					CHECK_ERROR(err);

					// Read back the results from the compute device
					err = clEnqueueReadBuffer(commands[j], d_r[j], CL_FALSE, 0,
						sz, h_r[i + j], 0, NULL, NULL);
					CHECK_ERROR(err);
				}
			}
		}
		else if (async_mode == ASYNC_V2) {
			// Async V2
			for (int i = 0; i < niterations; i += nstreams) {
				// H2D memcpy
				for (int j = 0; j < nstreams; j++) {
					err = clEnqueueWriteBuffer(commands[j], d_a[j], CL_FALSE, 0, sz, h_a[i + j], 0, NULL, NULL);
					CHECK_ERROR(err);

					err = clSetKernelArg(kernel[j], 0, sizeof(cl_mem), (void*)&d_a[j]);
					err |= clSetKernelArg(kernel[j], 1, sizeof(cl_mem), (void*)&d_r[j]);
					err |= clSetKernelArg(kernel[j], 2, sizeof(float), (void*)&factor);
					err |= clSetKernelArg(kernel[j], 3, sizeof(int), (void*)&N);
					CHECK_ERROR(err);
				}
				// Kernel call
				for (int j = 0; j < nstreams; j++) {
					err = clEnqueueNDRangeKernel(commands[j], kernel[j], ndim, NULL,
						global, local, 0, NULL, NULL);
					CHECK_ERROR(err);
				}
				// copy the result D2H(device to host)
				for (int j = 0; j < nstreams; j++) {
					err = clEnqueueReadBuffer(commands[j], d_r[j], CL_FALSE, 0,
						sz, h_r[i + j], 0, NULL, NULL);
					CHECK_ERROR(err);
				}
			}
		}
		else
		{
			// Async V3
			for (int i = 0; i < niterations; i += nstreams) {
				// H2D memcpy
				for (int j = 0; j < nstreams; j++) {
					// Set the arguments to our compute kernel
					err = clEnqueueWriteBuffer(commands[j], d_a[j], CL_FALSE, 0, sz, h_a[i + j], 0, NULL, NULL);
					CHECK_ERROR(err);

					err = clSetKernelArg(kernel[j], 0, sizeof(cl_mem), (void*)&d_a[j]);
					err |= clSetKernelArg(kernel[j], 1, sizeof(cl_mem), (void*)&d_r[j]);
					err |= clSetKernelArg(kernel[j], 2, sizeof(float), (void*)&factor);
					err |= clSetKernelArg(kernel[j], 3, sizeof(int), (void*)&N);
					CHECK_ERROR(err);
				}
				// Kernel call & D2H memcpy
				for (int j = 0; j < nstreams; j++) {
					err = clEnqueueNDRangeKernel(commands[j], kernel[j], ndim, NULL,
						global, local, 0, NULL, NULL);
					CHECK_ERROR(err);

					err = clEnqueueReadBuffer(commands[j], d_r[j], CL_FALSE, 0,
						sz, h_r[i + j], 0, NULL, NULL);
					CHECK_ERROR(err);
				}
			}
		}
	}

	for (int i = 0; i < nstreams; i++) {
		clFinish(commands[i]);
	}
	end = clock();
	timer = (double)(end - start) / CLOCKS_PER_SEC * 1e3;

	printf("GPU elapsed time: %.3lf\n", timer);

	clReleaseProgram(program);
	clReleaseContext(context);

	for (int i = 0; i < nstreams; i++)
	{
		clReleaseCommandQueue(commands[i]);
		clReleaseMemObject(d_a[i]);
		clReleaseMemObject(d_r[i]);
		clReleaseKernel(kernel[i]);
		free(h_a[i]);
		free(h_r[i]);
	}

	system("pause");

	return 0;
}
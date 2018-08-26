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

void additionCPU(float *a, float *b, float *r, int n)
{
	int i = 0;
	for (i = 0; i<n; i++) {
		r[i] = a[i] + b[i];
	}
}

int main(int argc, char* argv[])
{
	if (argc < 2) {
		puts("usage: vecAdd [N]");
		return 0;
	}

	int N = atoi(argv[1]);
	printf("N: %d\n", N);

	//Total size
	size_t sz = sizeof(float) * N;

	//Struct for time measure
	clock_t cstart, cend;
	double ctimer;
	cl_ulong estart, eend;
	double etimer;

	//-------------------------------------------------------------------------
	// Set up the OpenCL platform using whchever platform is "first"
	//-------------------------------------------------------------------------
	int err;
	int ndim = 1;
	cl_device_id        device_id;
	cl_context          context;
	cl_command_queue    commands;
	cl_program          program;
	cl_kernel           kernel;
	cl_uint				numPlatforms;
	cl_platform_id		firstPlatformId[3];
	cl_event			event;

	err = clGetPlatformIDs(3, firstPlatformId, &numPlatforms);
	CHECK_ERROR(err);

	err = clGetDeviceIDs(firstPlatformId[1], CL_DEVICE_TYPE_GPU, 1,
		&device_id, NULL);
	CHECK_ERROR(err);

	cl_context_properties context_props[] =
	{
		CL_CONTEXT_PLATFORM, (cl_context_properties)firstPlatformId[1], 0
	};
	cl_command_queue_properties queue_props = CL_QUEUE_PROFILING_ENABLE;

	context = clCreateContext(context_props, 1, &device_id, NULL, NULL, &err);
	CHECK_ERROR(err);
	commands = clCreateCommandQueue(context, device_id, queue_props, &err);
	CHECK_ERROR(err);

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
	cl_mem d_a, d_b, d_r;
	d_a = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR,
		sz, NULL, NULL);
	d_b = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR,
		sz, NULL, NULL);
	d_r = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR,
		sz, NULL, NULL);

	// Map a region of the buffer object into the host address space
	float *h_a, *h_b, *h_r;
	cstart = clock();
	h_a = (float*)clEnqueueMapBuffer(commands, d_a, CL_TRUE, 
		CL_MAP_READ | CL_MAP_WRITE, 0, sz, 0, NULL, NULL, &err);
	h_b = (float*)clEnqueueMapBuffer(commands, d_b, CL_TRUE, 
		CL_MAP_READ | CL_MAP_WRITE, 0, sz, 0, NULL, NULL, &err);
	h_r = (float*)clEnqueueMapBuffer(commands, d_r, CL_TRUE,
		CL_MAP_READ | CL_MAP_WRITE, 0, sz, 0, NULL, NULL, &err);
	CHECK_ERROR(err);
	cend = clock();
	ctimer = (double)(cend - cstart) / CLOCKS_PER_SEC * 1e3;
	printf("map               elapsed time: %.3lf\n", ctimer);

	//Memory allocation for cpu(host)
	float *h_r_cpu = (float*)malloc(sz);
	for (int i = 0; i < N; i++) {
		h_a[i] = i;
		h_b[i] = N - i;
		h_r[i] = 0.0;
		h_r_cpu[i] = 0.0;
	}

	// Need for Discrete GPU
	cstart = clock();
	err = clEnqueueUnmapMemObject(commands, d_a, h_a, 0, NULL, NULL);
	err = clEnqueueUnmapMemObject(commands, d_b, h_b, 0, NULL, NULL);
	err = clEnqueueUnmapMemObject(commands, d_r, h_r, 0, NULL, NULL);
	CHECK_ERROR(err);
	cend = clock();
	ctimer = (double)(cend - cstart) / CLOCKS_PER_SEC * 1e3;
	printf("unmap               elapsed time: %.3lf\n", ctimer);

	/*
	cstart = clock();
	err = clEnqueueWriteBuffer(commands, d_a, CL_TRUE, 0,
	sz, h_a, 0, NULL, NULL);
	err = clEnqueueWriteBuffer(commands, d_b, CL_TRUE, 0,
	sz, h_b, 0, NULL, NULL);
	err = clEnqueueWriteBuffer(commands, d_r, CL_TRUE, 0,
	sz, h_r, 0, NULL, NULL);
	CHECK_ERROR(err);
	cend = clock();
	ctimer = (double)(cend - cstart) / CLOCKS_PER_SEC * 1e3;
	printf("copy               elapsed time: %.3lf\n", ctimer);
	*/
	
	// Create the compute program from the source buffer
	std::string strSource = readKernel("kernel.cl");
	const char* source = strSource.c_str();
	program = clCreateProgramWithSource(context, 1,
		(const char **)&source, NULL, &err);
	CHECK_ERROR(err);

	// Build the program
	char kernel_name[1000];
	char build_option[1000] = { 0, };
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
	strcpy_s(kernel_name, "additionGPU");
	kernel = clCreateKernel(program, kernel_name, &err);
	CHECK_ERROR(err);

	/*************************************************
	GPU Vector addition
	*************************************************/
	size_t local[] = { 512 };							// number of work-items per work-group
	size_t global[] = { (size_t)roundUp(local[0], N) };	// number of total work-items

	// Set the arguments to our compute kernel
	err = 0;
	err |= clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_a);
	err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_b);
	err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_r);
	err |= clSetKernelArg(kernel, 3, sizeof(int), &N);
	CHECK_ERROR(err);

	err = clEnqueueNDRangeKernel(commands, kernel, ndim, NULL,
		global, local, 0, NULL, &event);
	CHECK_ERROR(err);

	// Wait for the event to be completed before reading back results
	err = clWaitForEvents(1, &event);
	CHECK_ERROR(err);

	// Get Profiling Info
	err = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START,
		sizeof(cl_ulong), &estart, NULL);
	err = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END,
		sizeof(cl_ulong), &eend, NULL);
	etimer = (double)(eend - estart) * 1e-6;
	CHECK_ERROR(err);

	// Read back the results from the compute device
	// Need for Discrete GPU
	h_r = (float*)clEnqueueMapBuffer(commands, d_r, CL_TRUE,
		CL_MAP_READ | CL_MAP_WRITE, 0, sz, 0, NULL, NULL, &err);
	CHECK_ERROR(err);

	/*
	err = clEnqueueReadBuffer(commands, d_r, CL_TRUE, 0,
		sz, h_result, 0, NULL, NULL);
	CHECK_ERROR(err);
	*/

	printf("GPU (event timer) elapsed time: %.3lf\n", etimer);

	/*************************************************
	CPU Vector addition
	*************************************************/
	cstart = clock();
	additionCPU(h_a, h_b, h_r_cpu, N);
	cend = clock();
	ctimer = (double)(cend - cstart) / CLOCKS_PER_SEC * 1e3;
	printf("CPU               elapsed time: %.3lf\n", ctimer);


	/*************************************************
	Verification
	*************************************************/
	for (int i = 0; i<N; i++) {
		if (h_r[i] != h_r_cpu[i]) {
			printf("Failed at %d\n", i);
			break;
		}
	}

	clReleaseProgram(program);
	clReleaseKernel(kernel);
	clReleaseMemObject(d_a);
	clReleaseMemObject(d_b);
	clReleaseMemObject(d_r);
	clReleaseCommandQueue(commands);
	clReleaseContext(context);
	clReleaseEvent(event);

	free(h_r_cpu);

	system("pause");

	return 0;
}
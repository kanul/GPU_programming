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

#define CHECK_ERROR(err) if( err != CL_SUCCESS ) { std::cout << __LINE__ << ", OpenCL Error: " << err << std::endl; exit(-1); }

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

std::string strSource = readKernel("kernel.cl");

#define THREADS 512
/*************************************************
main
*************************************************/

int main(int argc, char* argv[])
{
	if (argc < 2) {
		puts("usage: main [N]");
		return 0;
	}
	int N = atoi(argv[1]);
	printf("N: %d\n", N);

	//Total size
	size_t sz = sizeof(unsigned int) * N;

	unsigned int * data = (unsigned int*)malloc(sz);
	srand(time(NULL));
	for (int i = 0; i<N; i++) {
		data[i] = (unsigned int)(rand() % 10000000);
	}

	//Struct for time measure
	clock_t cstart, cend;
	double ctimer;
	cl_ulong estart, eend;
	double etimer;

	//Declare sum variable
	unsigned int sum;

	//-------------------------------------------------------------------------
	// Set up the OpenCL platform using whchever platform is "first"
	//-------------------------------------------------------------------------
	int err;
	int ndim = 1;
	cl_device_id        device_id;
	cl_context          context;
	cl_command_queue    commands;
	cl_program          program;
	cl_kernel           kernel_global;
	cl_kernel           kernel_shared;
	cl_kernel           kernel_binary;
	cl_uint				numPlatforms;
	cl_platform_id		firstPlatformId;
	cl_event			event;

	err = clGetPlatformIDs(1, &firstPlatformId, &numPlatforms);
	CHECK_ERROR(err);
	err = clGetDeviceIDs(firstPlatformId, CL_DEVICE_TYPE_GPU, 1,
		&device_id, NULL);
	CHECK_ERROR(err);

	cl_context_properties context_props[] =
	{
		CL_CONTEXT_PLATFORM, (cl_context_properties)firstPlatformId, 0
	};

	cl_queue_properties queue_props = CL_QUEUE_PROFILING_ENABLE;
	
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
	cl_mem d_data, d_sum;
	d_data = clCreateBuffer(context, CL_MEM_READ_ONLY,
		sz, NULL, NULL);

	d_sum = clCreateBuffer(context, CL_MEM_READ_ONLY,
		sizeof(unsigned int), NULL, NULL);


	// Write the A and B matrices into compute device memory
	err = clEnqueueWriteBuffer(commands, d_data, CL_TRUE, 0,
		sz, data, 0, NULL, NULL);
	CHECK_ERROR(err);

	// Create the compute program from the source buffer
	const char* source = strSource.c_str();
	program = clCreateProgramWithSource(context, 1,
		(const char **)&source, NULL, &err);
	CHECK_ERROR(err);


	// Build the program
	char kernel_name[1000];
	char build_option[1000] = { 0, };
	sprintf_s(build_option, "-D THREADS=%d", THREADS);
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
	strcpy_s(kernel_name, "getSum_global");
	kernel_global = clCreateKernel(program, kernel_name, &err);
	CHECK_ERROR(err);
	strcpy_s(kernel_name, "getSum_local");
	kernel_shared = clCreateKernel(program, kernel_name, &err);
	CHECK_ERROR(err);
	strcpy_s(kernel_name, "getSum_binary");
	kernel_binary = clCreateKernel(program, kernel_name, &err);
	CHECK_ERROR(err);


	/*************************************************
	CPU
	*************************************************/
	cstart = clock();
	//init sum
	sum = 0;
	for (int i = 0; i<N; i++) {
		sum += data[i];
	}
	//Time measure end
	cend = clock();
	ctimer = (double)(cend - cstart) / CLOCKS_PER_SEC * 1000.0;
	printf("CPU , elapsend time: %lf\n", ctimer);
	printf("CPU , sum value : %d\n", sum);

	/*************************************************
	Global sum
	*************************************************/
	size_t local[] = { 512 };
	size_t global[] = { (size_t)roundUp(local[0], N) };
	sum = 0;
	err = clEnqueueWriteBuffer(commands, d_sum, CL_TRUE, 0,
		sizeof(unsigned int), &sum, 0, NULL, NULL);
	CHECK_ERROR(err);

	// Set the arguments to our compute kernel
	err = 0;
	err |= clSetKernelArg(kernel_global, 0, sizeof(cl_mem), &d_data);
	err |= clSetKernelArg(kernel_global, 1, sizeof(cl_mem), &d_sum);
	err |= clSetKernelArg(kernel_global, 2, sizeof(int), &N);

	//Time measure start
	err = clEnqueueNDRangeKernel(commands, kernel_global, ndim, NULL,
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
	err = clEnqueueReadBuffer(commands, d_sum, CL_TRUE, 0,
		sizeof(unsigned int), &sum, 0, NULL, NULL);
	CHECK_ERROR(err);

	printf("Global sum, elapsend time: %lf\n", etimer);
	printf("Global sum, sum value : %d\n", sum);


	/*************************************************
	Local sum
	*************************************************/
	local[0] = 512;
	global[0] = (size_t)roundUp(local[0], N);
	sum = 0;
	err = clEnqueueWriteBuffer(commands, d_sum, CL_TRUE, 0,
		sizeof(unsigned int), &sum, 0, NULL, NULL);
	CHECK_ERROR(err);

	// Set the arguments to our compute kernel
	err = 0;
	err |= clSetKernelArg(kernel_shared, 0, sizeof(cl_mem), &d_data);
	err |= clSetKernelArg(kernel_shared, 1, sizeof(cl_mem), &d_sum);
	err |= clSetKernelArg(kernel_shared, 2, sizeof(int), &N);
	err |= clSetKernelArg(kernel_shared, 3, sizeof(unsigned int), NULL);
	err |= clSetKernelArg(kernel_shared, 4, sizeof(unsigned int) * THREADS, NULL);

	//Time measure start
	err = clEnqueueNDRangeKernel(commands, kernel_shared, ndim, NULL,
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
	err = clEnqueueReadBuffer(commands, d_sum, CL_TRUE, 0,
		sizeof(unsigned int), &sum, 0, NULL, NULL);
	CHECK_ERROR(err);

	printf("Local sum, elapsend time: %lf\n", etimer);
	printf("Local sum, sum value : %d\n", sum);

	/*************************************************
	Binary
	*************************************************/
	if (N % THREADS != 0) {
		puts("N must be times of THREADS when using binary reduction");
		goto out;
	}
	local[0] = THREADS;
	global[0] = (size_t)roundUp(local[0], N);
	sum = 0;
	err = clEnqueueWriteBuffer(commands, d_sum, CL_TRUE, 0,
		sizeof(unsigned int), &sum, 0, NULL, NULL);
	CHECK_ERROR(err);

	// Set the arguments to our compute kernel
	err = 0;
	err |= clSetKernelArg(kernel_binary, 0, sizeof(cl_mem), &d_data);
	err |= clSetKernelArg(kernel_binary, 1, sizeof(cl_mem), &d_sum);
	err |= clSetKernelArg(kernel_binary, 2, sizeof(int), &N);
	err |= clSetKernelArg(kernel_binary, 3, sizeof(unsigned int) * THREADS, NULL);

	//Time measure start
	err = clEnqueueNDRangeKernel(commands, kernel_binary, ndim, NULL,
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
	err = clEnqueueReadBuffer(commands, d_sum, CL_TRUE, 0,
		sizeof(unsigned int), &sum, 0, NULL, NULL);
	CHECK_ERROR(err);

	printf("Binary reduction, elapsend time: %lf\n", etimer);
	printf("Binary reduction, sum value : %d\n", sum);
out:

	clReleaseProgram(program);
	clReleaseKernel(kernel_global);
	clReleaseKernel(kernel_shared);
	clReleaseKernel(kernel_binary);
	clReleaseMemObject(d_data);
	clReleaseMemObject(d_sum);
	clReleaseCommandQueue(commands);
	clReleaseContext(context);
	clReleaseEvent(event);

	free(data);

	system("pause");

	return 0;
}
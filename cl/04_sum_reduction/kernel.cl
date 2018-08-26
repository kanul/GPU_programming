__kernel void getSum_global(__global unsigned int * d_data, __global unsigned int * d_sum, int n)
{

}

__kernel void getSum_local( __global unsigned int * d_data, __global unsigned int * d_sum, int n,
							__local unsigned int *_s_sum, __local unsigned int *s_data)
{

}

__kernel void getSum_binary(__global unsigned int * d_data, __global unsigned int * d_sum, int n,
							__local unsigned int * s_data)
{

}
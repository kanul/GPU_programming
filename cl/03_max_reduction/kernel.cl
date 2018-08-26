__kernel void getMax_global(__global unsigned int * d_data, __global unsigned int * d_max ,int n)
{

}

__kernel void getMax_local( __global unsigned int * d_data, __global unsigned int * d_max ,int n,
							__local unsigned int *_s_max, __local unsigned int *s_data)
{

}

__kernel void getMax_binary(__global unsigned int * d_data, __global unsigned int * d_max ,int n,
							__local unsigned int * s_data)
{

}
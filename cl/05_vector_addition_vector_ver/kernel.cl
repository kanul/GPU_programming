__kernel void additionGPU(__global float *a, __global float *b, __global float *r, int n)
{
	int gid = get_global_id(0);

	if( gid < n) {
		r[gid] = a[gid] + b[gid];
	}
}
__kernel void additionGPU_implicit(__global float4 *a, __global float4 *b, __global float4 *r, int n)
{
	int gid = get_global_id(0);

	if( gid < n) {
		r[gid] = a[gid] + b[gid];
	}
}
__kernel void additionGPU_explicit(__global float *a, __global float *b, __global float *r, int n)
{

}


__kernel void additionGPU(__global float *a, __global float *b, __global float *r, int n)
{
	int gid = get_global_id(0);

	if( gid < n) {
		r[gid] = a[gid] + b[gid];
	}
}

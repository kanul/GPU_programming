__kernel void timeBurningKernel(__global float *d_a, __global float *d_r, float factor, int N)
{
	int gid = get_global_id(0);

	if(gid < N) {
		for(int i = 0; i< BURNING; i++)
			d_r[gid] = d_a[gid] + factor * factor;
	}
}
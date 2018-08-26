__kernel void matmulGPU_normal(__global float *a, __global float *b, __global float *r, int n)
{
	int x = get_global_id(0);
	int y = get_global_id(1);

	if( x >= n || y >= n )
		return;

	float sum = 0;
	for(int i=0; i<n; i++)
		sum += (a[y * n + i] * b[i * n + x]); 

	r[y * n + x] = sum;
}

__kernel void matmulGPU_vector(__global float *a, __global float *b, __global float *r, int n)
{

}
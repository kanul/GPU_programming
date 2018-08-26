__kernel void matmulGPU_global(__global float *a, __global float *b, __global float *r, int n)
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


__kernel void matmulGPU_shared(__global float *a, __global float *b, __global float *r,int n, 
								__local float* s_a, __local float* s_b)
{
	int bx = get_group_id(0);
	int by = get_group_id(1);
	int tx = get_local_id(0);
	int ty = get_local_id(1);

	float sum = 0.0;
	int x,y;
	x = TILE_WIDTH * bx + tx;
	y = TILE_WIDTH * by + ty;

	for(int i=0; i < n/TILE_WIDTH; i++){
		s_a[ty * TILE_WIDTH + tx] = a[y * n + (i * TILE_WIDTH  + tx)];
		s_b[ty * TILE_WIDTH + tx] = b[(i * TILE_WIDTH  + ty) * n + x];
		barrier(CLK_LOCAL_MEM_FENCE);

		for(int j =0; j < TILE_WIDTH; j++){
			sum += s_a[ty * TILE_WIDTH  + j ] * s_b[j* TILE_WIDTH + tx];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	r[ y * n + x] = sum;
}


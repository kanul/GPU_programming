__kernel void getSum_global(__global unsigned int * d_data, __global unsigned int * d_sum, int n)
{
	int gid = get_global_id(0);
	if( gid < n) {
		atomic_add( d_sum , d_data[gid]);
	}
}

__kernel void getSum_local( __global unsigned int * d_data, __global unsigned int * d_sum, int n,
							__local unsigned int *_s_sum)
{
	int tid = get_local_id(0);
	int gid = get_global_id(0);

	//Shared variable init
	if( tid == 0 ){
		*_s_sum = 0;
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	float val = (gid < n) ? d_data[gid] : 0;
	
	atomic_add( _s_sum, val);
	barrier(CLK_LOCAL_MEM_FENCE);

	if( tid == 0 ){
		atomic_add( d_sum, *_s_sum);
	}
}

__kernel void getSum_binary(__global unsigned int * d_data, __global unsigned int * d_sum, int n,
							__local unsigned int * s_data)
{
	int tid = get_local_id(0);
	int gid = get_global_id(0);
	int stride = THREADS>>1;

	if (gid < n)
		s_data[tid] = d_data[gid];
	else
		s_data[tid] = 0;
	barrier(CLK_LOCAL_MEM_FENCE);

	for(;;){
		if(tid < stride ){
			s_data[tid] += s_data[tid+stride];
		}
		if( stride == 1)
			break;

		stride >>= 1;
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if( tid == 0 ){
		atomic_add( d_sum, s_data[0]);
	}
}
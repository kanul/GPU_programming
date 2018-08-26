#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h> 
#include <time.h>

#define KB 1024
#define MB 1048576

#define ASYNC_V1 1
#define ASYNC_V2 2
#define ASYNC_V3 3

static char *sAsyncMethod[] = 
{
	"0 (None, Sequential)",
    "1 (Async V1)",
    "2 (Async V2)",
    "3 (Async V3)",
    NULL
};

/*************************************************
   timeBurningKernel
 *************************************************/

#define BURNING 1050
__global__ void timeBurningKernel(float *d_a, float *d_r, float factor, int N)
{
	int gid = blockIdx.x * blockDim.x + threadIdx.x;

	if( gid < N ) {
		for( int i = 0 ; i < BURNING ; ++i ) 
			d_r[gid] = d_a[gid] + factor * factor;
	}
}


/*************************************************
   main
 *************************************************/
	
int main(int argc, char* argv[])
{
	if( argc < 2 ) {
		puts("usage: ./a.out [Async Mode]");
		return 0;
	}
    
	const int niterations = 2*3*4;	// number of iterations for the loop inside the kernel
	int nstreams = 3;
	int async_mode = ASYNC_V1;		//default Async_V1
	float factor = 1.1;

	int N = 4*MB;
	if( argc > 1 ) async_mode = atoi(argv[1]);
	if( async_mode == 0 ) nstreams = 1;

	printf("N: %d\n", N );
	printf("# BURNING: %d\n", BURNING );
	printf("# iterations: %d\n", niterations );
	printf("# streams: %d\n", nstreams );
	printf("ASync method: %s\n", sAsyncMethod[async_mode]);

	//Total size
	size_t sz = 128 * MB;
	if( (sz/sizeof(float)) < BURNING ) {
		printf("error: 'sz' must be larger than BURNING\n");
		exit(-1);
	}

	//Struct for time measure
	struct timeval start, end, timer;

    // TODO: allocate and initialize an array of stream handles
    cudaStream_t *streams = (cudaStream_t*) malloc(nstreams * sizeof(cudaStream_t));
	for(int i = 0; i < nstreams; i++) {
		cudaStreamCreate(&(streams[i]));
	}
	
	// Memory allocation for cpu (host)
	// Pinned memory (page-locked)
	float *h_a[niterations]; 
	float *h_r[niterations];
	for( int i = 0 ; i < niterations ; ++i ) {
		cudaMallocHost((void**)&h_a[i], sz);
		cudaMallocHost((void**)&h_r[i], sz);
	}

	srand(time(NULL));
	for( int j = 0 ; j < niterations ; ++j ) {
		for(int i = 0 ; i < N*N; i++ ) {
			h_a[j][i] = (float)(rand()%100);
			h_r[j][i] = 0.;
		}
	}

	//Memory allocation for gpu(device)
	float *d_a[nstreams], *d_r[nstreams];
	for( int j = 0 ; j < nstreams ; ++j ) {
		cudaMalloc((void **) &d_a[j], sz );
		cudaMalloc((void **) &d_r[j], sz );
	}

	/*************************************************
		Launching timeBurningKernel
	*************************************************/
	size_t dim_threads = 256;
	size_t dim_grid = ((N%dim_threads)? N/dim_threads+1 : N/dim_threads);

	cudaDeviceSynchronize();
	gettimeofday(&start, NULL);

	if(nstreams == 1 ) {
		for( int i =0 ; i < niterations ; i ++ ) {
		}
	}
	else {

		if(async_mode == ASYNC_V1 )
		{
			for(int i = 0; i < niterations; i += nstreams) {
				for(int j = 0; j < nstreams; ++j) {
					cudaMemcpyAsync(d_a[j], h_a[i+j], sz, cudaMemcpyHostToDevice, streams[j]);

					timeBurningKernel<<< dim_grid, dim_threads, 0, streams[j] >>>(d_a[j], d_r[j], factor, N);

					cudaMemcpyAsync(h_r[i+j], d_r[j], sz, cudaMemcpyDeviceToHost, streams[j]);
				}
			}
		}
		else if(async_mode == ASYNC_V2)
		{

		}
		else
			// Async V3
		{


		}
	}
	cudaDeviceSynchronize();
	gettimeofday(&end, NULL);
	timersub(&end,&start,&timer);
	printf("%d, elapsed time: %lf\n", niterations, (timer.tv_usec / 1000.0 + timer.tv_sec *1000.0) ); 

	for(int i=0; i<niterations; i++) {
		cudaFreeHost(h_r[i]);
		cudaFreeHost(h_a[i]);
	}
	for(int i=0; i<nstreams; i++) {
		cudaFree(d_r[i]);
		cudaFree(d_a[i]);
	}

	return 0;
}


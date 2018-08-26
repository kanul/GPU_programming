#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cublas_v2.h>
#include <cudnn.h>
#include <opencv2/opencv.hpp>

#define getMillisecond(start, end) \
    (end.tv_sec-start.tv_sec)*1000 + \
    (end.tv_usec-start.tv_usec)/1000.0

#define checkCUDA(expression)                             \
{                                                         \
	cudaError_t status = (expression);                      \
	if (status != cudaSuccess) {                            \
		printf("Error on line %d: err code %d (%s)\n",        \
				__LINE__, status, cudaGetErrorString(status));    \
		exit(EXIT_FAILURE);                                   \
	}                                                       \
}

#define checkCUBLAS(expression)                           \
{                                                         \
	cublasStatus_t status = (expression);                   \
	if (status != CUBLAS_STATUS_SUCCESS) {                  \
		printf("Error on line %d: err code %d\n",             \
				__LINE__, status);                                \
		exit(EXIT_FAILURE);                                   \
	}                                                       \
}

#define checkCUDNN(expression)                            \
{                                                         \
	cudnnStatus_t status = (expression);                    \
	if (status != CUDNN_STATUS_SUCCESS) {                   \
		printf("Error on line %d: err code %d (%s)\n",        \
				__LINE__, status, cudnnGetErrorString(status));   \
		exit(EXIT_FAILURE);                                   \
	}                                                       \
}

cv::Mat load_image(const char* image_path) {
	cv::Mat image = cv::imread(image_path, CV_LOAD_IMAGE_GRAYSCALE);
	image.convertTo(image, CV_32FC1);
	cv::normalize(image, image, 0, 1, cv::NORM_MINMAX);
	return image;
}

bool load_weight(float* p_weight, int elemCount, const char* filename)
{
	// Read weights file
	FILE *fp = fopen(filename, "rb");
	if (!fp)
	{
		printf("ERROR: Cannot open file %s\n", filename);
		return false;
	}
	fread(p_weight, sizeof(float), elemCount, fp);
	fclose(fp);

	return true;
}


int main(int argc, char const *argv[]) {
	cublasHandle_t cublas;
	cudnnHandle_t cudnn;
	checkCUBLAS(cublasCreate(&cublas));
	checkCUDNN(cudnnCreate(&cudnn));

	int    batch_size = 1;
	size_t workspace_bytes = 0;
	cv::Mat image = load_image("image/input.pgm");

	/* Input */

	int input_dim = 28;
	int input_channels = 1;

	float* d_input{NULL};
	int input_bytes = batch_size * input_channels * input_dim * input_dim
										* sizeof(float);
	cudaMalloc(&d_input, input_bytes);
	cudaMemcpy(d_input, image.ptr<float>(0), input_bytes, cudaMemcpyHostToDevice);

	// Input Tensor
	cudnnTensorDescriptor_t input_descriptor;
	checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
	checkCUDNN(cudnnSetTensor4dDescriptor(input_descriptor,
				/*format=*/CUDNN_TENSOR_NCHW,
				/*dataType=*/CUDNN_DATA_FLOAT,
				/*batch_size=*/batch_size,
				/*channels=*/input_channels,
				/*height=*/input_dim,
				/*width=*/input_dim));

	/* Layer 1. Convolution */

	int l1_kernel_dim = 5;
	int l1_pad = 0;
	int l1_stride = 1;
	int l1_dilation = 1;

	int l1_out_dim = 24;
	int l1_out_channels = 20;

	char* l1_weight_file = "pretrained/conv1.bin";
	char* l1_weight_bias_file = "pretrained/conv1.bias.bin";

	// Describing Operands

	cudnnTensorDescriptor_t l1_out_descriptor;
	checkCUDNN(cudnnCreateTensorDescriptor(&l1_out_descriptor));
	checkCUDNN(cudnnSetTensor4dDescriptor(l1_out_descriptor,
				/*format=*/CUDNN_TENSOR_NCHW,
				/*dataType=*/CUDNN_DATA_FLOAT,
				/*batch_size=*/batch_size,
				/*channels=*/l1_out_channels,
				/*height=*/l1_out_dim,
				/*width=*/l1_out_dim));

	cudnnTensorDescriptor_t l1_kernel_bias_descriptor;
	checkCUDNN(cudnnCreateTensorDescriptor(&l1_kernel_bias_descriptor));
	checkCUDNN(cudnnSetTensor4dDescriptor(l1_kernel_bias_descriptor,
				/*format=*/CUDNN_TENSOR_NCHW,
				/*dataType=*/CUDNN_DATA_FLOAT,
				/*batch_size=*/1,
				/*channels=*/l1_out_channels,
				/*height=*/1,
				/*width=*/1));

	cudnnFilterDescriptor_t l1_kernel_descriptor;
	checkCUDNN(cudnnCreateFilterDescriptor(&l1_kernel_descriptor));
	checkCUDNN(cudnnSetFilter4dDescriptor(l1_kernel_descriptor,
				/*dataType=*/CUDNN_DATA_FLOAT,
				/*format=*/CUDNN_TENSOR_NCHW,
				/*out_channels=*/l1_out_channels,
				/*in_channels=*/input_channels,
				/*kernel_height=*/l1_kernel_dim,
				/*kernel_width=*/l1_kernel_dim));

	// Describing the Convolution Kernel

	cudnnConvolutionDescriptor_t l1_convolution_descriptor;
	checkCUDNN(cudnnCreateConvolutionDescriptor(&l1_convolution_descriptor));
	checkCUDNN(cudnnSetConvolution2dDescriptor(l1_convolution_descriptor,
				/*pad_height=*/l1_pad,
				/*pad_width=*/l1_pad,
				/*vertical_stride=*/l1_stride,
				/*horizontal_stride=*/l1_stride,
				/*dilation_height=*/l1_dilation,
				/*dilation_width=*/l1_dilation,
				/*mode=*/CUDNN_CROSS_CORRELATION,
				/*dataType=*/CUDNN_DATA_FLOAT
				));

	cudnnConvolutionFwdAlgo_t l1_convolution_algorithm;
	checkCUDNN(
			cudnnGetConvolutionForwardAlgorithm(cudnn,
				input_descriptor,
				l1_kernel_descriptor,
				l1_convolution_descriptor,
				l1_out_descriptor,
				CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
				/*memoryLimitInBytes=*/0,
				&l1_convolution_algorithm));

	size_t l1_workspace_bytes = 0;
	checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn,
				input_descriptor,
				l1_kernel_descriptor,
				l1_convolution_descriptor,
				l1_out_descriptor,
				l1_convolution_algorithm,
				&l1_workspace_bytes));
	workspace_bytes = max(workspace_bytes, l1_workspace_bytes);

	/* Allocating Memory for Layer 1 */

	int l1_out_bytes = batch_size * l1_out_channels * l1_out_dim * l1_out_dim
										 * sizeof(float);
	float* d_l1_output{NULL};
	cudaMalloc(&d_l1_output, l1_out_bytes);
	cudaMemset(d_l1_output, 0, l1_out_bytes);

	int l1_kernel_bytes = input_channels * l1_out_channels * l1_kernel_dim
												* l1_kernel_dim * sizeof(float);
	float* l1_kernel = (float*)malloc (l1_kernel_bytes);
	float* l1_kernel_bias = (float*)malloc (l1_out_channels * sizeof(float));

	// load pretrained weight
	load_weight(l1_kernel,
							input_channels * l1_out_channels * l1_kernel_dim * l1_kernel_dim,
							l1_weight_file);
	load_weight(l1_kernel_bias, l1_out_channels, l1_weight_bias_file);

	float* d_l1_kernel{NULL};
	cudaMalloc(&d_l1_kernel, l1_kernel_bytes);
	cudaMemcpy(d_l1_kernel, l1_kernel, l1_kernel_bytes, cudaMemcpyHostToDevice);

	float* d_l1_kernel_bias{NULL};
	cudaMalloc(&d_l1_kernel_bias, l1_out_channels * sizeof(float));
	cudaMemcpy(d_l1_kernel_bias, l1_kernel_bias, l1_out_channels * sizeof(float),
						 cudaMemcpyHostToDevice);



	/* Layer 2. Max Pooling */

	int l2_pool_dim = 2;
	int l2_pad = 0;
	int l2_stride = 2;

	int l2_out_dim = (l1_out_dim + l2_pad*2) / l2_stride;
	int l2_out_channels = l1_out_channels;

	// Describing Operands
	cudnnTensorDescriptor_t l2_out_descriptor;
	checkCUDNN(cudnnCreateTensorDescriptor(&l2_out_descriptor));
	checkCUDNN(cudnnSetTensor4dDescriptor(l2_out_descriptor,
				/*format=*/CUDNN_TENSOR_NCHW,
				/*dataType=*/CUDNN_DATA_FLOAT,
				/*batch_size=*/batch_size,
				/*channels=*/l2_out_channels,
				/*height=*/l2_out_dim,
				/*width=*/l2_out_dim));

	cudnnPoolingDescriptor_t l2_pool_descriptor;
	checkCUDNN(cudnnCreatePoolingDescriptor(&l2_pool_descriptor));
	checkCUDNN(cudnnSetPooling2dDescriptor(l2_pool_descriptor,
				/*poolingMode=*/CUDNN_POOLING_MAX,
				/*NanPropagationMode=*/CUDNN_PROPAGATE_NAN,
				l2_pool_dim, l2_pool_dim,
				l2_pad,      l2_pad,
				l2_stride,   l2_stride));

	/* Allocating Memory for Layer 2 */

	int l2_out_bytes = batch_size * l2_out_channels * l2_out_dim * l2_out_dim
										 * sizeof(float);

	float* d_l2_output{NULL};
	cudaMalloc(&d_l2_output, l2_out_bytes);
	cudaMemset(d_l2_output, 0, l2_out_bytes);



	/* Layer 3. Convolution */
	int l3_kernel_dim =  
	int l3_pad = 
	int l3_stride = 
	int l3_dilation = 

	int l3_out_dim = 
	int l3_out_channels = 

	char* l3_weight_file = "pretrained/conv2.bin";
	char* l3_weight_bias_file = "pretrained/conv2.bias.bin";

	// Describing Operands

	cudnnTensorDescriptor_t l3_out_descriptor;
	checkCUDNN(cudnnCreateTensorDescriptor(&l3_out_descriptor));
	checkCUDNN(cudnnSetTensor4dDescriptor(l3_out_descriptor,
				/*format=*/  ,
				/*dataType=*/  ,
				/*batch_size=*/  ,
				/*channels=*/  ,
				/*height=*/  ,
				/*width=*/  ));

	cudnnTensorDescriptor_t l3_kernel_bias_descriptor;
	checkCUDNN(cudnnCreateTensorDescriptor(&l3_kernel_bias_descriptor));
	checkCUDNN(cudnnSetTensor4dDescriptor(l3_kernel_bias_descriptor,
				/*format=*/  ,
				/*dataType=*/  ,
				/*batch_size=*/  ,
				/*channels=*/  ,
				/*height=*/  ,
				/*width=*/  ));

	cudnnFilterDescriptor_t l3_kernel_descriptor;
	checkCUDNN(cudnnCreateFilterDescriptor(&l3_kernel_descriptor));
	checkCUDNN(cudnnSetFilter4dDescriptor(l3_kernel_descriptor,
				/*dataType=*/  ,
				/*format=*/  ,
				/*out_channels=*/  ,
				/*in_channels=*/  ,
				/*kernel_height=*/  ,
				/*kernel_width=*/  ));

	// Describing the Convolution Kernel

	cudnnConvolutionDescriptor_t l3_convolution_descriptor;
	checkCUDNN(cudnnCreateConvolutionDescriptor(&l3_convolution_descriptor));
	checkCUDNN(cudnnSetConvolution2dDescriptor(l3_convolution_descriptor,
				/*pad_height=*/  ,
				/*pad_width=*/  ,
				/*vertical_stride=*/  ,
				/*horizontal_stride=*/  ,
				/*dilation_height=*/  ,
				/*dilation_width=*/  ,
				/*mode=*/  ,
				/*dataType=*/  ));

	cudnnConvolutionFwdAlgo_t l3_convolution_algorithm;
	checkCUDNN(cudnnGetConvolutionForwardAlgorithm(cudnn,
				,
				,
				,
				,
				,
				/*memoryLimitInBytes=*/0,
				));

	size_t l3_workspace_bytes = 0;
	checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn,
				,
				,
				,
				,
				,
				));
	workspace_bytes = max(workspace_bytes, l3_workspace_bytes);

	/* Allocating Memory for Layer 3 */

	int l3_out_bytes = 
	float* d_l3_output{NULL};
	cudaMalloc(&d_l3_output, l3_out_bytes);
	cudaMemset(d_l3_output, 0, l3_out_bytes);

	int l3_kernel_bytes = 
	float* l3_kernel = (float*)malloc (l3_kernel_bytes);
	float* l3_kernel_bias = (float*)malloc ( );

	// load pretrained weight
	load_weight(l3_kernel,
							,
							l3_weight_file);
	load_weight(l3_kernel_bias,   , l3_weight_bias_file);

	float* d_l3_kernel{NULL};
	cudaMalloc(&d_l3_kernel, l3_kernel_bytes);
	cudaMemcpy(d_l3_kernel, l3_kernel, l3_kernel_bytes, cudaMemcpyHostToDevice);

	float* d_l3_kernel_bias{NULL};
	cudaMalloc(&d_l3_kernel_bias, );
	cudaMemcpy(d_l3_kernel_bias,
						 l3_kernel_bias,
						 ,
						 cudaMemcpyHostToDevice);



	/* Layer 4. Max Pooling */

	int l4_pool_dim = 
	int l4_pad = 
	int l4_stride = 

	int l4_out_dim = 
	int l4_out_channels = 

	// Describing Operands
	cudnnTensorDescriptor_t l4_out_descriptor;
	checkCUDNN(cudnnCreateTensorDescriptor(&l4_out_descriptor));
	checkCUDNN(cudnnSetTensor4dDescriptor(l4_out_descriptor,
				/*format=*/  ,
				/*dataType=*/  ,
				/*batch_size=*/  ,
				/*channels=*/  ,
				/*height=*/  ,
				/*width=*/  ));

	cudnnPoolingDescriptor_t l4_pool_descriptor;
	checkCUDNN(cudnnCreatePoolingDescriptor(&l4_pool_descriptor));
	checkCUDNN(cudnnSetPooling2dDescriptor(l4_pool_descriptor,
				/*poolingMode=*/CUDNN_POOLING_MAX,
				/*NanPropagationMode=*/CUDNN_PROPAGATE_NAN,
				, ,
				, ,
				, ));

	/* Allocating Memory for Layer 2 */

	int l4_out_bytes = 

	float* d_l4_output{NULL};
	cudaMalloc(&d_l4_output, l4_out_bytes);
	cudaMemset(d_l4_output, 0, l4_out_bytes);



	/* Layer 5. Fully Connected Layer */
	int l5_fc_in_dim = (l4_out_channels * l4_out_dim * l4_out_dim);
	int l5_fc_out_dim = 500;
	int l5_fc_neuron_size = l5_fc_in_dim * l5_fc_out_dim;

	char* l5_weight_file = "pretrained/ip1.bin";
	char* l5_weight_bias_file = "pretrained/ip1.bias.bin";

	cudnnActivationDescriptor_t l5_fc_activation_descriptor;
	checkCUDNN(cudnnCreateActivationDescriptor(&l5_fc_activation_descriptor));
	checkCUDNN(cudnnSetActivationDescriptor(l5_fc_activation_descriptor,
																					CUDNN_ACTIVATION_RELU,
																					CUDNN_PROPAGATE_NAN, 0.0));

	cudnnTensorDescriptor_t l5_relu_descriptor;
	checkCUDNN(cudnnCreateTensorDescriptor(&l5_relu_descriptor));
	checkCUDNN(cudnnSetTensor4dDescriptor(l5_relu_descriptor,
					CUDNN_TENSOR_NCHW,
					CUDNN_DATA_FLOAT,
					batch_size, l5_fc_out_dim, 1, 1));

	/* Allocating memory for Layer 5 */

	float* l5_fc_neuron = (float*)malloc (l5_fc_neuron_size * sizeof(float));
	load_weight(l5_fc_neuron, l5_fc_neuron_size, l5_weight_file);

	float* l5_fc_neuron_bias = (float*)malloc (l5_fc_out_dim * sizeof(float));
	load_weight(l5_fc_neuron_bias, l5_fc_out_dim, l5_weight_bias_file);

	float* d_l5_fc_neuron{NULL};
	checkCUDA(cudaMalloc(&d_l5_fc_neuron, l5_fc_neuron_size * sizeof(float)));
	checkCUDA(cudaMemcpy(d_l5_fc_neuron, l5_fc_neuron,
			l5_fc_neuron_size * sizeof(float), cudaMemcpyHostToDevice));

	float* d_l5_fc_neuron_bias{NULL};
	checkCUDA(cudaMalloc(&d_l5_fc_neuron_bias, l5_fc_out_dim * sizeof(float)));
	checkCUDA(cudaMemcpy(d_l5_fc_neuron_bias, l5_fc_neuron_bias,
			l5_fc_out_dim * sizeof(float), cudaMemcpyHostToDevice));

	float* d_l5_fc_output{NULL};
	checkCUDA(cudaMalloc(&d_l5_fc_output,
											 batch_size * l5_fc_out_dim * sizeof(float)));
	checkCUDA(cudaMemset(d_l5_fc_output, 0, l5_fc_out_dim * sizeof(float)));

	float* d_l5_relu_output{NULL};
	checkCUDA(cudaMalloc(&d_l5_relu_output,
											 batch_size * l5_fc_out_dim * sizeof(float)));
	checkCUDA(cudaMemset(d_l5_relu_output, 0, l5_fc_out_dim * sizeof(float)));



	/* Layer 6. Fully Connected Layer */
	int l6_fc_in_dim = l5_fc_out_dim;
	int l6_fc_out_dim = 10;
	int l6_fc_neuron_size = l6_fc_in_dim * l6_fc_out_dim;

	char* l6_weight_file = "pretrained/ip2.bin";
	char* l6_weight_bias_file = "pretrained/ip2.bias.bin";

	cudnnActivationDescriptor_t l6_fc_activation_descriptor;
	checkCUDNN(cudnnCreateActivationDescriptor(&l6_fc_activation_descriptor));
	checkCUDNN(cudnnSetActivationDescriptor(l6_fc_activation_descriptor,
																					CUDNN_ACTIVATION_RELU,
																					CUDNN_PROPAGATE_NAN, 0.0));

	cudnnTensorDescriptor_t l6_softmax_descriptor;
	checkCUDNN(cudnnCreateTensorDescriptor(&l6_softmax_descriptor));
	checkCUDNN(cudnnSetTensor4dDescriptor(l6_softmax_descriptor,
				CUDNN_TENSOR_NCHW,
				CUDNN_DATA_FLOAT,
				batch_size, l6_fc_out_dim, 1, 1));

	/* Allocating memory for Layer 6 */

	float* l6_fc_neuron = (float*)malloc (l6_fc_neuron_size * sizeof(float));
	load_weight(l6_fc_neuron, l6_fc_neuron_size, l6_weight_file);

	float* l6_fc_neuron_bias = (float*)malloc (l6_fc_out_dim * sizeof(float));
	load_weight(l6_fc_neuron_bias, l6_fc_out_dim, l6_weight_bias_file);

	float* l6_softmax_output = (float*)malloc (l6_fc_out_dim * sizeof(float));

	float* d_l6_fc_neuron{NULL};
	checkCUDA(cudaMalloc(&d_l6_fc_neuron, l6_fc_neuron_size * sizeof(float)));
	checkCUDA(cudaMemcpy(d_l6_fc_neuron, l6_fc_neuron,
			l6_fc_neuron_size * sizeof(float), cudaMemcpyHostToDevice));

	float* d_l6_fc_neuron_bias{NULL};
	checkCUDA(cudaMalloc(&d_l6_fc_neuron_bias, l6_fc_out_dim * sizeof(float)));
	checkCUDA(cudaMemcpy(d_l6_fc_neuron_bias, l6_fc_neuron_bias,
			l6_fc_out_dim * sizeof(float), cudaMemcpyHostToDevice));

	float* d_l6_fc_output{NULL};
	checkCUDA(cudaMalloc(&d_l6_fc_output,
											 batch_size * l6_fc_out_dim * sizeof(float)));
	checkCUDA(cudaMemset(d_l6_fc_output, 0, l6_fc_out_dim * sizeof(float)));

	float* d_l6_softmax_output{NULL};
	checkCUDA(cudaMalloc(&d_l6_softmax_output,
											 batch_size * l6_fc_out_dim * sizeof(float)));
	checkCUDA(cudaMemset(d_l6_softmax_output, 0, l6_fc_out_dim * sizeof(float)));



	/* Forward */

	struct timeval start, end;
	gettimeofday(&start, NULL);

	// Allocating Memory for Workspace
	void* d_workspace{NULL};
	cudaMalloc(&d_workspace, workspace_bytes);

	// One vector for FC
	float *d_onevec{NULL};
	checkCUDA(cudaMalloc(&d_onevec, batch_size * sizeof(float)));
	checkCUDA(cudaMemset(d_onevec, 1, batch_size * sizeof(float)));

	/* Layer 1. Convolution */
	const float alpha = 1, beta = 0;
	checkCUDNN(cudnnConvolutionForward(cudnn, &alpha, input_descriptor,
				                             /*input device mem=*/d_input,
				                             l1_kernel_descriptor,
				                             /*kernel device mem*/d_l1_kernel,
				                             l1_convolution_descriptor,
				                             l1_convolution_algorithm,
				                             d_workspace, workspace_bytes,
				                             &beta, l1_out_descriptor,
				                             /*output device mem=*/d_l1_output));
	// Add bias
	checkCUDNN(cudnnAddTensor(cudnn, &alpha, l1_kernel_bias_descriptor,
                            d_l1_kernel_bias, &alpha, l1_out_descriptor,
                            d_l1_output));

	/* Layer 2. Max Pooling */
	checkCUDNN(cudnnPoolingForward(cudnn, l2_pool_descriptor, &alpha,
																 l1_out_descriptor, d_l1_output, &beta,
																 l2_out_descriptor, d_l2_output));

	/* Layer 3. Convolution */
	checkCUDNN(cudnnConvolutionForward(cudnn,
				                             &alpha,
				                             l2_out_descriptor,
				                             /*input device mem=*/d_l2_output,
				                             l3_kernel_descriptor,
				                             /*kernel device mem*/d_l3_kernel,
				                             l3_convolution_descriptor,
				                             l3_convolution_algorithm,
				                             d_workspace,
				                             workspace_bytes,
				                             &beta,
				                             l3_out_descriptor,
				                             /*output device mem=*/d_l3_output));
	// Add bias
	checkCUDNN(cudnnAddTensor(cudnn, &alpha, l3_kernel_bias_descriptor,
                            d_l3_kernel_bias, &alpha, l3_out_descriptor,
                            d_l3_output));

	/* Layer 4. Max Pooling */
	checkCUDNN(cudnnPoolingForward(cudnn, l4_pool_descriptor, &alpha,
																 l3_out_descriptor, d_l3_output, &beta,
																 l4_out_descriptor, d_l4_output));

	/* Layer 5. Fully Connected */
	// FC1 layer
	// Forward propagate neurons using weights
	checkCUBLAS(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
				l5_fc_out_dim, batch_size, l5_fc_in_dim,
				&alpha,
				d_l5_fc_neuron, l5_fc_in_dim,
				d_l4_output, l5_fc_in_dim,
				&beta,
				d_l5_fc_output, l5_fc_out_dim));
	// Add bias using GEMM's "beta"
	checkCUBLAS(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
				l5_fc_out_dim, batch_size, 1,
				&alpha,
				d_l5_fc_neuron_bias, l5_fc_out_dim,
				d_onevec, 1,
				&alpha,
				d_l5_fc_output, l5_fc_out_dim));
	// ReLU activation
	checkCUDNN(cudnnActivationForward(cudnn, l5_fc_activation_descriptor, &alpha,
																		l5_relu_descriptor, d_l5_fc_output, &beta,
																		l5_relu_descriptor, d_l5_relu_output));

	/* Layer 6. Fully Connected (Softmax) */
	// FC2 layer
	checkCUBLAS(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
				l6_fc_out_dim, batch_size, l6_fc_in_dim,
				&alpha,
				d_l6_fc_neuron, l6_fc_in_dim,
				d_l5_relu_output, l6_fc_in_dim,
				&beta,
				d_l6_fc_output, l6_fc_out_dim));
	// Add bias using GEMM's "beta"
	checkCUBLAS(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
				l6_fc_out_dim, batch_size, 1,
				&alpha,
				d_l6_fc_neuron_bias, l6_fc_out_dim,
				d_onevec, 1,
				&alpha,
				d_l6_fc_output, l6_fc_out_dim));
	// Softmax loss
	checkCUDNN(cudnnSoftmaxForward(cudnn, CUDNN_SOFTMAX_ACCURATE,
																 CUDNN_SOFTMAX_MODE_CHANNEL,
																 &alpha, l6_softmax_descriptor,
																 d_l6_fc_output, &beta, l6_softmax_descriptor,
																 d_l6_softmax_output));

	/* Show result */

	checkCUDA(cudaMemcpy(l6_softmax_output, d_l6_softmax_output,
											 l6_fc_out_dim * sizeof(float), cudaMemcpyDeviceToHost));

	gettimeofday(&end, NULL);

	int i, chosen = 0;
	for (i = 0; i < l6_fc_out_dim; i++) {
		printf("%d: %.2f\n", i, l6_softmax_output[i]);
		if (l6_softmax_output[i] > l6_softmax_output[chosen])
			chosen = i;
	}
	printf("\nPredict: %d\n", chosen);
	printf("Time: %f\n", getMillisecond(start, end));



	/* Free */

	// input
	cudnnDestroyTensorDescriptor(input_descriptor);
	cudaFree(d_input);

	// Layer 1
	cudnnDestroyTensorDescriptor(l1_out_descriptor);
	cudnnDestroyFilterDescriptor(l1_kernel_descriptor);
	cudnnDestroyTensorDescriptor(l1_kernel_bias_descriptor);
	cudnnDestroyConvolutionDescriptor(l1_convolution_descriptor);
	cudaFree(d_l1_output);
	cudaFree(d_l1_kernel);
	cudaFree(d_l1_kernel_bias);
	free(l1_kernel);
	free(l1_kernel_bias);

	// Layer 2
	cudnnDestroyTensorDescriptor(l2_out_descriptor);
	cudnnDestroyPoolingDescriptor(l2_pool_descriptor);
	cudaFree(d_l2_output);

	// Layer 3
	cudnnDestroyTensorDescriptor(l3_out_descriptor);
	cudnnDestroyFilterDescriptor(l3_kernel_descriptor);
	cudnnDestroyTensorDescriptor(l3_kernel_bias_descriptor);
	cudnnDestroyConvolutionDescriptor(l3_convolution_descriptor);
	cudaFree(d_l3_output);
	cudaFree(d_l3_kernel);
	cudaFree(d_l3_kernel_bias);
	free(l3_kernel);
	free(l3_kernel_bias);

	// Layer 4
	cudnnDestroyTensorDescriptor(l4_out_descriptor);
	cudnnDestroyPoolingDescriptor(l4_pool_descriptor);
	cudaFree(d_l4_output);

	// Layer 5
	checkCUDNN(cudnnDestroyActivationDescriptor(l5_fc_activation_descriptor));
	checkCUDNN(cudnnDestroyTensorDescriptor(l5_relu_descriptor));
	checkCUDA(cudaFree(d_l5_fc_output));
	checkCUDA(cudaFree(d_l5_fc_neuron));
	checkCUDA(cudaFree(d_l5_fc_neuron_bias));
	checkCUDA(cudaFree(d_l5_relu_output));
	free(l5_fc_neuron);
	free(l5_fc_neuron_bias);

	// Layer 6
	checkCUDNN(cudnnDestroyActivationDescriptor(l6_fc_activation_descriptor));
	checkCUDNN(cudnnDestroyTensorDescriptor(l6_softmax_descriptor));
	checkCUDA(cudaFree(d_l6_fc_output));
	checkCUDA(cudaFree(d_l6_fc_neuron));
	checkCUDA(cudaFree(d_l6_fc_neuron_bias));
	checkCUDA(cudaFree(d_l6_softmax_output));
	free(l6_fc_neuron);
	free(l6_fc_neuron_bias);
	free(l6_softmax_output);

	// etc
	cudaFree(d_onevec);
	cudaFree(d_workspace);
	cublasDestroy(cublas);
	cudnnDestroy(cudnn);

	return 0;
}

#include <cudnn.h>
#include <cublas.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <opencv2/opencv.hpp>

#define checkCUDA(expression)                  \
{                                              \
	cudaError_t status = (expression);           \
	if (status != cudaSuccess) {                 \
		printf("Error on line %d: err code %d\n",  \
				__LINE__, status);                     \
		exit(EXIT_FAILURE);                        \
	}                                            \
}

#define checkCUBLAS(expression)                \
{                                              \
	cublasStatus_t status = (expression);        \
	if (status != CUBLAS_STATUS_SUCCESS) {       \
		printf("Error on line %d: err code %d\n",  \
				__LINE__, status);                     \
		exit(EXIT_FAILURE);                        \
	}                                            \
}

#define getMillisecond(start, end) \
	(end.tv_sec-start.tv_sec)*1000 + \
	(end.tv_usec-start.tv_usec)/1000.0

cv::Mat load_image(const char* image_path) {
	cv::Mat image = cv::imread(image_path, CV_LOAD_IMAGE_COLOR);
	image.convertTo(image, CV_32FC3);
	cv::normalize(image, image, 0, 1, cv::NORM_MINMAX);
	return image;
}

void save_image(const char* output_filename,
		float* buffer,
		int height,
		int width) {
	cv::Mat output_image(height, width, CV_32FC3, buffer);
	// Make negative values zero.
	cv::threshold(output_image,
			output_image,
			/*threshold=*/0,
			/*maxval=*/0,
			cv::THRESH_TOZERO);
	cv::normalize(output_image, output_image, 0.0, 255.0, cv::NORM_MINMAX);
	output_image.convertTo(output_image, CV_8UC3);
	cv::imwrite(output_filename, output_image);
}

void im2col_cpu(const float* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    float* data_col) {
  const int output_h = (height + 2 * pad_h -
    (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  const int output_w = (width + 2 * pad_w -
    (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
  const int channel_size = height * width;
  for (int channel = channels; channel--; data_im += channel_size) {
    for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
      for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
        int input_row = -pad_h + kernel_row * dilation_h;
        for (int output_rows = output_h; output_rows; output_rows--) {
          if (!(input_row < height)) {
            for (int output_cols = output_w; output_cols; output_cols--) {
              *(data_col++) = 0;
            }
          } else {
            int input_col = -pad_w + kernel_col * dilation_w;
            for (int output_col = output_w; output_col; output_col--) {
              if (input_col < width) {
                *(data_col++) = data_im[input_row * width + input_col];
              } else {
                *(data_col++) = 0;
              }
              input_col += stride_w;
            }
          }
          input_row += stride_h;
        }
      }
    }
  }
}

__global__ void im2col_gpu_kernel(const int n, const float* data_im,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
	const int height_col, const int width_col,
    float* data_col) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x;
		  index < n; index += blockDim.x * gridDim.x) {
    const int h_index = index / width_col;
    const int h_col = h_index % height_col;
    const int w_col = index % width_col;
    const int c_im = h_index / height_col;
    const int c_col = c_im * kernel_h * kernel_w;
    const int h_offset = h_col * stride_h - pad_h;
    const int w_offset = w_col * stride_w - pad_w;
    float* data_col_ptr = data_col;
    data_col_ptr += (c_col * height_col + h_col) * width_col + w_col;
    const float* data_im_ptr = data_im;
    data_im_ptr += (c_im * height + h_offset) * width + w_offset;

	// Fill kernel code !

	for (int i = 0; i < kernel_h; ++i) {
		for (int j = 0; j< kernel_w; ++j) {
			int h_im = h_offset + i * dilation_h;
			int w_im = w_offset + j * dilation_w;
			*data_col_ptr = (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width) ?
				data_im_ptr[i * dilation_h * width +j * dilation_w] : 0;
			data_col_ptr += height_col * width_col;
		}
	}
  }
}

void im2col_gpu(const float* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
	const int dilation_h, const int dilation_w,
    float* data_col) {
  int height_col = (height + 2 * pad_h - kernel_h) / stride_h + 1;
  int width_col = (width + 2 * pad_w - kernel_w) / stride_w + 1;
  int num_kernels = channels * height_col * width_col;

  int num_threads = 512;
  int num_blocks = (num_kernels + num_threads - 1) / num_threads;

  im2col_gpu_kernel<<<num_blocks, num_threads>>>(
      num_kernels, data_im, height, width, kernel_h, kernel_w, pad_h,
      pad_w, stride_h, stride_w, dilation_h, dilation_w, 
	  height_col, width_col, data_col);
}

void convert_image_layout_hwc_chw(const int channels, const int height, const int width,
		const float *src, float *dst) {
	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++)
			for (int k = 0; k < channels; k++)
				dst[k*height*width + i*width + j] = src[i*width*channels + j*channels + k];
}

void convert_image_layout_chw_hwc(const int channels, const int height, const int width,
		const float *src, float *dst) {
	for (int i = 0; i < channels; i++)
		for (int j = 0; j < height; j++)
			for (int k = 0; k < width; k++)
				dst[j*width*channels + k*channels + i] = src[i*height*width + j*width + k];
}

int main(int argc, char const *argv[]) {
	// variables for time check
	float ms = 0;
	struct timeval start, end;

	// Init cuBLAS
	cublasHandle_t handle;
	checkCUBLAS (cublasCreate (&handle));
	cv::Mat image = load_image("image/input.jpg");

	int img_h = image.rows;
	int img_w = image.cols;
	int img_c = image.channels();
	int img_bytes = img_c*img_h*img_w*sizeof(float);

	// convert hwc image layout to chw
	float *image_chw = (float *)malloc(img_bytes);
	convert_image_layout_hwc_chw(img_c, img_h, img_w, image.ptr<float>(0), image_chw);

	float* d_input{NULL};
	cudaMalloc(&d_input, img_bytes);
	cudaMemcpy(d_input, image_chw, img_bytes, cudaMemcpyHostToDevice);

	// Kernel for edge detection
	const float kernel_template[3][3] = {
		{1,  1, 1},
		{1, -8, 1},
		{1,  1, 1}
	};

	float h_kernel[3][3][3][3];
	for (int kernel = 0; kernel < 3; ++kernel) {
		for (int channel = 0; channel < 3; ++channel) {
			for (int row = 0; row < 3; ++row) {
				for (int column = 0; column < 3; ++column) {
					h_kernel[kernel][channel][row][column] = kernel_template[row][column];
				}
			}
		}
	}

	float* d_kernel{NULL};
	cudaMalloc(&d_kernel, sizeof(h_kernel));
	cudaMemcpy(d_kernel, h_kernel, sizeof(h_kernel), cudaMemcpyHostToDevice);

	// parameters for im2col
	int kernel_num = 3;
	int pad_h = 1;
	int pad_w = 1;
	int kernel_h = 3;
	int kernel_w = 3;
	int stride_h = 1;
	int stride_w = 1;
	int channels = 3;
	int dilation_h = 1;
	int dilation_w = 1;

  	int col_h = (img_h + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  	int col_w = (img_w + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;

	int col_bytes = channels * kernel_h * kernel_w * col_h * col_w * sizeof(float);

	// device memory to store im2col results
	float* d_col{NULL};
	cudaMalloc(&d_col, col_bytes);

	// cpu ver im2col

	float* h_col = (float*)malloc(col_bytes);
/*
	im2col_cpu(image_chw, channels,
		       img_h, img_w, kernel_h, kernel_w,
			   pad_h, pad_w,
			   stride_h, stride_w,
			   dilation_h, dilation_w,
			   h_col);
	cudaMemcpy(d_col, h_col, col_bytes, cudaMemcpyHostToDevice);
*/	

    // gpu ver im2col
	im2col_gpu(d_input, channels,
		       img_h, img_w, kernel_h, kernel_w,
			   pad_h, pad_w,
			   stride_h, stride_w,
			   dilation_h, dilation_w,
			   d_col);
	cudaDeviceSynchronize();

	// allocate cuda output buffer
	int output_bytes = kernel_num * col_h * col_w * sizeof(float);
	float* d_output{NULL};
	cudaMalloc(&d_output, output_bytes);

	// allocate host output buffer
	float* h_output = (float*)malloc(output_bytes);

	// Sgemm with GPU
	int M = kernel_num;
	int N = col_h * col_w;
	int K = channels * kernel_h * kernel_w;

	// cublas gemm
	// caution!! cublas use column major matrix!!
	float alpha = 1.0, beta = 0.0;
	gettimeofday(&start, NULL);
	checkCUBLAS (cublasSgemm (handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
				&alpha, d_col, N, d_kernel, K, &beta, d_output, N));
	gettimeofday(&end, NULL);

	// Print duration
	ms = getMillisecond(start, end);
	printf("GPU time: %f (ms)\n", ms);

	// Memcpy device to host
	checkCUBLAS (cublasGetMatrix (M, N, sizeof(float),
				d_output, M, h_output, M));

	// convert chw image layout to hwc
	float* h_output_hwc = (float*)malloc(output_bytes);
	convert_image_layout_chw_hwc(M, col_h, col_w, h_output, h_output_hwc);

	// Save output
	save_image("image/output.jpg", h_output_hwc, image.rows, image.cols);

	// Free
	checkCUDA(cudaFree(d_kernel));
	checkCUDA(cudaFree(d_input));
	checkCUDA(cudaFree(d_output));
	checkCUDA(cudaFree(d_col));
	free(image_chw);
	free(h_col);
	free(h_output);
	free(h_output_hwc);

	checkCUBLAS (cublasDestroy (handle));
	return EXIT_SUCCESS;
}

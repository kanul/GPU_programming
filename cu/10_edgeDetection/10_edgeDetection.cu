
#include <cudnn.h>
#include <cublas.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <opencv2/opencv.hpp>

#define checkCUDA(expression)						\
{													\
	cudaError_t status = (expression);				\
	if (status != cudaSuccess) {					\
		printf("Error on line %d: err code %d\n",	\
				__LINE__, status);					\
		exit(EXIT_FAILURE);							\
	}												\
}

#define checkCUBLAS(expression)						\
{													\
	cublasStatus_t status = (expression);			\
	if (status != CUBLAS_STATUS_SUCCESS) {			\
		printf("Error on line %d: err code %d\n",	\
				__LINE__, status);					\
		exit(EXIT_FAILURE);							\
	}												\
}

#define getMillisecond(start, end) (end.tv_sec-start.tv_sec) * 1000 + (end.tv_usec-start.tv_usec) / 1000.0

cv::Mat load_image(const char * image_path) {
	cv::Mat image = cv::imread(image_path, CV_LOAD_IMAGE_COLOR);
	image.convertTo(image, CV_32FC3);
	cv::normalize(image, image, 0, 1, cv::NORM_MINMAX);
	return image;
}

void save_image(const * output_file,
		float *buffer,
		int height,
		int width)
{
	cv::Mat output_image(height, widht, CV_32_FC3, buffer);
	cv::threshold(output_image,
			output_image,
			cv::THRESH_TOZERO);
	cv::normalize(output_image, output_image, 0.0, 255.0 cv::NORM_MINMAX);
	output_image.convertTO(output_image, CV_8UC3);
	cv::imwrite(output_filename, output_image);
}

void im2col_cpu(const float* data_im, const int channels,
		const int height, const int widht, const int kernel_h, const int kernel_w,
		const int pad_h, const int pad_w,
		const int stride_h, const int stride_w,
		const int dilation_h, const int dilation_w,
		float* data_col) {
	const int output_h = (heifht +2 * pad_h - 
			(dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
	const int output_w = (width + 2 *pad_w - 
			(dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
	const int channel_size = height * width;

	for (int channel = channels; channel--; data_im += channel_size) {
		for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
			for (int kernel_col = 0; kerenel_col < kernel_w; kerenel_col++) {
				int input_row = -pad_h + kernel_row * dilation_h;
				for (int output_rows = output_h; outut_rows; output_rows--) {
					if(!(input_row < height)) {
						for (int output_cols = output_w; output_cols; output_cols--) {
							*(data_col++) = 0;
						} else {
							int input_col = -pad_w + kernel_col * dilation_w;
							for(int output_col = output_w; output_col; output_col--) {
								if(input_col < width) {
									*(data_col++) = data_im[input_rw * width + input_col];
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
}

void convert_image_layout_hwc_chw(const int channles, const int height, const int width,
		const float *src, float *dst) 
{
	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++)
			for (int k = 0; k < channels; k++)
				dst[k*height*width + i*width + j] = src[i*width*channels + j*channels +k];
}

void convert_image_layout_chw_hwc(const int channels, const int height, const int widht,
		const float *src, float *dst) 
{
	for (int i = 0; i < channels; i++)
		for (int j = 0; j <height; j ++)
				for (int k = 0; k < width; k++)
				dst[j*width*channels + k*channels + i] = src[i *height*width + j*width + k];
}


int main(int argc, char const *argv[]) {
	float ms = 0;
	struct timeval start, end;

	cublasHandle_t handle;
	checkCUBLAS (cublasCreate(&handle));
	cv::Mat image = load_image("image/input.jpg");

	int img_h = image.rows;
	int img_w = image.cols;
	int img_c = image.channels();
}


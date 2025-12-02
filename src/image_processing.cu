#include "image_processing.h"
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

__global__
void gaussianKernel(const unsigned char* input, unsigned char* output, int w, int h)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    float kernel[3][3] = {
        {1, 2, 1},
        {2, 4, 2},
        {1, 2, 1}
    };

    float sum = 0.0f;
    float total = 0.0f;

    for (int ky = -1; ky <= 1; ky++) {
        for (int kx = -1; kx <= 1; kx++) {
            int px = min(max(x + kx, 0), w - 1);
            int py = min(max(y + ky, 0), h - 1);
            float kval = kernel[ky + 1][kx + 1];

            sum += kval * input[py * w + px];
            total += kval;
        }
    }

    output[y * w + x] = static_cast<unsigned char>(sum / total);
}

void applyGaussianBlurCUDA(const cv::Mat &input, cv::Mat &output)
{
    int w = input.cols;
    int h = input.rows;

    output.create(h, w, CV_8UC1);

    unsigned char *d_in, *d_out;

    cudaMalloc(&d_in, w * h);
    cudaMalloc(&d_out, w * h);

    cudaMemcpy(d_in, input.data, w * h, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((w + 15) / 16, (h + 15) / 16);

    gaussianKernel<<<grid, block>>>(d_in, d_out, w, h);

    cudaMemcpy(output.data, d_out, w * h, cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_out);
}

__global__
void sobelKernel(const unsigned char* input, unsigned char* output, int w, int h)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    int Gx[3][3] = {
        { -1, 0, 1 },
        { -2, 0, 2 },
        { -1, 0, 1 }
    };

    int Gy[3][3] = {
        { -1, -2, -1 },
        {  0,  0,  0 },
        {  1,  2,  1 }
    };

    float sumX = 0.0f;
    float sumY = 0.0f;

    for (int ky = -1; ky <= 1; ky++) {
        for (int kx = -1; kx <= 1; kx++) {
            int px = min(max(x + kx, 0), w - 1);
            int py = min(max(y + ky, 0), h - 1);

            int pixel = input[py * w + px];

            sumX += Gx[ky + 1][kx + 1] * pixel;
            sumY += Gy[ky + 1][kx + 1] * pixel;
        }
    }

    float mag = sqrtf(sumX * sumX + sumY * sumY);
    mag = fminf(255.0f, mag);

    output[y * w + x] = static_cast<unsigned char>(mag);
}

void applySobelEdgesCUDA(const cv::Mat &input, cv::Mat &output)
{
    int w = input.cols;
    int h = input.rows;

    output.create(h, w, CV_8UC1);

    unsigned char *d_in, *d_out;

    cudaMalloc(&d_in, w * h);
    cudaMalloc(&d_out, w * h);

    cudaMemcpy(d_in, input.data, w * h, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((w + 15) / 16, (h + 15) / 16);

    sobelKernel<<<grid, block>>>(d_in, d_out, w, h);

    cudaMemcpy(output.data, d_out, w * h, cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_out);
}

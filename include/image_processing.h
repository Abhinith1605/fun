#pragma once
#include <opencv2/opencv.hpp>

void applyGaussianBlurCUDA(const cv::Mat &input, cv::Mat &output);
void applySobelEdgesCUDA(const cv::Mat &input, cv::Mat &output);

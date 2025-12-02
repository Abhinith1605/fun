#include "image_processing.h"
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <iostream>

namespace fs = std::filesystem;

int main() {
    std::string inputDir = "data/input";
    std::string outputDir = "data/output";

    fs::create_directories(outputDir);

    for (const auto &file : fs::directory_iterator(inputDir)) {
        std::string path = file.path().string();
        std::string name = file.path().filename().string();

        cv::Mat img = cv::imread(path);
        if (img.empty()) continue;

        cv::Mat gray, blur, edges;

        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

        applyGaussianBlurCUDA(gray, blur);
        applySobelEdgesCUDA(gray, edges);

        fs::path out1 = fs::path(outputDir) / ("blur_" + name);
        fs::path out2 = fs::path(outputDir) / ("edges_" + name);

        cv::imwrite(out1.string(), blur);
        cv::imwrite(out2.string(), edges);
    }

    return 0;
}

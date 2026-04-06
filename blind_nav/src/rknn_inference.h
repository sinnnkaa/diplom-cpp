#pragma once
#include <rknn_api.h>
#include <opencv2/opencv.hpp>

class RKNNModel {
public:
    rknn_context ctx;
    rknn_input_output_num io_num;
    rknn_tensor_attr output_attr;

    bool load(const char* model_path);
    std::vector<int8_t> infer(cv::Mat& img);
};
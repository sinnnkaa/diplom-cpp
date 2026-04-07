#pragma once
#include <rknn_api.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

struct RKNNTensorAttr {
    float scale;
    int32_t zp;
};

class RKNNModel {
public:
    RKNNModel();
    ~RKNNModel();

    bool load(const std::string& model_path);
    std::vector<int8_t> infer(const cv::Mat& img);
    
    RKNNTensorAttr out_attr;
    int input_w, input_h;

private:
    rknn_context ctx;
    rknn_input_output_num io_num;
};

#pragma once
#include <rknn_api.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

struct RKNNOutputAttr {
    float scale = 1.0f;
    int zp = 0;
};

class RKNNModel {
public:
    RKNNModel();
    ~RKNNModel();

    bool load(const std::string& model_path);
    // ИЗМЕНЕНО: теперь возвращаем vector<float>
    std::vector<float> infer(const cv::Mat& img);
    
    RKNNOutputAttr out_attr;
    int input_w, input_h;

private:
    rknn_context ctx;
    rknn_input_output_num io_num;
};

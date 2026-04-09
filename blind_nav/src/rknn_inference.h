#ifndef RKNN_INFERENCE_H
#define RKNN_INFERENCE_H

#include "rknn_api.h"
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

class RKNNModel {
public:
    RKNNModel();
    ~RKNNModel();

    bool load(const std::string& model_path);
    
    // ИЗМЕНЕНО: Теперь возвращает 3 тензора вместо 1
    std::vector<std::vector<float>> infer(const cv::Mat& img);

    rknn_context get_ctx();

private:
    rknn_context ctx;
};
#endif

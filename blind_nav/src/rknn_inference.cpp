#include "rknn_inference.h"
#include <iostream>
#include <fstream>
#include <cstring>

RKNNModel::RKNNModel() : ctx(0) {}
RKNNModel::~RKNNModel() { if (ctx) rknn_destroy(ctx); }

bool RKNNModel::load(const std::string& model_path) {
    std::ifstream ifs(model_path, std::ios::binary | std::ios::ate);
    if (!ifs.is_open()) return false;
    std::streamsize size = ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    std::vector<char> buffer(size);
    ifs.read(buffer.data(), size);

    if (rknn_init(&ctx, buffer.data(), size, 0, NULL) < 0) return false;

    rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));

    rknn_tensor_attr in_attr;
    in_attr.index = 0;
    rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &in_attr, sizeof(in_attr));
    input_w = in_attr.dims[2]; 
    input_h = in_attr.dims[1];

    std::cout << "Model Loaded! Input: " << input_w << "x" << input_h << std::endl;
    return true;
}

// Теперь возвращаем std::vector<float>!
std::vector<float> RKNNModel::infer(const cv::Mat& img) {
    cv::Mat rgb;
    cv::cvtColor(img, rgb, cv::COLOR_BGR2RGB);

    rknn_input inputs[1];
    std::memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].size = rgb.total() * rgb.elemSize();
    inputs[0].buf = rgb.data;

    rknn_inputs_set(ctx, 1, inputs);
    rknn_run(ctx, NULL);

    rknn_output outputs[1];
    std::memset(outputs, 0, sizeof(outputs));
    // ГЛАВНЫЙ ФИКС: просим драйвер отдать float
    outputs[0].want_float = 1; 
    outputs[0].is_prealloc = 0;

    if (rknn_outputs_get(ctx, 1, outputs, NULL) < 0) return {};

    // Копируем как float
    int count = outputs[0].size / sizeof(float);
    std::vector<float> result((float*)outputs[0].buf, (float*)outputs[0].buf + count);
    
    rknn_outputs_release(ctx, 1, outputs);
    return result;
}

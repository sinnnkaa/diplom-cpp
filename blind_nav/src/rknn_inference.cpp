#include "rknn_inference.h"
#include <iostream>
#include <fstream>

RKNNModel::RKNNModel() : ctx(0) {}

RKNNModel::~RKNNModel() {
    if (ctx) rknn_destroy(ctx);
}
bool RKNNModel::load(const std::string& model_path) {
    std::ifstream ifs(model_path, std::ios::binary | std::ios::ate);
    if (!ifs.is_open()) return false;
    std::streamsize size = ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    std::vector<char> buffer(size);
    ifs.read(buffer.data(), size);

    if (rknn_init(&ctx, buffer.data(), size, 0, NULL) < 0) return false;

    rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    
    rknn_tensor_attr q_out_attr;
    q_out_attr.index = 0;
    rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &q_out_attr, sizeof(q_out_attr));
    
    out_attr.scale = q_out_attr.scale;
    out_attr.zp = q_out_attr.zp;

    // ВАЖНАЯ ДИАГНОСТИКА:
    std::cout << "--- Параметры выхода NPU ---" << std::endl;
    std::cout << "Количество выходов: " << io_num.n_output << std::endl;
    std::cout << "Размеры (dims): ";
    for(int i=0; i < q_out_attr.n_dims; i++) std::cout << q_out_attr.dims[i] << " ";
    std::cout << "\n---------------------------" << std::endl;

    input_w = 512; // или вытяни из input_attr
    input_h = 512;
    return true;
}

std::vector<int8_t> RKNNModel::infer(const cv::Mat& img) {
    cv::Mat rgb;
    cv::cvtColor(img, rgb, cv::COLOR_BGR2RGB);

    rknn_input inputs[1];
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].size = rgb.total() * rgb.elemSize();
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].buf = rgb.data;

    rknn_inputs_set(ctx, io_num.n_input, inputs);
    rknn_run(ctx, NULL);

    rknn_output outputs[1];
    memset(outputs, 0, sizeof(outputs));
    outputs[0].want_float = 0; // Оставляем int8

    rknn_outputs_get(ctx, io_num.n_output, outputs, NULL);
    
    std::vector<int8_t> res((int8_t*)outputs[0].buf, (int8_t*)outputs[0].buf + outputs[0].size);
    rknn_outputs_release(ctx, 1, outputs);
    
    return res;
}

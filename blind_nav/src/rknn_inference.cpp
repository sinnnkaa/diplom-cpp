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

std::vector<float> RKNNModel::infer(const cv::Mat& img) {
    // 1. Подготовка входа (RGB)
    cv::Mat rgb;
    cv::cvtColor(img, rgb, cv::COLOR_BGR2RGB);

    rknn_input inputs[1];
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].fmt = RKNN_TENSOR_NHWC; // YOLO ждет NHWC (HWC в C++)
    inputs[0].size = 512 * 512 * 3;
    inputs[0].buf = rgb.data;

    rknn_inputs_set(ctx, 1, inputs);
    rknn_run(ctx, nullptr);

    // 2. Получение выхода (ВАЖНОЕ ИЗМЕНЕНИЕ)
    rknn_output outputs[1];
    memset(outputs, 0, sizeof(outputs));
    outputs[0].want_float = 1; // Просим драйвер сделать деквантование в float
    outputs[0].is_prealloc = 0; // Драйвер сам выделит память

    int ret = rknn_outputs_get(ctx, 1, outputs, nullptr);
    if (ret < 0) {
        std::cout << "rknn_outputs_get error ret=" << ret << std::endl;
        return {};
    }

    // Копируем данные в наш вектор
    // Размер тензора: 14 * 5376
    int out_size = 14 * 5376;
    std::vector<float> result((float*)outputs[0].buf, (float*)outputs[0].buf + out_size);

    // ОСВОБОЖДАЕМ БУФЕР ДРАЙВЕРА (Обязательно!)
    rknn_outputs_release(ctx, 1, outputs);

    return result;
}

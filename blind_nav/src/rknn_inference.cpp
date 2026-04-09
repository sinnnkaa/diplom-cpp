#include "rknn_inference.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>

RKNNModel::RKNNModel() : ctx(0) {}

RKNNModel::~RKNNModel() {
    if (ctx > 0) rknn_destroy(ctx);
}

// Вспомогательная функция для чтения файла модели
static unsigned char* load_model_file(const char* filename, int* model_size) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file.is_open()) return nullptr;
    *model_size = file.tellg();
    file.seekg(0, std::ios::beg);
    unsigned char* data = (unsigned char*)malloc(*model_size);
    file.read((char*)data, *model_size);
    file.close();
    return data;
}

bool RKNNModel::load(const std::string& model_path) {
    int model_len = 0;
    unsigned char* model_data = load_model_file(model_path.c_str(), &model_len);
    if (!model_data) {
        std::cerr << "Failed to load model file: " << model_path << std::endl;
        return false;
    }

    int ret = rknn_init(&ctx, model_data, model_len, 0, NULL);
    free(model_data);

    if (ret < 0) {
        std::cerr << "rknn_init error ret=" << ret << std::endl;
        return false;
    }

    // Запрос версии SDK для лога
    rknn_sdk_version version;
    rknn_query(ctx, RKNN_QUERY_SDK_VERSION, &version, sizeof(rknn_sdk_version));
    std::cout << "RKNN SDK Version: " << version.api_version << std::endl;

    return true;
}

std::vector<float> RKNNModel::infer(const cv::Mat& img) {
    cv::Mat resized, rgb;
    cv::resize(img, resized, cv::Size(512, 512));
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);

    rknn_input inputs[1];
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].size = 512 * 512 * 3;
    inputs[0].buf = rgb.data;

    rknn_inputs_set(ctx, 1, inputs);
    rknn_run(ctx, nullptr);

    // ПОЛУЧАЕМ ПАРАМЕТРЫ КВАНТОВАНИЯ
    rknn_tensor_attr out_attr;
    out_attr.index = 0;
    rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &out_attr, sizeof(out_attr));
    float scale = out_attr.scale;
    int32_t zp = out_attr.zp;

    rknn_output outputs[1];
    memset(outputs, 0, sizeof(outputs));
    outputs[0].want_float = 1; // ЗАБИРАЕМ СЫРЫЕ БАЙТЫ

    rknn_outputs_get(ctx, 1, outputs, nullptr);

    // Вручную переводим int8 -> float
    int8_t* raw_buf = (int8_t*)outputs[0].buf;
    std::vector<float> dequantized(14 * 5376);
    for (int i = 0; i < 14 * 5376; i++) {
        dequantized[i] = ((float)raw_buf[i] - (float)zp) * scale;
    }

    rknn_outputs_release(ctx, 1, outputs);
    return dequantized;
}

rknn_context RKNNModel::get_ctx() {
    return ctx;
}

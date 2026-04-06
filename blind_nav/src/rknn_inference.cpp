#include "rknn_inference.h"
#include <fstream>
#include <iostream>

bool RKNNModel::load(const char* model_path) {
    std::ifstream file(model_path, std::ios::binary);
    std::vector<char> model((std::istreambuf_iterator<char>(file)),
                             std::istreambuf_iterator<char>());

    if (rknn_init(&ctx, model.data(), model.size(), 0, NULL) != 0) {
        std::cout << "RKNN init failed\n";
        return false;
    }

    rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));

    output_attr.index = 0;
    rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &output_attr, sizeof(output_attr));

    return true;
}

std::vector<int8_t> RKNNModel::infer(cv::Mat& img) {
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(512, 512));
    cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);

    std::vector<int8_t> input_data(512 * 512 * 3);
    for (int i = 0; i < 512 * 512 * 3; i++) {
        input_data[i] = (int8_t)(resized.data[i] - 128);
    }

    rknn_input input;
    memset(&input, 0, sizeof(input));

    input.index = 0;
    input.type = RKNN_TENSOR_INT8;
    input.size = input_data.size();
    input.fmt = RKNN_TENSOR_NHWC;
    input.buf = input_data.data();

    rknn_inputs_set(ctx, 1, &input);
    rknn_run(ctx, NULL);

    rknn_output output;
    memset(&output, 0, sizeof(output));
    output.want_float = 0;

    rknn_outputs_get(ctx, 1, &output, NULL);

    int8_t* data = (int8_t*)output.buf;
    std::vector<int8_t> result(data, data + 14 * 5376);

    rknn_outputs_release(ctx, 1, &output);

    return result;
}
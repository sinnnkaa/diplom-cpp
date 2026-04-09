#include "decode.h"
#include <cmath>
#include <algorithm>
#include <vector>
#include <iostream>

inline float fast_sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

float calculate_iou(const Detection& a, const Detection& b) {
    float x1 = std::max(a.x, b.x); float y1 = std::max(a.y, b.y);
    float x2 = std::min(a.x + a.w, b.x + b.w); float y2 = std::min(a.y + a.h, b.y + b.h);
    float w = std::max(0.0f, x2 - x1); float h = std::max(0.0f, y2 - y1);
    return (w * h) / (a.w * a.h + b.w * b.h - w * h + 1e-6f);
}

void apply_nms(std::vector<Detection>& input, float threshold) {
    std::sort(input.begin(), input.end(), [](const Detection& a, const Detection& b) { return a.score > b.score; });
    std::vector<bool> removed(input.size(), false);
    std::vector<Detection> result;
    for (size_t i = 0; i < input.size(); i++) {
        if (removed[i]) continue;
        result.push_back(input[i]);
        for (size_t j = i + 1; j < input.size(); j++) {
            if (input[i].class_id == input[j].class_id && calculate_iou(input[i], input[j]) > threshold) removed[j] = true;
        }
    }
    input = result;
}

std::vector<Detection> decode(float* output, int input_w, int input_h,
                              int orig_w, int orig_h, float threshold) {
    std::vector<Detection> all_dets;
    const int num_anchors = 5376;
    const int num_channels = 14; 

    // Расчет коэффициентов для правильного масштаба
    float gain = std::min((float)input_w / orig_w, (float)input_h / orig_h);
    float pad_x = (input_w - orig_w * gain) / 2.0f;
    float pad_y = (input_h - orig_h * gain) / 2.0f;

    for (int i = 0; i < num_anchors; i++) {
        float* ptr = output + (i * num_channels);

        // 1. Классы (каналы 4-13)
        float max_logit = -100.0f;
        int cls_id = -1;
        for (int c = 0; c < 10; c++) {
            if (ptr[c + 4] > max_logit) { max_logit = ptr[c + 4]; cls_id = c; }
        }
        float score = fast_sigmoid(max_logit);

        if (score > threshold) {
            // 2. Координаты (0-3)
            float cx = ptr[0]; float cy = ptr[1];
            float w  = ptr[2]; float h  = ptr[3];

            // ЕСЛИ КООРДИНАТЫ ОГРОМНЫЕ (как в твоем логе 514), значит это ПИКСЕЛИ.
            // Приводим их к диапазону 0..1 для универсальности:
            if (cx > 1.1f) { cx /= 512.0f; cy /= 512.0f; w /= 512.0f; h /= 512.0f; }

            // 3. ПРАВИЛЬНЫЙ пересчет в пиксели оригинала
            float real_x = (cx * input_w - pad_x) / gain;
            float real_y = (cy * input_h - pad_y) / gain;
            float real_w = (w * input_w) / gain;
            float real_h = (h * input_h) / gain;

            // Центр -> Левый верхний угол
            float x1 = real_x - real_w / 2.0f;
            float y1 = real_y - real_h / 2.0f;

            if (real_w > 10 && real_h > 10 && x1 < orig_w && y1 < orig_h) {
                all_dets.push_back({cls_id, score, x1, y1, real_w, real_h});
            }
        }
    }

    // Жесткий NMS (0.3), чтобы убрать горы рамок
    apply_nms(all_dets, 0.30f);
    return all_dets;
}

#include "decode.h"
#include <cmath>
#include <algorithm>
#include <vector>
#include <iostream>

static float fast_sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

float calculate_iou(const Detection& a, const Detection& b) {
    float x1 = std::max(a.x, b.x); float y1 = std::max(a.y, b.y);
    float x2 = std::min(a.x + a.w, b.x + b.w); float y2 = std::min(a.y + a.h, b.y + b.h);
    float w = std::max(0.0f, x2 - x1); float h = std::max(0.0f, y2 - y1);
    float inter = w * h;
    return inter / (a.w * a.h + b.w * b.h - inter + 1e-6f);
}

void apply_nms(std::vector<Detection>& input, float threshold) {
    std::sort(input.begin(), input.end(), [](const Detection& a, const Detection& b) {
        return a.score > b.score;
    });
    std::vector<bool> removed(input.size(), false);
    std::vector<Detection> result;
    for (size_t i = 0; i < input.size(); i++) {
        if (removed[i]) continue;
        result.push_back(input[i]);
        for (size_t j = i + 1; j < input.size(); j++) {
            if (input[i].class_id == input[j].class_id) {
                if (calculate_iou(input[i], input[j]) > threshold) removed[j] = true;
            }
        }
    }
    input = result;
}

std::vector<Detection> decode(float* output, int input_w, int input_h,
                              int orig_w, int orig_h, float threshold) {
    std::vector<Detection> all_dets;
    const int num_classes = 10;
    const int num_anchors = 5376;

    // Параметры Letterbox
    float scale_l = std::min((float)input_w / orig_w, (float)input_h / orig_h);
    float off_x = (input_w - orig_w * scale_l) / 2.0f;
    float off_y = (input_h - orig_h * scale_l) / 2.0f;

    for (int i = 0; i < num_anchors; i++) {
        float max_prob = 0.0f;
        int cls_id = -1;

        // 1. Ищем лучший класс в каналах 0-9 (ПЛАНАРНО)
        for (int c = 0; c < num_classes; c++) {
            float prob = output[c * num_anchors + i]; 
            if (prob > max_prob) {
                max_prob = prob;
                cls_id = c;
            }
        }

        // Если Sigmoid уже внутри, то prob будет от 0 до 1. 
        // Если нет - примени fast_sigmoid(max_prob)
        if (max_prob > threshold) {
            // 2. Декодируем координаты из каналов 10, 11, 12, 13
            float x = output[10 * num_anchors + i];
            float y = output[11 * num_anchors + i];
            float w = output[12 * num_anchors + i];
            float h = output[13 * num_anchors + i];

            // Восстановление из Letterbox (YOLO обычно выдает cx, cy, w, h)
            float real_w = w / scale_l;
            float real_h = h / scale_l;
            float real_x = (x - off_x) / scale_l - (real_w / 2.0f);
            float real_y = (y - off_y) / scale_l - (real_h / 2.0f);

            all_dets.push_back({cls_id, max_prob, real_x, real_y, real_w, real_h});
        }
    }

    apply_nms(all_dets, 0.45f);
    return all_dets;
}

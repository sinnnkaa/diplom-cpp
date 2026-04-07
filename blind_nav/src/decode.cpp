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
    const int num_anchors = 5376;

    float max_val_found = -100.0f;

    for (int i = 0; i < num_anchors; i++) {
        float max_logit = -100.0f;
        int cls_id = -1;

        // Ищем в каналах 4-13
        for (int c = 0; c < 10; c++) {
            float val = output[(c + 4) * num_anchors + i];
            if (val > max_logit) { max_logit = val; cls_id = c; }
        }

        if (max_logit > max_val_found) max_val_found = max_logit;

        // СНИЖАЕМ ПОРОГ до минимума, чтобы увидеть, ЧТО она видит
        float score = fast_sigmoid(max_logit);
        if (score > 0.2f) { // Порог 0.2 для теста
            float x1 = output[0 * num_anchors + i];
            float y1 = output[1 * num_anchors + i];
            float x2 = output[2 * num_anchors + i];
            float y2 = output[3 * num_anchors + i];

            // Формат YOLOv11: обычно cx, cy, w, h
            float w = x2; float h = y2;
            float x = x1 - w/2.0f; float y = y1 - h/2.0f;

            float scale_l = std::min((float)input_w / orig_w, (float)input_h / orig_h);
            all_dets.push_back({cls_id, score, (x-((input_w-orig_w*scale_l)/2))/scale_l, (y-((input_h-orig_h*scale_l)/2))/scale_l, w/scale_l, h/scale_l});
        }
    }
    std::cout << "Max Logit in Class Channels: " << max_val_found << std::endl;

    apply_nms(all_dets, 0.45f);
    return all_dets;
}

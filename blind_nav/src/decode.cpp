#include "decode.h"
#include <cmath>
#include <algorithm>
#include <vector>
#include <iostream>

static float fast_sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

// IoU и NMS оставь как были...
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
    const int num_classes = 10;
    const int num_anchors = 5376;

    float scale_l = std::min((float)input_w / orig_w, (float)input_h / orig_h);
    float off_x = (input_w - orig_w * scale_l) / 2.0f;
    float off_y = (input_h - orig_h * scale_l) / 2.0f;

    float max_prob_found = 0.0f;

    for (int i = 0; i < num_anchors; i++) {
        float best_score = 0.0f;
        int cls_id = -1;

        // Сканируем каналы классов (4-13)
        for (int c = 0; c < num_classes; c++) {
            float score = output[(0 + c) * num_anchors + i];
            if (score > best_score) {
                best_score = score;
                cls_id = c;
            }
        }

        if (best_score > max_prob_found) max_prob_found = best_score;

        // ВНИМАНИЕ: Снижаем порог до 0.15 для отладки
        if (best_score > 0.15f) {
            float x1 = output[0 * num_anchors + i];
            float y1 = output[1 * num_anchors + i];
            float x2 = output[2 * num_anchors + i];
            float y2 = output[3 * num_anchors + i];

            float rx1 = (x1 - off_x) / scale_l;
            float ry1 = (y1 - off_y) / scale_l;
            float rx2 = (x2 - off_x) / scale_l;
            float ry2 = (y2 - off_y) / scale_l;

            all_dets.push_back({cls_id, best_score, rx1, ry1, rx2 - rx1, ry2 - ry1});
        }
    }

    std::cout << "Debug: Max class probability in tensor: " << max_prob_found << std::endl;

    apply_nms(all_dets, 0.45f);
    return all_dets;
}

#include "decode.h"
#include <cmath>
#include <algorithm>
#include <vector>
#include <iostream>

static float fast_sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

float calculate_iou(const Detection& a, const Detection& b) {
    float x1 = std::max(a.x, b.x);
    float y1 = std::max(a.y, b.y);
    float x2 = std::min(a.x + a.w, b.x + b.w);
    float y2 = std::min(a.y + a.h, b.y + b.h);
    float w = std::max(0.0f, x2 - x1);
    float h = std::max(0.0f, y2 - y1);
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

std::vector<Detection> decode(int8_t* output, float scale, int zp,
                              int input_w, int input_h,
                              int orig_w, int orig_h, float threshold) {
    std::vector<Detection> all_dets;
    const int num_classes = 10;
    const int num_anchors = 5376; 
    
    float sw = (float)orig_w / input_w;
    float sh = (float)orig_h / input_h;

    for (int a_idx = 0; a_idx < num_anchors; a_idx++) {
        // 1. Ищем лучший класс. Они начинаются с 4-го канала (0,1,2,3 - это боксы)
        float max_score = -10.0f;
        int cls_id = -1;

        for (int c = 0; c < num_classes; c++) {
            int channel_idx = 4 + c; // 4 координаты + индекс класса
            float logit = (output[channel_idx * num_anchors + a_idx] - zp) * scale;
            if (logit > max_score) {
                max_score = logit;
                cls_id = c;
            }
        }

        float prob = fast_sigmoid(max_score);

        // Если нашли объект
        if (prob > threshold) {
            // 2. Декодируем бокс. В 14-канальной YOLO это обычно cx, cy, w, h
            float cx = (output[0 * num_anchors + a_idx] - zp) * scale;
            float cy = (output[1 * num_anchors + a_idx] - zp) * scale;
            float w  = (output[2 * num_anchors + a_idx] - zp) * scale;
            float h  = (output[3 * num_anchors + a_idx] - zp) * scale;

            // Пересчитываем в экранные координаты (x_min, y_min, width, height)
            float x = (cx - w / 2.0f) * sw;
            float y = (cy - h / 2.0f) * sh;
            float width = w * sw;
            float height = h * sh;

            all_dets.push_back({cls_id, prob, x, y, width, height});
        }
    }

    apply_nms(all_dets, 0.45f);
    return all_dets;
}

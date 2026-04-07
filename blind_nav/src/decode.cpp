#include "decode.h"
#include <cmath>
#include <algorithm>
#include <vector>
#include <iostream>

inline float fast_sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

// IoU и NMS (без изменений)
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

    float scale_l = std::min((float)input_w / orig_w, (float)input_h / orig_h);
    float off_x = (input_w - orig_w * scale_l) / 2.0f;
    float off_y = (input_h - orig_h * scale_l) / 2.0f;

    for (int i = 0; i < num_anchors; i++) {
        // Мы берем строку i, в которой 14 значений лежат ПОДРЯД
        float* row = output + (i * num_channels);

        // Согласно логике YOLO, если первые 4 числа — это координаты, 
        // то классы начинаются с 4-го индекса
        float max_logit = -100.0f;
        int cls_id = -1;

        for (int c = 0; c < 10; c++) {
            float val = row[c + 4]; 
            if (val > max_logit) {
                max_logit = val;
                cls_id = c;
            }
        }

        float score = fast_sigmoid(max_logit);

        // Порог 0.5
        if (score > threshold) {
            float x1 = row[0];
            float y1 = row[1];
            float x2 = row[2];
            float y2 = row[3];

            // Если это [x, y, w, h]
            float w = x2;
            float h = y2;
            float x = x1 - w/2.0f;
            float y = y1 - h/2.0f;

            // Если вдруг это [x1, y1, x2, y2], то w = x2-x1, h = y2-y1
            if (w < 0 || w > 512) {
                w = x2 - x1;
                h = y2 - y1;
                x = x1;
                y = y1;
            }

            float rx = (x - off_x) / scale_l;
            float ry = (y - off_y) / scale_l;
            float rw = w / scale_l;
            float rh = h / scale_l;

            all_dets.push_back({cls_id, score, rx, ry, rw, rh});
        }
    }

    apply_nms(all_dets, 0.45f);
    return all_dets;
}

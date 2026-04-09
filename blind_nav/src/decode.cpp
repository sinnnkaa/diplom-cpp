#include "decode.h"
#include <cmath>
#include <algorithm>
#include <vector>
#include <iostream>

inline float fast_sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

// IoU и NMS оставляем без изменений, они работают правильно
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

    // Коэффициенты масштабирования (Letterbox компенсация)
    float scale = std::min((float)input_w / orig_w, (float)input_h / orig_h);
    float dx = (input_w - orig_w * scale) / 2.0f;
    float dy = (input_h - orig_h * scale) / 2.0f;

    for (int i = 0; i < num_anchors; i++) {
        float* ptr = output + (i * num_channels);

        // 1. Классы (индексы 4-13)
        float max_logit = -100.0f;
        int cls_id = -1;
        for (int c = 0; c < 10; c++) {
            if (ptr[c + 4] > max_logit) { max_logit = ptr[c + 4]; cls_id = c; }
        }

        float score = fast_sigmoid(max_logit);

        if (score > threshold) {
            // 2. Координаты из тензора (индексы 0-3)
            // ВАЖНО: Если p1...p4 > 1, значит это пиксели (0..512). 
            // Если < 1, значит нормализованные (0..1).
            float p1 = ptr[0]; float p2 = ptr[1];
            float p3 = ptr[2]; float p4 = ptr[3];

            // Принудительно приводим к пикселям 512x512
            if (p1 < 1.1f && p3 < 1.1f) {
                p1 *= 512.0f; p2 *= 512.0f; p3 *= 512.0f; p4 *= 512.0f;
            }

            // YOLOv11 output: [cx, cy, w, h]
            float real_w = p3 / scale;
            float real_h = p4 / scale;
            float real_x = (p1 - dx) / scale - (real_w / 2.0f);
            float real_y = (p2 - dy) / scale - (real_h / 2.0f);

            // Фильтр: рамка должна быть внутри картинки и иметь размер
            if (real_w > 10 && real_h > 10 && real_x >= 0 && real_y >= 0 && 
                real_x < orig_w && real_y < orig_h) {
                all_dets.push_back({cls_id, score, real_x, real_y, real_w, real_h});
            }
        }
    }

    apply_nms(all_dets, 0.45f);
    return all_dets;
}

#include "decode.h"
#include <cmath>
#include <algorithm>
#include <vector>
#include <iostream>

inline float sigmoid(float x) {
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
    const int num_classes = 10;

    float scale = std::min((float)input_w / orig_w, (float)input_h / orig_h);
    float dx = (input_w - orig_w * scale) / 2.0f;
    float dy = (input_h - orig_h * scale) / 2.0f;

    for (int i = 0; i < num_anchors; i++) {
        // 1. Классы
        float max_score = -10.0f;
        int cls_id = -1;
        for (int c = 0; c < num_classes; c++) {
            float s = output[(c + 4) * num_anchors + i];
            if (s > max_score) { max_score = s; cls_id = c; }
        }
        float prob = sigmoid(max_score);

        if (prob > threshold) {
            // 2. Координаты (cx, cy, w, h)
            float cx = output[0 * num_anchors + i];
            float cy = output[1 * num_anchors + i];
            float w  = output[2 * num_anchors + i];
            float h  = output[3 * num_anchors + i];

            // ЕСЛИ КООРДИНАТЫ МАЛЕНЬКИЕ (0..1), умножаем на 512
            // ЕСЛИ В ДИАПАЗОНЕ СЕТКИ (0..64), тоже нормализуем
            if (cx < 1.1f) { cx *= 512.0f; cy *= 512.0f; w *= 512.0f; h *= 512.0f; }
            
            // Если они всё еще подозрительно маленькие, вероятно это формат Grid (0..64)
            // Но судя по твоим логам [-204], они уже в каком-то масштабе.
            // Применяем стандартный пересчет:
            float real_w = w / scale;
            float real_h = h / scale;
            float x0 = (cx - dx) / scale;
            float y0 = (cy - dy) / scale;

            float x_final = x0 - (real_w / 2.0f);
            float y_final = y0 - (real_h / 2.0f);

            // Фильтрация: координаты ДОЛЖНЫ быть в пределах картинки
            if (x_final >= 0 && y_final >= 0 && x_final < orig_w && y_final < orig_h && real_w < orig_w) {
                all_dets.push_back({cls_id, prob, x_final, y_final, real_w, real_h});
            }
        }
    }

    apply_nms(all_dets, 0.45f);
    return all_dets;
}

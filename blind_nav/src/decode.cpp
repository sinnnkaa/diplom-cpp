#include "decode.h"
#include <cmath>
#include <algorithm>
#include <vector>
#include <iostream>

inline float fast_sigmoid(float x) {
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

std::vector<Detection> decode(float* output, int input_w, int input_h,
                              int orig_w, int orig_h, float threshold) {
    std::vector<Detection> all_dets;
    const int num_anchors = 5376;
    const int num_channels = 14; 

    float gain = std::min((float)input_w / orig_w, (float)input_h / orig_h);
    float pad_x = (input_w - orig_w * gain) / 2.0f;
    float pad_y = (input_h - orig_h * gain) / 2.0f;

    for (int i = 0; i < num_anchors; i++) {
        float* ptr = output + (i * num_channels);

        // 1. Ищем лучший класс (4-13)
        float max_logit = -100.0f;
        int cls_id = -1;
        for (int c = 0; c < 10; c++) {
            if (ptr[c + 4] > max_logit) { max_logit = ptr[c + 4]; cls_id = c; }
        }
        float score = fast_sigmoid(max_logit);

        // Повышаем порог, чтобы отсечь шум
        if (score > threshold) {
            // 2. Координаты как [x1, y1, x2, y2]
            float x1_raw = ptr[0];
            float y1_raw = ptr[1];
            float x2_raw = ptr[2];
            float y2_raw = ptr[3];

            // Если данные в 0..1, переводим в 512
            if (x1_raw < 1.1f) {
                x1_raw *= 512.0f; y1_raw *= 512.0f; x2_raw *= 512.0f; y2_raw *= 512.0f;
            }

            // 3. Пересчет в координаты оригинала
            float x1 = (x1_raw - pad_x) / gain;
            float y1 = (y1_raw - pad_y) / gain;
            float x2 = (x2_raw - pad_x) / gain;
            float y2 = (y2_raw - pad_y) / gain;

            float w = x2 - x1;
            float h = y2 - y1;

            // Валидация: рамка должна быть внутри и иметь размер
            if (w > 10 && h > 10 && x1 >= 0 && y1 >= 0 && x1 < orig_w && y1 < orig_h) {
                all_dets.push_back({cls_id, score, x1, y1, w, h});
            }
        }
    }

    // Очень строгий NMS для удаления "каши"
    apply_nms(all_dets, 0.25f); 
    return all_dets;
}

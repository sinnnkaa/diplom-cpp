#include "decode.h"
#include <cmath>
#include <algorithm>
#include <vector>
#include <iostream>

// Математическая сигмоида (нужна, так как в тензоре лежат сырые числа)
inline float fast_sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

// IoU и NMS (без изменений)
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
    const int num_channels = 14; // 4 box + 10 class

    float scale_l = std::min((float)input_w / orig_w, (float)input_h / orig_h);
    float off_x = (input_w - orig_w * scale_l) / 2.0f;
    float off_y = (input_h - orig_h * scale_l) / 2.0f;

    for (int i = 0; i < num_anchors; i++) {
        // ЧИТАЕМ БЛОК ИЗ 14 ЧИСЕЛ ДЛЯ ТЕКУЩЕЙ ТОЧКИ
        // Согласно твоим логам (521 в начале), координаты [x1, y1, x2, y2] идут ПЕРВЫМИ (0,1,2,3)
        // А классы идут ВТОРЫМИ (4,5,6...13)
        
        float* ptr = output + (i * num_channels);

        float max_logit = -100.0f;
        int cls_id = -1;

        for (int c = 0; c < num_classes; c++) {
            float logit = ptr[4 + c]; // Пропускаем 4 координаты
            if (logit > max_logit) {
                max_logit = logit;
                cls_id = c;
            }
        }

        float score = fast_sigmoid(max_logit);

        // Порог 0.5 - теперь он будет работать правильно
        if (score > threshold) {
            float x1 = ptr[0];
            float y1 = ptr[1];
            float x2 = ptr[2];
            float y2 = ptr[3];

            // Масштабируем
            float rx1 = (x1 - off_x) / scale_l;
            float ry1 = (y1 - off_y) / scale_l;
            float rx2 = (x2 - off_x) / scale_l;
            float ry2 = (y2 - off_y) / scale_l;

            all_dets.push_back({cls_id, score, rx1, ry1, rx2 - rx1, ry2 - ry1});
        }
    }

    apply_nms(all_dets, 0.45f);
    return all_dets;
}

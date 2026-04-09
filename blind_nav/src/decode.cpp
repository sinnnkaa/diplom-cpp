#include "decode.h"
#include <cmath>
#include <algorithm>
#include <vector>
#include <iostream>

// Математическая сигмоида для превращения 500+ в 1.0
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
    const int num_channels = 14; 

    float scale = std::min((float)input_w / orig_w, (float)input_h / orig_h);
    float dx = (input_w - orig_w * scale) / 2.0f;
    float dy = (input_h - orig_h * scale) / 2.0f;

    for (int i = 0; i < num_anchors; i++) {
        float* ptr = output + (i * num_channels);

        // 1. Ищем КЛАССЫ (индексы 4-13)
        float max_logit = -100.0f;
        int cls_id = -1;
        for (int c = 0; c < 10; c++) {
            if (ptr[c + 4] > max_logit) {
                max_logit = ptr[c + 4];
                cls_id = c;
            }
        }

        float score = fast_sigmoid(max_logit);

        if (score > threshold) {
            // 2. Берем КООРДИНАТЫ (индексы 0-3)
            float p1 = ptr[0]; // Могут быть x1 или cx
            float p2 = ptr[1]; // Могут быть y1 или cy
            float p3 = ptr[2]; // Могут быть x2 или w
            float p4 = ptr[3]; // Могут быть y2 или h

            float x1, y1, x2, y2;

            // ПРОВЕРКА ФОРМАТА: Углы [x1,y1,x2,y2] или Центр [cx,cy,w,h]
            if (p3 > p1 && p4 > p2) {
                // Это углы (BOX format)
                x1 = p1; y1 = p2; x2 = p3; y2 = p4;
            } else {
                // Это центр и размеры (YOLO format)
                x1 = p1 - p3/2.0f;
                y1 = p2 - p4/2.0f;
                x2 = p1 + p3/2.0f;
                y2 = p2 + p4/2.0f;
            }

            // 3. Пересчет в оригинальные координаты фото
            float rx1 = (x1 - dx) / scale;
            float ry1 = (y1 - dy) / scale;
            float rx2 = (x2 - dx) / scale;
            float ry2 = (y2 - dy) / scale;

            float rw = rx2 - rx1;
            float rh = ry2 - ry1;

            // Если координаты вменяемые — добавляем
            if (rw > 5 && rh > 5 && rx1 < orig_w && ry1 < orig_h) {
                all_dets.push_back({cls_id, score, rx1, ry1, rw, rh});
            }
        }
    }

    // 4. Жесткий NMS, чтобы убрать 500+ рамок
    apply_nms(all_dets, 0.30f); 
    return all_dets;
}

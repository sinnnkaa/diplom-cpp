#include "decode.h"
#include <cmath>
#include <algorithm>
#include <vector>
#include <iostream>

// IoU для очистки дубликатов
float calculate_iou(const Detection& a, const Detection& b) {
    float x1 = std::max(a.x, b.x);
    float y1 = std::max(a.y, b.y);
    float x2 = std::min(a.x + a.w, b.x + b.w);
    float y2 = std::min(a.y + a.h, b.y + b.h);
    float w = std::max(0.0f, x2 - x1);
    float h = std::max(0.0f, y2 - y1);
    return (w * h) / (a.w * a.h + b.w * b.h - w * h + 1e-6f);
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
            if (calculate_iou(input[i], input[j]) > threshold) removed[j] = true;
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
    
    // ПРОВЕРКА: Если у тебя YOLO11 с DFL, каналов 74 (64+10). Если без - 14.
    // Мы попробуем прочитать классы с конца тензора - это самое надежное.
    
    float sw = (float)orig_w / input_w;
    float sh = (float)orig_h / input_h;

    int strides[] = {8, 16, 32};
    int offset = 0;

    for (int stride : strides) {
        int gw = input_w / stride;
        int gh = input_h / stride;
        
        for (int i = 0; i < gw * gh; i++) {
            int a_idx = offset + i;

            // Читаем вероятность объекта (в YOLO11 это макс. из классов)
            float max_score = 0;
            int cls_id = -1;

            for (int c = 0; c < num_classes; c++) {
                // Пытаемся планарное чтение: классы лежат после координат
                // Предполагаем, что каналов 74. Классы - это каналы с 64 по 73.
                int channel = 64 + c; 
                float s = (output[channel * num_anchors + a_idx] - zp) * scale;
                float prob = 1.0f / (1.0f + std::exp(-s)); // Sigmoid

                if (prob > max_score) {
                    max_score = prob;
                    cls_id = c;
                }
            }

            if (max_score > threshold) {
                // Декодируем бокс (упрощенно)
                float d0 = (output[0 * num_anchors + a_idx] - zp) * scale;
                float d1 = (output[1 * num_anchors + a_idx] - zp) * scale;
                float d2 = (output[2 * num_anchors + a_idx] - zp) * scale;
                float d3 = (output[3 * num_anchors + a_idx] - zp) * scale;

                float cx = ((i % gw) + 0.5f) * stride;
                float cy = ((i / gw) + 0.5f) * stride;

                float x1 = (cx - d0 * stride) * sw;
                float y1 = (cy - d1 * stride) * sh;
                float x2 = (cx + d2 * stride) * sw;
                float y2 = (cy + d3 * stride) * sh;

                all_dets.push_back({cls_id, max_score, x1, y1, x2 - x1, y2 - y1});
            }
        }
        offset += gw * gh;
    }

    apply_nms(all_dets, 0.45f);
    return all_dets;
}

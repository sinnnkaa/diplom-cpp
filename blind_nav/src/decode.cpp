
#include "decode.h"
#include <cmath>
#include <algorithm>
#include <vector>
#include <iostream>

static float fast_sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

// Функции calculate_iou и apply_nms оставь как были раньше

float calculate_iou(const Detection& a, const Detection& b) {
    float x1 = std::max(a.x, b.x);
    float y1 = std::max(a.y, b.y);
    float x2 = std::min(a.x + a.w, b.x + b.w);
    float y2 = std::min(a.y + a.h, b.y + b.h);
    float w = std::max(0.0f, x2 - x1);
    float h = std::max(0.0f, y2 - y1);
    float inter = w * h;
    return inter / (a.w * a.h + b.w * b.h - inter);
}

// Реализация NMS (Non-Maximum Suppression)
void apply_nms(std::vector<Detection>& input, float iou_threshold) {
    std::sort(input.begin(), input.end(), [](const Detection& a, const Detection& b) {
        return a.score > b.score;
    });
    std::vector<bool> is_suppressed(input.size(), false);
    std::vector<Detection> result;
    for (size_t i = 0; i < input.size(); ++i) {
        if (is_suppressed[i]) continue;
        result.push_back(input[i]);
        for (size_t j = i + 1; j < input.size(); ++j) {
            if (!is_suppressed[j] && input[i].class_id == input[j].class_id) {
                if (calculate_iou(input[i], input[j]) > iou_threshold) {
                    is_suppressed[j] = true;
                }
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
    const int channels = 64 + num_classes; // Всего 74 канала для YOLO11

    float scale_w = (float)orig_w / input_w;
    float scale_h = (float)orig_h / input_h;

    int strides[] = {8, 16, 32};
    int current_offset = 0;

    for (int stride : strides) {
        int grid_w = input_w / stride;
        int grid_h = input_h / stride;
        int grid_size = grid_w * grid_h;

        for (int g = 0; g < grid_size; g++) {
            int a_idx = current_offset + g;

            // 1. Ищем лучший класс (они начинаются ПОСЛЕ 64 каналов боксов)
            float max_logit = -100.0f;
            int cls_id = -1;

            for (int c = 0; c < num_classes; c++) {
                // Сдвигаемся на 64 (DFL) и берем канал класса c
                int channel_idx = 64 + c;
                float logit = (output[channel_idx * num_anchors + a_idx] - zp) * scale;
                if (logit > max_logit) {
                    max_logit = logit;
                    cls_id = c;
                }
            }

            float score = fast_sigmoid(max_logit);

            // Если нашли что-то похожее на объект
            if (score > threshold) {
                // 2. Декодируем координаты (упрощенно из DFL)
                // Берем среднее из 16 значений для каждой стороны
                float d[4] = {0, 0, 0, 0};
                for (int side = 0; side < 4; side++) {
                    float sum = 0;
                    for (int j = 0; j < 16; j++) {
                        sum += (output[(side * 16 + j) * num_anchors + a_idx] - zp) * scale;
                    }
                    d[side] = sum / 16.0f; // Упрощенное чтение DFL
                }

                int gy = g / grid_w;
                int gx = g % grid_w;

                // Восстанавливаем бокс
                float x1 = (gx + 0.5f - d[0]) * stride * scale_w;
                float y1 = (gy + 0.5f - d[1]) * stride * scale_h;
                float x2 = (gx + 0.5f + d[2]) * stride * scale_w;
                float y2 = (gy + 0.5f + d[3]) * stride * scale_h;

                all_dets.push_back({cls_id, score, x1, y1, x2 - x1, y2 - y1});
            }
        }
        current_offset += grid_size;
    }

    apply_nms(all_dets, 0.45f);
    return all_dets;
}

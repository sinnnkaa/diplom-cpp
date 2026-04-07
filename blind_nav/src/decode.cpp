
#include "decode.h"
#include <cmath>
#include <algorithm>
#include <vector>

static float fast_sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

// IoU и NMS (оставь те же, что были раньше)
// ... calculate_iou ...
// ... apply_nms ...

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
    
    float scale_w = (float)orig_w / input_w;
    float scale_h = (float)orig_h / input_h;

    // Сетки для YOLO11
    int strides[] = {8, 16, 32};
    int current_offset = 0;

    for (int stride : strides) {
        int grid_w = input_w / stride;
        int grid_h = input_h / stride;
        int grid_size = grid_w * grid_h;

        for (int g = 0; g < grid_size; g++) {
            int a_idx = current_offset + g;

            // 1. Читаем логиты классов (ПЛАНАРНЫЙ формат)
            float max_logit = -100.0f;
            int cls_id = -1;

            for (int c = 0; c < num_classes; c++) {
                // Канал смещается на grid_size * канал
                float logit = (output[(4 + c) * num_anchors + a_idx] - zp) * scale;
                if (logit > max_logit) {
                    max_logit = logit;
                    cls_id = c;
                }
            }

            float score = fast_sigmoid(max_logit);

            // Порог 0.6, чтобы отсечь мусор
            if (score > 0.6f) {
                // 2. Декодируем координаты
                float d0 = (output[0 * num_anchors + a_idx] - zp) * scale;
                float d1 = (output[1 * num_anchors + a_idx] - zp) * scale;
                float d2 = (output[2 * num_anchors + a_idx] - zp) * scale;
                float d3 = (output[3 * num_anchors + a_idx] - zp) * scale;

                int gy = g / grid_w;
                int gx = g % grid_w;

                // Центр ячейки
                float cx = (gx + 0.5f) * stride;
                float cy = (gy + 0.5f) * stride;

                // Восстановление бокса (x1, y1, x2, y2)
                float x1 = (cx - d0 * stride) * scale_w;
                float y1 = (cy - d1 * stride) * scale_h;
                float x2 = (cx + d2 * stride) * scale_w;
                float y2 = (cy + d3 * stride) * scale_h;

                all_dets.push_back({cls_id, score, x1, y1, x2 - x1, y2 - y1});
            }
        }
        current_offset += grid_size;
    }

    apply_nms(all_dets, 0.45f);
    return all_dets;
}

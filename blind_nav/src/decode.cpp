#include "decode.h"
#include <cmath>
#include <algorithm>
#include <vector>

// Вспомогательная функция для IoU (нужна для NMS)
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

static float fast_sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

std::vector<Detection> decode(int8_t* output, float scale, int zp,
                              int input_w, int input_h,
                              int orig_w, int orig_h, float threshold) {
    std::vector<Detection> all_dets;
    const int num_classes = 10;
    const int num_anchors = 5376; // Для 512x512 это (64*64 + 32*32 + 16*16)
    
    // Коэффициенты масштабирования координат обратно к оригиналу
    float scale_w = (float)orig_w / input_w;
    float scale_h = (float)orig_h / input_h;

    // Сетки (strides) YOLO11
    int strides[] = {8, 16, 32};
    int current_anchor_offset = 0;

    for (int stride : strides) {
        int grid_w = input_w / stride;
        int grid_h = input_h / stride;
        int grid_size = grid_w * grid_h;

        for (int g = 0; g < grid_size; g++) {
            int anchor_idx = current_anchor_offset + g;

            // 1. Ищем лучший класс
            float max_logit = -100.0f;
            int cls_id = -1;
            for (int c = 0; c < num_classes; c++) {
                // Данные в RKNN обычно лежат планарно: [C][H*W]
                float logit = (output[(4 + c) * num_anchors + anchor_idx] - zp) * scale;
                if (logit > max_logit) {
                    max_logit = logit;
                    cls_id = c;
                }
            }

            float score = fast_sigmoid(max_logit);
            if (score > threshold) {
                // 2. Декодируем Bounding Box (Box Transformation)
                // В YOLO11/v8 это расстояние до границ от центра ячейки
                float d0 = (output[0 * num_anchors + anchor_idx] - zp) * scale;
                float d1 = (output[1 * num_anchors + anchor_idx] - zp) * scale;
                float d2 = (output[2 * num_anchors + anchor_idx] - zp) * scale;
                float d3 = (output[3 * num_anchors + anchor_idx] - zp) * scale;

                // Вычисляем координаты ячейки (grid x, grid y)
                int gy = g / grid_w;
                int gx = g % grid_w;

                // Формула: координата = (центр_ячейки + смещение) * stride
                // Для упрощенного экспорта rknn без DFL:
                float x_center = (gx + 0.5f) * stride;
                float y_center = (gy + 0.5f) * stride;
                
                float x1 = (x_center - d0 * stride) * scale_w;
                float y1 = (y_center - d1 * stride) * scale_h;
                float x2 = (x_center + d2 * stride) * scale_w;
                float y2 = (y_center + d3 * stride) * scale_h;

                all_dets.push_back({cls_id, score, x1, y1, x2 - x1, y2 - y1});
            }
        }
        current_anchor_offset += grid_size;
    }

    apply_nms(all_dets, 0.45f);
    return all_dets;
}

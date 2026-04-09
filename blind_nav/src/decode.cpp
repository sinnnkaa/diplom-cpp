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
    const int num_classes = 10;

    float gain = std::min((float)input_w / orig_w, (float)input_h / orig_h);
    float pad_x = (input_w - orig_w * gain) / 2.0f;
    float pad_y = (input_h - orig_h * gain) / 2.0f;

    int mesh_grids[] = {64, 32, 16}; 
    int strides[] = {8, 16, 32};
    int anchor_offset = 0;

    for (int g = 0; g < 3; g++) {
        int grid_size = mesh_grids[g];
        int stride = strides[g];

        for (int i = 0; i < grid_size; i++) {
            for (int j = 0; j < grid_size; j++) {
                int idx = anchor_offset + i * grid_size + j;

                // КОРРЕКТНЫЙ ДОСТУП NCHW:
                // output[канал * 5376 + индекс_анкора]
                
                // 1. Ищем лучший класс (каналы 4-13)
                float max_score = -10.0f;
                int cls_id = -1;
                for (int c = 0; c < num_classes; c++) {
                    float s = output[(c + 4) * num_anchors + idx];
                    if (s > max_score) { max_score = s; cls_id = c; }
                }

                float final_prob = sigmoid(max_score);

                if (final_prob > threshold) {
                    // 2. Декодируем LTRB из каналов 0, 1, 2, 3
                    float l = output[0 * num_anchors + idx];
                    float t = output[1 * num_anchors + idx];
                    float r = output[2 * num_anchors + idx];
                    float b = output[3 * num_anchors + idx];

                    // Формула из postprocess.cc: привязка к сетке
                    float cx = (j + 0.5f) * stride;
                    float cy = (i + 0.5f) * stride;

                    float x1_raw = cx - l * stride;
                    float y1_raw = cy - t * stride;
                    float x2_raw = cx + r * stride;
                    float y2_raw = cy + b * stride;

                    // 3. Пересчет в пиксели оригинала
                    float x1 = (x1_raw - pad_x) / gain;
                    float y1 = (y1_raw - pad_y) / gain;
                    float x2 = (x2_raw - pad_x) / gain;
                    float y2 = (y2_raw - pad_y) / gain;

                    float rw = x2 - x1;
                    float rh = y2 - y1;

                    if (rw > 5 && rh > 5 && x1 < orig_w && y1 < orig_h && x1 >= 0) {
                        all_dets.push_back({cls_id, final_prob, x1, y1, rw, rh});
                    }
                }
            }
        }
        anchor_offset += grid_size * grid_size;
    }

    apply_nms(all_dets, 0.45f);
    return all_dets;
}

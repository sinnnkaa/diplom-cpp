#include "decode.h"
#include <cmath>
#include <algorithm>
#include <vector>
#include <iostream>

inline float fast_sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

// Стандартные IoU и NMS
float calculate_overlap(float xmin0, float ymin0, float xmax0, float ymax0, 
                        float xmin1, float ymin1, float xmax1, float ymax1) {
    float w = std::max(0.f, std::min(xmax0, xmax1) - std::max(xmin0, xmin1));
    float h = std::max(0.f, std::min(ymax0, ymax1) - std::max(ymin0, ymin1));
    float i = w * h;
    float u = (xmax0 - xmin0) * (ymax0 - ymin0) + (xmax1 - xmin1) * (ymax1 - ymin1) - i;
    return u <= 0.f ? 0.f : (i / u);
}

void apply_nms(std::vector<Detection>& input, float threshold) {
    std::sort(input.begin(), input.end(), [](const Detection& a, const Detection& b) {
        return a.score > b.score;
    });
    std::vector<bool> removed(input.size(), false);
    for (size_t i = 0; i < input.size(); i++) {
        if (removed[i]) continue;
        for (size_t j = i + 1; j < input.size(); j++) {
            if (removed[j] || input[i].class_id != input[j].class_id) continue;
            float iou = calculate_overlap(input[i].x, input[i].y, input[i].x + input[i].w, input[i].y + input[i].h,
                                          input[j].x, input[j].y, input[j].x + input[j].w, input[j].y + input[j].h);
            if (iou > threshold) removed[j] = true;
        }
    }
    std::vector<Detection> result;
    for (size_t i = 0; i < input.size(); i++) if (!removed[i]) result.push_back(input[i]);
    input = result;
}

std::vector<Detection> decode(float* output, int input_w, int input_h, int orig_w, int orig_h, float threshold) {
    std::vector<Detection> all_dets;
    const int num_classes = 10;
    const int num_channels = 14; 
    
    // Параметры сеток YOLOv11 (80x80, 40x40, 20x20 для входа 640, но у нас 512)
    // Для 512x512 это сетки: 64x64, 32x32, 16x16
    int grids[] = {64, 32, 16};
    int strides[] = {8, 16, 32};
    int anchor_offset = 0;

    float gain = std::min((float)input_w / orig_w, (float)input_h / orig_h);
    float pad_x = (input_w - orig_w * gain) / 2.0f;
    float pad_y = (input_h - orig_h * gain) / 2.0f;

    for (int g = 0; g < 3; g++) {
        int grid_size = grids[g];
        int stride = strides[g];
        
        for (int i = 0; i < grid_size; i++) {
            for (int j = 0; j < grid_size; j++) {
                int anchor_idx = anchor_offset + i * grid_size + j;
                float* ptr = output + (anchor_idx * num_channels);

                // 1. Ищем лучший класс (индексы 4-13)
                float max_logit = -100.0f;
                int cls_id = -1;
                for (int c = 0; c < num_classes; c++) {
                    if (ptr[c + 4] > max_logit) { max_logit = ptr[c + 4]; cls_id = c; }
                }
                float score = fast_sigmoid(max_logit);

                if (score > threshold) {
                    // 2. Декодируем Box по формуле Rockchip:
                    // x1 = (grid_j + 0.5 - box[0]) * stride
                    float x1_raw = (j + 0.5f - ptr[0]) * stride;
                    float y1_raw = (i + 0.5f - ptr[1]) * stride;
                    float x2_raw = (j + 0.5f + ptr[2]) * stride;
                    float y2_raw = (i + 0.5f + ptr[3]) * stride;

                    // 3. Пересчет в координаты оригинала с учетом Letterbox
                    float x1 = (x1_raw - pad_x) / gain;
                    float y1 = (y1_raw - pad_y) / gain;
                    float x2 = (x2_raw - pad_x) / gain;
                    float y2 = (y2_raw - pad_y) / gain;

                    float w = x2 - x1;
                    float h = y2 - y1;

                    if (w > 5 && h > 5 && x1 >= 0 && y1 >= 0 && x1 < orig_w) {
                        all_dets.push_back({cls_id, score, x1, y1, w, h});
                    }
                }
            }
        }
        anchor_offset += grid_size * grid_size;
    }

    apply_nms(all_dets, 0.35f);
    return all_dets;
}

#include "decode.h"
#include <cmath>
#include <algorithm>
#include <vector>

// Быстрая сигмоида для вероятностей
inline float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

// Расчет пересечения для NMS
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
    const int num_classes = 10;
    const int num_channels = 14; // 4 бокса + 10 классов

    // Коэффициенты Letterbox
    float scale = std::min((float)input_w / orig_w, (float)input_h / orig_h);
    float dx = (input_w - orig_w * scale) / 2.0f;
    float dy = (input_h - orig_h * scale) / 2.0f;

    // Сетки YOLOv11 для входа 512x512
    int mesh_grids[] = {64, 32, 16}; 
    int strides[] = {8, 16, 32};
    int anchor_offset = 0;

    for (int g = 0; g < 3; g++) {
        int grid_size = mesh_grids[g];
        int stride = strides[g];

        for (int i = 0; i < grid_size; i++) {
            for (int j = 0; j < grid_size; j++) {
                int index = anchor_offset + i * grid_size + j;
                float* ptr = output + (index * num_channels);

                // 1. Ищем лучший класс
                float max_score = 0;
                int cls_id = -1;
                for (int c = 0; c < num_classes; c++) {
                    float s = sigmoid(ptr[c + 4]);
                    if (s > max_score) { max_score = s; cls_id = c; }
                }

                if (max_score > threshold) {
                    // 2. Декодируем LTRB (Left, Top, Right, Bottom)
                    // Эти значения в тензоре — смещения от центра ячейки
                    float l = ptr[0]; float t = ptr[1];
                    float r = ptr[2]; float b = ptr[3];

                    // Центр текущей ячейки в пикселях 512x512
                    float anchor_x = (j + 0.5f) * stride;
                    float anchor_y = (i + 0.5f) * stride;

                    // Координаты углов в 512x512
                    float x1_raw = anchor_x - l * stride;
                    float y1_raw = anchor_y - t * stride;
                    float x2_raw = anchor_x + r * stride;
                    float y2_raw = anchor_y + b * stride;

                    // 3. Перенос на оригинальное фото
                    float x1 = (x1_raw - dx) / scale;
                    float y1 = (y1_raw - dy) / scale;
                    float x2 = (x2_raw - dx) / scale;
                    float y2 = (y2_raw - dy) / scale;

                    if (x2 > x1 && y2 > y1 && x1 < orig_w && y1 < orig_h) {
                        all_dets.push_back({cls_id, max_score, x1, y1, x2 - x1, y2 - y1});
                    }
                }
            }
        }
        anchor_offset += grid_size * grid_size;
    }

    apply_nms(all_dets, 0.45f);
    return all_dets;
}

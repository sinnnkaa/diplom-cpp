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
    return (w * h) / (a.w * a.h + b.w * b.h - w * h + 1e-6f);
}

void apply_nms(std::vector<Detection>& input, float threshold) {
    std::sort(input.begin(), input.end(), [](const Detection& a, const Detection& b) { return a.score > b.score; });
    std::vector<bool> removed(input.size(), false);
    std::vector<Detection> result;
    for (size_t i = 0; i < input.size(); i++) {
        if (removed[i]) continue;
        result.push_back(input[i]);
        for (size_t j = i + 1; j < input.size(); j++) {
            if (input[i].class_id == input[j].class_id && calculate_iou(input[i], input[j]) > threshold) removed[j] = true;
        }
    }
    input = result;
}

// Вычисление DFL (Distribution Focal Loss) для одной координаты
inline float compute_dfl(const float* tensor, int step) {
    float max_val = -10000.0f;
    for (int i = 0; i < 16; i++) {
        max_val = std::max(max_val, tensor[i * step]);
    }
    
    float sum = 0.0f;
    float exp_vals[16];
    for (int i = 0; i < 16; i++) {
        exp_vals[i] = std::exp(tensor[i * step] - max_val);
        sum += exp_vals[i];
    }
    
    float res = 0.0f;
    for (int i = 0; i < 16; i++) {
        res += (exp_vals[i] / sum) * i;
    }
    return res;
}

// Декодирование одного масштаба (P3, P4 или P5)
void decode_single_output(const float* output, int grid_size, int stride, 
                          float conf_thresh, std::vector<Detection>& proposals) {
    int num_classes = 10;
    int reg_max = 16;
    int area = grid_size * grid_size;
    
    // Формат памяти NCHW: сначала 64 канала DFL боксов, затем 10 каналов классов
    const float* box_ptr = output;
    const float* cls_ptr = output + (4 * reg_max) * area; 

    for (int i = 0; i < area; i++) {
        int grid_y = i / grid_size;
        int grid_x = i % grid_size;

        // Ищем максимальную вероятность среди 10 классов
        float max_conf = -1.0f;
        int class_id = -1;
        for (int c = 0; c < num_classes; c++) {
            float conf = sigmoid(cls_ptr[c * area + i]);
            if (conf > max_conf) {
                max_conf = conf;
                class_id = c;
            }
        }

        // Если уверенность выше порога, декодируем координаты
        if (max_conf > conf_thresh) {
            float dfl[4]; // [left, top, right, bottom]
            for (int k = 0; k < 4; k++) {
                const float* dfl_start = box_ptr + (k * 16) * area + i;
                dfl[k] = compute_dfl(dfl_start, area);
            }

            // Перевод координат в размер 512x512
            float xmin = (grid_x - dfl[0]) * stride;
            float ymin = (grid_y - dfl[1]) * stride;
            float xmax = (grid_x + dfl[2]) * stride;
            float ymax = (grid_y + dfl[3]) * stride;

            proposals.push_back({class_id, max_conf, xmin, ymin, xmax - xmin, ymax - ymin});
        }
    }
}

std::vector<Detection> decode(const std::vector<std::vector<float>>& outputs, 
                              int input_w, int input_h,
                              int orig_w, int orig_h, float threshold) {
    std::vector<Detection> all_dets;
    
    // Шаги сетки и её размеры для входа 512x512
    int strides[] = {8, 16, 32};
    int grid_sizes[] = {input_w / 8, input_w / 16, input_w / 32}; // 64, 32, 16

    // 1. Проходим по всем трем выходам
    for (int i = 0; i < 3; i++) {
        decode_single_output(outputs[i].data(), grid_sizes[i], strides[i], threshold, all_dets);
    }

    // 2. Возвращаем координаты в исходный размер картинки
    float scale = std::min((float)input_w / orig_w, (float)input_h / orig_h);
    float dx = (input_w - orig_w * scale) / 2.0f;
    float dy = (input_h - orig_h * scale) / 2.0f;

    std::vector<Detection> final_dets;
    for (auto& det : all_dets) {
        float real_w = det.w / scale;
        float real_h = det.h / scale;
        float real_x = (det.x - dx) / scale;
        float real_y = (det.y - dy) / scale;

        // Проверка выхода за границы кадра
        if (real_x >= 0 && real_y >= 0 && real_x < orig_w && real_y < orig_h && real_w < orig_w) {
            final_dets.push_back({det.class_id, det.score, real_x, real_y, real_w, real_h});
        }
    }

    // 3. Применяем NMS (удаление дубликатов)
    apply_nms(final_dets, 0.45f);
    
    return final_dets;
}

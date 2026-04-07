#include "decode.h"
#include <cmath>
#include <algorithm>
#include <vector>
#include <iostream>

static float fast_sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

float calculate_iou(const Detection& a, const Detection& b) {
    float x1 = std::max(a.x, b.x); float y1 = std::max(a.y, b.y);
    float x2 = std::min(a.x + a.w, b.x + b.w); float y2 = std::min(a.y + a.h, b.y + b.h);
    float w = std::max(0.0f, x2 - x1); float h = std::max(0.0f, y2 - y1);
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
            if (input[i].class_id == input[j].class_id && calculate_iou(input[i], input[j]) > threshold) 
                removed[j] = true;
        }
    }
    input = result;
}

std::vector<Detection> decode(float* output, int input_w, int input_h,
                              int orig_w, int orig_h, float threshold) {
    std::vector<Detection> all_dets;
    const int num_classes = 10;
    const int num_anchors = 5376;
    
    float scale_l = std::min((float)input_w / orig_w, (float)input_h / orig_h);
    float off_x = (input_w - orig_w * scale_l) / 2.0f;
    float off_y = (input_h - orig_h * scale_l) / 2.0f;

    // ПРОВЕРКА ПОРЯДКА (Planar vs Interleaved)
    // Посмотрим на первые 4 значения. Если они > 10, это скорее всего координаты (Planar)
    bool is_planar = (std::abs(output[0]) > 10.0f || std::abs(output[num_anchors]) > 10.0f);
    
    std::cout << "Debug: Interpretation Mode: " << (is_planar ? "PLANAR" : "INTERLEAVED") << std::endl;

    for (int i = 0; i < num_anchors; i++) {
        float max_logit = -100.0f;
        int cls_id = -1;
        float x, y, w, h;

        if (is_planar) {
            // PLANAR: [Channel 0...5375][Channel 1...5375]...
            // Судя по логам, координаты в 0,1,2,3, классы в 4...13
            for (int c = 0; c < num_classes; c++) {
                float val = output[(4 + c) * num_anchors + i];
                if (val > max_logit) { max_logit = val; cls_id = c; }
            }
            x = output[0 * num_anchors + i];
            y = output[1 * num_anchors + i];
            w = output[2 * num_anchors + i];
            h = output[3 * num_anchors + i];
        } else {
            // INTERLEAVED: [Anchor 0: 14 values][Anchor 1: 14 values]...
            float* ptr = output + (i * 14);
            for (int c = 0; c < num_classes; c++) {
                float val = ptr[4 + c];
                if (val > max_logit) { max_logit = val; cls_id = c; }
            }
            x = ptr[0]; y = ptr[1]; w = ptr[2]; h = ptr[3];
        }

        float score = fast_sigmoid(max_logit);

        // Порог ставим 0.5f, но если ничего не найдет, в логе увидим почему
        if (score > threshold) {
            // Масштабируем cx, cy, w, h -> x_min, y_min, w, h
            float real_w = w / scale_l;
            float real_h = h / scale_l;
            float real_x = (x - off_x) / scale_l - (real_w / 2.0f);
            float real_y = (y - off_y) / scale_l - (real_h / 2.0f);

            // Фильтр от "мусорных" огромных рамок
            if (real_w > 5 && real_h > 5 && real_w < orig_w && real_h < orig_h) {
                all_dets.push_back({cls_id, score, real_x, real_y, real_w, real_h});
            }
        }
    }

    apply_nms(all_dets, 0.45f);
    return all_dets;
}

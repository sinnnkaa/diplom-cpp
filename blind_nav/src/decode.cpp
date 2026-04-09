#include "decode.h"
#include <cmath>
#include <algorithm>
#include <vector>
#include <iostream>

// Оставляем IoU и NMS как были (они понадобятся, когда пойдут детекции)
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

std::vector<Detection> decode(float* output, int input_w, int input_h,
                              int orig_w, int orig_h, float threshold) {
    std::vector<Detection> all_dets;
    const int num_anchors = 5376;
    const int num_channels = 14;

    // --- СУПЕР-ДИАГНОСТИКА КАНАЛОВ ---
    float channel_max[14];
    for(int j=0; j<14; j++) channel_max[j] = -1000.0f;

    // Мы проверяем версию INTERLEAVED (подряд по 14 чисел)
    for (int i = 0; i < num_anchors; i++) {
        for (int c = 0; c < num_channels; c++) {
            float val = output[i * num_channels + c];
            if (val > channel_max[c]) channel_max[c] = val;
        }
    }

    std::cout << "--- CHANNEL ANALYSIS (Interleaved Mode) ---" << std::endl;
    for(int j=0; j<14; j++) {
        std::cout << "CH[" << j << "] Max: " << channel_max[j] << " | ";
        if (j == 6) std::cout << "\n";
    }
    std::cout << "\n-------------------------------------------" << std::endl;

    // Теперь пробуем декодировать, предполагая, что классы там, где значения < 1.1
    // А координаты там, где значения большие. 
    // По логу ONNX мы знаем: Классы (10), Боксы (4).
    
    for (int i = 0; i < num_anchors; i++) {
        float* ptr = output + (i * num_channels);
        
        // Согласно твоему логу конвертации Concat_3: 4 бокса + 10 классов
        // Попробуем прочитать классы из ПЕРВЫХ 10 (индексы 0-9), а боксы из 10-13
        float max_score = 0.0f;
        int cls_id = -1;
        for (int c = 0; c < 10; c++) {
            if (ptr[c] > max_score) {
                max_score = ptr[c];
                cls_id = c;
            }
        }

        if (max_score > 0.3f) { // Снизили порог для теста
            float cx = ptr[10]; float cy = ptr[11];
            float w  = ptr[12]; float h  = ptr[13];
            
            // Если координаты нормализованы (0..1)
            if (cx < 1.1f) { cx *= 512; cy *= 512; w *= 512; h *= 512; }

            float scale = std::min((float)input_w / orig_w, (float)input_h / orig_h);
            float dx = (input_w - orig_w * scale) / 2.0f;
            float dy = (input_h - orig_h * scale) / 2.0f;

            all_dets.push_back({cls_id, max_score, (cx-dx)/scale - (w/(2*scale)), (cy-dy)/scale - (h/(2*scale)), w/scale, h/scale});
        }
    }

    apply_nms(all_dets, 0.45f);
    return all_dets;
}

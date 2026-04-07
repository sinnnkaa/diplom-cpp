#pragma once
#include <vector>
#include <cstdint>

struct Detection {
    int class_id;
    float score;
    float x, y, w, h;
};

// Декодирование выходного тензора модели YOLO
std::vector<Detection> decode(int8_t* output, float scale, int zp,
                              int input_w, int input_h,
                              int orig_w, int orig_h,
                              float score_threshold = 0.5f);

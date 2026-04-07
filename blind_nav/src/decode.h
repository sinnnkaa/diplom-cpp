#pragma once
#include <vector>
#include <cstdint>

struct Detection {
    int class_id;
    float score;
    float x, y, w, h;
};

// ИЗМЕНЕНО: убрали scale и zp, заменили int8* на float*
std::vector<Detection> decode(float* output, int input_w, int input_h,
                              int orig_w, int orig_h, float threshold);

#pragma once
#include <vector>
#include <cstdint>

struct Detection {
    int class_id;
    float score;
    float x, y, w, h;
};

// ИЗМЕНЕНО: Теперь принимает вектор из 3-х массивов float
std::vector<Detection> decode(const std::vector<std::vector<float>>& outputs, 
                              int input_w, int input_h,
                              int orig_w, int orig_h, float threshold);

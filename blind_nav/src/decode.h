#pragma once
#include <vector>

struct Detection {
    float x, y, w, h;
    float score;
    int class_id;
};

std::vector<Detection> decode(
    int8_t* output,
    float scale,
    int zp
);
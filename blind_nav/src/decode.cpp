#include "decode.h"
#include <cmath>
#include <algorithm>
#include <vector>
#include <iostream>

static float fast_sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

// IoU
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

// NMS
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
    const int num_anchors = 5376;

    float scale_letterbox = std::min((float)input_w / orig_w, (float)input_h / orig_h);
    float offset_x = (input_w - orig_w * scale_letterbox) / 2.0f;
    float offset_y = (input_h - orig_h * scale_letterbox) / 2.0f;

    for (int i = 0; i < num_anchors; i++) {
        float max_logit = -100.0f;
        int cls_id = -1;

        for (int c = 0; c < num_classes; c++) {
            float logit = output[(4 + c) * num_anchors + i];
            if (logit > max_logit) {
                max_logit = logit;
                cls_id = c;
            }
        }

        float score = fast_sigmoid(max_logit);

        if (score > threshold) {
            float cx = output[0 * num_anchors + i];
            float cy = output[1 * num_anchors + i];
            float w  = output[2 * num_anchors + i];
            float h  = output[3 * num_anchors + i];

            float real_cx = (cx - offset_x) / scale_letterbox;
            float real_cy = (cy - offset_y) / scale_letterbox;
            float real_w  = w / scale_letterbox;
            float real_h  = h / scale_letterbox;

            all_dets.push_back({cls_id, score, real_cx - real_w/2.0f, real_cy - real_h/2.0f, real_w, real_h});
        }
    }

    apply_nms(all_dets, 0.45f);
    return all_dets;
}

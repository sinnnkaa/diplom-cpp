#include "decode.h"
#include <cmath>
#include <algorithm>

float sigmoid(float x) {
    return 1.f / (1.f + exp(-x));
}

float iou(const Detection& a, const Detection& b) {
    float x1 = std::max(a.x, b.x);
    float y1 = std::max(a.y, b.y);
    float x2 = std::min(a.x + a.w, b.x + b.w);
    float y2 = std::min(a.y + a.h, b.y + b.h);

    float inter = std::max(0.f, x2 - x1) * std::max(0.f, y2 - y1);
    float union_ = a.w * a.h + b.w * b.h - inter;

    return inter / union_;
}

std::vector<Detection> nms(std::vector<Detection>& dets) {
    std::sort(dets.begin(), dets.end(),
              [](auto& a, auto& b) { return a.score > b.score; });

    std::vector<Detection> result;
    std::vector<bool> removed(dets.size(), false);

    for (size_t i = 0; i < dets.size(); i++) {
        if (removed[i]) continue;
        result.push_back(dets[i]);

        for (size_t j = i + 1; j < dets.size(); j++) {
            if (iou(dets[i], dets[j]) > 0.5) {
                removed[j] = true;
            }
        }
    }
    return result;
}

std::vector<Detection> decode(
    int8_t* output,
    float scale,
    int zp
) {
    std::vector<Detection> dets;

    int num_preds = 5376;
    int num_classes = 10;

    for (int i = 0; i < num_preds; i++) {
        float data[14];

        for (int j = 0; j < 14; j++) {
            data[j] = (output[j * num_preds + i] - zp) * scale;
        }

        float x = data[0];
        float y = data[1];
        float w = data[2];
        float h = data[3];

        int best_cls = -1;
        float best_score = 0;

        for (int c = 0; c < num_classes; c++) {
            float score = sigmoid(data[4 + c]);
            if (score > best_score) {
                best_score = score;
                best_cls = c;
            }
        }

        if (best_score > 0.4) {
            Detection d;

            d.x = (x - w / 2) * 512;
            d.y = (y - h / 2) * 512;
            d.w = w * 512;
            d.h = h * 512;

            d.score = best_score;
            d.class_id = best_cls;

            dets.push_back(d);
        }
    }

    return nms(dets);
}
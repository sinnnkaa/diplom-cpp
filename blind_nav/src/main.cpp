#include "rknn_inference.h"
#include "decode.h"
#include "audio.h"
#include <opencv2/opencv.hpp>
#include <iostream>

std::string class_names[] = {
    "person", "car", "curb", "pole",
    "traffic_sign", "traffic_light",
    "trash_can", "bench", "sidewalk", "crosswalk"
};

int main() {
    RKNNModel model;

    if (!model.load("model/yolo11_blind.rknn")) {
        return -1;
    }

    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cout << "Camera error\n";
        return -1;
    }

    while (true) {
        cv::Mat frame;
        cap >> frame;

        if (frame.empty()) continue;

        auto output = model.infer(frame);

        float scale = model.output_attr.scale;
        int zp = model.output_attr.zp;

        auto detections = decode(output.data(), scale, zp);

        for (auto& d : detections) {
            std::string label = class_names[d.class_id];

            cv::rectangle(frame,
                cv::Rect(d.x, d.y, d.w, d.h),
                cv::Scalar(0,255,0), 2);

            cv::putText(frame, label,
                cv::Point(d.x, d.y - 5),
                cv::FONT_HERSHEY_SIMPLEX,
                0.5, cv::Scalar(0,255,0), 1);

            if (d.score > 0.6) {
                speak(label);
            }
        }

        cv::imshow("result", frame);
        if (cv::waitKey(1) == 27) break;
    }

    return 0;
}
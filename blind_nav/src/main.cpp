#include "rknn_inference.h"
#include "decode.h"
#include "audio.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>

std::string class_names[] = {
    "person", "car", "curb", "pole",
    "traffic_sign", "traffic_light",
    "trash_can", "bench", "sidewalk", "crosswalk"
};

int main() {
    RKNNModel model;

    // Загрузка модели (укажи полный путь к файлу .rknn)
    if (!model.load("/root/diplom-cpp/blind_nav/model/yolo11_blind.rknn")) {
        std::cerr << "Failed to load RKNN model\n";
        return -1;
    }

    // Подключение камеры
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Camera error\n";
        return -1;
    }

    int frame_count = 0;

    while (true) {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) continue;

        // Инференс
        auto output = model.infer(frame);

        float scale = model.output_attr.scale;
        int zp = model.output_attr.zp;

        auto detections = decode(output.data(), scale, zp);

        // Обработка детекций
        for (auto& d : detections) {
            std::string label = class_names[d.class_id];

            // Рисуем рамку
            cv::rectangle(frame,
                          cv::Rect(d.x, d.y, d.w, d.h),
                          cv::Scalar(0, 255, 0), 2);

            if (d.score > 0.6) {
                std::cout << "Detected: " << label
                          << " score: " << d.score
                          << " x:" << d.x << " y:" << d.y
                          << " w:" << d.w << " h:" << d.h << std::endl;
                speak(label); // озвучивание
            }
        }

        // Сохраняем кадр с рамками
        std::string filename = "frame_" + std::to_string(frame_count) + ".jpg";
        cv::imwrite(filename, frame);
        frame_count++;

        // На Orange Pi нет GUI, поэтому imshow отключен
        // if (cv::waitKey(1) == 27) break; // ESC для выхода, если нужно
    }

    return 0;
}

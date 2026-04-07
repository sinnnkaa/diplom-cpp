#include "rknn_inference.h"
#include "decode.h"
#include "audio.h"
#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    std::cout << "--- Запуск диплома ---" << std::endl;
    
    RKNNModel model;
    if (!model.load("/root/diplom-cpp/blind_nav/model/yolo11_blind.rknn")) {
        std::cerr << "Fatal: Model not found!" << std::endl;
        return -1;
    }

    cv::Mat frame = cv::imread("/root/diplom-cpp/blind_nav/-0GQmYRienNVqEKiQ0Mkyw.jpg");
    if (frame.empty()) {
        std::cerr << "Fatal: Image not found!" << std::endl;
        return -1;
    }

    // --- РЕАЛИЗАЦИЯ LETTERBOX ---
    int img_w = frame.cols;
    int img_h = frame.rows;
    float scale = std::min((float)model.input_w / img_w, (float)model.input_h / img_h);
    int new_w = img_w * scale;
    int new_h = img_h * scale;

    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(new_w, new_h));

    // Создаем черный квадрат 512x512
    cv::Mat input_img = cv::Mat::zeros(model.input_h, model.input_w, CV_8UC3);
    // Копируем картинку в центр
    resized.copyTo(input_img(cv::Rect((model.input_w - new_w) / 2, (model.input_h - new_h) / 2, new_w, new_h)));

    // Инференс
    auto raw_out = model.infer(input_img);

    // Декодирование (порог 0.5, так как ZP -127 очень резкий)
    auto results = decode(raw_out.data(), model.out_attr.scale, model.out_attr.zp,
                          model.input_w, model.input_h, img_w, img_h, 0.2f);

    std::cout << "Найдено: " << results.size() << " объектов." << std::endl;

    for (auto& d : results) {
        std::cout << "Объект [" << d.class_id << "] Уверенность: " << d.score << std::endl;
        cv::rectangle(frame, cv::Rect(d.x, d.y, d.w, d.h), cv::Scalar(0, 255, 0), 3);
    }

    cv::imwrite("result.jpg", frame);
    std::cout << "Файл result.jpg сохранен." << std::endl;

    return 0;
}

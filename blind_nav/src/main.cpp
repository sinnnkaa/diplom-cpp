#include "rknn_inference.h"
#include "decode.h"
#include "audio.h"
#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    std::cout << "--- Запуск программы ---" << std::endl;
    
    RKNNModel model;
    std::string model_path = "/root/diplom-cpp/blind_nav/model/yolo11_blind.rknn";
    
    std::cout << "Загрузка модели: " << model_path << "..." << std::endl;
    if (!model.load(model_path)) {
        std::cerr << "Ошибка: Не удалось загрузить RKNN модель!" << std::endl;
        return -1;
    }
    std::cout << "Модель успешно загружена." << std::endl;

    std::string img_path = "/root/diplom-cpp/blind_nav/-0GQmYRienNVqEKiQ0Mkyw.jpg";
    std::cout << "Чтение картинки: " << img_path << "..." << std::endl;
    cv::Mat frame = cv::imread(img_path);
    
    if (frame.empty()) {
        std::cerr << "Ошибка: Картинка не найдена по пути " << img_path << std::endl;
        return -1;
    }
    std::cout << "Картинка загружена: " << frame.cols << "x" << frame.rows << std::endl;

    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(model.input_w, model.input_h));
    std::cout << "Ресайз выполнен. Запуск инференса..." << std::endl;

    auto raw_out = model.infer(resized);
    
    if (raw_out.empty()) {
        std::cerr << "Ошибка: Инференс вернул пустой результат!" << std::endl;
        return -1;
    }
    std::cout << "Инференс завершен. Декодирование..." << std::endl;

    // Устанавливаем порог повыше (0.8), чтобы не было ложных срабатываний
    auto results = decode(raw_out.data(), model.out_attr.scale, model.out_attr.zp,
                          model.input_w, model.input_h, frame.cols, frame.rows, 0.25f);

    std::cout << "Найдено объектов: " << results.size() << std::endl;

    for (auto& d : results) {
        std::cout << "Детекция: Класс " << d.class_id << " Уверенность: " << d.score << std::endl;
        cv::rectangle(frame, cv::Rect(d.x, d.y, d.w, d.h), cv::Scalar(0, 255, 0), 2);
    }

    cv::imwrite("result.jpg", frame);
    std::cout << "Результат сохранен в build/result.jpg" << std::endl;

    return 0;
}

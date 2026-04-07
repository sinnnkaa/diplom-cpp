#include "rknn_inference.h"
#include "decode.h"
#include "audio.h"
#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    std::cout << "--- Запуск программы (Float Mode) ---" << std::endl;
    
    RKNNModel model;
    if (!model.load("/root/diplom-cpp/blind_nav/model/yolo11_blind.rknn")) {
        std::cerr << "Не удалось загрузить модель!" << std::endl;
        return -1;
    }

    cv::Mat frame = cv::imread("/root/diplom-cpp/blind_nav/_4g6iIzKJPj_8_PTRPIV3Q_aug_0.jpg");
    if (frame.empty()) {
        std::cerr << "Не удалось прочитать картинку!" << std::endl;
        return -1;
    }

    // --- ПОДГОТОВКА КАРТИНКИ (Letterbox) ---
    int img_w = frame.cols;
    int img_h = frame.rows;
    float scale_factor = std::min((float)model.input_w / img_w, (float)model.input_h / img_h);
    int new_w = img_w * scale_factor;
    int new_h = img_h * scale_factor;

    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(new_w, new_h));

    // Создаем черный квадрат и копируем туда изображение по центру
    cv::Mat input_img = cv::Mat::zeros(model.input_h, model.input_w, CV_8UC3);
    resized.copyTo(input_img(cv::Rect((model.input_w - new_w) / 2, (model.input_h - new_h) / 2, new_w, new_h)));
    
    // Сохраняем для визуальной проверки входа NPU
    cv::imwrite("npu_input_check.jpg", input_img); 

    // --- ИНФЕРЕНС (Один вызов) ---
    std::vector<float> raw_out = model.infer(input_img);

    if (raw_out.empty()) {
        std::cerr << "Ошибка инференса NPU!" << std::endl;
        return -1;
    }

    // --- ДЕКОДИРОВАНИЕ ---
    // Используем порог 0.5 (для float это стандарт)
    auto results = decode(raw_out.data(), model.input_w, model.input_h, img_w, img_h, 0.2f);

    std::cout << "Найдено реальных объектов: " << results.size() << std::endl;

    for (auto& d : results) {
        std::cout << "Детекция: Класс " << d.class_id << " Уверенность: " << d.score << std::endl;
        cv::rectangle(frame, cv::Rect(d.x, d.y, d.w, d.h), cv::Scalar(0, 255, 0), 3);
    }

    cv::imwrite("result.jpg", frame);
    std::cout << "Готово! Проверьте файл result.jpg" << std::endl;

    return 0;
}

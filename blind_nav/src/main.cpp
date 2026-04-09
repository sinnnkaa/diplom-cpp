#include <iostream>
#include <opencv2/opencv.hpp>
#include "rknn_inference.h"
#include "decode.h"

int main() {
    std::cout << "--- Запуск программы (Diplom Navigation) ---" << std::endl;

    RKNNModel model;
    if (!model.load("/root/diplom-cpp/blind_nav/model/yolo11_final.rknn")) {
        std::cerr << "Не удалось загрузить модель!" << std::endl;
        return -1;
    }
    std::cout << "Model Loaded!" << std::endl;

    // Загружаем тестовое изображение
    std::string img_path = "/root/diplom-cpp/blind_nav/-0GQmYRienNVqEKiQ0Mkyw.jpg";
    cv::Mat img = cv::imread(img_path);
    if (img.empty()) {
        std::cerr << "Не удалось загрузить картинку: " << img_path << std::endl;
        return -1;
    }

    int img_w = img.cols;
    int img_h = img.rows;

    // Инференс (теперь внутри сам делается resize до 512x512)
    std::vector<float> raw_out = model.infer(img);

    if (raw_out.empty()) {
        std::cerr << "Ошибка инференса!" << std::endl;
        return -1;
    }

    // Декодирование результатов
    // Передаем 512, 512 как размеры входа модели
    auto results = decode(raw_out.data(), 512, 512, img_w, img_h, 0.4f);

    std::cout << "Найдено реальных объектов: " << results.size() << std::endl;

    // Отрисовка
    for (const auto& det : results) {
        std::cout << "Рисую: Класс " << det.class_id 
        << " [" << det.x << ", " << det.y << ", " << det.w << ", " << det.h << "]" << std::endl;
        cv::Rect box(det.x, det.y, det.w, det.h);
        cv::rectangle(img, box, cv::Scalar(0, 255, 0), 2);
        std::string label = "ID:" + std::to_string(det.class_id) + " " + std::to_string(det.score).substr(0, 4);
        cv::putText(img, label, cv::Point(det.x, det.y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
    }

    cv::imwrite("result.jpg", img);
    std::cout << "Готово! Проверьте файл result.jpg" << std::endl;

    return 0;
}

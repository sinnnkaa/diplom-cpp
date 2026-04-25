#include <iostream>
#include <fstream>
#include <chrono>
#include <vector>
#include <string>
#include <cstdio>  
#include <cstdlib> 
#include <opencv2/opencv.hpp>
#include "rknn_inference.h"
#include "decode.h"

const float FOCAL_LENGTH = 450.0f;
const int FRAME_WIDTH = 640;
const int FRAME_HEIGHT = 480;

float get_temperature() {
    std::ifstream temp_file("/sys/class/thermal/thermal_zone0/temp");
    if (!temp_file.is_open()) return 0.0f;
    float temp;
    temp_file >> temp;
    return temp / 1000.0f;
}

float get_real_height(int class_id) {
    switch(class_id) {
        case 0: return 1.70f; // person (человек)
        case 1: return 1.50f; // car (машина)
        case 2: return 0.15f; // curb (бордюр)
        case 3: return 3.00f; // pole (столб)
        case 4: return 0.70f; // traffic_sign (дорожный знак)
        case 5: return 0.80f; // traffic_light (светофор)
        case 6: return 0.60f; // trash_can (урна)
        case 7: return 0.60f; // bench (скамья)
        case 8: return 0.05f; // sidewalk (тротуар)
        case 9: return 0.05f; // crosswalk (пешеходный переход)
        default: return 1.00f;
    }
}

float get_class_weight(int class_id) {
    switch(class_id) {
        case 1: return 3.0f; // car 
        case 5: return 2.5f; // traffic_light 
        case 0: return 2.0f; // person
        case 4: return 1.8f; // traffic_sign
        case 2: return 1.5f; // curb 
        case 3: return 1.2f; // pole
        case 6: return 1.2f; // trash_can
        case 7: return 1.0f; // bench
        case 9: return 1.0f; // crosswalk
        case 8: return 0.5f; // sidewalk 
        default: return 1.0f;
    }
}

std::string get_class_name_ru(int class_id) {
    switch(class_id) {
        case 0: return "Человек";
        case 1: return "Машина";
        case 2: return "Бордюр";
        case 3: return "Столб";
        case 4: return "Знак";
        case 5: return "Светофор";
        case 6: return "Урна";
        case 7: return "Скамья";
        case 8: return "Тротуар";
        case 9: return "Переход";
        default: return "Объект";
    }
}

std::string get_plural_meters(int dist) {
    if (dist % 10 == 1 && dist % 100 != 11) return "метр";
    if (dist % 10 >= 2 && dist % 10 <= 4 && (dist % 100 < 10 || dist % 100 >= 20)) return "метра";
    return "метров";
}

int main(int argc, char** argv) {
    std::cout << "--- Запуск навигации (10 Классов + Сектора + Piper TTS) ---" << std::endl;

    RKNNModel model;
    if (!model.load("/root/diplom-cpp/blind_nav/model/yolo11_final.rknn")) {
        std::cerr << "Ошибка загрузки модели!" << std::endl;
        return -1;
    }

    cv::VideoCapture cap(0, cv::CAP_V4L2);
    if (!cap.isOpened()) {
        std::cerr << "Ошибка: Камера не найдена!" << std::endl;
        return -1;
    }

    cap.set(cv::CAP_PROP_FRAME_WIDTH, FRAME_WIDTH);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT);
    cap.set(cv::CAP_PROP_FPS, 30);

    int frame_count = 0;
    auto last_speech_time = std::chrono::steady_clock::now(); 

    while (true) {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) continue;

        auto start_time = std::chrono::high_resolution_clock::now();

        cv::Mat rgb_frame;
        cv::cvtColor(frame, rgb_frame, cv::COLOR_BGR2RGB);

        auto raw_out = model.infer(rgb_frame);
        auto results = decode(raw_out, 512, 512, rgb_frame.cols, rgb_frame.rows, 0.25f);

        float max_danger = -1.0f;
        int best_class_id = -1;
        float best_distance = 0.0f;
        std::string best_sector = "прямо";
        std::string priority_info = "Чисто";

        for (const auto& det : results) {
            float H = get_real_height(det.class_id);
            float distance = (FOCAL_LENGTH * H) / det.h;

            float center_x = det.x + (det.w / 2.0f);
            std::string current_sector = "прямо";
            float W_pos = 1.0f;

            if (center_x < FRAME_WIDTH / 3.0f) {
                current_sector = "слева";
            } else if (center_x > 2.0f * FRAME_WIDTH / 3.0f) {
                current_sector = "справа";
            } else {
                current_sector = "прямо";
                W_pos = 1.5f;
            }

            float W_class = get_class_weight(det.class_id);
            float danger_score = (W_class * W_pos) / distance;

            if (danger_score > max_danger) {
                max_danger = danger_score;
                best_class_id = det.class_id;
                best_distance = distance;
                best_sector = current_sector;
                priority_info = get_class_name_ru(det.class_id) + " " + best_sector + " " + std::to_string(distance).substr(0, 3) + "m";
            }

            cv::Rect box(det.x, det.y, det.w, det.h);
            cv::rectangle(frame, box, cv::Scalar(0, 255, 0), 2);
            std::string label = get_class_name_ru(det.class_id) + " " + current_sector + " " + std::to_string(distance).substr(0, 3) + "m";
            cv::putText(frame, label, cv::Point(det.x, det.y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
        }

        auto current_time = std::chrono::steady_clock::now();
        std::chrono::duration<float> elapsed_since_speech = current_time - last_speech_time;

        if (max_danger > 0 && elapsed_since_speech.count() > 3.0f) {
            int dist_m = static_cast<int>(best_distance + 0.5f); 

            std::string text_to_speak = get_class_name_ru(best_class_id) + " " + best_sector + ", " + 
                                        std::to_string(dist_m) + " " + get_plural_meters(dist_m);

            std::string command = "echo \"" + text_to_speak + "\" | /root/diplom-cpp/piper/piper/piper "
                                  "--model /root/diplom-cpp/piper/ru_RU-irina-medium.onnx "
                                  "--output-raw | aplay -r 22050 -f S16_LE -t raw -c 1 2>/dev/null &";
            
            std::system(command.c_str());
            last_speech_time = current_time;
        }

        std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, 70};
        cv::imwrite("/dev/shm/stream_tmp.jpg", frame, params);
        std::rename("/dev/shm/stream_tmp.jpg", "/dev/shm/stream.jpg");

        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float, std::milli> total_ms = end_time - start_time;

        if (frame_count % 10 == 0) { 
            std::cout << "\r[Кадр " << frame_count << "] Объектов: " << results.size() 
                      << " | Temp: " << get_temperature() << " C " 
                      << " | Главная цель: " << priority_info << "          " << std::flush;
        }
        frame_count++;
    }

    cap.release();
    return 0;
}
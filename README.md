# BlindNav: Edge AI Navigation System

![C++](https://img.shields.io/badge/C++-17-blue.svg?style=flat-square&logo=c%2B%2B)
![Python](https://img.shields.io/badge/Python-3.10-blue.svg?style=flat-square&logo=python)
![YOLOv11](https://img.shields.io/badge/Model-YOLOv11--Nano-orange.svg?style=flat-square)
![Platform](https://img.shields.io/badge/Platform-Orange%20Pi%203B%20(RK3566)-lightgrey.svg?style=flat-square)
![Status](https://img.shields.io/badge/Status-Active_Research-success.svg?style=flat-square)

Аппаратно-программный комплекс семантической навигации для людей с нарушениями зрения. Проект представляет собой полностью автономное (Offline) носимое устройство, объединяющее компьютерное зрение на базе нейропроцессора (NPU), спутниковую навигацию и нейросетевой синтез речи.

Разработано в рамках дипломного проектирования и научных исследований ГУАП.

## Ключевые возможности

* **Hardware-Accelerated Vision:** Обнаружение препятствий в реальном времени с использованием модели `YOLOv11-Nano` (квантованной в INT8). Инференс выполняется на NPU Rockchip RK3566, обеспечивая высокую частоту кадров при минимальном энергопотреблении.
* **Семантический анализ среды:** Распознавание 10 классов городских объектов (люди, машины, столбы, светофоры, бордюры и др.) с расчетом дистанции до них и приоритезацией опасности по секторам (лево/центр/право).
* **Локальный Neural TTS:** Мгновенное голосовое оповещение об опасности с использованием нейросетевого движка [Piper](https://github.com/rhasspy/piper) (голос "Ирина", формат ONNX).
* **Offline GPS Маршрутизация:** Прокладка пешеходных маршрутов (на примере г. Санкт-Петербург) без доступа к интернету с помощью локального сервера **OSRM** и кастомного геокодера.
* **Голосовое управление (в разработке):** Активация по физической кнопке (GPIO) и распознавание адреса с помощью локальной STT-модели Vosk.
* **Headless-режим:** Автоматический запуск всего конвейера обработки данных при подаче питания через фоновую службу `systemd`.

## Аппаратное обеспечение

* **Микрокомпьютер:** Orange Pi 3B (ARM Cortex-A55, 0.8 TOPS NPU).
* **Камера:** Стандартная USB Web-камера (V4L2).
* **Навигация:** USB/UART GPS-приемник.
* **Аудио:** Подключение наушников/динамика через разъем 3.5 мм или USB.
* **Управление:** Тактовая кнопка (GPIO).

## Архитектура программного комплекса

Проект разделен на несколько независимых подсистем:
1. **Computer Vision (C++):** Захват видеопотока, инференс через RKNN API, пространственный алгоритм оценки опасности, формирование команд для TTS.
2. **Web-Streaming (Python):** Легкий `mjpeg` сервер для отладки видения NPU через браузер (через `/dev/shm`).
3. **Routing Engine (Docker/OSRM):** Локальный сервер построения пешеходных маршрутов на базе графа OSM (`.pbf`).
4. **Offline Geocoder (Python):** Скрипт `extract_addresses.py` для создания легковесной (O(1)) хэш-таблицы адресов из сырых картографических данных.

## Установка и сборка

### 1. Зависимости
Для компиляции C++ ядра требуются библиотеки RKNN Toolkit 2 и OpenCV:
```bash
sudo apt update
sudo apt install cmake gcc g++ libopencv-dev
```

### 2. Сборка проекта  
```bash
git clone [https://github.com/sinnnkaa/diplom-cpp.git](https://github.com/sinnnkaa/diplom-cpp.git)
cd diplom-cpp/blind_nav/build
cmake ..
make -j4
```

### 3. Настройка служб  
Система спроектирована для работы в фоне. Установка systemd службы:  
```bash
sudo cp blind_nav.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now blind_nav.service
```

### 4. Структура директорий

```text
diplom-cpp/
├── blind_nav/
│   ├── src/                 # Исходный код C++
│   │   ├── main.cpp         # Главный цикл (Камера -> NPU -> Логика)
│   │   ├── decode.cpp       # Постобработка YOLO (NMS, Bounding Boxes)
│   │   └── rknn_inference.cpp # Обертка для работы с Rockchip NPU
│   ├── model/               # Скомпилированные веса
│   │   └── yolo11_final.rknn # Модель INT8 для RK3566
│   ├── map/                 # Оффлайн навигация
│   │   ├── spb.pbf          # Исходный граф OpenStreetMap
│   │   └── addresses.json   # Локальная хэш-таблица адресов
│   └── build/               # Скомпилированные бинарники и службы
│       ├── stream.py        # MJPEG-сервер трансляции
│       └── start_nav.sh     # Скрипт запуска для systemd
├── piper/                   # Локальный нейросетевой TTS-движок
├── vosk/                    # Модель распознавания речи (STT)
└── README.md
```


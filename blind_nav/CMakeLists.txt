cmake_minimum_required(VERSION 3.10)
project(blind_nav)

set(CMAKE_CXX_STANDARD 17)

# RKNN
include_directories(/usr/include)
link_directories(/usr/lib)

# OpenCV
find_package(OpenCV REQUIRED)

add_executable(blind_nav
    src/main.cpp
    src/rknn_inference.cpp
    src/decode.cpp
    src/audio.cpp
)

target_link_libraries(blind_nav
    ${OpenCV_LIBS}
    rknnrt
)
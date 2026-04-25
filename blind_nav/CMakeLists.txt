cmake_minimum_required(VERSION 3.10)
project(blind_nav)

set(CMAKE_CXX_STANDARD 17)

# RKNN
# link_directories(/usr/lib)

set(RKNN_API_PATH /root/diplom/rknn-toolkit2/rknpu2/runtime/Linux/librknn_api/aarch64/librknnrt.so)

include_directories(
    /root/diplom/rknn-toolkit2/rknpu2/runtime/Linux/librknn_api/include
)

# OpenCV
find_package(OpenCV REQUIRED)

add_executable(blind_nav
    src/main.cpp
    src/rknn_inference.cpp
    src/decode.cpp
    
)

target_link_libraries(blind_nav
    ${OpenCV_LIBS}
    ${RKNN_API_PATH}
)

cmake_minimum_required(VERSION 3.12)
project(cuda_image_processor CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(include)

add_executable(cuda_image_processor
    src/main.cu
    src/image_processing.cu
)

target_link_libraries(cuda_image_processor ${OpenCV_LIBS})

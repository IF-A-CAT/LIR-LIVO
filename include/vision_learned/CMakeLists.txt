cmake_minimum_required(VERSION 3.6)
project(plnet)

set(CMAKE_CXX_STANDARD 17)

if(POLICY CMP0146)
    cmake_policy(SET CMP0146 OLD)
endif()

find_package(CUDA REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")

include_directories(
    ${PROJECT_SOURCE_DIR}/include
    
)

set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

file(GLOB SOURCES
    "src/*.cpp"
    "src/*.cc"
)

set(TENSORRT_LIB_DIR "/home/sjz/Desktop/TensorRT-8.6.1.6/lib")
# 查找 TensorRT 中的具体库文件
find_library(NVINFER_LIB nvinfer HINTS ${TENSORRT_LIB_DIR})
find_library(NVINFER_PLUGIN_LIB nvinfer_plugin HINTS ${TENSORRT_LIB_DIR})
find_library(NVONNXPARSER_LIB nvonnxparser HINTS ${TENSORRT_LIB_DIR})
find_library(NVPARSERS_LIB nvparsers HINTS ${TENSORRT_LIB_DIR})


add_library(vision_learned SHARED ${SOURCES})

target_link_libraries(vision_learned ${CUDA_LIBRARIES} ${NVINFER_LIB} ${NVINFER_PLUGIN_LIB} ${NVONNXPARSER_LIB} ${NVPARSERS_LIB})




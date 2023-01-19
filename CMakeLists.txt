project(Deployment)
set(CMAKE_CXX_STANDARD 14)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/examples)

set(CMAKE_BUILD_TYPE "Debug")

set(OpenCV_DIR "/usr/local/lib/cmake/opencv4/")
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")
set(CUDNN_DIR "/usr/local/cuda")
set(TENSORRT_DIR "/usr/local/TensorRT")

find_package(OpenCV)
find_package(CUDA)

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${CUDA_TOOLKIT_ROOT_DIR}/include
    ${PROTOBUF_DIR}/include
    ${TENSORRT_DIR}/include
    ${CUDNN_DIR}/include
)

link_directories(
    ${PROTOBUF_DIR}/lib
    ${TENSORRT_DIR}/lib
    ${CUDA_TOOLKIT_ROOT_DIR}/lib64
    ${CUDNN_DIR}/lib64
)

list(APPEND SOCK_SRCS
    ${PROJECT_SOURCE_DIR}/src/net/buffer.cpp
    ${PROJECT_SOURCE_DIR}/src/net/channel.cpp
    ${PROJECT_SOURCE_DIR}/src/net/conn.cpp
    ${PROJECT_SOURCE_DIR}/src/net/event_base.cpp
    ${PROJECT_SOURCE_DIR}/src/net/logging.cpp
    ${PROJECT_SOURCE_DIR}/src/net/net.cpp
    ${PROJECT_SOURCE_DIR}/src/net/threads.cpp
    ${PROJECT_SOURCE_DIR}/src/net/utils.cpp
)

list(APPEND TRT_SRCS
    ${PROJECT_SOURCE_DIR}/src/application/cuda_tools.cpp
    ${PROJECT_SOURCE_DIR}/src/application/tensor.cpp
    ${PROJECT_SOURCE_DIR}/src/application/trt_infer.cpp
    ${PROJECT_SOURCE_DIR}/src/application/arcface/arcface.cpp
)

option(BUILD_SHARED_LIBRARY "Build Shared Library" OFF)
option(BUILD_STATIC_LIBRARY "Build Static Library" ON)
option(BUILD_EXAMPLES "Build Examples" ON)

##Shared Library
if(BUILD_SHARED_LIBRARY)
    add_library(p_sock SHARED ${SOCK_SRCS})
    target_link_libraries(p_sock pthread)
    target_include_directories(p_sock PUBLIC ${PROJECT_SOURCE_DIR}/src)

    add_library(trt SHARED ${TRT_SRCS})
    target_link_libraries(trt nvinfer nvinfer_plugin)
    target_link_libraries(trt cuda cublas cudart cudnn)
    target_link_libraries(trt protobuf)
    target_link_libraries(trt opencv_core opencv_imgproc opencv_imgcodecs)
    target_link_libraries(trt p_sock)
    target_include_directories(trt PUBLIC ${PROJECT_SOURCE_DIR}/src)
endif(BUILD_SHARED_LIBRARY)

#Static library
if(BUILD_STATIC_LIBRARY)
    add_library(p_sock_s STATIC ${SOCK_SRCS})
    target_link_libraries(p_sock_s pthread)
    target_include_directories(p_sock_s PUBLIC ${PROJECT_SOURCE_DIR}/src)

    add_library(trt_s STATIC ${TRT_SRCS})
    target_link_libraries(trt_s nvinfer nvinfer_plugin)
    target_link_libraries(trt_s cuda cublas cudart cudnn)
    target_link_libraries(trt_s protobuf)
    target_link_libraries(trt_s opencv_core opencv_imgproc opencv_imgcodecs)
    target_link_libraries(trt_s p_sock_s)
    target_include_directories(trt_s PUBLIC ${PROJECT_SOURCE_DIR}/src)
endif(BUILD_STATIC_LIBRARY)



function(add_p_sock_executable EXECUTABLE_NAME EXECUTABLE_SOURCES)
    add_executable(${EXECUTABLE_NAME} ${EXECUTABLE_SOURCES})
    target_link_libraries(${EXECUTABLE_NAME} p_sock_s)
    target_include_directories(${EXECUTABLE_NAME} PUBLIC ${PROJECT_SOURCE_DIR})
endfunction(add_p_sock_executable)

function(add_trt_executable EXECUTABLE_NAME EXECUTABLE_SOURCES)
    add_executable(${EXECUTABLE_NAME} ${EXECUTABLE_SOURCES})
    target_link_libraries(${EXECUTABLE_NAME} trt_s)
    target_include_directories(${EXECUTABLE_NAME} PUBLIC ${PROJECT_SOURCE_DIR})
endfunction(add_trt_executable)

if(BUILD_EXAMPLES)
    add_p_sock_executable(cli.o examples/cli.cpp)
    add_p_sock_executable(svr.o examples/svr.cpp)
    add_trt_executable(arcface_test.o examples/arcface_test.cpp)
endif(BUILD_EXAMPLES)
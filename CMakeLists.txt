project(test_json)
cmake_minimum_required(VERSION 3.0)
find_package(OpenCV REQUIRED)
find_package(CUDA)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(/data/Code/tensorrt_7.0/TensorRT-7.0.0.11/include)
include_directories(${PROJECT_BINARY_DIR}/../ ${PROJECT_BINARY_DIR}/../jsoncpp-00.11.0/include)
include_directories(${PROJECT_BINARY_DIR}/../node_info)
include_directories(${PROJECT_BINARY_DIR}/../node_create)
include_directories(${PROJECT_BINARY_DIR}/../cuda_impl)
include_directories(${PROJECT_BINARY_DIR}/../node_info/plugin)
include_directories(/usr/local/cuda-10.0/include)
include_directories(${PROJECT_BINARY_DIR}/../cub-1.8.0)

link_directories(/data/Code/tensorrt_7.0/TensorRT-7.0.0.11/lib)
link_directories(/usr/local/cuda-10.0/lib64)
link_directories(${PROJECT_BINARY_DIR}/../jsoncpp-00.11.0/build-shared/)
add_compile_options(-std=c++11 -w)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11 -w" )
cuda_add_executable(test_json test_json.cpp
                         weights_graph_parse.cpp
                         tensorrt_engine.cpp
                         ${PROJECT_BINARY_DIR}/../node_info/node_info.cpp
                         ${PROJECT_BINARY_DIR}/../node_info/conv2d_node_info.cpp
                         ${PROJECT_BINARY_DIR}/../node_info/activation_node_info.cpp
                         ${PROJECT_BINARY_DIR}/../node_info/elementwise_node_info.cpp
                         ${PROJECT_BINARY_DIR}/../node_info/shuffle_node_info.cpp
                         ${PROJECT_BINARY_DIR}/../node_info/padding_node_info.cpp
                         ${PROJECT_BINARY_DIR}/../node_info/unary_node_info.cpp
                         ${PROJECT_BINARY_DIR}/../node_info/softmax_node_info.cpp
                         ${PROJECT_BINARY_DIR}/../node_info/pooling_node_info.cpp
                         ${PROJECT_BINARY_DIR}/../node_info/reduce_node_info.cpp
                         ${PROJECT_BINARY_DIR}/../node_info/slice_node_info.cpp
                         ${PROJECT_BINARY_DIR}/../node_info/identity_node_info.cpp
                         ${PROJECT_BINARY_DIR}/../node_info/nonzero_node_info.cpp
                         ${PROJECT_BINARY_DIR}/../node_create/create_node.cpp
                         ${PROJECT_BINARY_DIR}/../node_create/create_activation_node.cpp
                         ${PROJECT_BINARY_DIR}/../node_create/create_elementwise_node.cpp
                         ${PROJECT_BINARY_DIR}/../node_create/create_unary_node.cpp
                         ${PROJECT_BINARY_DIR}/../node_create/create_padding_node.cpp
                         ${PROJECT_BINARY_DIR}/../node_create/create_reduce_node.cpp
                         ${PROJECT_BINARY_DIR}/../node_create/create_softmax_node.cpp
                         ${PROJECT_BINARY_DIR}/../node_create/create_shuffle_node.cpp
                         ${PROJECT_BINARY_DIR}/../node_create/create_conv2d_node.cpp
                         ${PROJECT_BINARY_DIR}/../node_create/create_slice_node.cpp
                         ${PROJECT_BINARY_DIR}/../node_create/create_identity_node.cpp
                         ${PROJECT_BINARY_DIR}/../node_create/create_pooling_node.cpp
                         ${PROJECT_BINARY_DIR}/../node_create/create_nonzero_node.cpp
                         ${PROJECT_BINARY_DIR}/../node_create/plugin/nonzero_plugin.cu
                         ${PROJECT_BINARY_DIR}/../cuda_impl/nonzero_cuda_impl.cu
                         ${PROJECT_BINARY_DIR}/../cuda_impl/convert_cuda_impl.cu)
target_link_libraries(test_json jsoncpp)
target_link_libraries(test_json nvinfer)
target_link_libraries(test_json cudart)
target_link_libraries(test_json ${OpenCV_LIBS})
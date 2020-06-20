#ifndef __CREATE_NODE_HPP__
#define __CREATE_NODE_HPP__
#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "node_info.hpp"
#include "weights_graph_parse.hpp"
#include "utils.hpp"
#include <iostream>
using namespace std;

namespace tensorrtInference
{
    enum OnnxDataType {
        DEFAULT,
        FLOAT,
        UINT8,
        INT8,
        UINT16,
        INT16,
        INT32,
        INT64,
        STRING,
        BOOL,
        FLOAT16,
        DOUBLE,
        UINT32,
        UINT64,
        COMPLEX64,
        COMPLEX128,
        BFLOAT16,
    };
    extern int onnxDataTypeEleCount[];
    extern nvinfer1::ILayer* createNode(nvinfer1::INetworkDefinition* network, std::map<std::string, nvinfer1::ITensor*>& tensors, 
        tensorrtInference::nodeInfo* nodeConfInfo, std::map<std::string, tensorrtInference::weightInfo>& nodeWeightsInfo);
    extern std::vector<float> parseFloatArrayValue(int dataType, char* data, int byteCount, std::vector<int> shape);
    extern std::vector<int> parseIntArrayValue(int dataType, char* data, int byteCount, std::vector<int> shape);
    extern int getTensorrtDataType(OnnxDataType onnxDataType);
}

#endif
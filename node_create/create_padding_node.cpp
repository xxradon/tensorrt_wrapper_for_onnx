#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "weights_graph_parse.hpp"
#include "create_node.hpp"
#include "create_padding_node.hpp"

namespace tensorrtInference
{
    nvinfer1::ILayer* createPaddingNode(nvinfer1::INetworkDefinition* network, std::map<std::string, nvinfer1::ITensor*>& tensors,
        tensorrtInference::nodeInfo* nodeConfInfo, std::map<std::string, tensorrtInference::weightInfo>& nodeWeightsInfo)
    {
        auto subType = nodeConfInfo->getSubNodeType();
        auto inputs = nodeConfInfo->getInputs();
        CHECK_ASSERT(inputs.size(), "Padding node must have 2 inputs\n");
        nvinfer1::ITensor* inputTensors = tensors[inputs[0]];
        auto shape = nodeWeightsInfo[inputs[1]].shape;
        CHECK_ASSERT(shape.size() == 1 && shape[0] == 8, "Pads value must be 8 (Nbegin, Cbegin, Hbegin, Wbegin, Nend, Cend, Hend, Wend)\n");
        auto pads = parseIntArrayValue(nodeWeightsInfo[inputs[1]].dataType, nodeWeightsInfo[inputs[1]].data,
                         nodeWeightsInfo[inputs[1]].byteCount, shape);
        nvinfer1::IPaddingLayer* padding = network->addPadding(*inputTensors, nvinfer1::DimsHW{pads[2], pads[3]}, nvinfer1::DimsHW{pads[6], pads[7]});
        CHECK_ASSERT(padding != nullptr, "create Padding node fail\n");
        return padding;
    }
}
#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "weights_graph_parse.hpp"
#include "create_node.hpp"
#include "create_conv2d_node.hpp"
#include "conv2d_node_info.hpp"

namespace tensorrtInference
{
    nvinfer1::ILayer* createConv2dNode(nvinfer1::INetworkDefinition* network, std::map<std::string, nvinfer1::ITensor*>& tensors,
        tensorrtInference::nodeInfo* nodeConfInfo, std::map<std::string, tensorrtInference::weightInfo>& nodeWeightsInfo)
    {
        auto conv2dNodeInfo = (Conv2dNodeInfo *)nodeConfInfo;
        auto inputs = conv2dNodeInfo->getInputs();
        CHECK_ASSERT(inputs.size() >= 2, "conv2d inputs must greater than 2\n");
        auto kernelShape = conv2dNodeInfo->getKernelShape();
        CHECK_ASSERT(kernelShape.size() == 2, "conv2d kernel shape must be 2\n");

        nvinfer1::IConvolutionLayer* conv2d = nullptr;
        nvinfer1::ITensor* inputTensors = tensors[inputs[0]];
        nvinfer1::DataType dataType = (nodeWeightsInfo[inputs[1]].dataType == tensorrtInference::OnnxDataType::FLOAT) ? 
                                        nvinfer1::DataType::kFLOAT : nvinfer1::DataType::kHALF;
        int weightEleCount = onnxDataTypeEleCount[nodeWeightsInfo[inputs[1]].dataType];
        CHECK_ASSERT(nodeWeightsInfo[inputs[1]].byteCount % weightEleCount == 0,
            "weights byte count shoud be mulptile of element byte count\n");
        nvinfer1::Weights wt{dataType, nullptr, 0};
        wt.type   = dataType;
        wt.values = nodeWeightsInfo[inputs[1]].data;
        wt.count  = nodeWeightsInfo[inputs[1]].byteCount / weightEleCount;
        int nbOutputMaps = nodeWeightsInfo[inputs[1]].shape[0];
        if(inputs.size() > 2)
        {
            int biasEleCount = onnxDataTypeEleCount[nodeWeightsInfo[inputs[2]].dataType];
            CHECK_ASSERT(nodeWeightsInfo[inputs[2]].byteCount % biasEleCount == 0,
                "bias byte count shoud be mulptile of element byte count\n");
            nvinfer1::Weights bias{dataType, nullptr, 0};
            bias.type = dataType;
            bias.values = nodeWeightsInfo[inputs[2]].data;
            bias.count = nodeWeightsInfo[inputs[2]].byteCount / biasEleCount;
            conv2d = network->addConvolution(*inputTensors, nbOutputMaps, nvinfer1::DimsHW{kernelShape[0], kernelShape[1]}, wt, bias);
        }
        else
        {
            nvinfer1::Weights bias{dataType, nullptr, 0};
            conv2d = network->addConvolution(*inputTensors, nbOutputMaps, nvinfer1::DimsHW{kernelShape[0], kernelShape[1]}, wt, bias);
        }
        CHECK_ASSERT(conv2d, "create conv2d node fail\n");
        auto group = conv2dNodeInfo->getGroup();
        auto strides = conv2dNodeInfo->getStrides();
        auto pads = conv2dNodeInfo->getPads();
        auto dilation = conv2dNodeInfo->getDilation();
        if(group > 1)
            conv2d->setNbGroups(group);
        if(strides.size())
            conv2d->setStride(nvinfer1::DimsHW{strides[0], strides[1]});
        if(pads.size() && pads.size() == 4)
        {
            CHECK_ASSERT(pads[0] == pads[1], "conv2d only support symmetric padding %d %d %d %d\n", pads[0], pads[1], pads[2], pads[3]);
            CHECK_ASSERT(pads[2] == pads[3], "conv2d only support symmetric padding %d %d %d %d\n", pads[0], pads[1], pads[2], pads[3]);
            CHECK_ASSERT(pads[0] == pads[2], "conv2d only support symmetric padding %d %d %d %d\n", pads[0], pads[1], pads[2], pads[3]);
            conv2d->setPadding(nvinfer1::DimsHW{pads[0], pads[1]});
        }
        
        if(dilation.size())
            conv2d->setDilation(nvinfer1::DimsHW{dilation[0], dilation[1]});
        
        return conv2d;
    }
}
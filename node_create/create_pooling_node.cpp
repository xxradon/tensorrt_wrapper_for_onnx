#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "weights_graph_parse.hpp"
#include "create_node.hpp"
#include "pooling_node_info.hpp"
#include "create_pooling_node.hpp"

namespace tensorrtInference
{
    nvinfer1::ILayer* createPoolingNode(nvinfer1::INetworkDefinition* network, std::map<std::string, nvinfer1::ITensor*>& tensors,
        tensorrtInference::nodeInfo* nodeConfInfo, std::map<std::string, tensorrtInference::weightInfo>& nodeWeightsInfo)
    {
        PoolingNodeInfo *nodeConfigInfo = (PoolingNodeInfo *)nodeConfInfo;
        auto inputs = nodeConfigInfo->getInputs();
        nvinfer1::ITensor* inputTensor = tensors[inputs[0]];

        auto kernelShape = nodeConfigInfo->getKernelShape();
        auto pads        = nodeConfigInfo->getPads();
        auto strides     = nodeConfigInfo->getStrides();
        auto subNodeType = nodeConfInfo->getSubNodeType();
        nvinfer1::IPoolingLayer* pooling = nullptr;
        if(subNodeType.compare("MaxPool") == 0)
        {
            pooling = network->addPooling(*inputTensor, nvinfer1::PoolingType::kMAX, nvinfer1::DimsHW{kernelShape[0], kernelShape[1]});
        }
        else
            LOG("current noly support max pooling!\n");
        CHECK_ASSERT(pooling, "create pooling node fail\n");
        pooling->setStride(nvinfer1::DimsHW{strides[0], strides[1]});
        pooling->setPadding(nvinfer1::DimsHW{pads[0], pads[1]});
        return pooling;
    }
}
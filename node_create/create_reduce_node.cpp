#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "weights_graph_parse.hpp"
#include "create_reduce_node.hpp"
#include "reduce_node_info.hpp"
namespace tensorrtInference
{
    nvinfer1::ILayer* createReduceNode(nvinfer1::INetworkDefinition* network, std::map<std::string, nvinfer1::ITensor*>& tensors,
        tensorrtInference::nodeInfo* nodeConfInfo, std::map<std::string, tensorrtInference::weightInfo>& nodeWeightsInfo)
    {
        auto reduceNodeInfo = (ReduceNodeInfo*)nodeConfInfo;
        auto subType = reduceNodeInfo->getSubNodeType();
        nvinfer1::ReduceOperation operation;
        //ReduceSum
        if(subType.compare("ReduceSum") == 0) {
            operation = nvinfer1::ReduceOperation::kSUM;
        }
        else {
            LOG("Current not support unary operation(%s) \n", subType);
            return nullptr;
        }
        auto inputs = reduceNodeInfo->getInputs();
        nvinfer1::ITensor* inputTensors = tensors[inputs[0]];
        auto axesNodeConfig = reduceNodeInfo->getAxes();
        unsigned int axes = 0;
        for(int i = 0; i < axesNodeConfig.size(); i++)
        {
            axes |= (1 << axesNodeConfig[i]);
        }
        bool keepdims = reduceNodeInfo->getKeepdims();
        nvinfer1::IReduceLayer* reduce = network->addReduce(*inputTensors, operation, axes, keepdims);
        CHECK_ASSERT(reduce, "create reduce node fail\n");
        return reduce;
    }
}
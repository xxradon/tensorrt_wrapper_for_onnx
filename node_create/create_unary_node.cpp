#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "weights_graph_parse.hpp"
#include "create_node.hpp"
#include "create_unary_node.hpp"

namespace tensorrtInference
{
    nvinfer1::ILayer* createUnaryNode(nvinfer1::INetworkDefinition* network, std::map<std::string, nvinfer1::ITensor*>& tensors,
        tensorrtInference::nodeInfo* nodeConfInfo, std::map<std::string, tensorrtInference::weightInfo>& nodeWeightsInfo)
    {
        auto subType = nodeConfInfo->getSubNodeType();
        nvinfer1::UnaryOperation operation;
        //Sqrt Reciprocal Abs
        if(subType.compare("Sqrt") == 0) {
            operation = nvinfer1::UnaryOperation::kSQRT;
        }
        else if(subType.compare("Reciprocal") == 0) {
            operation = nvinfer1::UnaryOperation::kRECIP;
        }
        else if(subType.compare("Abs") == 0) {
            operation = nvinfer1::UnaryOperation::kABS;
        }        
        else {
            LOG("Current not support unary operation(%s) \n", subType);
            return nullptr;
        }
        auto inputs = nodeConfInfo->getInputs();
        nvinfer1::ITensor* inputTensors = tensors[inputs[0]];
        nvinfer1::IUnaryLayer* unary = network->addUnary(*inputTensors, operation);
        CHECK_ASSERT(unary, "create unary node fail\n");
        return unary;
    }
}
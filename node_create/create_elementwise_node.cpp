#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "weights_graph_parse.hpp"
#include "create_node.hpp"
#include "create_elementwise_node.hpp"

namespace tensorrtInference
{
    nvinfer1::ILayer* createElementWiseNode(nvinfer1::INetworkDefinition* network, std::map<std::string, nvinfer1::ITensor*>& tensors,
        tensorrtInference::nodeInfo* nodeConfInfo, std::map<std::string, tensorrtInference::weightInfo>& nodeWeightsInfo)
    {
        auto subType = nodeConfInfo->getSubNodeType();
        nvinfer1::ElementWiseOperation operation;
        //Sub Div Add Mul Equal Greater Max
        if(subType.compare("Sub") == 0) {
            operation = nvinfer1::ElementWiseOperation::kSUB;
        }
        else if(subType.compare("Div") == 0) {
            operation = nvinfer1::ElementWiseOperation::kDIV;
        }
        else if(subType.compare("Add") == 0) {
            operation = nvinfer1::ElementWiseOperation::kSUM;
        }
        else if(subType.compare("Mul") == 0) {
            operation = nvinfer1::ElementWiseOperation::kPROD;
        }
        else if(subType.compare("Max") == 0) {
            operation = nvinfer1::ElementWiseOperation::kMAX;
        }        
        else if(subType.compare("Equal") == 0) {
            operation = nvinfer1::ElementWiseOperation::kEQUAL;
        }
        else if(subType.compare("Greater") == 0) {
            operation = nvinfer1::ElementWiseOperation::kGREATER;
        }        
        else {
            LOG("Current not support elementwise operation(%s) \n", subType);
            return nullptr;
        }
        auto inputs = nodeConfInfo->getInputs();
        nvinfer1::ITensor* inputTensors1 = tensors[inputs[0]];
        nvinfer1::ITensor* inputTensors2 = tensors[inputs[1]];
        nvinfer1::IElementWiseLayer* ew = network->addElementWise(*inputTensors1, *inputTensors2, operation);
        CHECK_ASSERT(ew, "create ElementWise node fail\n");
        if(subType.compare("Greater") == 0)
        {
            auto dataType = ew->getOutputType(0);
            auto tensorFormat = ew->getOutput(0)->getAllowedFormats();
            printf("run here!\n");
        }
        return ew;
    }
}
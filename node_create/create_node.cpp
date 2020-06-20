#include "create_node.hpp"
#include "create_elementwise_node.hpp"
#include "create_padding_node.hpp"
#include "create_reduce_node.hpp"
#include "create_softmax_node.hpp"
#include "create_unary_node.hpp"
#include "create_shuffle_node.hpp"
#include "create_activation_node.hpp"
#include "create_conv2d_node.hpp"
#include "create_slice_node.hpp"
#include "create_identity_node.hpp"
#include "create_pooling_node.hpp"
#include "create_nonzero_node.hpp"

namespace tensorrtInference
{
    typedef nvinfer1::ILayer* (*func)(nvinfer1::INetworkDefinition* network, std::map<std::string, nvinfer1::ITensor*>& tensors,
        tensorrtInference::nodeInfo* nodeConfInfo, std::map<std::string, tensorrtInference::weightInfo>& nodeWeightsInfo);
    
    static std::map<std::string, func> createNodeFuncMap;

    int onnxDataTypeEleCount[] = {0, 4, 1, 1, 2, 2, 4, 8, 0, 1, 2, 8, 4, 8, 8, 16, 2};

    nvinfer1::ILayer* createNode(nvinfer1::INetworkDefinition* network, std::map<std::string, nvinfer1::ITensor*>& tensors,
     tensorrtInference::nodeInfo* nodeConfInfo, std::map<std::string, tensorrtInference::weightInfo>& nodeWeightsInfo)
    {
        if(createNodeFuncMap.size() == 0)
        {
            createNodeFuncMap["ElementWise"] = tensorrtInference::createElementWiseNode;
            createNodeFuncMap["Activation"]  = tensorrtInference::createActivationNode;
            createNodeFuncMap["Padding"]     = tensorrtInference::createPaddingNode;
            createNodeFuncMap["Reduce"]      = tensorrtInference::createReduceNode;
            createNodeFuncMap["Softmax"]     = tensorrtInference::createSoftmaxNode;
            createNodeFuncMap["Unary"]       = tensorrtInference::createUnaryNode;
            createNodeFuncMap["Shuffle"]     = tensorrtInference::createShuffleNode;
            createNodeFuncMap["Conv2d"]      = tensorrtInference::createConv2dNode;
            createNodeFuncMap["Slice"]       = tensorrtInference::createSliceNode;
            createNodeFuncMap["Identity"]    = tensorrtInference::createIdentityNode;
            createNodeFuncMap["Pooling"]     = tensorrtInference::createPoolingNode;
            createNodeFuncMap["NonZero"]     = tensorrtInference::createNonZeroNode;
        }
        auto inputs = nodeConfInfo->getInputs();
        for(int i = 0; i < inputs.size(); i++)
        {
            if(tensors.count(inputs[i]) == 0 && nodeWeightsInfo.count(inputs[i]) == 0)
            {
                CHECK_ASSERT(0, "topo order wrong!\n");
            }
        }
        auto nodeType = nodeConfInfo->getNodeType();
        nvinfer1::ILayer* layer = nullptr;
        if(createNodeFuncMap.count(nodeType) != 0)
        {
            layer = createNodeFuncMap[nodeType](network, tensors, nodeConfInfo, nodeWeightsInfo);
        }
        else
            LOG("current not support node type (%s)\n", nodeType.c_str());
        
        return layer;
    }

    std::vector<float> parseFloatArrayValue(int dataType, char* data, int byteCount, std::vector<int> shape)
    {
        bool supportFlag = (dataType == int(OnnxDataType::FLOAT) || dataType == int(OnnxDataType::DOUBLE));
        CHECK_ASSERT(supportFlag , "only support FLOAT and DOUBLE\n");
        int eleCount = onnxDataTypeEleCount[dataType];
        int size = shape.size();
        int shapeCount = 1;
        std::vector<float> arrValue;
        for(int i = 0; i < size; i++)
        {
            shapeCount *= shape[i];
        }
        CHECK_ASSERT((shapeCount * eleCount) == byteCount , "shapeCount * eleCount not equal to byteCount\n");
        if(dataType == int(OnnxDataType::FLOAT))
        {
            float *floatData = (float *)data;
            for(int i = 0; i < shapeCount; i++)
            {
                arrValue.push_back(floatData[i]);
            }
        }
        else
        {
            double *doubleData = (double *)data;
            for(int i = 0; i < shapeCount; i++)
            {
                arrValue.push_back(doubleData[i]);
            }
        }
        return arrValue;
    }

    std::vector<int> parseIntArrayValue(int dataType, char* data, int byteCount, std::vector<int> shape)
    {
        bool supportFlag = (dataType == int(OnnxDataType::INT32) || dataType == int(OnnxDataType::INT64));
        CHECK_ASSERT(supportFlag , "only support int32 and int64\n");
        int eleCount = onnxDataTypeEleCount[dataType];
        int size = shape.size();
        int shapeCount = 1;
        std::vector<int> arrValue;
        for(int i = 0; i < size; i++)
        {
            shapeCount *= shape[i];
        }
        CHECK_ASSERT((shapeCount * eleCount) == byteCount , "shapeCount * eleCount not equal to byteCount\n");
        if(dataType == int(OnnxDataType::INT32))
        {
            int *intData = (int *)data;
            for(int i = 0; i < shapeCount; i++)
            {
                arrValue.push_back(intData[i]);
            }
        }
        else
        {
            int64_t *int64Data = (int64_t *)data;
            for(int i = 0; i < shapeCount; i++)
            {
                arrValue.push_back(int64Data[i]);
            }
        }
        return arrValue;
    }
    int getTensorrtDataType(OnnxDataType onnxDataType)
    {
        switch(onnxDataType)
        {
            case OnnxDataType::FLOAT:
                return int(nvinfer1::DataType::kFLOAT);
            case OnnxDataType::FLOAT16:
                return int(nvinfer1::DataType::kHALF);
            default:
                return -1;
        }
    }
}
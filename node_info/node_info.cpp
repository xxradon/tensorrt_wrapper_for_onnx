#include "node_info.hpp"
#include "conv2d_node_info.hpp"
#include "elementwise_node_info.hpp"
#include "activation_node_info.hpp"
#include "shuffle_node_info.hpp"
#include "padding_node_info.hpp"
#include "unary_node_info.hpp"
#include "softmax_node_info.hpp"
#include "reduce_node_info.hpp"
#include "pooling_node_info.hpp"
#include "slice_node_info.hpp"
#include "identity_node_info.hpp"
#include "nonzero_node_info.hpp"
#include <assert.h>

namespace tensorrtInference 
{
    nodeInfo::nodeInfo() {
        inputs.clear();
        outputs.clear();
    }
    nodeInfo::~nodeInfo() {
        inputs.clear();
        outputs.clear();
    }
    std::string nodeInfo::getNodeType() { return nodeType; }
    void nodeInfo::setNodeType(std::string type) { nodeType = type; }
    std::string nodeInfo::getSubNodeType() { return subNodeType; }
    void nodeInfo::setSubNodeType(std::string type) { subNodeType = type; }
    std::vector<std::string> nodeInfo::getOutputs() { return outputs; }
    std::vector<std::string> nodeInfo::getInputs() { return inputs; }
    void nodeInfo::addInput(std::string input) { inputs.push_back(input); }
    void nodeInfo::addOutput(std::string output) { outputs.push_back(output); }
    // std::vector<std::string> nodeInfo::getDependNodes() { return dependNodes; }
    void nodeInfo::printNodeInfo() {
        LOG("###################NODE INFO######################\n");
        LOG("currend node type is %s , sub node type is %s\n", nodeType.c_str(), subNodeType.c_str());
        auto input = getInputs();
        LOG("Input tensor size is %d\n", input.size());
        for(int i = 0; i < input.size(); i++) {
            LOG("----index %d tensor : %s\n", i, input[i].c_str());
        }
        auto output = getOutputs();
        LOG("Output tensor size is %d\n", output.size());
        for(int i = 0; i < output.size(); i++) {
            LOG("----index %d tensor : %s\n", i, output[i].c_str());
        }
    }

    nodeInfo* parseConv2dNodeInfo(std::string type, Json::Value& root)
    {
        nodeInfo* node = new Conv2dNodeInfo();
        if(node->parseNodeInfoFromJson(type, root))
            return node;
        else
            delete node;
        return nullptr;
    }
    nodeInfo* parseElementWiseNodeInfo(std::string type, Json::Value& root)
    {
        nodeInfo* node = new ElementWiseNodeInfo();
        if(node->parseNodeInfoFromJson(type, root))
            return node;
        else
            delete node;
        return nullptr;
    }
    nodeInfo* parseActivationNodeInfo(std::string type, Json::Value& root)
    {
        nodeInfo* node = new ActivationNodeInfo();
        if(node->parseNodeInfoFromJson(type, root))
            return node;
        else
            delete node;
        return nullptr;
    }    
    nodeInfo* parseShuffleNodeInfo(std::string type, Json::Value& root)
    {
        nodeInfo* node = new ShuffleNodeInfo();
        if(node->parseNodeInfoFromJson(type, root))
            return node;
        else
            delete node;
        return nullptr;
    }
    nodeInfo* parsePaddingNodeInfo(std::string type, Json::Value& root)
    {
        nodeInfo* node = new PaddingNodeInfo();
        if(node->parseNodeInfoFromJson(type, root))
            return node;
        else
            delete node;
        return nullptr;
    }
    nodeInfo* parseUnaryNodeInfo(std::string type, Json::Value& root)
    {
        nodeInfo* node = new UnaryNodeInfo();
        if(node->parseNodeInfoFromJson(type, root))
            return node;
        else
            delete node;
        return nullptr;
    }
    nodeInfo* parseSoftmaxNodeInfo(std::string type, Json::Value& root)
    {
        nodeInfo* node = new SoftmaxNodeInfo();
        if(node->parseNodeInfoFromJson(type, root))
            return node;
        else
            delete node;
        return nullptr;
    }    
    nodeInfo* parseReduceNodeInfo(std::string type, Json::Value& root)
    {
        nodeInfo* node = new ReduceNodeInfo();
        if(node->parseNodeInfoFromJson(type, root))
            return node;
        else
            delete node;
        return nullptr;
    }
    nodeInfo* parsePoolingNodeInfo(std::string type, Json::Value& root)
    {
        nodeInfo* node = new PoolingNodeInfo();
        if(node->parseNodeInfoFromJson(type, root))
            return node;
        else
            delete node;
        return nullptr;
    }
    nodeInfo* parseSliceNodeInfo(std::string type, Json::Value& root)
    {
        nodeInfo* node = new SliceNodeInfo();
        if(node->parseNodeInfoFromJson(type, root))
            return node;
        else
            delete node;
        return nullptr;
    }
    nodeInfo* parseIdentityNodeInfo(std::string type, Json::Value& root)
    {
        nodeInfo* node = new IdentityNodeInfo();
        if(node->parseNodeInfoFromJson(type, root))
            return node;
        else
            delete node;
        return nullptr;
    }
    nodeInfo* parseNonZeroNodeInfo(std::string type, Json::Value& root)
    {
        nodeInfo* node = new NonZeroNodeInfo();
        if(node->parseNodeInfoFromJson(type, root))
            return node;
        else
            delete node;
        return nullptr;
    }    
    nodeInfo* (*parseNodeInfoFromJsonFuncArr[])(std::string, Json::Value&) = {
        parseConv2dNodeInfo,         // Conv
        parseElementWiseNodeInfo,    // Sub Div Add Mul Equal Greater
        parseActivationNodeInfo,     // Clip
        parseShuffleNodeInfo,        // Reshape transpose
        parsePaddingNodeInfo,        // Pad
        parseUnaryNodeInfo,          // Sqrt Reciprocal Abs
        parseSoftmaxNodeInfo,        // Softmax
        parseReduceNodeInfo,         // ReduceSum
        parsePoolingNodeInfo,        // MaxPool
        parseSliceNodeInfo,          // Slice
        parseIdentityNodeInfo,       // Cast
        parseNonZeroNodeInfo,        // NonZero
    };
}
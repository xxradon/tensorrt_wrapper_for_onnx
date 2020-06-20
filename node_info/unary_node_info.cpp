#include "unary_node_info.hpp"
#include "utils.hpp"

namespace tensorrtInference
{
    // Unary Node
    UnaryNodeInfo::UnaryNodeInfo()
    {
        setNodeType("Unary");
        setSubNodeType("");
    }
    UnaryNodeInfo::~UnaryNodeInfo()
    {  
        
    }
    bool UnaryNodeInfo::parseNodeInfoFromJson(std::string type, Json::Value &root)
    {
        setSubNodeType(type);
        auto inputSize = root["inputs"].size();
        CHECK_ASSERT(inputSize == 1, "Unary node must have 1 inputs\n");
        for(int i = 0; i < inputSize; i++)
        {
            addInput(root["inputs"][i].asString());
        }
        auto outputSize = root["outputs"].size();
        CHECK_ASSERT(outputSize == 1, "Unary node must have 1 output\n");
        auto nodeOutputs = getOutputs();
        for(int i = 0; i < outputSize; i++)
        {
            addOutput(root["outputs"][i].asString());
        }
        return true;
    }
}
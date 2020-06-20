#include "activation_node_info.hpp"
#include "utils.hpp"

namespace tensorrtInference
{
    // Activation Node
    ActivationNodeInfo::ActivationNodeInfo()
    {
        setNodeType("Activation");
        setSubNodeType("");
    }
    ActivationNodeInfo::~ActivationNodeInfo()
    {

    }
    bool ActivationNodeInfo::parseNodeInfoFromJson(std::string type, Json::Value &root)
    {
        setSubNodeType(type);
        auto inputSize = root["inputs"].size();
        CHECK_ASSERT(inputSize <= 3, "Activation node must less than 3 inputs\n");
        for(int i = 0; i < inputSize; i++)
        {
            addInput(root["inputs"][i].asString());
        }
        auto outputSize = root["outputs"].size();
        CHECK_ASSERT(outputSize == 1, "Activation node must have 1 output\n");
        auto nodeOutputs = getOutputs();
        for(int i = 0; i < outputSize; i++)
        {
            addOutput(root["outputs"][i].asString());
        }
        return true;
    }
}
#include "identity_node_info.hpp"
#include "utils.hpp"

namespace tensorrtInference
{
    // Identity Node
    IdentityNodeInfo::IdentityNodeInfo()
    {
        setNodeType("Identity");
        setSubNodeType("");
        dataType = 0;
    }
    IdentityNodeInfo::~IdentityNodeInfo()
    {
        dataType = 0;
    }
    bool IdentityNodeInfo::parseNodeInfoFromJson(std::string type, Json::Value &root)
    {
        setSubNodeType(type);
        auto inputSize = root["inputs"].size();
        CHECK_ASSERT(inputSize == 1, "Identity node must have 1 inputs\n");
        for(int i = 0; i < inputSize; i++)
        {
            addInput(root["inputs"][i].asString());
        }
        auto outputSize = root["outputs"].size();
        CHECK_ASSERT(outputSize == 1, "Identity node must have 1 output\n");
        auto nodeOutputs = getOutputs();
        for(int i = 0; i < outputSize; i++)
        {
            addOutput(root["outputs"][i].asString());
        }
        auto attr = root["attributes"];
        for (auto elem : attr.getMemberNames())
        {
            if(elem.compare("to") == 0)
            {
                auto size = attr[elem].size();
                CHECK_ASSERT(size == 1, "Identity node's to must have 1 element\n");
                dataType = attr[elem][0].asInt();
            }
            else
            {
                LOG("currnet Identity node not support %s \n", elem.c_str());
            }
        }
        return true;
    }
    void IdentityNodeInfo::printNodeInfo()
    {
        nodeInfo::printNodeInfo();
        LOG("node attribute is as follows:\n");
        LOG("----dataType is : %d \n", dataType);
    }
}
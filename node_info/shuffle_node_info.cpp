#include "shuffle_node_info.hpp"
#include "utils.hpp"

namespace tensorrtInference
{
    // Shuffle Node
    ShuffleNodeInfo::ShuffleNodeInfo()
    {
        setNodeType("Shuffle");
        setSubNodeType("");
    }
    ShuffleNodeInfo::~ShuffleNodeInfo()
    {  

    }
    bool ShuffleNodeInfo::parseNodeInfoFromJson(std::string type, Json::Value &root)
    {
        setSubNodeType(type);
        auto inputSize = root["inputs"].size();
        CHECK_ASSERT(inputSize <= 2, "Shuffle node must less than 2 inputs\n");
        for(int i = 0; i < inputSize; i++)
        {
            addInput(root["inputs"][i].asString());
        }
        auto outputSize = root["outputs"].size();
        CHECK_ASSERT(outputSize == 1, "Shuffle node must have 1 output\n");
        auto nodeOutputs = getOutputs();
        for(int i = 0; i < outputSize; i++)
        {
            addOutput(root["outputs"][i].asString());
        }
        auto attr = root["attributes"];
        for (auto elem : attr.getMemberNames())
        {
            if(elem.compare("perm") == 0)
            {
                auto size = attr[elem].size();
                for(int i = 0; i < size; i++)
                {
                    perm.push_back(attr[elem][i].asInt());
                }
            }
            else
            {
                LOG("current Shuffle node not support %s \n", elem.c_str());
            }
        }
        return true;
    }
    void ShuffleNodeInfo::printNodeInfo()
    {
        nodeInfo::printNodeInfo();
        LOG("node attribute is as follows:\n");
        LOG("----perm is : ");
        for(int i = 0; i < perm.size(); i++) {
            LOG("%d ", perm[i]);
        }
        LOG("\n");
    }        
}
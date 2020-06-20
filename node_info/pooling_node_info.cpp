#include "pooling_node_info.hpp"
#include "utils.hpp"

namespace tensorrtInference
{
    // Pooling Node
    PoolingNodeInfo::PoolingNodeInfo()
    {
        kernelShape.clear();
        pads.clear();
        strides.clear();
        setNodeType("Pooling");
        setSubNodeType("");
    }
    PoolingNodeInfo::~PoolingNodeInfo()
    {
        kernelShape.clear();
        pads.clear();
        strides.clear();
    }
    bool PoolingNodeInfo::parseNodeInfoFromJson(std::string type, Json::Value &root)
    {
        setSubNodeType(type);
        auto inputSize = root["inputs"].size();
        CHECK_ASSERT(inputSize == 1, "Pooling node must have 1 inputs\n");
        for(int i = 0; i < inputSize; i++)
        {
            addInput(root["inputs"][i].asString());
        }
        auto outputSize = root["outputs"].size();
        CHECK_ASSERT(outputSize == 1, "Pooling node must have 1 output\n");
        auto nodeOutputs = getOutputs();
        for(int i = 0; i < outputSize; i++)
        {
            addOutput(root["outputs"][i].asString());
        }
        auto attr = root["attributes"];
        for (auto elem : attr.getMemberNames())
        {
            if(elem.compare("kernel_shape") == 0)
            {
                auto size = attr[elem].size();
                for(int i = 0; i < size; i++)
                {
                    kernelShape.push_back(attr[elem][i].asInt());
                }
            }
            else if(elem.compare("pads") == 0)
            {
                auto size = attr[elem].size();
                for(int i = 0; i < size; i++)
                {
                    pads.push_back(attr[elem][i].asInt());
                }
            }
            else if(elem.compare("strides") == 0)
            {
                auto size = attr[elem].size();
                for(int i = 0; i < size; i++)
                {
                    strides.push_back(attr[elem][i].asInt());
                }
            }
            else
            {
                LOG("currnet Pooling node not support %s \n", elem.c_str());
            }
        }
        return true;
    }
    void PoolingNodeInfo::printNodeInfo()
    {
        nodeInfo::printNodeInfo();
        LOG("node attribute is as follows:\n");
        LOG("----kernelShape is : ");
        for(int i = 0; i < kernelShape.size(); i++) {
            LOG("%d ", kernelShape[i]);
        }
        LOG("\n----pads is : ");
        for(int i = 0; i < pads.size(); i++) {
            LOG("%d ", pads[i]);  
        }
        LOG("\n----strides is : ");
        for(int i = 0; i < strides.size(); i++) {
            LOG("%d ", strides[i]);  
        }
        LOG("\n");
    }    
}
#include "conv2d_node_info.hpp"
#include "utils.hpp"

namespace tensorrtInference
{
    // Conv2d Node
    Conv2dNodeInfo::Conv2dNodeInfo()
    {
        group = 0;
        kernel_shape.clear();
        pads.clear();
        strides.clear();
        dilation.clear();
        setNodeType("Conv2d");
        setSubNodeType("");
    }
    Conv2dNodeInfo::~Conv2dNodeInfo()
    {
        group = 0;
        kernel_shape.clear();
        pads.clear();
        strides.clear();
        dilation.clear();
    }
    bool Conv2dNodeInfo::parseNodeInfoFromJson(std::string type, Json::Value &root)
    {
        setSubNodeType(type);
        auto inputSize = root["inputs"].size();
        CHECK_ASSERT(inputSize == 3, "conv2d node must have 3 inputs\n");
        for(int i = 0; i < inputSize; i++)
        {
            addInput(root["inputs"][i].asString());
        }
        auto outputSize = root["outputs"].size();
        CHECK_ASSERT(outputSize == 1, "conv2d node must have 1 output\n");
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
                    kernel_shape.push_back(attr[elem][i].asInt());
                }
            }
            else if(elem.compare("dilations") == 0 )
            {
                auto size = attr[elem].size();
                for(int i = 0; i < size; i++)
                {
                    dilation.push_back(attr[elem][i].asInt());
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
            else if(elem.compare("group") == 0)
            {
                auto size = attr[elem].size();
                CHECK_ASSERT(size <= 1, "conv2d node's group must less than 1 element\n");
                if(size)
                    group = attr[elem][0].asInt();
            }
            else if(elem.compare("pads") == 0)
            {
                auto size = attr[elem].size();
                for(int i = 0; i < size; i++)
                {
                    pads.push_back(attr[elem][i].asInt());
                }                
            }
            else
            {
                LOG("currnet conv2d node not support %s \n", elem.c_str());
            }
        }
        return true;
    }
    void Conv2dNodeInfo::printNodeInfo()
    {
        nodeInfo::printNodeInfo();
        LOG("node attribute is as follows:\n");
        LOG("----group is %d \n", group);
        LOG("----kernel_shape is : ");
        for(int i = 0; i < kernel_shape.size(); i++) {
            LOG("%d ", kernel_shape[i]);  
        }
        LOG("\n----pads is : ");
        for(int i = 0; i < pads.size(); i++) {
            LOG("%d ", pads[i]);  
        }
        LOG("\n----stride is : ");
        for(int i = 0; i < strides.size(); i++) {
            LOG("%d ", strides[i]);  
        }
        LOG("\n----dilation is : ");
        for(int i = 0; i < dilation.size(); i++) {
            LOG("%d ", dilation[i]);
        }
        LOG("\n");
    }
}
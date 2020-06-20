
#ifndef __CONV2D_NODE_INFO_HPP__
#define __CONV2D_NODE_INFO_HPP__

#include "node_info.hpp"

namespace tensorrtInference
{
    class Conv2dNodeInfo : public nodeInfo
    {
    public:
        Conv2dNodeInfo();
        ~Conv2dNodeInfo();
        virtual bool parseNodeInfoFromJson(std::string type, Json::Value &root) override;
        void printNodeInfo();
        int getGroup() { return group; }
        std::vector<int> getKernelShape() { return kernel_shape; }
        std::vector<int> getPads() { return pads; }
        std::vector<int> getStrides() { return strides; }
        std::vector<int> getDilation() { return dilation; }
    private:
        int group;
        std::vector<int> kernel_shape;
        std::vector<int> pads;
        std::vector<int> strides;
        std::vector<int> dilation;
    };
} // tensorrtInference
#endif //__CONV2D_NODE_INFO_HPP__
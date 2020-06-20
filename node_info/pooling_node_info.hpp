
#ifndef __POOLING_NODE_INFO_HPP__
#define __POOLING_NODE_INFO_HPP__

#include "node_info.hpp"

namespace tensorrtInference
{
    class PoolingNodeInfo : public nodeInfo
    {
    public:
        PoolingNodeInfo();
        ~PoolingNodeInfo();
        virtual bool parseNodeInfoFromJson(std::string type, Json::Value &root) override;
        void printNodeInfo();
        std::vector<int> getKernelShape() { return kernelShape; }
        std::vector<int> getPads() { return pads; }
        std::vector<int> getStrides() { return strides; }
    private:
        std::vector<int> kernelShape;
        std::vector<int> pads;
        std::vector<int> strides;
    };
} // tensorrtInference
#endif //__POOLING_NODE_INFO_HPP__
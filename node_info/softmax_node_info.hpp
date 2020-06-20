
#ifndef __SOFTMAX_NODE_INFO_HPP__
#define __SOFTMAX_NODE_INFO_HPP__

#include "node_info.hpp"

namespace tensorrtInference
{
    class SoftmaxNodeInfo : public nodeInfo
    {
    public:
        SoftmaxNodeInfo();
        ~SoftmaxNodeInfo();
        virtual bool parseNodeInfoFromJson(std::string type, Json::Value &root) override;
        void printNodeInfo();
        int getAxis(){return axis;}
    private:
        int axis;
    };
} // tensorrtInference
#endif //__SOFTMAX_NODE_INFO_HPP__
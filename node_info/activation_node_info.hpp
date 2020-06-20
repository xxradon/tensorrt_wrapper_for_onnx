
#ifndef __CLIP_NODE_INFO_HPP__
#define __CLIP_NODE_INFO_HPP__

#include "node_info.hpp"

namespace tensorrtInference
{
    class ActivationNodeInfo : public nodeInfo
    {
    public:
        ActivationNodeInfo();
        ~ActivationNodeInfo();
        virtual bool parseNodeInfoFromJson(std::string type, Json::Value &root) override;
    };
} // tensorrtInference
#endif // __CLIP_NODE_INFO_HPP__
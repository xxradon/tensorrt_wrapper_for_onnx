
#ifndef __PADDING_NODE_INFO_HPP__
#define __PADDING_NODE_INFO_HPP__

#include "node_info.hpp"

namespace tensorrtInference
{
    class PaddingNodeInfo : public nodeInfo
    {
    public:
        PaddingNodeInfo();
        ~PaddingNodeInfo();
        virtual bool parseNodeInfoFromJson(std::string type, Json::Value &root) override;
    private:

    };
} //tensorrtInference
#endif // __PADDING_NODE_INFO_HPP__
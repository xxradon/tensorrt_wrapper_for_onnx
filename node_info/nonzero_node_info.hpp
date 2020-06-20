
#ifndef __NONZERO_NODE_INFO_HPP__
#define __NONZERO_NODE_INFO_HPP__

#include "node_info.hpp"

namespace tensorrtInference
{
    class NonZeroNodeInfo : public nodeInfo
    {
    public:
        NonZeroNodeInfo();
        ~NonZeroNodeInfo();
        virtual bool parseNodeInfoFromJson(std::string type, Json::Value &root) override;
    private:
    };
} // tensorrtInference
#endif //__NONZERO_NODE_INFO_HPP__
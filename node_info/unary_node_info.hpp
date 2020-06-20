
#ifndef __UNARY_NODE_INFO_HPP__
#define __UNARY_NODE_INFO_HPP__

#include "node_info.hpp"

namespace tensorrtInference
{
    class UnaryNodeInfo : public nodeInfo
    {
    public:
        UnaryNodeInfo();
        ~UnaryNodeInfo();
        virtual bool parseNodeInfoFromJson(std::string type, Json::Value &root) override;
    private:
        
    };
} // tensorrtInference
#endif //__UNARY_NODE_INFO_HPP__
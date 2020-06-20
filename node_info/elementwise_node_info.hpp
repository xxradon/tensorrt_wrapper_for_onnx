
#ifndef __ELEMENTWISE_NODE_INFO_HPP__
#define __ELEMENTWISE_NODE_INFO_HPP__

#include "node_info.hpp"

namespace tensorrtInference
{
    class ElementWiseNodeInfo : public nodeInfo
    {
    public:
        ElementWiseNodeInfo();
        ~ElementWiseNodeInfo();
        virtual bool parseNodeInfoFromJson(std::string type, Json::Value &root) override;
    private:
        
    };
} // tensorrtInference
#endif //__ELEMENTWISE_NODE_INFO_HPP__
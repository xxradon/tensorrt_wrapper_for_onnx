
#ifndef __IDENTITY_NODE_INFO_HPP__
#define __IDENTITY_NODE_INFO_HPP__

#include "node_info.hpp"

namespace tensorrtInference
{
    class IdentityNodeInfo : public nodeInfo
    {
    public:
        IdentityNodeInfo();
        ~IdentityNodeInfo();
        virtual bool parseNodeInfoFromJson(std::string type, Json::Value &root) override;
        void printNodeInfo();
        int getDataType() {return dataType;}
    private:
        int dataType;
    };
} // tensorrtInference
#endif //__IDENTITY_NODE_INFO_HPP__
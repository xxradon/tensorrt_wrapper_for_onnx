
#ifndef __SLICE_NODE_INFO_HPP__
#define __SLICE_NODE_INFO_HPP__

#include "node_info.hpp"

namespace tensorrtInference
{
    class SliceNodeInfo : public nodeInfo
    {
    public:
        SliceNodeInfo();
        ~SliceNodeInfo();
        virtual bool parseNodeInfoFromJson(std::string type, Json::Value &root) override;
    private:
        
    };
} // tensorrtInference
#endif //__SLICE_NODE_INFO_HPP__

#ifndef __SHUFFLE_NODE_INFO_HPP__
#define __SHUFFLE_NODE_INFO_HPP__

#include "node_info.hpp"

namespace tensorrtInference
{
    class ShuffleNodeInfo : public nodeInfo
    {
    public:
        ShuffleNodeInfo();
        ~ShuffleNodeInfo();
        virtual bool parseNodeInfoFromJson(std::string type, Json::Value &root) override;
        void printNodeInfo();
        std::vector<int> getPerm() { return perm; }
    private:
        std::vector<int> perm;
    };
} // tensorrtInference
#endif // __SHUFFLE_NODE_INFO_HPP__

#ifndef __REDUCE_NODE_INFO_HPP__
#define __REDUCE_NODE_INFO_HPP__

#include "node_info.hpp"

namespace tensorrtInference
{
    class ReduceNodeInfo : public nodeInfo
    {
    public:
        ReduceNodeInfo();
        ~ReduceNodeInfo();
        virtual bool parseNodeInfoFromJson(std::string type, Json::Value &root) override;
        void printNodeInfo();
        std::vector<int> getAxes() {return axes;}
        bool getKeepdims() {return keepdims == 1;}
    private:
        std::vector<int> axes;
        int keepdims;
    };
} // tensorrtInference
#endif //__REDUCE_NODE_INFO_HPP__
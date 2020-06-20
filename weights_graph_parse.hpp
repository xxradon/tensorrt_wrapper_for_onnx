#ifndef __WEIGHTS_GRAPH_PARSE_HPP__
#define __WEIGHTS_GRAPH_PARSE_HPP__
#include <iostream>
#include <string>
#include <vector>
#include <map>
#include "utils.hpp"
#include "json/json.h"
#include "node_info.hpp"

namespace tensorrtInference
{
    struct weightInfo {
        weightInfo() {
            byteCount = 0;
            dataType = 0;
            data = nullptr;
            shape.clear();
        }
        int byteCount;
        int dataType;
        std::vector<int> shape;
        char* data;
        
    };

    class weightsAndGraphParse {
    public:
        weightsAndGraphParse(std::string &jsonFile, std::string &weightsFile);
        ~weightsAndGraphParse();
        std::vector<std::string>& getNetInputBlobNames();
        std::vector<std::string>& getNetOutputBlobNames();
        const std::vector<std::string>& getTopoNodeOrder();
        const std::map<std::string, char*>& getWeightsData();
        const std::map<std::string, weightInfo>& getWeightsInfo();
        const std::map<std::string, std::shared_ptr<nodeInfo>>& getNodeInfoMap();
        std::vector<std::string> getConstWeightTensorNames();
        bool getInitFlag() {return initFlag;}
        bool getFp16Flag() {return fp16Flag;}
    private:
        bool extractNodeInfo(Json::Value &root);
        void generateNodeTypeMap();
        std::vector<std::string> topoNodeOrder;
        std::map<std::string, std::shared_ptr<nodeInfo>> nodeInfoMap;
        std::map<std::string, weightInfo> netWeightsInfo;
        std::vector<std::string> inputTensorNames;
        std::vector<std::string> outputTensorNames;
        std::map<std::string, char*> weightsData;
        std::map<std::string, NodeType> mapStringToNodeType;
        bool initFlag = false;
        bool fp16Flag = false;
    };
} //tensorrtInference

#endif //__WEIGHTS_GRAPH_PARSE_HPP__
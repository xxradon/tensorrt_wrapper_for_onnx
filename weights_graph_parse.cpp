#include "weights_graph_parse.hpp"
#include <fstream>
using namespace std;

namespace tensorrtInference
{
    weightsAndGraphParse::weightsAndGraphParse(std::string &jsonFile, std::string &weightsFile)
    {
        ifstream jsonStream;
        jsonStream.open(jsonFile);
        if(!jsonStream.is_open())
        {
            std::cout << "open json file " << jsonFile << " fail!!!" << std::endl;
            return;
        }
        ifstream weightStream;
        weightStream.open(weightsFile, ios::in | ios::binary);
        if(!weightStream.is_open())
        {
            jsonStream.close();
            std::cout << "open weights file " << weightsFile << " fail!!!" << std::endl;
            return;
        }

        Json::Reader reader;
        Json::Value root;
        if (!reader.parse(jsonStream, root, false))
        {
            jsonStream.close();
            weightStream.close();
            return;
        }
        //generate string to NodeType map
        {
            generateNodeTypeMap();
        }
        //get fp16 flag
        {
            fp16Flag = root["fp16_flag"].asBool();
        }
        //extract topo node order
        {
            int size = root["topo_order"].size();
            for(int i = 0; i < size; i++)
            {
                std::string nodeName;
                nodeName = root["topo_order"][i].asString();
                topoNodeOrder.push_back(nodeName);
            }
        }
        // //extrac depend nodes info
        // {
        //     auto depend_nodes = root["depend_nodes"];
        //     for (auto elem : depend_nodes.getMemberNames())
        //     {
        //         int size = depend_nodes[elem].size();
        //         std::vector<std::string> nodeNames;
        //         for(int i = 0; i < size; i++)
        //         {
        //             auto nodeName = depend_nodes[elem][i].asString();
        //             nodeNames.push_back(nodeName);
        //         }
        //         dependNodes[elem] = nodeNames;
        //     }
        // }
        //extract weight info 
        {
            auto weihtsInfo = root["weights_info"];
            weightInfo nodeWeightInfo;
            for (auto elem : weihtsInfo.getMemberNames())
            {
                if(elem.compare("net_output") != 0)
                {
                    auto offset = weihtsInfo[elem]["offset"].asInt();
                    auto byteCount  = weihtsInfo[elem]["count"].asInt();
                    auto dataType = weihtsInfo[elem]["data_type"].asInt();
                    std::vector<int> shape;
                    int size = weihtsInfo[elem]["tensor_shape"].size();
                    for(int i = 0; i < size; i++)
                    {
                        auto dim = weihtsInfo[elem]["tensor_shape"][i].asInt();
                        shape.push_back(dim);
                    }
                    nodeWeightInfo.byteCount = byteCount;
                    nodeWeightInfo.dataType = dataType;
                    nodeWeightInfo.shape = shape;
                    char* data = nullptr;
                    if(offset != -1)
                    {
                        data = (char*)malloc(byteCount);
                        CHECK_ASSERT(data, "malloc memory fail!!!!\n");
                        weightStream.seekg(offset, ios::beg);
                        weightStream.read(data, byteCount);
                        weightsData[elem] = data;
                    }
                    nodeWeightInfo.data = data;
                    netWeightsInfo[elem] = nodeWeightInfo;
                    if(offset == -1)
                    {
                        inputTensorNames.push_back(elem);
                    }
                }
                else
                {
                    int size = weihtsInfo[elem].size();
                    for(int i = 0; i < size; i++)
                    {
                        std::string tensorName;
                        tensorName = weihtsInfo[elem][i].asString();
                        outputTensorNames.push_back(tensorName);
                    }
                }
            }
        }
        // extra node info 
        initFlag = extractNodeInfo(root["nodes_info"]);
        jsonStream.close();
        weightStream.close();
        return;
    }
    weightsAndGraphParse::~weightsAndGraphParse()
    {
        for(auto it : weightsData) {
            if(it.second != nullptr)
                free(it.second);
        }
        weightsData.clear();
    }
    void weightsAndGraphParse::generateNodeTypeMap()
    {
        mapStringToNodeType["Conv"]       = NodeType::Conv2d;
        mapStringToNodeType["Add"]        = NodeType::ElementWise;
        mapStringToNodeType["Sub"]        = NodeType::ElementWise;
        mapStringToNodeType["Mul"]        = NodeType::ElementWise;
        mapStringToNodeType["Div"]        = NodeType::ElementWise;
        mapStringToNodeType["Max"]        = NodeType::ElementWise;
        mapStringToNodeType["Equal"]      = NodeType::ElementWise;
        mapStringToNodeType["Greater"]    = NodeType::ElementWise;
        mapStringToNodeType["Clip"]       = NodeType::Activation;
        mapStringToNodeType["Reshape"]    = NodeType::Shuffle;
        mapStringToNodeType["Transpose"]  = NodeType::Shuffle;
        mapStringToNodeType["Pad"]        = NodeType::Padding;
        mapStringToNodeType["Sqrt"]       = NodeType::Unary;
        mapStringToNodeType["Reciprocal"] = NodeType::Unary;
        mapStringToNodeType["Abs"]        = NodeType::Unary;
        mapStringToNodeType["Softmax"]    = NodeType::Softmax;
        mapStringToNodeType["ReduceSum"]  = NodeType::Reduce;
        mapStringToNodeType["MaxPool"]    = NodeType::Pooling;
        mapStringToNodeType["Slice"]      = NodeType::Slice;
        mapStringToNodeType["Cast"]       = NodeType::Identity;
        mapStringToNodeType["NonZero"]    = NodeType::NonZero;
    }
    bool weightsAndGraphParse::extractNodeInfo(Json::Value &root)
    {
        for (auto elem : root.getMemberNames()) {
            if(root[elem]["op_type"].isString())
            {
                auto op_type = root[elem]["op_type"].asString();
                // if(op_type.compare("NonZero") == 0)
                //     printf("run here!\n");
                if(mapStringToNodeType.count(op_type) != 0)
                {
                    NodeType nodeType = mapStringToNodeType[op_type];
                    std::shared_ptr<nodeInfo> node;
                    auto curr_node = parseNodeInfoFromJsonFuncArr[nodeType](op_type, root[elem]);
                    if(curr_node == nullptr)
                        return false;
                    // curr_node->printNodeInfo();
                    node.reset(curr_node);
                    nodeInfoMap[elem] = node;
                }
                else
                {
                    LOG("current not support %s node type\n", op_type.c_str());
                }
            }
        }
        return true;
    }

    std::vector<std::string>& weightsAndGraphParse::getNetInputBlobNames()
    {
        return inputTensorNames;
    }
    std::vector<std::string>& weightsAndGraphParse::getNetOutputBlobNames()
    {
        return outputTensorNames;
    }    
    const std::vector<std::string>& weightsAndGraphParse::getTopoNodeOrder()
    {
        return topoNodeOrder;
    }
    const std::map<std::string, char*>& weightsAndGraphParse::getWeightsData()
    {
        return weightsData;
    }
    const std::map<std::string, weightInfo>& weightsAndGraphParse::getWeightsInfo()
    {
        return netWeightsInfo;
    }
    const std::map<std::string, std::shared_ptr<nodeInfo>>& weightsAndGraphParse::getNodeInfoMap()
    {
        return nodeInfoMap;
    }
    std::vector<std::string> weightsAndGraphParse::getConstWeightTensorNames()
    {
        std::vector<std::string> constTensorNames;
        for(auto it : nodeInfoMap)
        {
            auto nodeType = it.second->getNodeType();
            auto subNodeType = it.second->getSubNodeType();
            // LOG("node type: %s , sub node type: %s\n", nodeType.c_str(), subNodeType.c_str());
            // if(nodeType.compare("ElementWise") == 0 || subNodeType.compare("Reshape") == 0)
            if(nodeType.compare("ElementWise") == 0)
            {
                auto inputs = it.second->getInputs();
                int size = inputs.size();
                for(int i = 0; i < size; i++)
                {
                    if(netWeightsInfo.count(inputs[i]))
                    {
                        auto weight = netWeightsInfo[inputs[i]];
                        if(weight.byteCount == 0)
                            continue;
                        else
                            constTensorNames.push_back(inputs[i]);
                    }
                }
            }
        }
        return constTensorNames;
    }

}
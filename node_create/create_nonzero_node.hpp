#ifndef __CREATE_NONZERO_NODE_HPP__
#define __CREATE_NONZERO_NODE_HPP__

//plugin currently not support DataType::kBOOL
namespace tensorrtInference
{
    extern nvinfer1::ILayer* createNonZeroNode(nvinfer1::INetworkDefinition* network, std::map<std::string, nvinfer1::ITensor*>& tensors,  
        tensorrtInference::nodeInfo* nodeConfInfo, std::map<std::string, tensorrtInference::weightInfo>& nodeWeightsInfo);
}

#endif
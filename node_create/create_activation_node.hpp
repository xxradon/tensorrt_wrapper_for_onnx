#ifndef __CREATE_ACTIVATION_NODE_HPP__
#define __CREATE_ACTIVATION_NODE_HPP__


namespace tensorrtInference
{
    extern nvinfer1::ILayer* createActivationNode(nvinfer1::INetworkDefinition* network, std::map<std::string, nvinfer1::ITensor*>& tensors, 
        tensorrtInference::nodeInfo* nodeConfInfo, std::map<std::string, tensorrtInference::weightInfo>& nodeWeightsInfo);
}

#endif
#ifndef __CREATE_SOFTMAX_NODE_HPP__
#define __CREATE_SOFTMAX_NODE_HPP__


namespace tensorrtInference
{
    extern nvinfer1::ILayer* createSoftmaxNode(nvinfer1::INetworkDefinition* network, std::map<std::string, nvinfer1::ITensor*>& tensors,  
        tensorrtInference::nodeInfo* nodeConfInfo, std::map<std::string, tensorrtInference::weightInfo>& nodeWeightsInfo);
}

#endif
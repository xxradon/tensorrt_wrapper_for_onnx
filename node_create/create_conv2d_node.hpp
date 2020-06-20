#ifndef __CREATE_CONV2D_NODE_HPP__
#define __CREATE_CONV2D_NODE_HPP__


namespace tensorrtInference
{
    extern nvinfer1::ILayer* createConv2dNode(nvinfer1::INetworkDefinition* network, std::map<std::string, nvinfer1::ITensor*>& tensors,  
        tensorrtInference::nodeInfo* nodeConfInfo, std::map<std::string, tensorrtInference::weightInfo>& nodeWeightsInfo);
}

#endif
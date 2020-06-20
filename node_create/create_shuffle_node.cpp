#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "weights_graph_parse.hpp"
#include "create_node.hpp"
#include "shuffle_node_info.hpp"
namespace tensorrtInference
{
    nvinfer1::ILayer* createShuffleNode(nvinfer1::INetworkDefinition* network, std::map<std::string, nvinfer1::ITensor*>& tensors,
        tensorrtInference::nodeInfo* nodeConfInfo, std::map<std::string, tensorrtInference::weightInfo>& nodeWeightsInfo)
    {
        auto shuffleNodeInfo = (ShuffleNodeInfo*)nodeConfInfo;
        auto inputs = shuffleNodeInfo->getInputs();
        nvinfer1::IShuffleLayer* shuffle = nullptr;
        nvinfer1::ITensor* inputTensors = tensors[inputs[0]];
        if(inputs.size() == 1)
        {
            auto perm = shuffleNodeInfo->getPerm();
            shuffle = network->addShuffle(*inputTensors);
            CHECK_ASSERT(shuffle, "create shuffle node fail\n");
            CHECK_ASSERT(perm.size() == 4, "perm dims must equal to 4\n");
            nvinfer1::Dims dims = inputTensors->getDimensions();
            shuffle->setFirstTranspose(nvinfer1::Permutation{perm[0], perm[1], perm[2], perm[3]});
            if(dims.nbDims == 4)
                shuffle->setReshapeDimensions(nvinfer1::Dims4(dims.d[perm[0]], dims.d[perm[1]], dims.d[perm[2]], dims.d[perm[3]]));
            else
                CHECK_ASSERT(0, "current only support 4 dims in transpose\n");
        }
        else
        {
            auto dims = parseIntArrayValue(nodeWeightsInfo[inputs[1]].dataType, nodeWeightsInfo[inputs[1]].data, 
                            nodeWeightsInfo[inputs[1]].byteCount, nodeWeightsInfo[inputs[1]].shape);
            shuffle = network->addShuffle(*inputTensors);
            CHECK_ASSERT(shuffle, "create shuffle node fail\n");
            auto tensorDims = inputTensors->getDimensions();
            if(dims.size() == 4 && tensorDims.nbDims == 4)
                shuffle->setReshapeDimensions(nvinfer1::Dims4(dims[0], dims[1], dims[2], dims[3]));
            else if(dims.size() == 3 && tensorDims.nbDims == 3)
                shuffle->setReshapeDimensions(nvinfer1::DimsCHW(dims[0], dims[1], dims[2]));
            else if(dims.size() == 2 && tensorDims.nbDims == 2)
                shuffle->setReshapeDimensions(nvinfer1::DimsHW(dims[0], dims[1]));
            else if(dims.size() < tensorDims.nbDims)
            {
                nvinfer1::Dims temp;
                temp.nbDims = tensorDims.nbDims;
                for(int i = 0; i < temp.nbDims; i++)
                {
                    if(i < dims.size())
                        temp.d[i] = 1;
                    else
                        temp.d[i] = dims[i-dims.size()];
                }
                shuffle->setReshapeDimensions(temp);
            }
            else
            {
                CHECK_ASSERT(0, "current only support 2 3 or 4 dims for reshape\n");
            }
        }
        return shuffle;
    }
}
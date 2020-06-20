#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "weights_graph_parse.hpp"
#include "create_node.hpp"
#include "create_softmax_node.hpp"
#include "softmax_node_info.hpp"

namespace tensorrtInference
{
    nvinfer1::ILayer* createSoftmaxNode(nvinfer1::INetworkDefinition* network, std::map<std::string, nvinfer1::ITensor*>& tensors,
        tensorrtInference::nodeInfo* nodeConfInfo, std::map<std::string, tensorrtInference::weightInfo>& nodeWeightsInfo)
    {
        auto softmaxNodeInfo = (SoftmaxNodeInfo*)nodeConfInfo;
        auto inputs = softmaxNodeInfo->getInputs();
        int axes = softmaxNodeInfo->getAxis();
        // CHECK_ASSERT(axes >= 0, "axes only support positive\n");
        nvinfer1::ITensor* inputTensor = tensors[inputs[0]];
        nvinfer1::Dims dims = inputTensor->getDimensions();
        nvinfer1::ISoftMaxLayer* softmax = network->addSoftMax(*inputTensor);
        CHECK_ASSERT(softmax, "create softmax node fail\n");
        if(axes < 0)
        {
            axes = dims.nbDims + axes;
            CHECK_ASSERT(axes >= 0, "axes value wrong\n");
        }
        softmax->setAxes(1 << axes);
        return softmax;
    }
}
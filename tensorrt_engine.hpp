#ifndef __TENSORRT_ENGINE_HPP__
#define __TENSORRT_ENGINE_HPP__

#include <iostream>
#include <string>
#include <vector>
#include <map>
#include "utils.hpp"
#include "weights_graph_parse.hpp"
#include "cuda_runtime_api.h"

using namespace nvinfer1;
using namespace std;

namespace tensorrtInference
{
    class tensorrtEngine
    {
    public:
        tensorrtEngine(std::string jsonFile, std::string weightsFile);
        tensorrtEngine(std::string engineFile);
        ~tensorrtEngine();
        bool saveEnginePlanFile(std::string saveFile);
        void doInference(std::vector<void*> data, int batchSize, bool fp16InferenceFlag);
        void createEngine(unsigned int maxBatchSize);
        std::map<std::string, int> getBindingNamesIndexMap();
    private:
        void initConstTensors(std::map<std::string, nvinfer1::ITensor*>& tensors, nvinfer1::INetworkDefinition* network);
        void setNetInput(std::map<std::string, nvinfer1::ITensor*>& tensors, nvinfer1::INetworkDefinition* network);
        void createNetBackbone(std::map<std::string, nvinfer1::ITensor*>& tensors, nvinfer1::INetworkDefinition* network);
        std::vector<int> getBindingByteCount();
        void preprocessInputData();
        void postprocessOutputData();
        // void setNetOutput(nvinfer1::INetworkDefinition* network, nvinfer1::DataType dataType);
        Logger mLogger;
        std::shared_ptr<tensorrtInference::weightsAndGraphParse> weightsAndGraph;
        nvinfer1::IRuntime* runtime;
        nvinfer1::IBuilder* builder;
        nvinfer1::ICudaEngine* cudaEngine;
        nvinfer1::IExecutionContext* context;
    };
}

#endif
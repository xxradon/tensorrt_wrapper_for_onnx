#include "tensorrt_engine.hpp"
#include "create_node.hpp"
#include "nonzero_cuda_impl.hpp"
#include "convert_cuda_impl.hpp"
#include <fstream>

using namespace std;

namespace tensorrtInference 
{
    tensorrtEngine::tensorrtEngine(std::string jsonFile, std::string weightsFile)
    {
        builder = nullptr;
        cudaEngine = nullptr;
        runtime = nullptr;
        context = nullptr;
        builder = createInferBuilder(mLogger);
        CHECK_ASSERT(builder != nullptr, "create builder fail!\n");
        weightsAndGraph.reset(new weightsAndGraphParse(jsonFile, weightsFile));
        CHECK_ASSERT((weightsAndGraph.get()->getInitFlag() != false), "init jsonFile and weightsFile fail!!\n");
        createEngine(1);
    }
    tensorrtEngine::tensorrtEngine(std::string engineFile)
    {
        char *trtModelStream = nullptr;
        size_t size = 0;
        std::ifstream file(engineFile.c_str(), std::ios::binary);
        if (file.good()) {
            file.seekg(0, file.end);
            size = file.tellg();
            file.seekg(0, file.beg);
            trtModelStream = new char[size];
            CHECK_ASSERT(trtModelStream, "malloc fail !\n");
            file.read(trtModelStream, size);
            file.close();
        }
        IRuntime* runtime = createInferRuntime(mLogger);
        CHECK_ASSERT(runtime != nullptr, "create runtime fail!\n");
        cudaEngine = runtime->deserializeCudaEngine(trtModelStream, size, nullptr);
        CHECK_ASSERT(cudaEngine != nullptr, "create engine fail!\n");
        context = cudaEngine->createExecutionContext();
        CHECK_ASSERT(context != nullptr, "create context fail!\n");
        delete[] trtModelStream;
    }
    tensorrtEngine::~tensorrtEngine()
    {
        if(builder != nullptr)
            builder->destroy();
        if(context != nullptr)
            context->destroy();
        if(cudaEngine != nullptr)
            cudaEngine->destroy();
        if(runtime != nullptr)
            runtime->destroy();
    }
    bool tensorrtEngine::saveEnginePlanFile(std::string saveFile)
    {
        IHostMemory* modelStream = nullptr;
        if(cudaEngine == nullptr)
        {
            LOG("please create net engine first!\n");
            return false;
        }
        // Serialize the engine
        modelStream = cudaEngine->serialize();
        std::ofstream plan(saveFile);
        if (!plan)
        {
            LOG("could not open plan engine file\n");
            return false;
        }
        plan.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
        if(modelStream != nullptr)
            modelStream->destroy();
        return true;
    }

    void tensorrtEngine::initConstTensors(std::map<std::string, nvinfer1::ITensor*> &tensors, nvinfer1::INetworkDefinition* network)
    {
        auto constWeightTensors = weightsAndGraph.get()->getConstWeightTensorNames();
        auto weightsInfo = weightsAndGraph.get()->getWeightsInfo();
        auto size = constWeightTensors.size();
        for(int i = 0; i < size; i++)
        {
            if(tensors.count(constWeightTensors[i]))
                continue;
            LOG("create const tensor %s \n", constWeightTensors[i].c_str());
            auto shape = weightsInfo[constWeightTensors[i]].shape;
            CHECK_ASSERT((shape.size() <= 4), "const tensor shape must less than 3!\n");
            int count = 1;
            for(int j = 0; j < shape.size(); j++)
                count *= shape[j];
            
            nvinfer1::DataType dataType = (weightsInfo[constWeightTensors[i]].dataType == tensorrtInference::OnnxDataType::FLOAT) ? 
                                nvinfer1::DataType::kFLOAT : nvinfer1::DataType::kHALF;
            nvinfer1::Weights weights{dataType, weightsInfo[constWeightTensors[i]].data, count};
            nvinfer1::ILayer* constLayer = nullptr;
            if(shape.size() == 4 && shape[0] == 1)
            {
                nvinfer1::Dims dims;
                dims.nbDims = 4;
                dims.d[0] = shape[0];
                dims.d[1] = shape[1];
                dims.d[2] = shape[2];
                dims.d[3] = shape[3];
                constLayer = network->addConstant(dims, weights);
                // constLayer = network->addConstant(nvinfer1::Dims4(shape[0], shape[1], shape[2], shape[3]), weights);
            }
            else if(shape.size() == 3)
            {
                nvinfer1::Dims dims;
                dims.nbDims = 4;
                dims.d[0] = 1;
                dims.d[1] = shape[0];
                dims.d[2] = shape[1];
                dims.d[3] = shape[2];
                constLayer = network->addConstant(dims, weights);
                // constLayer = network->addConstant(nvinfer1::Dims4(1, shape[0], shape[1], shape[2]), weights);
            }
            else if(shape.size() == 2)
            {
                nvinfer1::Dims dims;
                dims.nbDims = 4;
                dims.d[0] = 1;
                dims.d[1] = 1;
                dims.d[2] = shape[0];
                dims.d[3] = shape[1];
                constLayer = network->addConstant(dims, weights);
                // constLayer = network->addConstant(nvinfer1::Dims4(1, 1, shape[0], shape[1]), weights);
            }
            else if(shape.size() == 1)
            {
                nvinfer1::Dims dims;
                dims.nbDims = 4;
                dims.d[0] = 1;
                dims.d[1] = 1;
                dims.d[2] = 1;
                dims.d[3] = shape[0];
                constLayer = network->addConstant(dims, weights);
                // constLayer = network->addConstant(nvinfer1::Dims4(1, 1, 1, shape[0]), weights);
            }
            else
                LOG("const tensor shape size is %d (%d %d %d %d), \n", shape.size(), shape[0], shape[1], shape[2], shape[3]);
            CHECK_ASSERT(constLayer, "create const tensor (%s) fail\n");
            tensors[constWeightTensors[i]] = constLayer->getOutput(0);
        }
    }
    void tensorrtEngine::setNetInput(std::map<std::string, nvinfer1::ITensor*> &tensors, nvinfer1::INetworkDefinition* network)
    {
        int channel, height, width;
        auto inputBlobNames = weightsAndGraph.get()->getNetInputBlobNames();
        auto weightsInfo = weightsAndGraph.get()->getWeightsInfo();
        int size = inputBlobNames.size();
        for(int i = 0; i < size; i++)
        {
            auto shape = weightsInfo[inputBlobNames[i]].shape;
            if(shape.size() != 4 || inputBlobNames[i].compare("") == 0)
            {
                LOG("input blob shape or input blob name error!\n");
            }
            channel = shape[1];
            height = shape[2];
            width = shape[3];
            nvinfer1::DataType dataType = (weightsInfo[inputBlobNames[i]].dataType == tensorrtInference::OnnxDataType::FLOAT) ? 
                                nvinfer1::DataType::kFLOAT : nvinfer1::DataType::kHALF;
            
            nvinfer1::ITensor* data = network->addInput(inputBlobNames[i].c_str(), dataType, nvinfer1::Dims4{1, channel, height, width});
            CHECK_ASSERT(data!=nullptr, "setNetInput fail\n");
            tensors[inputBlobNames[i]] = data;
        }
    }
    void tensorrtEngine::createNetBackbone(std::map<std::string, nvinfer1::ITensor*>& tensors, 
        nvinfer1::INetworkDefinition* network)
    {
        auto topoOrder = weightsAndGraph.get()->getTopoNodeOrder();
        auto weightsInfo = weightsAndGraph.get()->getWeightsInfo();
        auto nodeInfoMap = weightsAndGraph.get()->getNodeInfoMap();
        std::map<std::string, nvinfer1::ILayer*> netNode;
        for(int i = 0; i < topoOrder.size(); i++)
        {
            std::string nodeName = topoOrder[i];
            LOG("create %s node\n", nodeName.c_str());
            // if(nodeName.compare("prefix/pred/global_head/vlad/Reshape") == 0)
            //     LOG("run here\n");
            auto nodeConfigInfo = nodeInfoMap[nodeName];
            nvinfer1::ILayer* layer = createNode(network, tensors, nodeConfigInfo.get(), weightsInfo);
            layer->setName(nodeName.c_str());
            CHECK_ASSERT(layer != nullptr, "create %s node fail\n", nodeName);
            netNode[nodeName] = layer;
            auto outputs = nodeConfigInfo.get()->getOutputs();
            for(int i = 0; i < outputs.size(); i++)
            {
                tensors[outputs[i]] = layer->getOutput(i);
                nvinfer1::ITensor *tensor = layer->getOutput(i);
                tensor->setName(outputs[i].c_str());
                nvinfer1::Dims dims = layer->getOutput(i)->getDimensions();
                if(dims.nbDims == 4)
                    LOG("tensor %s  shape is %d %d %d %d\n", outputs[i].c_str(), dims.d[0], dims.d[1], dims.d[2], dims.d[3]);
            }
        }
    }
    void tensorrtEngine::createEngine(unsigned int maxBatchSize)
    {
        bool ret = true;
        std::map<std::string, nvinfer1::ITensor*> tensors;
        nvinfer1::INetworkDefinition* network = builder->createNetwork();

        //init constant tensors
        initConstTensors(tensors, network);
        
        //set network input tensor
        setNetInput(tensors, network);
        
        //set network backbone 
        createNetBackbone(tensors, network);

        //mark network output
        auto outputTensors = weightsAndGraph.get()->getNetOutputBlobNames();
        for(int i = 0; i < outputTensors.size(); i++)
        {
            nvinfer1::ITensor* tensor = tensors[outputTensors[i]];
            network->markOutput(*tensor);
        }
        
        // Build engine
        builder->setMaxBatchSize(maxBatchSize);
        builder->setMaxWorkspaceSize(1 << 20);
        builder->setFp16Mode(true);
        cudaEngine = builder->buildCudaEngine(*network);
        CHECK_ASSERT(cudaEngine != nullptr, "createEngine fail!\n");
        LOG("createEngine success!\n");

        // Don't need the network any more
        network->destroy();
    }
    std::map<std::string, int> tensorrtEngine::getBindingNamesIndexMap()
    {
        std::map<std::string, int> bindingNamesIndexMap;
        if(cudaEngine == nullptr)
            LOG("create engine first!\n");
        int nbBinding = cudaEngine->getNbBindings();
        for(int i = 0; i < nbBinding; i++)
        {
            std::string tensorName(cudaEngine->getBindingName(i));
            bindingNamesIndexMap[tensorName] = i;
        }
        return bindingNamesIndexMap;
    }
    std::vector<int> tensorrtEngine::getBindingByteCount()
    {
        std::vector<int> byteCount;
        const nvinfer1::ICudaEngine& engine = context->getEngine();
        int nbBinding = engine.getNbBindings();
        for(int i = 0; i < nbBinding; i++)
        {
            int totalByteCount = 1;
            int eleSize = 1;
            nvinfer1::Dims dims = engine.getBindingDimensions(i);
            nvinfer1::DataType dataType = engine.getBindingDataType(i);
            int eleByteCount = 1;
            if(dataType == nvinfer1::DataType::kFLOAT)
                eleByteCount = 4;
            else if(dataType == nvinfer1::DataType::kHALF)
                eleByteCount = 2;
            else if(dataType == nvinfer1::DataType::kINT32)
                eleByteCount = 4;
            else if(dataType == nvinfer1::DataType::kBOOL)
                eleByteCount = 1;                
            else
                CHECK_ASSERT(0, "current only support float/half!\n");
            for(int i = 0; i < dims.nbDims; i++) {
                eleSize *= dims.d[i];
            }
            totalByteCount = eleSize * eleByteCount;
            byteCount.push_back(totalByteCount);
        }
        return byteCount;
    }
    void tensorrtEngine::doInference(std::vector<void*> data, int batchSize, bool fp16InferenceFlag)
    {
        const nvinfer1::ICudaEngine& engine = context->getEngine();
        std::vector<void*> input;
        std::vector<void*> output;
        for(int i = 0; i < data.size(); i++)
        {
            if(engine.bindingIsInput(i) == true)
                input.push_back(data[i]);
            else
                output.push_back(data[i]);
        }
        int inputSize = input.size();
        int outputSize = output.size();
        // Pointers to input and output device buffers to pass to engine.
        // Engine requires exactly IEngine::getNbBindings() number of buffers.
        CHECK_ASSERT(engine.getNbBindings() == (inputSize + outputSize), "binding size must be equal to inputSize + outputSize \n");
        std::vector<void*> bufferArr(inputSize + outputSize);
        std::vector<void*> bufferFp16Arr(inputSize + outputSize);
        std::vector<int> bindingByteCount = getBindingByteCount();
        for(int i = 0; i < inputSize; i++)
        {
            void* buffer = nullptr;
            int count = bindingByteCount[i];
            cudaError_t cudastatus = cudaMalloc(&buffer, batchSize * count);
            CHECK_ASSERT(cudastatus == cudaSuccess, "net input tensor %s malloc fail: %s\n", engine.getBindingName(i),
                         cudaGetErrorString(cudastatus));
            bufferArr[i] = buffer;
            if(fp16InferenceFlag)
            {
                buffer = nullptr;
                cudastatus = cudaMalloc(&buffer, batchSize * count * 2);
                CHECK_ASSERT(cudastatus == cudaSuccess, "fp16 inference : net input tensor %s malloc fail: %s\n", engine.getBindingName(i),
                            cudaGetErrorString(cudastatus));
                bufferFp16Arr[i] = buffer;
            }
        }
        for(int i = inputSize; i < inputSize + outputSize; i++)
        {
            void* buffer = nullptr;
            int count = bindingByteCount[i];
            cudaError_t cudastatus = cudaMalloc(&buffer, batchSize * count);
            CHECK_ASSERT(cudastatus == cudaSuccess, "net output tensor %s malloc fail: %s\n", engine.getBindingName(i),
                         cudaGetErrorString(cudastatus));
            bufferArr[i] = buffer;
            if(fp16InferenceFlag)
            {
                buffer = nullptr;
                cudastatus = cudaMalloc(&buffer, batchSize * count * 2);
                CHECK_ASSERT(cudastatus == cudaSuccess, "fp16 inference : net input tensor %s malloc fail: %s\n", engine.getBindingName(i),
                            cudaGetErrorString(cudastatus));
                bufferFp16Arr[i] = buffer;
            }            
        }

        // Create stream
        cudaStream_t stream;
        cudaError_t cudastatus = cudaStreamCreate(&stream);
        CHECK_ASSERT(cudastatus == cudaSuccess, "create cuda stream fail: %s\n", cudaGetErrorString(cudastatus));
        if(fp16InferenceFlag)
        {
            // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
            for(int i = 0; i < inputSize; i++)
            {
                int byteCount = bindingByteCount[i] * batchSize;
                int eleCount = byteCount / 2;
                cudaError_t cudastatus = cudaMemcpyAsync(bufferFp16Arr[i], input[i], byteCount * 2, cudaMemcpyHostToDevice, stream);
                CHECK_ASSERT(cudastatus == cudaSuccess, "tensor %s copy to device fail: %s\n", engine.getBindingName(i),
                            cudaGetErrorString(cudastatus));
                CudaImpl::ConvertFp32ToFp16CudaImpl(bufferFp16Arr[i], eleCount, bufferArr[i], stream);
            }
        }
        else
        {
            // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
            for(int i = 0; i < inputSize; i++)
            {
                int count = bindingByteCount[i];
                cudaError_t cudastatus = cudaMemcpyAsync(bufferArr[i], input[i], batchSize * count, cudaMemcpyHostToDevice, stream);
                CHECK_ASSERT(cudastatus == cudaSuccess, "tensor %s copy to device fail: %s\n", engine.getBindingName(i),
                            cudaGetErrorString(cudastatus));
            }            
        }


        context->enqueue(batchSize, &bufferArr[0], stream, nullptr);

//temp add for hfnet
#if 1
        void* buffer = nullptr;
        int index = getBindingNamesIndexMap()["prefix/pred/keypoint_extraction/Greater:0"];
        cudastatus = cudaMalloc(&buffer, bindingByteCount[index] * sizeof(int));
        CHECK_ASSERT(cudastatus == cudaSuccess, "cudaMalloc execute fail\n");

        if(fp16InferenceFlag)
        {
            CudaImpl::NoneZeroCudaImpl((const unsigned char*)bufferArr[index], bindingByteCount[index], (int*)buffer, stream);
            cudastatus = cudaMemcpyAsync(output[index - inputSize], buffer, bindingByteCount[index] * sizeof(int), cudaMemcpyDeviceToHost, stream);
            CHECK_ASSERT(cudastatus == cudaSuccess, "cudaMemcpyAsync execute fail\n");
            for(int i = 0; i < outputSize; i++)
            {
                if((i + inputSize) == index)
                    continue;
                int bufferIndex = inputSize + i;
                int byteCount = bindingByteCount[bufferIndex];
                CudaImpl::ConvertFp16ToFp32CudaImpl(bufferArr[bufferIndex], byteCount / 2, bufferFp16Arr[bufferIndex], stream);
                cudastatus = cudaMemcpyAsync(output[i], bufferFp16Arr[bufferIndex], batchSize * byteCount * 2 , cudaMemcpyDeviceToHost, stream);
                CHECK_ASSERT(cudastatus == cudaSuccess, "tensor %s copy from device fail: %s\n", engine.getBindingName(bufferIndex),
                            cudaGetErrorString(cudastatus));
            }
        }
        else
        {
            CudaImpl::NoneZeroCudaImpl((const unsigned char*)bufferArr[index], bindingByteCount[index], (int*)buffer, stream);
            cudastatus = cudaMemcpyAsync(output[index - inputSize], buffer, bindingByteCount[index] * sizeof(int), cudaMemcpyDeviceToHost, stream);
            CHECK_ASSERT(cudastatus == cudaSuccess, "cudaMemcpyAsync execute fail\n");
            for(int i = 0; i < outputSize; i++)
            {
                if((i + inputSize) == index)
                    continue;
                int bufferIndex = inputSize + i;
                int count = bindingByteCount[bufferIndex];
                cudastatus = cudaMemcpyAsync(output[i], bufferArr[bufferIndex], batchSize * count, cudaMemcpyDeviceToHost, stream);
                CHECK_ASSERT(cudastatus == cudaSuccess, "tensor %s copy from device fail: %s\n", engine.getBindingName(bufferIndex),
                            cudaGetErrorString(cudastatus));
            }
        }
        cudaStreamSynchronize(stream);
        // Release stream and buffers
        cudaStreamDestroy(stream);
        cudaFree(buffer);
        if(fp16InferenceFlag)
        {
            for(int i = 0; i < inputSize + outputSize; i++)
            {
                cudaFree(bufferFp16Arr[i]);
            }
        }
        for(int i = 0; i < inputSize + outputSize; i++)
        {
            cudaFree(bufferArr[i]);
        }
#else
        for(int i = 0; i < outputSize; i++)
        {
            int bufferIndex = inputSize + i;
            int count = bindingByteCount[bufferIndex];
            cudaError_t cudastatus = cudaMemcpyAsync(output[i], bufferArr[bufferIndex], batchSize * count, cudaMemcpyDeviceToHost, stream);
            CHECK_ASSERT(cudastatus == cudaSuccess, "tensor %s copy from device fail: %s\n", engine.getBindingName(bufferIndex),
                         cudaGetErrorString(cudastatus));
        }
        cudaStreamSynchronize(stream);
        // Release stream and buffers
        cudaStreamDestroy(stream);

        //
        cudaFree(bufferArr[i]);
        for(int i = 0; i < inputSize + outputSize; i++)
        {
            cudaFree(bufferArr[i]);
        }        
#endif
    }
}
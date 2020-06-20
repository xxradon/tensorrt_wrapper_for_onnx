#ifndef __NONZERO_PLUGIN_HPP__
#define __NONZERO_PLUGIN_HPP__

#include <string>
#include <vector>
#include "NvInfer.h"
#include "NvInferPlugin.h"


class NonZeroPlugin: public nvinfer1::IPluginV2IOExt
{
    public:
        explicit NonZeroPlugin();
        NonZeroPlugin(const void* data, size_t length);

        ~NonZeroPlugin();

        int getNbOutputs() const override;

        nvinfer1::Dims getOutputDimensions(int index, const nvinfer1::Dims* inputs, int nbInputDims) override;

        int initialize() override;

        virtual void terminate() override;

        virtual size_t getWorkspaceSize(int maxBatchSize) const override;

        virtual int enqueue(int batchSize, const void*const * inputs, void** outputs, void* workspace, cudaStream_t stream) override;

        virtual size_t getSerializationSize() const override;

        virtual void serialize(void* buffer) const override;

        bool supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) const override {
            if(pos == 0)
                return inOut[pos].format == nvinfer1::TensorFormat::kLINEAR && inOut[pos].type == nvinfer1::DataType::kBOOL;
            else
                return inOut[pos].format == nvinfer1::TensorFormat::kLINEAR && inOut[pos].type == nvinfer1::DataType::kINT32;
        }

        const char* getPluginType() const override;

        const char* getPluginVersion() const override;

        void destroy() override;

        nvinfer1::IPluginV2IOExt* clone() const override;

        void setPluginNamespace(const char* pluginNamespace) override;

        const char* getPluginNamespace() const override;

        nvinfer1::DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const override;

        bool isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const override;

        bool canBroadcastInputAcrossBatch(int inputIndex) const override;

        void attachToContext(cudnnContext* cudnnContext, cublasContext* cublasContext, nvinfer1::IGpuAllocator* gpuAllocator) override;

        void configurePlugin(const nvinfer1::PluginTensorDesc* in, int nbInput, const nvinfer1::PluginTensorDesc* out, int nbOutput) override;

        void detachFromContext() override;

        int inputSize;
    private:
        void forwardGpu(const unsigned char *const * inputs, int* output, cudaStream_t stream, int batchSize = 1);
        const char* mPluginNamespace;
};

class NonZeroPluginCreator : public nvinfer1::IPluginCreator
{
    public:
        NonZeroPluginCreator();

        ~NonZeroPluginCreator() override = default;

        const char* getPluginName() const override;

        const char* getPluginVersion() const override;

        const nvinfer1::PluginFieldCollection* getFieldNames() override;

        nvinfer1::IPluginV2IOExt* createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc) override;

        nvinfer1::IPluginV2IOExt* deserializePlugin(const char* name, const void* serialData, size_t serialLength) override;

        void setPluginNamespace(const char* libNamespace) override
        {
            mNamespace = libNamespace;
        }

        const char* getPluginNamespace() const override
        {
            return mNamespace.c_str();
        }

    private:
        std::string mNamespace;
        nvinfer1::PluginFieldCollection mFC;
        std::vector<nvinfer1::PluginField> mPluginAttributes;
};

#endif 

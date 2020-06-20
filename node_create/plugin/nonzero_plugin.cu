#include <cmath>
#include <stdio.h>
#include <cassert>
#include <iostream>
#include "utils.hpp"
#include "nonzero_plugin.hpp"
#include <cub/cub.cuh>

#define THREADS_PER_BLOCK 256

NonZeroPlugin::NonZeroPlugin()
{
}

NonZeroPlugin::~NonZeroPlugin()
{
}

// create the plugin at runtime from a byte stream
NonZeroPlugin::NonZeroPlugin(const void* data, size_t length)
{
    assert(length == sizeof(inputSize));
    inputSize = *reinterpret_cast<const int*>(data);
}


/********************************************************************************
/////////////////inherited from nvinfer1::IPluginV2Ext///////////////////////////
********************************************************************************/
// Return the DataType of the plugin output at the requested index
nvinfer1::DataType NonZeroPlugin::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const
{
    return nvinfer1::DataType::kINT32;
}

// Return true if output tensor is broadcast across a batch.
bool NonZeroPlugin::isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const
{
    return false;
}

// Return true if plugin can use input that is broadcast across batch without replication.
bool NonZeroPlugin::canBroadcastInputAcrossBatch(int inputIndex) const
{
    return false;
}

void NonZeroPlugin::configurePlugin(const nvinfer1::PluginTensorDesc* in, int nbInput, const nvinfer1::PluginTensorDesc* out, int nbOutput)
{
    // int pos = 0;
    // in[pos]->format = nvinfer1::DataType::kINT8;
    // return;
}

// Attach the plugin object to an execution context and grant the plugin the access to some context resource.
void NonZeroPlugin::attachToContext(cudnnContext* cudnnContext, cublasContext* cublasContext, nvinfer1::IGpuAllocator* gpuAllocator)
{
}

// Detach the plugin object from its execution context.
void NonZeroPlugin::detachFromContext()
{
    return;
}

// Clone the plugin
nvinfer1::IPluginV2IOExt* NonZeroPlugin::clone() const
{
    NonZeroPlugin *p = new NonZeroPlugin();
    p->inputSize = inputSize;
    p->setPluginNamespace(mPluginNamespace);
    return p;
}
/********************************************************************************
/////////////////inherited from nvinfer1::IPluginV2//////////////////////////////
********************************************************************************/

const char* NonZeroPlugin::getPluginType() const
{
    return "NonZero_TRT";
}

const char* NonZeroPlugin::getPluginVersion() const
{
    return "1";
}

int NonZeroPlugin::getNbOutputs() const
{
    return 1;
}

nvinfer1::Dims NonZeroPlugin::getOutputDimensions(int index, const nvinfer1::Dims* inputs, int nbInputDims)
{
    CHECK_ASSERT(nbInputDims == 1 && index == 0, "nonzero only support 1 input!\n");
    inputSize = 1;
    nvinfer1::Dims dims;
    for(int i = 0; i < inputs[0].nbDims; i++)
    {
        inputSize *= inputs[0].d[i];
        dims.d[i] = inputs[0].d[i];
    }
    dims.nbDims = inputs[0].nbDims;
    // Output dimensions
    return dims;
}

int NonZeroPlugin::initialize()
{ 
    return 0;
}
void NonZeroPlugin::terminate()
{
    return;
}

size_t NonZeroPlugin::getWorkspaceSize(int maxBatchSize) const
{
    return 0;
}

cudaError_t NonZeroCalcPrefixSumTempStorageBytes(int* prefix_counts, int number_of_blocks,
            size_t& temp_storage_bytes, cudaStream_t stream)
{
  temp_storage_bytes = 0;
  return cub::DeviceScan::InclusiveSum(
      nullptr, temp_storage_bytes, prefix_counts, prefix_counts, number_of_blocks, stream);
}

cudaError_t NonZeroInclusivePrefixSum(void* d_temp_storage, size_t temp_storage_bytes, int* prefix_counts,
             int number_of_blocks, cudaStream_t stream)
{
  return cub::DeviceScan::InclusiveSum(
      d_temp_storage, temp_storage_bytes, prefix_counts, prefix_counts, number_of_blocks, stream);
}

__global__ void NonZeroCountEachBlockKernel(const unsigned char* x, int x_size, int* count_in_blocks) {
    typedef cub::BlockReduce<int, THREADS_PER_BLOCK, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY> BlockReduceT;
    __shared__ typename BlockReduceT::TempStorage temp_storage;
  
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int nz = 0;
    if (index < x_size && x[index] == 1)
        ++nz;

    int count = BlockReduceT(temp_storage).Sum(nz);
  
    if (threadIdx.x == 0) {
      count_in_blocks[blockIdx.x] = count;
    }
}

__global__ void NonZeroOutputPositionsKernel(const unsigned char* x, const int x_size, const int* prefix_counts, int* results)
{
    typedef cub::BlockScan<int, THREADS_PER_BLOCK> BlockScanT;
    __shared__ typename BlockScanT::TempStorage temp_storage;

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int nz = 0;
    if (index < x_size && x[index] == 1)
        ++nz;
    int pos_in_block = 0;
    BlockScanT(temp_storage).InclusiveSum(nz, pos_in_block);

    int result_position = ((blockIdx.x == 0) ? 0 : prefix_counts[blockIdx.x - 1]) + pos_in_block - nz;

    if (index < x_size && x[index] == 1) {
        results[result_position] = index;
        // printf("result_position %d index %d\n");
    }
}

void NonZeroPlugin::forwardGpu(const unsigned char *const * inputs, int* output, cudaStream_t stream, int batchSize) {
    const int threadCount = 128;
    int blockSize = threadCount;
    int gridSize = (inputSize * batchSize + blockSize - 1) / blockSize;
    int index = inputSize / 2;
    size_t tempStorageBytes = 0;
    NonZeroCountEachBlockKernel<<<gridSize, blockSize, 0, stream>>>(inputs[0], inputSize * batchSize, output + index);
    NonZeroCalcPrefixSumTempStorageBytes(output + index, gridSize, tempStorageBytes, stream);
    NonZeroInclusivePrefixSum(output + index + gridSize, tempStorageBytes, output + index, gridSize, stream);
    NonZeroOutputPositionsKernel<<<gridSize, blockSize, 0, stream>>>(inputs[0], inputSize * batchSize, output + index, output);
}

int NonZeroPlugin::enqueue(int batchSize, const void*const * inputs, void** outputs, void* workspace, cudaStream_t stream)
{
    //assert(batchSize == 1);
    //GPU
    forwardGpu((const unsigned char *const *)inputs, (int*)outputs[0], stream, batchSize);
    return 0;
}

size_t NonZeroPlugin::getSerializationSize() const
{  
    return sizeof(inputSize);
}

void NonZeroPlugin::serialize(void* buffer) const
{
    *reinterpret_cast<int*>(buffer) = inputSize;
}

void NonZeroPlugin::destroy()
{
    delete this;
}

// Set plugin namespace
void NonZeroPlugin::setPluginNamespace(const char* pluginNamespace)
{
    mPluginNamespace = pluginNamespace;
}

const char* NonZeroPlugin::getPluginNamespace() const
{
    return mPluginNamespace;
}



/********************************************************************************
/////////////////inherited from nvinfer1::IPluginCreator/////////////////////////
********************************************************************************/
NonZeroPluginCreator::NonZeroPluginCreator()
{
    mPluginAttributes.clear();
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* NonZeroPluginCreator::getPluginName() const
{
    return "NonZero_TRT";
}

const char* NonZeroPluginCreator::getPluginVersion() const
{
    return "1";
}

const nvinfer1::PluginFieldCollection* NonZeroPluginCreator::getFieldNames()
{
    return &mFC;
}

nvinfer1::IPluginV2IOExt* NonZeroPluginCreator::createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc)
{
    NonZeroPlugin* obj = new NonZeroPlugin();
    obj->setPluginNamespace(name);
    return obj;
}

nvinfer1::IPluginV2IOExt* NonZeroPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength)
{
    // This object will be deleted when the network is destroyed, which will
    // call NonZeroPlugin::destroy()
    NonZeroPlugin* obj = new NonZeroPlugin(serialData, serialLength);
    obj->setPluginNamespace(name);
    return obj;
}


REGISTER_TENSORRT_PLUGIN(NonZeroPluginCreator);
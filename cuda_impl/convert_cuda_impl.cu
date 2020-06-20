#include <cmath>
#include <stdio.h>
#include <cassert>
#include <iostream>
#include "utils.hpp"
#include <cuda_fp16.h>
#include "convert_cuda_impl.hpp"
#define THREADS_PER_BLOCK 256
#define ELEMENTS_PER_THREAD 4


namespace CudaImpl
{

    __global__ void fp32ToFp16Kernel(const float* input_data, const int input_size, half* output_data)
    {
        int start = ELEMENTS_PER_THREAD * THREADS_PER_BLOCK * blockIdx.x + threadIdx.x;
        float value[ELEMENTS_PER_THREAD];
      
        int id = start;
        #pragma unroll
        for (int i = 0; i < ELEMENTS_PER_THREAD; i++)
        {
            if (id < input_size)
            {
                value[i] = input_data[id];
                id += THREADS_PER_BLOCK;
            }
        }
      
        id = start;
        #pragma unroll
        for (int i = 0; i < ELEMENTS_PER_THREAD; i++)
        {
            if (id < input_size)
            {
                output_data[id] = half(value[i]);
                id += THREADS_PER_BLOCK;
            }
        }        
    }

    __global__ void fp16ToFp32Kernel(const half* input_data, const int input_size, float* output_data)
    {
        int start = ELEMENTS_PER_THREAD * THREADS_PER_BLOCK * blockIdx.x + threadIdx.x;
        half value[ELEMENTS_PER_THREAD];
      
        int id = start;
        #pragma unroll
        for (int i = 0; i < ELEMENTS_PER_THREAD; i++)
        {
            if (id < input_size)
            {
                value[i] = input_data[id];
                id += THREADS_PER_BLOCK;
            }
        }
      
        id = start;
        #pragma unroll
        for (int i = 0; i < ELEMENTS_PER_THREAD; i++)
        {
            if (id < input_size)
            {
                output_data[id] = float(value[i]);
                id += THREADS_PER_BLOCK;
            }
        }        
    }

    void ConvertFp16ToFp32CudaImpl(const void* input, const int inputSize, void* output, cudaStream_t stream)
    {
        int blockSize = THREADS_PER_BLOCK;
        int gridSize = inputSize / (blockSize * ELEMENTS_PER_THREAD) + 1;
        fp16ToFp32Kernel<<<gridSize, blockSize, 0, stream>>>((half*)input, inputSize, (float*)output);
        cudaError_t cudastatus = cudaGetLastError();
        CHECK_ASSERT(cudastatus == cudaSuccess, "launch failed: %s\n", cudaGetErrorString(cudastatus));
    }

    void ConvertFp32ToFp16CudaImpl(const void* input, const int inputSize, void* output, cudaStream_t stream)
    {
        int blockSize = THREADS_PER_BLOCK;
        int gridSize = inputSize / (blockSize * ELEMENTS_PER_THREAD) + 1;
        fp32ToFp16Kernel<<<gridSize, blockSize, 0, stream>>>((float*)input, inputSize, (half*)output);
        cudaError_t cudastatus = cudaGetLastError();
        CHECK_ASSERT(cudastatus == cudaSuccess, "launch failed: %s\n", cudaGetErrorString(cudastatus));
    }    
}

#ifndef __CONVERT_CUDA_IMPL_HPP__
#define __CONVERT_CUDA_IMPL_HPP__

#include <string>
#include <vector>
#include <cuda_runtime_api.h>

namespace CudaImpl
{
    extern void ConvertFp16ToFp32CudaImpl(const void* input, const int inputSize, void* output, cudaStream_t stream);
    extern void ConvertFp32ToFp16CudaImpl(const void* input, const int inputSize, void* output, cudaStream_t stream);
}


#endif 

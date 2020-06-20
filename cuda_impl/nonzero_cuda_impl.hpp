#ifndef __NONZERO_CUDA_IMPL_HPP__
#define __NONZERO_CUDA_IMPL_HPP__

#include <string>
#include <vector>
#include <cuda_runtime_api.h>

namespace CudaImpl
{
    extern void NoneZeroCudaImpl(const unsigned char* inputs, const int inputSize, int* output, cudaStream_t stream);
}


#endif 

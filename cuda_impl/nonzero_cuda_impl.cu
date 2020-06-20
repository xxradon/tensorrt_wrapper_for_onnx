#include <cmath>
#include <stdio.h>
#include <cassert>
#include <iostream>
#include "utils.hpp"
#include <cub/cub.cuh>
#include "nonzero_cuda_impl.hpp"
#define THREADS_PER_BLOCK 256


namespace CudaImpl
{

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

    __global__ void NonZeroCountEachBlockKernel(const unsigned char* x, int x_size, int* count_in_blocks)
    {
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

    void NoneZeroCudaImpl(const unsigned char* input, const int inputSize, int* output, cudaStream_t stream)
    {
        const int threadCount = THREADS_PER_BLOCK;
        int blockSize = threadCount;
        int gridSize = (inputSize + blockSize - 1) / blockSize;
        int index = inputSize / 2;
        size_t tempStorageBytes = 0;
        NonZeroCountEachBlockKernel<<<gridSize, blockSize, 0, stream>>>(input, inputSize, output + index);
        NonZeroCalcPrefixSumTempStorageBytes(output + index, gridSize, tempStorageBytes, stream);
        NonZeroInclusivePrefixSum(output + index + gridSize, tempStorageBytes, output + index, gridSize, stream);
        NonZeroOutputPositionsKernel<<<gridSize, blockSize, 0, stream>>>(input, inputSize, output + index, output);
        cudaError_t cudastatus = cudaGetLastError();
        CHECK_ASSERT(cudastatus == cudaSuccess, "launch failed: %s\n", cudaGetErrorString(cudastatus));
    }
}

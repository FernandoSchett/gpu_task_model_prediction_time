#ifndef CUDA_KERNELS_CUH
#define CUDA_KERNELS_CUH

#include "config.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

struct CudaKernelResources {
    unsigned long long *sink = nullptr;
    float *memory_buffer = nullptr;
    std::size_t memory_element_count = 0;
};

std::size_t default_kernel_memory_elements();

cudaError_t allocate_kernel_resources(CudaKernelResources *resources,
                                      std::size_t requested_memory_elements);
cudaError_t free_kernel_resources(CudaKernelResources *resources);

cudaError_t launch_configurable_kernel(cudaStream_t stream,
                                       const CudaKernelResources &resources,
                                       KernelType kernel_type,
                                       std::uint64_t target_duration_us,
                                       int device_clock_rate_khz,
                                       int blocks_x,
                                       int threads_per_block,
                                       int grid_z);

#endif

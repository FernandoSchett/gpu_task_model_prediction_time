#ifndef CUDA_KERNELS_CUH
#define CUDA_KERNELS_CUH

#include <cuda_runtime.h>

#include <cstdint>

cudaError_t allocate_busy_wait_sink(unsigned long long **sink);
cudaError_t free_busy_wait_sink(unsigned long long *sink);

cudaError_t launch_busy_wait_kernel(cudaStream_t stream,
                                    unsigned long long *sink,
                                    std::uint64_t target_duration_us,
                                    int device_clock_rate_khz,
                                    int blocks_x,
                                    int threads_per_block,
                                    int grid_z);

#endif

#include "cuda_kernels.cuh"

#include <algorithm>

namespace {

__global__ void busy_wait_kernel(unsigned long long target_cycles, unsigned long long *sink) {
    const unsigned long long start = clock64();
    unsigned long long elapsed = 0;
    unsigned long long value =
        static_cast<unsigned long long>(blockIdx.x) +
        static_cast<unsigned long long>(gridDim.x) * static_cast<unsigned long long>(blockIdx.z) +
        static_cast<unsigned long long>(threadIdx.x);

    while (elapsed < target_cycles) {
        elapsed = clock64() - start;
        value ^= elapsed + static_cast<unsigned long long>(threadIdx.x + 1);
    }

    if (threadIdx.x == 0) {
        atomicAdd(sink, value | 1ULL);
    }
}

unsigned long long microseconds_to_cycles(std::uint64_t duration_us, int clock_rate_khz) {
    if (duration_us == 0 || clock_rate_khz <= 0) {
        return 0ULL;
    }

    const unsigned long long us = static_cast<unsigned long long>(duration_us);
    const unsigned long long khz = static_cast<unsigned long long>(clock_rate_khz);
    return std::max(1ULL, (us * khz + 999ULL) / 1000ULL);
}

}  // namespace

cudaError_t allocate_busy_wait_sink(unsigned long long **sink) {
    const cudaError_t alloc_error = cudaMalloc(reinterpret_cast<void **>(sink), sizeof(unsigned long long));
    if (alloc_error != cudaSuccess) {
        return alloc_error;
    }
    return cudaMemset(*sink, 0, sizeof(unsigned long long));
}

cudaError_t free_busy_wait_sink(unsigned long long *sink) {
    if (sink == nullptr) {
        return cudaSuccess;
    }
    return cudaFree(sink);
}

cudaError_t launch_busy_wait_kernel(cudaStream_t stream,
                                    unsigned long long *sink,
                                    std::uint64_t target_duration_us,
                                    int device_clock_rate_khz,
                                    int blocks_x,
                                    int threads_per_block,
                                    int grid_z) {
    const dim3 grid(static_cast<unsigned int>(blocks_x), 1U, static_cast<unsigned int>(grid_z));
    const dim3 block(static_cast<unsigned int>(threads_per_block), 1U, 1U);
    const unsigned long long target_cycles = microseconds_to_cycles(target_duration_us, device_clock_rate_khz);

    busy_wait_kernel<<<grid, block, 0, stream>>>(target_cycles, sink);
    return cudaGetLastError();
}

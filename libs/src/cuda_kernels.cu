#include "cuda_kernels.cuh"

#include <algorithm>
#include <cstddef>

namespace {

constexpr std::size_t kDefaultMemoryElements = 1ULL << 20;

__device__ __forceinline__ unsigned long long linear_thread_id() {
    return (static_cast<unsigned long long>(blockIdx.z) * static_cast<unsigned long long>(gridDim.x) +
            static_cast<unsigned long long>(blockIdx.x)) *
               static_cast<unsigned long long>(blockDim.x) +
           static_cast<unsigned long long>(threadIdx.x);
}

__device__ __forceinline__ unsigned long long grid_thread_count() {
    return static_cast<unsigned long long>(gridDim.x) *
           static_cast<unsigned long long>(gridDim.z) *
           static_cast<unsigned long long>(blockDim.x);
}

__device__ __forceinline__ unsigned long long initial_memory_index(unsigned long long tid,
                                                                   unsigned long long memory_mask) {
    return (tid * 1315423911ULL) & memory_mask;
}

__device__ __forceinline__ unsigned long long memory_access_stride(unsigned long long total_threads,
                                                                   unsigned long long memory_mask) {
    return (total_threads * 17ULL + 1ULL) & memory_mask;
}

__device__ __forceinline__ unsigned long long next_memory_index(unsigned long long index,
                                                                unsigned long long stride,
                                                                unsigned long long elapsed,
                                                                unsigned long long memory_mask) {
    return (index + stride + (elapsed & 1023ULL)) & memory_mask;
}

__device__ __forceinline__ void publish_thread_value(unsigned long long value,
                                                     unsigned long long *sink) {
    if (threadIdx.x == 0) {
        atomicAdd(sink, value | 1ULL);
    }
}

__global__ void busy_wait_kernel(unsigned long long target_cycles, unsigned long long *sink) {
    const unsigned long long start = clock64();
    unsigned long long elapsed = 0;
    unsigned long long value = linear_thread_id();

    while (elapsed < target_cycles) {
        elapsed = clock64() - start;
        value ^= elapsed + static_cast<unsigned long long>(threadIdx.x + 1);
    }

    publish_thread_value(value, sink);
}

__global__ void compute_kernel(unsigned long long target_cycles, unsigned long long *sink) {
    const unsigned long long start = clock64();
    unsigned long long elapsed = 0;
    const unsigned long long tid = linear_thread_id();
    float x = static_cast<float>((tid & 255ULL) + 1ULL) * 0.001f;
    float y = static_cast<float>(((tid >> 8) & 255ULL) + 1ULL) * 0.002f;

    while (elapsed < target_cycles) {
#pragma unroll
        for (int i = 0; i < 32; ++i) {
            x = fmaf(x, 1.000001f, y + 0.000001f);
            y = fmaf(y, 0.999999f, x + 0.000002f);
        }
        elapsed = clock64() - start;
    }

    publish_thread_value(static_cast<unsigned long long>(__float_as_uint(x + y)), sink);
}

__global__ void memory_kernel(unsigned long long target_cycles,
                              float *memory_buffer,
                              unsigned long long memory_mask,
                              unsigned long long *sink) {
    const unsigned long long start = clock64();
    unsigned long long elapsed = 0;
    const unsigned long long tid = linear_thread_id();
    const unsigned long long total_threads = grid_thread_count();
    unsigned long long index = initial_memory_index(tid, memory_mask);
    const unsigned long long stride = memory_access_stride(total_threads, memory_mask);
    float acc = static_cast<float>((tid & 1023ULL) + 1ULL);
    volatile float *volatile_memory = memory_buffer;

    while (elapsed < target_cycles) {
        index = next_memory_index(index, stride, elapsed, memory_mask);
        const float loaded = volatile_memory[index];
        const float updated = loaded + acc * 0.000001f + 1.0f;
        volatile_memory[index] = updated;
        acc += updated;
        elapsed = clock64() - start;
    }

    publish_thread_value(static_cast<unsigned long long>(__float_as_uint(acc)), sink);
}

__global__ void mixed_kernel(unsigned long long target_cycles,
                             float *memory_buffer,
                             unsigned long long memory_mask,
                             unsigned long long *sink) {
    const unsigned long long start = clock64();
    unsigned long long elapsed = 0;
    const unsigned long long tid = linear_thread_id();
    const unsigned long long total_threads = grid_thread_count();
    unsigned long long index = initial_memory_index(tid, memory_mask);
    const unsigned long long stride = memory_access_stride(total_threads, memory_mask);
    float x = static_cast<float>((tid & 511ULL) + 1ULL) * 0.003f;
    float y = static_cast<float>(((tid >> 9) & 511ULL) + 1ULL) * 0.004f;
    volatile float *volatile_memory = memory_buffer;

    while (elapsed < target_cycles) {
#pragma unroll
        for (int i = 0; i < 12; ++i) {
            x = fmaf(x, 1.000003f, y + 0.000003f);
            y = fmaf(y, 0.999997f, x + 0.000004f);
        }
        index = next_memory_index(index, stride, elapsed, memory_mask);
        const float loaded = volatile_memory[index];
        const float updated = loaded + x * 0.000001f + y * 0.000002f;
        volatile_memory[index] = updated;
        x += loaded * 0.000001f;
        elapsed = clock64() - start;
    }

    publish_thread_value(static_cast<unsigned long long>(__float_as_uint(x + y)), sink);
}

unsigned long long microseconds_to_cycles(std::uint64_t duration_us, int clock_rate_khz) {
    if (duration_us == 0 || clock_rate_khz <= 0) {
        return 0ULL;
    }

    const unsigned long long us = static_cast<unsigned long long>(duration_us);
    const unsigned long long khz = static_cast<unsigned long long>(clock_rate_khz);
    return std::max(1ULL, (us * khz + 999ULL) / 1000ULL);
}

std::size_t next_power_of_two(std::size_t value) {
    if (value <= 1) {
        return 1;
    }

    --value;
    for (std::size_t shift = 1; shift < sizeof(std::size_t) * 8; shift <<= 1) {
        value |= value >> shift;
    }
    return value + 1;
}

cudaError_t first_error(cudaError_t first, cudaError_t second) {
    return first != cudaSuccess ? first : second;
}

}  // namespace

std::size_t default_kernel_memory_elements() {
    return kDefaultMemoryElements;
}

cudaError_t allocate_kernel_resources(CudaKernelResources *resources,
                                      std::size_t requested_memory_elements) {
    if (resources == nullptr) {
        return cudaErrorInvalidValue;
    }

    resources->sink = nullptr;
    resources->memory_buffer = nullptr;
    resources->memory_element_count = next_power_of_two(requested_memory_elements);

    cudaError_t error = cudaMalloc(reinterpret_cast<void **>(&resources->sink),
                                   sizeof(unsigned long long));
    if (error != cudaSuccess) {
        return error;
    }

    error = cudaMemset(resources->sink, 0, sizeof(unsigned long long));
    if (error != cudaSuccess) {
        free_kernel_resources(resources);
        return error;
    }

    error = cudaMalloc(reinterpret_cast<void **>(&resources->memory_buffer),
                       resources->memory_element_count * sizeof(float));
    if (error != cudaSuccess) {
        free_kernel_resources(resources);
        return error;
    }

    error = cudaMemset(resources->memory_buffer,
                      0,
                      resources->memory_element_count * sizeof(float));
    if (error != cudaSuccess) {
        free_kernel_resources(resources);
        return error;
    }

    return cudaSuccess;
}

cudaError_t free_kernel_resources(CudaKernelResources *resources) {
    if (resources == nullptr) {
        return cudaErrorInvalidValue;
    }

    cudaError_t error = cudaSuccess;
    if (resources->memory_buffer != nullptr) {
        error = first_error(error, cudaFree(resources->memory_buffer));
        resources->memory_buffer = nullptr;
    }
    if (resources->sink != nullptr) {
        error = first_error(error, cudaFree(resources->sink));
        resources->sink = nullptr;
    }
    resources->memory_element_count = 0;
    return error;
}

cudaError_t launch_configurable_kernel(cudaStream_t stream,
                                       const CudaKernelResources &resources,
                                       KernelType kernel_type,
                                       std::uint64_t target_duration_us,
                                       int device_clock_rate_khz,
                                       int blocks_x,
                                       int threads_per_block,
                                       int grid_z) {
    if (resources.sink == nullptr) {
        return cudaErrorInvalidDevicePointer;
    }

    const dim3 grid(static_cast<unsigned int>(blocks_x), 1U, static_cast<unsigned int>(grid_z));
    const dim3 block(static_cast<unsigned int>(threads_per_block), 1U, 1U);
    const unsigned long long target_cycles = microseconds_to_cycles(target_duration_us, device_clock_rate_khz);

    switch (kernel_type) {
        case KernelType::BusyWait:
            busy_wait_kernel<<<grid, block, 0, stream>>>(target_cycles, resources.sink);
            break;
        case KernelType::Compute:
            compute_kernel<<<grid, block, 0, stream>>>(target_cycles, resources.sink);
            break;
        case KernelType::Memory:
            if (resources.memory_buffer == nullptr || resources.memory_element_count == 0) {
                return cudaErrorInvalidDevicePointer;
            }
            memory_kernel<<<grid, block, 0, stream>>>(target_cycles,
                                                      resources.memory_buffer,
                                                      static_cast<unsigned long long>(resources.memory_element_count - 1),
                                                      resources.sink);
            break;
        case KernelType::Mixed:
            if (resources.memory_buffer == nullptr || resources.memory_element_count == 0) {
                return cudaErrorInvalidDevicePointer;
            }
            mixed_kernel<<<grid, block, 0, stream>>>(target_cycles,
                                                     resources.memory_buffer,
                                                     static_cast<unsigned long long>(resources.memory_element_count - 1),
                                                     resources.sink);
            break;
    }

    return cudaGetLastError();
}

cudaError_t compute_theoretical_occupancy(KernelType kernel_type,
                                          int threads_per_block,
                                          int sm_count,
                                          int *max_active_blocks_per_sm,
                                          double *theoretical_occupancy) {
    if (max_active_blocks_per_sm == nullptr || theoretical_occupancy == nullptr ||
        threads_per_block <= 0 || sm_count <= 0) {
        return cudaErrorInvalidValue;
    }

    int active_blocks = 0;
    cudaError_t error = cudaSuccess;
    switch (kernel_type) {
        case KernelType::BusyWait:
            error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active_blocks,
                                                                  busy_wait_kernel,
                                                                  threads_per_block,
                                                                  0);
            break;
        case KernelType::Compute:
            error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active_blocks,
                                                                  compute_kernel,
                                                                  threads_per_block,
                                                                  0);
            break;
        case KernelType::Memory:
            error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active_blocks,
                                                                  memory_kernel,
                                                                  threads_per_block,
                                                                  0);
            break;
        case KernelType::Mixed:
            error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active_blocks,
                                                                  mixed_kernel,
                                                                  threads_per_block,
                                                                  0);
            break;
    }
    if (error != cudaSuccess) {
        return error;
    }

    cudaDeviceProp prop{};
    int device_id = 0;
    error = cudaGetDevice(&device_id);
    if (error != cudaSuccess) {
        return error;
    }
    error = cudaGetDeviceProperties(&prop, device_id);
    if (error != cudaSuccess) {
        return error;
    }

    *max_active_blocks_per_sm = active_blocks;
    const int active_threads_per_sm = active_blocks * threads_per_block;
    *theoretical_occupancy =
        prop.maxThreadsPerMultiProcessor > 0
            ? static_cast<double>(active_threads_per_sm) /
                  static_cast<double>(prop.maxThreadsPerMultiProcessor)
            : 0.0;
    return cudaSuccess;
}

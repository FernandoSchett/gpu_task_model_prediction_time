#include "experiment.hpp"

#include "csv_writer.hpp"
#include "cuda_kernels.cuh"
#include "timer.hpp"

#include <cuda_runtime.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <iostream>
#include <mutex>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr std::uint64_t kRankSeedPrime = 1000003ULL;
constexpr std::uint64_t kThreadSeedPrime = 1000033ULL;
constexpr std::uint64_t kWarmupSeedSalt = 0x9E3779B97F4A7C15ULL;

std::string cuda_error_message(cudaError_t error) {
    return std::string(cudaGetErrorString(error));
}

void throw_on_cuda_error(cudaError_t error, const std::string &operation) {
    if (error != cudaSuccess) {
        throw std::runtime_error(operation + " failed: " + cuda_error_message(error));
    }
}

std::string sanitize_filename_component(const std::string &input) {
    std::string output;
    output.reserve(input.size());

    for (const char ch : input) {
        const bool is_alnum =
            (ch >= 'a' && ch <= 'z') ||
            (ch >= 'A' && ch <= 'Z') ||
            (ch >= '0' && ch <= '9');
        if (is_alnum || ch == '-' || ch == '_' || ch == '.') {
            output += ch;
        } else {
            output += '_';
        }
    }

    return output.empty() ? "experiment" : output;
}

std::filesystem::path output_file_path(const ExperimentConfig &config,
                                       int mpi_rank,
                                       const std::string &run_timestamp) {
    const std::string safe_name = sanitize_filename_component(config.experiment_name);
    std::ostringstream filename;
    filename << "resultados_experimentos_"
             << safe_name
             << "_seed_"
             << config.seed
             << "_"
             << run_timestamp
             << "_rank_"
             << mpi_rank
             << ".csv";

    return std::filesystem::path(config.output_dir) / filename.str();
}

std::uint64_t compute_global_kernel_id(const ExperimentConfig &config,
                                       int mpi_rank,
                                       int host_thread_id,
                                       int kernel_index) {
    const std::uint64_t kernels_per_rank =
        static_cast<std::uint64_t>(config.threads_per_process) *
        static_cast<std::uint64_t>(config.kernels_per_thread);
    return static_cast<std::uint64_t>(mpi_rank) * kernels_per_rank +
           static_cast<std::uint64_t>(host_thread_id) * static_cast<std::uint64_t>(config.kernels_per_thread) +
           static_cast<std::uint64_t>(kernel_index);
}

void validate_launch_config(const ExperimentConfig &config, const cudaDeviceProp &prop) {
    if (config.threads_per_block > prop.maxThreadsPerBlock) {
        std::ostringstream out;
        out << "--threads-per-block (" << config.threads_per_block
            << ") exceeds device maxThreadsPerBlock (" << prop.maxThreadsPerBlock << ")";
        throw std::runtime_error(out.str());
    }
    if (config.blocks_x > prop.maxGridSize[0]) {
        std::ostringstream out;
        out << "--blocks-x (" << config.blocks_x
            << ") exceeds device maxGridSize[0] (" << prop.maxGridSize[0] << ")";
        throw std::runtime_error(out.str());
    }
    if (config.grid_z > prop.maxGridSize[2]) {
        std::ostringstream out;
        out << "--grid-z (" << config.grid_z
            << ") exceeds device maxGridSize[2] (" << prop.maxGridSize[2] << ")";
        throw std::runtime_error(out.str());
    }
}

void record_thread_error(std::vector<std::string> &errors,
                         std::mutex &errors_mutex,
                         int host_thread_id,
                         const std::string &message) {
    std::lock_guard<std::mutex> lock(errors_mutex);
    std::ostringstream out;
    out << "host_thread_id=" << host_thread_id << ": " << message;
    errors.push_back(out.str());
}

cudaError_t launch_and_sync(cudaStream_t stream,
                            const CudaKernelResources &resources,
                            KernelType kernel_type,
                            std::uint64_t requested_busy_wait_us,
                            int device_clock_rate_khz,
                            int blocks_x,
                            int threads_per_block,
                            int grid_z) {
    const cudaError_t launch_error = launch_configurable_kernel(stream,
                                                                resources,
                                                                kernel_type,
                                                                requested_busy_wait_us,
                                                                device_clock_rate_khz,
                                                                blocks_x,
                                                                threads_per_block,
                                                                grid_z);
    if (launch_error != cudaSuccess) {
        return launch_error;
    }
    return cudaStreamSynchronize(stream);
}

void run_host_thread(const ExperimentConfig &config,
                     int mpi_world_size,
                     int mpi_rank,
                     int host_thread_id,
                     int device_clock_rate_khz,
                     CsvWriter &writer,
                     std::vector<std::string> &thread_errors,
                     std::mutex &thread_errors_mutex) {
    try {
        throw_on_cuda_error(cudaSetDevice(config.device_id), "cudaSetDevice");

        cudaStream_t stream = nullptr;
        throw_on_cuda_error(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
                            "cudaStreamCreateWithFlags");

        CudaKernelResources resources;
        try {
            throw_on_cuda_error(allocate_kernel_resources(&resources, default_kernel_memory_elements()),
                                "allocate_kernel_resources");

            const std::uint64_t thread_seed =
                config.seed +
                static_cast<std::uint64_t>(mpi_rank) * kRankSeedPrime +
                static_cast<std::uint64_t>(host_thread_id) * kThreadSeedPrime;
            std::mt19937_64 rng(thread_seed);
            std::mt19937_64 warmup_rng(thread_seed ^ kWarmupSeedSalt);
            std::uniform_real_distribution<double> arrival_dist(config.arrival_min_ms,
                                                                config.arrival_max_ms);
            std::uniform_int_distribution<std::uint64_t> kernel_dist(config.kernel_min_us,
                                                                     config.kernel_max_us);

            const std::uint64_t total_blocks =
                static_cast<std::uint64_t>(config.blocks_x) * static_cast<std::uint64_t>(config.grid_z);
            const std::uint64_t total_cuda_threads =
                total_blocks * static_cast<std::uint64_t>(config.threads_per_block);

            for (int warmup_index = 0; warmup_index < config.warmup_kernels; ++warmup_index) {
                const std::uint64_t warmup_duration_us = kernel_dist(warmup_rng);
                const cudaError_t warmup_error = launch_and_sync(stream,
                                                                 resources,
                                                                 config.kernel_type,
                                                                 warmup_duration_us,
                                                                 device_clock_rate_khz,
                                                                 config.blocks_x,
                                                                 config.threads_per_block,
                                                                 config.grid_z);
                throw_on_cuda_error(warmup_error, "warm-up kernel");
            }

            for (int kernel_index = 0; kernel_index < config.kernels_per_thread; ++kernel_index) {
                const double arrival_wait_ms = arrival_dist(rng);
                const std::uint64_t requested_busy_wait_us = kernel_dist(rng);

                std::this_thread::sleep_for(std::chrono::duration<double, std::milli>(arrival_wait_ms));

                const std::int64_t submit_time_ns = Timer::now_ns();
                const cudaError_t launch_error = launch_configurable_kernel(stream,
                                                                            resources,
                                                                            config.kernel_type,
                                                                            requested_busy_wait_us,
                                                                            device_clock_rate_khz,
                                                                            config.blocks_x,
                                                                            config.threads_per_block,
                                                                            config.grid_z);
                const std::int64_t launch_return_time_ns = Timer::now_ns();

                cudaError_t final_error = launch_error;
                if (launch_error == cudaSuccess) {
                    final_error = cudaStreamSynchronize(stream);
                }
                const std::int64_t completion_time_ns = Timer::now_ns();

                KernelRecord record;
                record.experiment_name = config.experiment_name;
                record.global_seed = config.seed;
                record.warmup_kernels = config.warmup_kernels;
                record.mpi_world_size = mpi_world_size;
                record.mpi_rank = mpi_rank;
                record.host_thread_id = host_thread_id;
                record.kernel_index_in_thread = kernel_index;
                record.global_kernel_id = compute_global_kernel_id(config,
                                                                    mpi_rank,
                                                                    host_thread_id,
                                                                    kernel_index);
                record.cuda_device_id = config.device_id;
                record.arrival_wait_ms = arrival_wait_ms;
                record.requested_busy_wait_us = requested_busy_wait_us;
                record.kernel_type = kernel_type_to_string(config.kernel_type);
                record.blocks_x = config.blocks_x;
                record.threads_per_block = config.threads_per_block;
                record.grid_z = config.grid_z;
                record.total_blocks = total_blocks;
                record.total_cuda_threads = total_cuda_threads;
                record.submit_time_ns = submit_time_ns;
                record.completion_time_ns = completion_time_ns;
                record.response_time_us =
                    static_cast<double>(completion_time_ns - submit_time_ns) / 1000.0;
                record.launch_overhead_us =
                    static_cast<double>(launch_return_time_ns - submit_time_ns) / 1000.0;
                record.cuda_error_code = static_cast<int>(final_error);
                record.cuda_error_string = cuda_error_message(final_error);

                writer.write(record);
            }
        } catch (...) {
            free_kernel_resources(&resources);
            cudaStreamDestroy(stream);
            throw;
        }

        const cudaError_t free_error = free_kernel_resources(&resources);
        const cudaError_t destroy_error = cudaStreamDestroy(stream);
        if (free_error != cudaSuccess) {
            record_thread_error(thread_errors,
                                thread_errors_mutex,
                                host_thread_id,
                                "cudaFree failed: " + cuda_error_message(free_error));
        }
        if (destroy_error != cudaSuccess) {
            record_thread_error(thread_errors,
                                thread_errors_mutex,
                                host_thread_id,
                                "cudaStreamDestroy failed: " + cuda_error_message(destroy_error));
        }
    } catch (const std::exception &ex) {
        record_thread_error(thread_errors, thread_errors_mutex, host_thread_id, ex.what());
    } catch (...) {
        record_thread_error(thread_errors, thread_errors_mutex, host_thread_id, "unknown exception");
    }
}

}  // namespace

void run_experiment(const ExperimentConfig &config,
                    int mpi_world_size,
                    int mpi_rank,
                    const std::string &run_timestamp) {
    const unsigned int schedule_flag =
        config.sync_mode == SyncMode::Blocking ? cudaDeviceScheduleBlockingSync : cudaDeviceScheduleSpin;

    throw_on_cuda_error(cudaSetDeviceFlags(schedule_flag), "cudaSetDeviceFlags");

    int device_count = 0;
    throw_on_cuda_error(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    if (device_count <= 0) {
        throw std::runtime_error("No CUDA devices were found");
    }
    if (config.device_id >= device_count) {
        std::ostringstream out;
        out << "--device " << config.device_id << " is not available; detected "
            << device_count << " CUDA device(s)";
        throw std::runtime_error(out.str());
    }

    throw_on_cuda_error(cudaSetDevice(config.device_id), "cudaSetDevice");

    cudaDeviceProp prop{};
    throw_on_cuda_error(cudaGetDeviceProperties(&prop, config.device_id), "cudaGetDeviceProperties");
    validate_launch_config(config, prop);

    throw_on_cuda_error(cudaFree(nullptr), "CUDA context initialization");

    std::filesystem::create_directories(config.output_dir);
    const std::filesystem::path csv_path = output_file_path(config, mpi_rank, run_timestamp);
    CsvWriter writer(csv_path.string());

    if (mpi_rank == 0) {
        std::cerr << "Running experiment '" << config.experiment_name
                  << "' with sync-mode=" << sync_mode_to_string(config.sync_mode)
                  << ", kernel-type=" << kernel_type_to_string(config.kernel_type)
                  << ", warmup-kernels=" << config.warmup_kernels
                  << ", output-dir=" << config.output_dir << '\n';
    }
    std::cerr << "[rank " << mpi_rank << "] writing " << csv_path.string() << '\n';

    std::vector<std::thread> workers;
    workers.reserve(static_cast<std::size_t>(config.threads_per_process));

    std::vector<std::string> thread_errors;
    std::mutex thread_errors_mutex;

    for (int host_thread_id = 0; host_thread_id < config.threads_per_process; ++host_thread_id) {
        workers.emplace_back(run_host_thread,
                             std::cref(config),
                             mpi_world_size,
                             mpi_rank,
                             host_thread_id,
                             prop.clockRate,
                             std::ref(writer),
                             std::ref(thread_errors),
                             std::ref(thread_errors_mutex));
    }

    for (auto &worker : workers) {
        worker.join();
    }

    writer.flush();

    if (!thread_errors.empty()) {
        std::ostringstream out;
        out << "One or more host threads failed on rank " << mpi_rank << ':';
        for (const std::string &error : thread_errors) {
            out << "\n  " << error;
        }
        throw std::runtime_error(out.str());
    }
}

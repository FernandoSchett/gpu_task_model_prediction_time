#include "experiment.hpp"

#include "csv_writer.hpp"
#include "cuda_kernels.cuh"
#include "gpu_telemetry.hpp"
#include "timer.hpp"

#include <cuda_runtime.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
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

struct ExperimentRuntimeState {
    std::atomic<std::uint64_t> rank_submitted_count{0};
    std::atomic<std::uint64_t> rank_completed_count{0};
    std::atomic<std::uint64_t> active_kernels{0};
    std::atomic<std::int64_t> measurement_start_ns{0};
};

struct DeviceInfo {
    std::string gpu_name;
    int sm_count = 0;
    int clock_rate_khz = 0;
    int cuda_runtime_version = 0;
    int cuda_driver_version = 0;
};

class MeasurementStartBarrier {
public:
    explicit MeasurementStartBarrier(int participants)
        : participants_(participants > 0 ? participants : 1) {}

    bool arrive_and_wait(ExperimentRuntimeState &runtime_state) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (aborted_) {
            return false;
        }

        const int generation = generation_;
        ++arrived_;
        if (arrived_ == participants_) {
            runtime_state.measurement_start_ns.store(Timer::now_ns(), std::memory_order_release);
            arrived_ = 0;
            ++generation_;
            condition_.notify_all();
            return true;
        }

        condition_.wait(lock, [this, generation]() {
            return aborted_ || generation_ != generation;
        });
        return !aborted_;
    }

    void abort() {
        std::lock_guard<std::mutex> lock(mutex_);
        aborted_ = true;
        ++generation_;
        condition_.notify_all();
    }

private:
    const int participants_;
    int arrived_ = 0;
    int generation_ = 0;
    bool aborted_ = false;
    std::mutex mutex_;
    std::condition_variable condition_;
};

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

std::filesystem::path telemetry_file_path(const ExperimentConfig &config,
                                          int mpi_rank,
                                          const std::string &run_timestamp) {
    const std::string safe_name = sanitize_filename_component(config.experiment_name);
    std::ostringstream filename;
    filename << "gpu_telemetry_"
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

std::uint64_t estimate_global_count(std::uint64_t rank_local_count, int mpi_world_size) {
    return rank_local_count * static_cast<std::uint64_t>(mpi_world_size > 0 ? mpi_world_size : 1);
}

std::uint64_t ceil_div(std::uint64_t numerator, std::uint64_t denominator) {
    return denominator == 0 ? 0 : (numerator + denominator - 1) / denominator;
}

DeviceInfo collect_device_info(const ExperimentConfig &config, const cudaDeviceProp &prop) {
    DeviceInfo info;
    info.gpu_name = prop.name;
    info.sm_count = prop.multiProcessorCount;

    cudaError_t error = cudaDeviceGetAttribute(&info.clock_rate_khz,
                                               cudaDevAttrClockRate,
                                               config.device_id);
    throw_on_cuda_error(error, "cudaDeviceGetAttribute(cudaDevAttrClockRate)");

    error = cudaDeviceGetAttribute(&info.sm_count,
                                   cudaDevAttrMultiProcessorCount,
                                   config.device_id);
    throw_on_cuda_error(error, "cudaDeviceGetAttribute(cudaDevAttrMultiProcessorCount)");

    throw_on_cuda_error(cudaRuntimeGetVersion(&info.cuda_runtime_version),
                        "cudaRuntimeGetVersion");
    throw_on_cuda_error(cudaDriverGetVersion(&info.cuda_driver_version),
                        "cudaDriverGetVersion");

    return info;
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
                     const DeviceInfo &device_info,
                     CsvWriter &writer,
                     ExperimentRuntimeState &runtime_state,
                     MeasurementStartBarrier &measurement_start_barrier,
                     std::vector<std::string> &thread_errors,
                     std::mutex &thread_errors_mutex) {
    try {
        throw_on_cuda_error(cudaSetDevice(config.device_id), "cudaSetDevice");

        cudaStream_t stream = nullptr;
        throw_on_cuda_error(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
                            "cudaStreamCreateWithFlags");

        cudaEvent_t start_event = nullptr;
        cudaEvent_t stop_event = nullptr;
        CudaKernelResources resources;
        try {
            throw_on_cuda_error(cudaEventCreate(&start_event), "cudaEventCreate(start_event)");
            throw_on_cuda_error(cudaEventCreate(&stop_event), "cudaEventCreate(stop_event)");
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
                                                                 device_info.clock_rate_khz,
                                                                 config.blocks_x,
                                                                 config.threads_per_block,
                                                                 config.grid_z);
                throw_on_cuda_error(warmup_error, "warm-up kernel");
            }

            if (!measurement_start_barrier.arrive_and_wait(runtime_state)) {
                throw std::runtime_error("measurement start barrier aborted");
            }

            for (int kernel_index = 0; kernel_index < config.kernels_per_thread; ++kernel_index) {
                const double arrival_wait_ms = arrival_dist(rng);
                const std::uint64_t requested_busy_wait_us = kernel_dist(rng);

                std::this_thread::sleep_for(std::chrono::duration<double, std::milli>(arrival_wait_ms));

                const std::int64_t submit_time_ns = Timer::now_ns();
                const std::uint64_t completed_before_rank =
                    runtime_state.rank_completed_count.load(std::memory_order_acquire);
                const std::uint64_t submitted_before_rank =
                    runtime_state.rank_submitted_count.fetch_add(1, std::memory_order_acq_rel);
                const std::uint64_t active_before_rank =
                    runtime_state.active_kernels.fetch_add(1, std::memory_order_acq_rel);

                cudaError_t final_error = cudaEventRecord(start_event, stream);
                if (final_error == cudaSuccess) {
                    final_error = launch_configurable_kernel(stream,
                                                             resources,
                                                             config.kernel_type,
                                                             requested_busy_wait_us,
                                                             device_info.clock_rate_khz,
                                                             config.blocks_x,
                                                             config.threads_per_block,
                                                             config.grid_z);
                }
                const std::int64_t launch_return_time_ns = Timer::now_ns();

                if (final_error == cudaSuccess) {
                    final_error = cudaEventRecord(stop_event, stream);
                }
                if (final_error == cudaSuccess) {
                    final_error = cudaStreamSynchronize(stream);
                } else {
                    cudaStreamSynchronize(stream);
                }
                const std::int64_t completion_time_ns = Timer::now_ns();

                float cuda_event_elapsed_ms = -1.0f;
                if (final_error == cudaSuccess) {
                    const cudaError_t elapsed_error =
                        cudaEventElapsedTime(&cuda_event_elapsed_ms, start_event, stop_event);
                    if (elapsed_error != cudaSuccess) {
                        final_error = elapsed_error;
                    }
                }

                runtime_state.active_kernels.fetch_sub(1, std::memory_order_acq_rel);
                runtime_state.rank_completed_count.fetch_add(1, std::memory_order_acq_rel);

                const std::uint64_t submitted_before_global =
                    estimate_global_count(submitted_before_rank, mpi_world_size);
                const std::uint64_t completed_before_global =
                    estimate_global_count(completed_before_rank, mpi_world_size);
                const std::uint64_t active_before_global =
                    estimate_global_count(active_before_rank, mpi_world_size);
                const std::int64_t measurement_start_ns =
                    runtime_state.measurement_start_ns.load(std::memory_order_acquire);
                const std::uint64_t warps_per_block =
                    ceil_div(static_cast<std::uint64_t>(config.threads_per_block), 32ULL);
                const std::uint64_t total_warps = total_blocks * warps_per_block;
                const double estimated_waves =
                    device_info.sm_count > 0
                        ? static_cast<double>(
                              static_cast<std::uint64_t>(config.blocks_x) *
                              static_cast<std::uint64_t>(config.grid_z)) /
                              static_cast<double>(device_info.sm_count)
                        : 0.0;
                const double response_time_us =
                    static_cast<double>(completion_time_ns - submit_time_ns) / 1000.0;
                const std::int64_t cuda_event_elapsed_time_ns =
                    cuda_event_elapsed_ms >= 0.0f
                        ? static_cast<std::int64_t>(static_cast<double>(cuda_event_elapsed_ms) * 1000000.0)
                        : -1;
                const std::int64_t device_end_time_ns_approx =
                    cuda_event_elapsed_time_ns >= 0 ? completion_time_ns : 0;
                const std::int64_t device_start_time_ns_approx =
                    cuda_event_elapsed_time_ns >= 0
                        ? device_end_time_ns_approx - cuda_event_elapsed_time_ns
                        : 0;

                KernelRecord record;
                record.experiment_name = config.experiment_name;
                record.global_seed = config.seed;
                record.warmup_kernels = config.warmup_kernels;
                record.mpi_world_size = mpi_world_size;
                record.mpi_rank = mpi_rank;
                record.threads_per_process = config.threads_per_process;
                record.kernels_per_thread = config.kernels_per_thread;
                record.host_thread_id = host_thread_id;
                record.kernel_index_in_thread = kernel_index;
                record.thread_local_kernel_index = kernel_index;
                record.global_kernel_id = compute_global_kernel_id(config,
                                                                    mpi_rank,
                                                                    host_thread_id,
                                                                    kernel_index);
                record.cuda_device_id = config.device_id;
                record.arrival_wait_ms = arrival_wait_ms;
                record.requested_busy_wait_us = requested_busy_wait_us;
                record.kernel_type = kernel_type_to_string(config.kernel_type);
                record.gpu_name = device_info.gpu_name;
                record.cuda_runtime_version = device_info.cuda_runtime_version;
                record.cuda_driver_version = device_info.cuda_driver_version;
                record.sm_count = device_info.sm_count;
                record.device_clock_rate_khz = device_info.clock_rate_khz;
                record.blocks_x = config.blocks_x;
                record.threads_per_block = config.threads_per_block;
                record.grid_z = config.grid_z;
                record.total_blocks = total_blocks;
                record.total_cuda_threads = total_cuda_threads;
                record.total_warps = total_warps;
                record.warps_per_block = static_cast<int>(warps_per_block);
                record.estimated_waves = estimated_waves;
                record.active_kernels_estimate = active_before_global;
                record.submitted_before_global = submitted_before_global;
                record.completed_before_global = completed_before_global;
                record.inflight_kernels_estimate =
                    submitted_before_global >= completed_before_global
                        ? submitted_before_global - completed_before_global
                        : 0;
                record.concurrent_kernels_estimate =
                    active_before_global + static_cast<std::uint64_t>(mpi_world_size > 0 ? mpi_world_size : 1);
                record.logical_stream_id = host_thread_id;
                record.measurement_start_time_ns = measurement_start_ns;
                record.time_since_experiment_start_us =
                    static_cast<double>(submit_time_ns - measurement_start_ns) / 1000.0;
                record.rank_local_submitted_count = submitted_before_rank;
                record.rank_local_completed_count = completed_before_rank;
                record.submit_time_ns = submit_time_ns;
                record.launch_return_time_ns = launch_return_time_ns;
                record.completion_time_ns = completion_time_ns;
                record.device_start_time_ns_approx = device_start_time_ns_approx;
                record.device_end_time_ns_approx = device_end_time_ns_approx;
                record.host_submit_time_ns = submit_time_ns;
                record.host_completion_time_ns = completion_time_ns;
                record.response_time_us = response_time_us;
                record.launch_overhead_us =
                    static_cast<double>(launch_return_time_ns - submit_time_ns) / 1000.0;
                record.cuda_event_elapsed_time_us =
                    cuda_event_elapsed_ms >= 0.0f
                        ? static_cast<double>(cuda_event_elapsed_ms) * 1000.0
                        : -1.0;
                record.queueing_delay_us =
                    response_time_us - static_cast<double>(requested_busy_wait_us);
                record.slowdown =
                    requested_busy_wait_us > 0
                        ? response_time_us / static_cast<double>(requested_busy_wait_us)
                        : 0.0;
                record.cuda_error_code = static_cast<int>(final_error);
                record.cuda_error_string = cuda_error_message(final_error);

                writer.write(record);
            }
        } catch (...) {
            free_kernel_resources(&resources);
            if (stop_event != nullptr) {
                cudaEventDestroy(stop_event);
            }
            if (start_event != nullptr) {
                cudaEventDestroy(start_event);
            }
            cudaStreamDestroy(stream);
            throw;
        }

        const cudaError_t free_error = free_kernel_resources(&resources);
        const cudaError_t stop_event_destroy_error = cudaEventDestroy(stop_event);
        const cudaError_t start_event_destroy_error = cudaEventDestroy(start_event);
        const cudaError_t destroy_error = cudaStreamDestroy(stream);
        if (free_error != cudaSuccess) {
            record_thread_error(thread_errors,
                                thread_errors_mutex,
                                host_thread_id,
                                "cudaFree failed: " + cuda_error_message(free_error));
        }
        if (stop_event_destroy_error != cudaSuccess) {
            record_thread_error(thread_errors,
                                thread_errors_mutex,
                                host_thread_id,
                                "cudaEventDestroy(stop_event) failed: " +
                                    cuda_error_message(stop_event_destroy_error));
        }
        if (start_event_destroy_error != cudaSuccess) {
            record_thread_error(thread_errors,
                                thread_errors_mutex,
                                host_thread_id,
                                "cudaEventDestroy(start_event) failed: " +
                                    cuda_error_message(start_event_destroy_error));
        }
        if (destroy_error != cudaSuccess) {
            record_thread_error(thread_errors,
                                thread_errors_mutex,
                                host_thread_id,
                                "cudaStreamDestroy failed: " + cuda_error_message(destroy_error));
        }
    } catch (const std::exception &ex) {
        measurement_start_barrier.abort();
        record_thread_error(thread_errors, thread_errors_mutex, host_thread_id, ex.what());
    } catch (...) {
        measurement_start_barrier.abort();
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
    const DeviceInfo device_info = collect_device_info(config, prop);

    throw_on_cuda_error(cudaFree(nullptr), "CUDA context initialization");

    std::filesystem::create_directories(config.output_dir);
    const std::filesystem::path csv_path = output_file_path(config, mpi_rank, run_timestamp);
    const std::filesystem::path telemetry_path = telemetry_file_path(config, mpi_rank, run_timestamp);
    CsvWriter writer(csv_path.string(), static_cast<std::uint64_t>(config.flush_every));
    GpuTelemetryMonitor gpu_telemetry(config,
                                      mpi_world_size,
                                      mpi_rank,
                                      telemetry_path.string());

    if (mpi_rank == 0) {
        std::cerr << "Running experiment '" << config.experiment_name
                  << "' with sync-mode=" << sync_mode_to_string(config.sync_mode)
                  << ", kernel-type=" << kernel_type_to_string(config.kernel_type)
                  << ", warmup-kernels=" << config.warmup_kernels
                  << ", flush-every=" << config.flush_every
                  << ", gpu-telemetry=" << (config.gpu_telemetry_enabled ? "on" : "off")
                  << ", gpu-telemetry-during=" << (config.gpu_telemetry_during ? "on" : "off")
                  << ", output-dir=" << config.output_dir << '\n';
    }
    std::cerr << "[rank " << mpi_rank << "] writing " << csv_path.string() << '\n';
    if (gpu_telemetry.enabled()) {
        std::cerr << "[rank " << mpi_rank << "] writing " << gpu_telemetry.path() << '\n';
    }

    std::vector<std::thread> workers;
    workers.reserve(static_cast<std::size_t>(config.threads_per_process));

    std::vector<std::string> thread_errors;
    std::mutex thread_errors_mutex;
    ExperimentRuntimeState runtime_state;
    MeasurementStartBarrier measurement_start_barrier(config.threads_per_process);

    gpu_telemetry.sample_once("before");
    if (config.gpu_telemetry_during) {
        gpu_telemetry.start();
    }

    for (int host_thread_id = 0; host_thread_id < config.threads_per_process; ++host_thread_id) {
        workers.emplace_back(run_host_thread,
                             std::cref(config),
                             mpi_world_size,
                             mpi_rank,
                             host_thread_id,
                             std::cref(device_info),
                             std::ref(writer),
                             std::ref(runtime_state),
                             std::ref(measurement_start_barrier),
                             std::ref(thread_errors),
                             std::ref(thread_errors_mutex));
    }

    for (auto &worker : workers) {
        worker.join();
    }

    if (config.gpu_telemetry_during) {
        gpu_telemetry.stop();
    }
    gpu_telemetry.sample_once("after");

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

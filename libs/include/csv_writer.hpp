#ifndef CSV_WRITER_HPP
#define CSV_WRITER_HPP

#include <cstdint>
#include <fstream>
#include <mutex>
#include <string>

struct KernelRecord {
    std::string experiment_name;
    std::uint64_t global_seed = 0;
    int warmup_kernels = 0;
    int mpi_world_size = 0;
    int mpi_rank = 0;
    int threads_per_process = 0;
    int kernels_per_thread = 0;
    int host_thread_id = 0;
    int kernel_index_in_thread = 0;
    int thread_local_kernel_index = 0;
    std::uint64_t global_kernel_id = 0;
    int cuda_device_id = 0;
    double arrival_wait_ms = 0.0;
    std::uint64_t requested_busy_wait_us = 0;
    std::string kernel_type;
    std::string gpu_name;
    int cuda_runtime_version = 0;
    int cuda_driver_version = 0;
    int sm_count = 0;
    int device_clock_rate_khz = 0;
    int blocks_x = 0;
    int threads_per_block = 0;
    int grid_z = 0;
    std::uint64_t total_blocks = 0;
    std::uint64_t total_cuda_threads = 0;
    std::uint64_t total_warps = 0;
    int warps_per_block = 0;
    double estimated_waves = 0.0;
    std::uint64_t active_kernels_estimate = 0;
    std::uint64_t submitted_before_global = 0;
    std::uint64_t completed_before_global = 0;
    std::uint64_t inflight_kernels_estimate = 0;
    std::uint64_t concurrent_kernels_estimate = 0;
    double time_since_experiment_start_us = 0.0;
    std::uint64_t rank_local_submitted_count = 0;
    std::uint64_t rank_local_completed_count = 0;
    std::int64_t submit_time_ns = 0;
    std::int64_t completion_time_ns = 0;
    std::int64_t host_submit_time_ns = 0;
    std::int64_t host_completion_time_ns = 0;
    double response_time_us = 0.0;
    double launch_overhead_us = 0.0;
    double cuda_event_elapsed_time_us = -1.0;
    double queueing_delay_us = 0.0;
    double slowdown = 0.0;
    int cuda_error_code = 0;
    std::string cuda_error_string;
};

class CsvWriter {
public:
    explicit CsvWriter(const std::string &path, std::uint64_t flush_every);
    void write(const KernelRecord &record);
    void flush();

private:
    std::ofstream file_;
    std::mutex mutex_;
    std::uint64_t flush_every_ = 1000;
    std::uint64_t rows_since_flush_ = 0;
};

#endif

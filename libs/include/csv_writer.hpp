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
    std::uint64_t execution_order = 0;
    std::uint64_t repetition_id = 0;
    std::uint64_t block_id = 0;
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
    double blocks_per_sm = 0.0;
    double total_blocks_per_sm = 0.0;
    int effective_workers = 0;
    double requested_busy_wait_s = 0.0;
    double workers_x_requested_busy_wait_us = 0.0;
    double workers_x_total_warps = 0.0;
    double workers_x_blocks_per_sm = 0.0;
    double requested_busy_wait_us_per_arrival_ms = 0.0;
    int logical_stream_id = 0;
    std::int64_t measurement_start_time_ns = 0;
    double time_since_experiment_start_us = 0.0;
    double time_since_previous_submit_us = -1.0;
    std::uint64_t rank_local_submitted_count = 0;
    std::uint64_t rank_local_completed_count = 0;
    std::uint64_t rank_local_backlog_at_launch = 0;
    std::string gpu_clock_sm_mhz;
    std::string gpu_clock_mem_mhz;
    std::string gpu_temperature_c;
    std::string gpu_power_w;
    std::string gpu_power_limit_w;
    std::string gpu_sm_utilization_percent;
    std::string gpu_memory_utilization_percent;
    std::string gpu_telemetry_status;
    std::int64_t submit_time_ns = 0;
    std::int64_t launch_return_time_ns = 0;
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

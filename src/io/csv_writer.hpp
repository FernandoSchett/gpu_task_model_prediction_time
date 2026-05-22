#ifndef CSV_WRITER_HPP
#define CSV_WRITER_HPP

#include <cstdint>
#include <fstream>
#include <mutex>
#include <string>

struct KernelRecord {
    std::string experiment_name;
    std::uint64_t global_seed = 0;
    int mpi_world_size = 0;
    int mpi_rank = 0;
    int host_thread_id = 0;
    int kernel_index_in_thread = 0;
    std::uint64_t global_kernel_id = 0;
    int cuda_device_id = 0;
    double arrival_wait_ms = 0.0;
    std::uint64_t requested_busy_wait_us = 0;
    int blocks_x = 0;
    int threads_per_block = 0;
    int grid_z = 0;
    std::uint64_t total_blocks = 0;
    std::uint64_t total_cuda_threads = 0;
    std::int64_t submit_time_ns = 0;
    std::int64_t completion_time_ns = 0;
    double response_time_us = 0.0;
    double launch_overhead_us = 0.0;
    int cuda_error_code = 0;
    std::string cuda_error_string;
};

class CsvWriter {
public:
    explicit CsvWriter(const std::string &path);
    void write(const KernelRecord &record);
    void flush();

private:
    std::ofstream file_;
    std::mutex mutex_;
};

#endif

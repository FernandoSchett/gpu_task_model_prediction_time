#include "csv_writer.hpp"

#include <iomanip>
#include <stdexcept>

namespace {

std::string csv_escape(const std::string &value) {
    bool needs_quotes = false;
    for (const char ch : value) {
        if (ch == ',' || ch == '"' || ch == '\n' || ch == '\r') {
            needs_quotes = true;
            break;
        }
    }

    if (!needs_quotes) {
        return value;
    }

    std::string escaped = "\"";
    for (const char ch : value) {
        if (ch == '"') {
            escaped += "\"\"";
        } else {
            escaped += ch;
        }
    }
    escaped += '"';
    return escaped;
}

}  // namespace

CsvWriter::CsvWriter(const std::string &path, std::uint64_t flush_every)
    : file_(path), flush_every_(flush_every) {
    if (!file_) {
        throw std::runtime_error("Could not open CSV file: " + path);
    }

    file_ << "experiment_name,"
          << "global_seed,"
          << "warmup_kernels,"
          << "mpi_world_size,"
          << "mpi_rank,"
          << "threads_per_process,"
          << "kernels_per_thread,"
          << "host_thread_id,"
          << "kernel_index_in_thread,"
          << "thread_local_kernel_index,"
          << "global_kernel_id,"
          << "cuda_device_id,"
          << "arrival_wait_ms,"
          << "requested_busy_wait_us,"
          << "kernel_type,"
          << "gpu_name,"
          << "cuda_runtime_version,"
          << "cuda_driver_version,"
          << "sm_count,"
          << "device_clock_rate_khz,"
          << "blocks_x,"
          << "threads_per_block,"
          << "grid_z,"
          << "total_blocks,"
          << "total_cuda_threads,"
          << "total_warps,"
          << "warps_per_block,"
          << "blocks_per_sm,"
          << "total_blocks_per_sm,"
          << "effective_workers,"
          << "requested_busy_wait_s,"
          << "workers_x_requested_busy_wait_us,"
          << "workers_x_total_warps,"
          << "workers_x_blocks_per_sm,"
          << "requested_busy_wait_us_per_arrival_ms,"
          << "logical_stream_id,"
          << "measurement_start_time_ns,"
          << "time_since_experiment_start_us,"
          << "rank_local_submitted_count,"
          << "rank_local_completed_count,"
          << "submit_time_ns,"
          << "launch_return_time_ns,"
          << "completion_time_ns,"
          << "host_submit_time_ns,"
          << "host_completion_time_ns,"
          << "response_time_us,"
          << "launch_overhead_us,"
          << "cuda_event_elapsed_time_us,"
          << "queueing_delay_us,"
          << "slowdown,"
          << "cuda_error_code,"
          << "cuda_error_string\n";
}

void CsvWriter::write(const KernelRecord &record) {
    std::lock_guard<std::mutex> lock(mutex_);

    file_ << csv_escape(record.experiment_name) << ','
          << record.global_seed << ','
          << record.warmup_kernels << ','
          << record.mpi_world_size << ','
          << record.mpi_rank << ','
          << record.threads_per_process << ','
          << record.kernels_per_thread << ','
          << record.host_thread_id << ','
          << record.kernel_index_in_thread << ','
          << record.thread_local_kernel_index << ','
          << record.global_kernel_id << ','
          << record.cuda_device_id << ','
          << std::fixed << std::setprecision(6) << record.arrival_wait_ms << ','
          << record.requested_busy_wait_us << ','
          << csv_escape(record.kernel_type) << ','
          << csv_escape(record.gpu_name) << ','
          << record.cuda_runtime_version << ','
          << record.cuda_driver_version << ','
          << record.sm_count << ','
          << record.device_clock_rate_khz << ','
          << record.blocks_x << ','
          << record.threads_per_block << ','
          << record.grid_z << ','
          << record.total_blocks << ','
          << record.total_cuda_threads << ','
          << record.total_warps << ','
          << record.warps_per_block << ','
          << std::fixed << std::setprecision(6) << record.blocks_per_sm << ','
          << std::fixed << std::setprecision(6) << record.total_blocks_per_sm << ','
          << record.effective_workers << ','
          << std::fixed << std::setprecision(6) << record.requested_busy_wait_s << ','
          << std::fixed << std::setprecision(3) << record.workers_x_requested_busy_wait_us << ','
          << std::fixed << std::setprecision(3) << record.workers_x_total_warps << ','
          << std::fixed << std::setprecision(6) << record.workers_x_blocks_per_sm << ','
          << std::fixed << std::setprecision(6) << record.requested_busy_wait_us_per_arrival_ms << ','
          << record.logical_stream_id << ','
          << record.measurement_start_time_ns << ','
          << std::fixed << std::setprecision(3) << record.time_since_experiment_start_us << ','
          << record.rank_local_submitted_count << ','
          << record.rank_local_completed_count << ','
          << record.submit_time_ns << ','
          << record.launch_return_time_ns << ','
          << record.completion_time_ns << ','
          << record.host_submit_time_ns << ','
          << record.host_completion_time_ns << ','
          << std::fixed << std::setprecision(3) << record.response_time_us << ','
          << std::fixed << std::setprecision(3) << record.launch_overhead_us << ','
          << std::fixed << std::setprecision(3) << record.cuda_event_elapsed_time_us << ','
          << std::fixed << std::setprecision(3) << record.queueing_delay_us << ','
          << std::fixed << std::setprecision(6) << record.slowdown << ','
          << record.cuda_error_code << ','
          << csv_escape(record.cuda_error_string) << '\n';

    ++rows_since_flush_;
    if (rows_since_flush_ >= flush_every_) {
        file_.flush();
        rows_since_flush_ = 0;
    }
}

void CsvWriter::flush() {
    std::lock_guard<std::mutex> lock(mutex_);
    file_.flush();
}

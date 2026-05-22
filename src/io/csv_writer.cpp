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

CsvWriter::CsvWriter(const std::string &path) : file_(path) {
    if (!file_) {
        throw std::runtime_error("Could not open CSV file: " + path);
    }

    file_ << "experiment_name,"
          << "global_seed,"
          << "warmup_kernels,"
          << "mpi_world_size,"
          << "mpi_rank,"
          << "host_thread_id,"
          << "kernel_index_in_thread,"
          << "global_kernel_id,"
          << "cuda_device_id,"
          << "arrival_wait_ms,"
          << "requested_busy_wait_us,"
          << "kernel_type,"
          << "blocks_x,"
          << "threads_per_block,"
          << "grid_z,"
          << "total_blocks,"
          << "total_cuda_threads,"
          << "submit_time_ns,"
          << "completion_time_ns,"
          << "response_time_us,"
          << "launch_overhead_us,"
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
          << record.host_thread_id << ','
          << record.kernel_index_in_thread << ','
          << record.global_kernel_id << ','
          << record.cuda_device_id << ','
          << std::fixed << std::setprecision(6) << record.arrival_wait_ms << ','
          << record.requested_busy_wait_us << ','
          << csv_escape(record.kernel_type) << ','
          << record.blocks_x << ','
          << record.threads_per_block << ','
          << record.grid_z << ','
          << record.total_blocks << ','
          << record.total_cuda_threads << ','
          << record.submit_time_ns << ','
          << record.completion_time_ns << ','
          << std::fixed << std::setprecision(3) << record.response_time_us << ','
          << std::fixed << std::setprecision(3) << record.launch_overhead_us << ','
          << record.cuda_error_code << ','
          << csv_escape(record.cuda_error_string) << '\n';

    file_.flush();
}

void CsvWriter::flush() {
    std::lock_guard<std::mutex> lock(mutex_);
    file_.flush();
}

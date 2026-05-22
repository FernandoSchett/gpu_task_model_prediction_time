#include "gpu_telemetry.hpp"

#include "timer.hpp"

#include <array>
#include <chrono>
#include <cstdio>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {

std::string trim(const std::string &input) {
    auto first = input.begin();
    while (first != input.end() && (*first == ' ' || *first == '\t' || *first == '\r' || *first == '\n')) {
        ++first;
    }

    auto last = input.end();
    while (last != first && (*(last - 1) == ' ' || *(last - 1) == '\t' ||
                             *(last - 1) == '\r' || *(last - 1) == '\n')) {
        --last;
    }

    return std::string(first, last);
}

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

std::vector<std::string> split_csv_line(const std::string &line) {
    std::vector<std::string> fields;
    std::string current;
    bool in_quotes = false;

    for (std::size_t i = 0; i < line.size(); ++i) {
        const char ch = line[i];
        if (ch == '"') {
            if (in_quotes && i + 1 < line.size() && line[i + 1] == '"') {
                current += '"';
                ++i;
            } else {
                in_quotes = !in_quotes;
            }
        } else if (ch == ',' && !in_quotes) {
            fields.push_back(trim(current));
            current.clear();
        } else {
            current += ch;
        }
    }
    fields.push_back(trim(current));
    return fields;
}

std::string first_non_empty_line(const std::string &text) {
    std::istringstream input(text);
    std::string line;
    while (std::getline(input, line)) {
        line = trim(line);
        if (!line.empty()) {
            return line;
        }
    }
    return "";
}

std::string run_nvidia_smi_query(int device_id, int *exit_code) {
    std::ostringstream command;
    command << "nvidia-smi -i " << device_id
            << " --query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw,power.limit,"
            << "utilization.gpu,utilization.memory --format=csv,noheader,nounits 2>&1";

#if defined(_WIN32)
    FILE *pipe = _popen(command.str().c_str(), "r");
#else
    FILE *pipe = popen(command.str().c_str(), "r");
#endif
    if (pipe == nullptr) {
        if (exit_code != nullptr) {
            *exit_code = -1;
        }
        return "failed to start nvidia-smi";
    }

    std::array<char, 512> buffer{};
    std::string output;
    while (fgets(buffer.data(), static_cast<int>(buffer.size()), pipe) != nullptr) {
        output += buffer.data();
    }

#if defined(_WIN32)
    const int status = _pclose(pipe);
#else
    const int status = pclose(pipe);
#endif
    if (exit_code != nullptr) {
        *exit_code = status;
    }
    return output;
}

}  // namespace

GpuTelemetryMonitor::GpuTelemetryMonitor(const ExperimentConfig &config,
                                         int mpi_world_size,
                                         int mpi_rank,
                                         const std::string &path)
    : config_(config),
      mpi_world_size_(mpi_world_size),
      mpi_rank_(mpi_rank),
      path_(path),
      monitor_start_ns_(Timer::now_ns()) {
    if (!config_.gpu_telemetry_enabled) {
        return;
    }

    file_.open(path_);
    if (!file_) {
        throw std::runtime_error("Could not open GPU telemetry file: " + path_);
    }
    write_header();
}

GpuTelemetryMonitor::~GpuTelemetryMonitor() {
    stop();
}

void GpuTelemetryMonitor::sample_once(const std::string &phase) {
    if (!enabled()) {
        return;
    }

    const std::int64_t sample_time_ns = Timer::now_ns();
    int exit_code = 0;
    const std::string raw_output = run_nvidia_smi_query(config_.device_id, &exit_code);
    const std::string first_line = first_non_empty_line(raw_output);
    const std::vector<std::string> fields = split_csv_line(first_line);

    std::string values[7] = {"", "", "", "", "", "", ""};
    std::string status = exit_code == 0 ? "ok" : "nvidia-smi-error";
    if (fields.size() >= 7) {
        for (std::size_t i = 0; i < 7; ++i) {
            values[i] = fields[i];
        }
    } else if (status == "ok") {
        status = "parse-error";
    }

    write_sample(phase, sample_time_ns, status, trim(raw_output), values);
}

void GpuTelemetryMonitor::start() {
    if (!enabled()) {
        return;
    }

    bool expected = false;
    if (!running_.compare_exchange_strong(expected, true)) {
        return;
    }

    worker_ = std::thread(&GpuTelemetryMonitor::run, this);
}

void GpuTelemetryMonitor::stop() {
    if (!enabled()) {
        return;
    }

    const bool was_running = running_.exchange(false);
    if (was_running && worker_.joinable()) {
        worker_.join();
    }

    std::lock_guard<std::mutex> lock(file_mutex_);
    if (file_) {
        file_.flush();
    }
}

bool GpuTelemetryMonitor::enabled() const {
    return config_.gpu_telemetry_enabled && file_.is_open();
}

const std::string &GpuTelemetryMonitor::path() const {
    return path_;
}

void GpuTelemetryMonitor::run() {
    while (running_.load()) {
        sample_once("during");

        const auto interval = std::chrono::milliseconds(config_.telemetry_interval_ms);
        const auto sleep_step = std::chrono::milliseconds(50);
        auto slept = std::chrono::milliseconds(0);
        while (running_.load() && slept < interval) {
            const auto remaining = interval - slept;
            const auto step = remaining < sleep_step ? remaining : sleep_step;
            std::this_thread::sleep_for(step);
            slept += step;
        }
    }
}

void GpuTelemetryMonitor::write_header() {
    file_ << "experiment_name,"
          << "global_seed,"
          << "mpi_world_size,"
          << "mpi_rank,"
          << "cuda_device_id,"
          << "sample_phase,"
          << "sample_time_ns,"
          << "time_since_monitor_start_us,"
          << "gpu_clock_sm_mhz,"
          << "gpu_clock_mem_mhz,"
          << "temperature_c,"
          << "power_w,"
          << "power_limit_w,"
          << "gpu_utilization,"
          << "memory_utilization,"
          << "nvidia_smi_status,"
          << "nvidia_smi_raw_output\n";
}

void GpuTelemetryMonitor::write_sample(const std::string &phase,
                                       std::int64_t sample_time_ns,
                                       const std::string &status,
                                       const std::string &raw_output,
                                       const std::string values[7]) {
    std::lock_guard<std::mutex> lock(file_mutex_);
    if (!file_) {
        return;
    }

    file_ << csv_escape(config_.experiment_name) << ','
          << config_.seed << ','
          << mpi_world_size_ << ','
          << mpi_rank_ << ','
          << config_.device_id << ','
          << csv_escape(phase) << ','
          << sample_time_ns << ','
          << static_cast<double>(sample_time_ns - monitor_start_ns_) / 1000.0 << ','
          << csv_escape(values[0]) << ','
          << csv_escape(values[1]) << ','
          << csv_escape(values[2]) << ','
          << csv_escape(values[3]) << ','
          << csv_escape(values[4]) << ','
          << csv_escape(values[5]) << ','
          << csv_escape(values[6]) << ','
          << csv_escape(status) << ','
          << csv_escape(raw_output) << '\n';
    file_.flush();
}

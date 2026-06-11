#ifndef GPU_TELEMETRY_HPP
#define GPU_TELEMETRY_HPP

#include "config.hpp"

#include <atomic>
#include <cstdint>
#include <fstream>
#include <mutex>
#include <string>
#include <thread>

struct GpuTelemetrySnapshot {
    std::string gpu_clock_sm_mhz;
    std::string gpu_clock_mem_mhz;
    std::string temperature_c;
    std::string power_w;
    std::string power_limit_w;
    std::string gpu_utilization;
    std::string memory_utilization;
    std::string status = "disabled";
};

class GpuTelemetryMonitor {
public:
    GpuTelemetryMonitor(const ExperimentConfig &config,
                        int mpi_world_size,
                        int mpi_rank,
                        const std::string &path);
    ~GpuTelemetryMonitor();

    GpuTelemetryMonitor(const GpuTelemetryMonitor &) = delete;
    GpuTelemetryMonitor &operator=(const GpuTelemetryMonitor &) = delete;

    void sample_once(const std::string &phase);
    void start();
    void stop();

    bool enabled() const;
    GpuTelemetrySnapshot latest_snapshot() const;
    const std::string &path() const;

private:
    void run();
    void write_header();
    void write_sample(const std::string &phase,
                      std::int64_t sample_time_ns,
                      const std::string &status,
                      const std::string &raw_output,
                      const std::string values[7]);

    const ExperimentConfig &config_;
    int mpi_world_size_ = 1;
    int mpi_rank_ = 0;
    std::string path_;
    std::ofstream file_;
    std::mutex file_mutex_;
    mutable std::mutex latest_mutex_;
    GpuTelemetrySnapshot latest_snapshot_;
    std::thread worker_;
    std::atomic<bool> running_{false};
    std::int64_t monitor_start_ns_ = 0;
};

#endif

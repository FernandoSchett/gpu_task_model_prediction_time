#ifndef CONFIG_HPP
#define CONFIG_HPP

#include "env_loader.hpp"

#include <cstdint>
#include <string>

enum class SyncMode {
    Blocking,
    Spin
};

struct ExperimentConfig {
    int threads_per_process = 1;
    int kernels_per_thread = 1;
    double arrival_min_ms = 1.0;
    double arrival_max_ms = 1.0;
    std::uint64_t kernel_min_us = 100;
    std::uint64_t kernel_max_us = 100;
    int blocks_x = 1;
    int threads_per_block = 256;
    int grid_z = 1;
    std::uint64_t seed = 42;
    std::string experiment_name = "experiment";
    std::string output_dir = "resultados";
    int device_id = 0;
    SyncMode sync_mode = SyncMode::Blocking;
    bool help_requested = false;
};

bool parse_command_line(int argc,
                        char **argv,
                        const EnvMap &env,
                        ExperimentConfig &config,
                        std::string &error);

std::string usage(const char *program_name);
std::string sync_mode_to_string(SyncMode mode);

#endif

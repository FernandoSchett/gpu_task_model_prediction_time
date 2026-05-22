#include "config.hpp"

#include <cstdlib>
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {

bool env_to_uint64(const EnvMap &env, const std::string &key, std::uint64_t &value) {
    const auto it = env.find(key);
    if (it == env.end()) {
        return true;
    }
    try {
        if (!it->second.empty() && it->second.front() == '-') {
            return false;
        }
        size_t pos = 0;
        const unsigned long long parsed = std::stoull(it->second, &pos, 10);
        if (pos != it->second.size()) {
            return false;
        }
        value = static_cast<std::uint64_t>(parsed);
        return true;
    } catch (...) {
        return false;
    }
}

bool env_to_int(const EnvMap &env, const std::string &key, int &value) {
    const auto it = env.find(key);
    if (it == env.end()) {
        return true;
    }
    try {
        size_t pos = 0;
        const int parsed = std::stoi(it->second, &pos, 10);
        if (pos != it->second.size()) {
            return false;
        }
        value = parsed;
        return true;
    } catch (...) {
        return false;
    }
}

void apply_env_defaults(const EnvMap &env, ExperimentConfig &config, std::string &error) {
    if (!env_to_uint64(env, "SEED", config.seed)) {
        error = "Invalid SEED value in .env";
        return;
    }
    if (!env_to_int(env, "DEFAULT_DEVICE", config.device_id)) {
        error = "Invalid DEFAULT_DEVICE value in .env";
        return;
    }

    const auto output_it = env.find("OUTPUT_DIR");
    if (output_it != env.end() && !output_it->second.empty()) {
        config.output_dir = output_it->second;
    }
}

bool parse_int_value(const std::string &text, int &value) {
    try {
        size_t pos = 0;
        const int parsed = std::stoi(text, &pos, 10);
        if (pos != text.size()) {
            return false;
        }
        value = parsed;
        return true;
    } catch (...) {
        return false;
    }
}

bool parse_uint64_value(const std::string &text, std::uint64_t &value) {
    try {
        if (!text.empty() && text.front() == '-') {
            return false;
        }
        size_t pos = 0;
        const unsigned long long parsed = std::stoull(text, &pos, 10);
        if (pos != text.size()) {
            return false;
        }
        value = static_cast<std::uint64_t>(parsed);
        return true;
    } catch (...) {
        return false;
    }
}

bool parse_double_value(const std::string &text, double &value) {
    try {
        size_t pos = 0;
        const double parsed = std::stod(text, &pos);
        if (pos != text.size() || !std::isfinite(parsed)) {
            return false;
        }
        value = parsed;
        return true;
    } catch (...) {
        return false;
    }
}

bool require_value(int argc, char **argv, int &index, std::string &value, std::string &error) {
    if (index + 1 >= argc) {
        error = std::string("Missing value for ") + argv[index];
        return false;
    }
    value = argv[++index];
    return true;
}

bool validate_config(const ExperimentConfig &config, std::string &error) {
    if (config.threads_per_process <= 0) {
        error = "--threads-per-process must be greater than zero";
        return false;
    }
    if (config.kernels_per_thread <= 0) {
        error = "--kernels-per-thread must be greater than zero";
        return false;
    }
    if (config.warmup_kernels < 0) {
        error = "--warmup-kernels must be zero or greater";
        return false;
    }
    if (config.flush_every <= 0) {
        error = "--flush-every must be greater than zero";
        return false;
    }
    if (config.telemetry_interval_ms <= 0) {
        error = "--telemetry-interval-ms must be greater than zero";
        return false;
    }
    if (config.arrival_min_ms < 0.0 || config.arrival_max_ms < 0.0) {
        error = "Arrival ranges must be non-negative";
        return false;
    }
    if (config.arrival_min_ms > config.arrival_max_ms) {
        error = "--arrival-min-ms cannot be greater than --arrival-max-ms";
        return false;
    }
    if (config.kernel_min_us > config.kernel_max_us) {
        error = "--kernel-min-us cannot be greater than --kernel-max-us";
        return false;
    }
    if (config.blocks_x <= 0) {
        error = "--blocks-x must be greater than zero";
        return false;
    }
    if (config.threads_per_block <= 0) {
        error = "--threads-per-block must be greater than zero";
        return false;
    }
    if (config.grid_z <= 0) {
        error = "--grid-z must be greater than zero";
        return false;
    }
    if (config.device_id < 0) {
        error = "--device must be non-negative";
        return false;
    }
    if (config.experiment_name.empty()) {
        error = "--experiment-name cannot be empty";
        return false;
    }
    if (config.output_dir.empty()) {
        error = "--output-dir cannot be empty";
        return false;
    }
    return true;
}

bool parse_kernel_type_value(const std::string &value, KernelType &kernel_type) {
    if (value == "busy_wait") {
        kernel_type = KernelType::BusyWait;
        return true;
    }
    if (value == "compute") {
        kernel_type = KernelType::Compute;
        return true;
    }
    if (value == "memory") {
        kernel_type = KernelType::Memory;
        return true;
    }
    if (value == "mixed") {
        kernel_type = KernelType::Mixed;
        return true;
    }
    return false;
}

}  // namespace

bool parse_command_line(int argc,
                        char **argv,
                        const EnvMap &env,
                        ExperimentConfig &config,
                        std::string &error) {
    apply_env_defaults(env, config, error);
    if (!error.empty()) {
        return false;
    }

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        std::string value;

        if (arg == "--help") {
            config.help_requested = true;
            return true;
        } else if (arg == "--threads-per-process") {
            if (!require_value(argc, argv, i, value, error) || !parse_int_value(value, config.threads_per_process)) {
                error = error.empty() ? "Invalid value for --threads-per-process" : error;
                return false;
            }
        } else if (arg == "--kernels-per-thread") {
            if (!require_value(argc, argv, i, value, error) || !parse_int_value(value, config.kernels_per_thread)) {
                error = error.empty() ? "Invalid value for --kernels-per-thread" : error;
                return false;
            }
        } else if (arg == "--warmup-kernels") {
            if (!require_value(argc, argv, i, value, error) || !parse_int_value(value, config.warmup_kernels)) {
                error = error.empty() ? "Invalid value for --warmup-kernels" : error;
                return false;
            }
        } else if (arg == "--flush-every") {
            if (!require_value(argc, argv, i, value, error) || !parse_int_value(value, config.flush_every)) {
                error = error.empty() ? "Invalid value for --flush-every" : error;
                return false;
            }
        } else if (arg == "--gpu-telemetry") {
            if (!require_value(argc, argv, i, value, error)) {
                return false;
            }
            if (value == "on") {
                config.gpu_telemetry_enabled = true;
            } else if (value == "off") {
                config.gpu_telemetry_enabled = false;
            } else {
                error = "--gpu-telemetry must be either on or off";
                return false;
            }
        } else if (arg == "--telemetry-interval-ms") {
            if (!require_value(argc, argv, i, value, error) ||
                !parse_int_value(value, config.telemetry_interval_ms)) {
                error = error.empty() ? "Invalid value for --telemetry-interval-ms" : error;
                return false;
            }
        } else if (arg == "--arrival-min-ms") {
            if (!require_value(argc, argv, i, value, error) || !parse_double_value(value, config.arrival_min_ms)) {
                error = error.empty() ? "Invalid value for --arrival-min-ms" : error;
                return false;
            }
        } else if (arg == "--arrival-max-ms") {
            if (!require_value(argc, argv, i, value, error) || !parse_double_value(value, config.arrival_max_ms)) {
                error = error.empty() ? "Invalid value for --arrival-max-ms" : error;
                return false;
            }
        } else if (arg == "--kernel-min-us") {
            if (!require_value(argc, argv, i, value, error) || !parse_uint64_value(value, config.kernel_min_us)) {
                error = error.empty() ? "Invalid value for --kernel-min-us" : error;
                return false;
            }
        } else if (arg == "--kernel-max-us") {
            if (!require_value(argc, argv, i, value, error) || !parse_uint64_value(value, config.kernel_max_us)) {
                error = error.empty() ? "Invalid value for --kernel-max-us" : error;
                return false;
            }
        } else if (arg == "--blocks-x") {
            if (!require_value(argc, argv, i, value, error) || !parse_int_value(value, config.blocks_x)) {
                error = error.empty() ? "Invalid value for --blocks-x" : error;
                return false;
            }
        } else if (arg == "--threads-per-block") {
            if (!require_value(argc, argv, i, value, error) || !parse_int_value(value, config.threads_per_block)) {
                error = error.empty() ? "Invalid value for --threads-per-block" : error;
                return false;
            }
        } else if (arg == "--grid-z") {
            if (!require_value(argc, argv, i, value, error) || !parse_int_value(value, config.grid_z)) {
                error = error.empty() ? "Invalid value for --grid-z" : error;
                return false;
            }
        } else if (arg == "--seed") {
            if (!require_value(argc, argv, i, value, error) || !parse_uint64_value(value, config.seed)) {
                error = error.empty() ? "Invalid value for --seed" : error;
                return false;
            }
        } else if (arg == "--experiment-name") {
            if (!require_value(argc, argv, i, config.experiment_name, error)) {
                return false;
            }
        } else if (arg == "--output-dir") {
            if (!require_value(argc, argv, i, config.output_dir, error)) {
                return false;
            }
        } else if (arg == "--device") {
            if (!require_value(argc, argv, i, value, error) || !parse_int_value(value, config.device_id)) {
                error = error.empty() ? "Invalid value for --device" : error;
                return false;
            }
        } else if (arg == "--sync-mode") {
            if (!require_value(argc, argv, i, value, error)) {
                return false;
            }
            if (value == "blocking") {
                config.sync_mode = SyncMode::Blocking;
            } else if (value == "spin") {
                config.sync_mode = SyncMode::Spin;
            } else {
                error = "--sync-mode must be either blocking or spin";
                return false;
            }
        } else if (arg == "--kernel-type") {
            if (!require_value(argc, argv, i, value, error)) {
                return false;
            }
            if (!parse_kernel_type_value(value, config.kernel_type)) {
                error = "--kernel-type must be one of: busy_wait, compute, memory, mixed";
                return false;
            }
        } else {
            error = "Unknown argument: " + arg;
            return false;
        }
    }

    return validate_config(config, error);
}

std::string usage(const char *program_name) {
    std::ostringstream out;
    out << "Usage:\n"
        << "  mpirun -np N " << program_name << " [options]\n\n"
        << "Options:\n"
        << "  --threads-per-process T     Host threads created by each MPI rank (default: 1)\n"
        << "  --kernels-per-thread K      Sequential CUDA kernel requests per host thread (default: 1)\n"
        << "  --warmup-kernels W          Warm-up kernels per host thread, excluded from CSV (default: 20)\n"
        << "  --flush-every N             Flush CSV output every N rows per rank (default: 1000)\n"
        << "  --gpu-telemetry on|off      Collect nvidia-smi telemetry in a side CSV (default: on)\n"
        << "  --telemetry-interval-ms MS  GPU telemetry sampling interval (default: 1000)\n"
        << "  --arrival-min-ms a          Minimum inter-arrival wait in milliseconds (default: 1)\n"
        << "  --arrival-max-ms b          Maximum inter-arrival wait in milliseconds (default: 1)\n"
        << "  --kernel-min-us d           Minimum requested busy-wait duration in microseconds (default: 100)\n"
        << "  --kernel-max-us e           Maximum requested busy-wait duration in microseconds (default: 100)\n"
        << "  --blocks-x x                CUDA gridDim.x (default: 1)\n"
        << "  --threads-per-block y       CUDA blockDim.x (default: 256)\n"
        << "  --grid-z z                  CUDA gridDim.z (default: 1)\n"
        << "  --seed S                    Global seed (default: SEED from .env or 42)\n"
        << "  --experiment-name NAME      Name stored in CSV and used in output filename (default: experiment)\n"
        << "  --output-dir DIR            Output directory (default: OUTPUT_DIR from .env or resultados)\n"
        << "  --device DEVICE_ID          CUDA device ID (default: DEFAULT_DEVICE from .env or 0)\n"
        << "  --sync-mode blocking|spin   CUDA host synchronization mode (default: blocking)\n"
        << "  --kernel-type TYPE          busy_wait, compute, memory, or mixed (default: busy_wait)\n"
        << "  --help                      Show this message\n";
    return out.str();
}

std::string sync_mode_to_string(SyncMode mode) {
    return mode == SyncMode::Blocking ? "blocking" : "spin";
}

std::string kernel_type_to_string(KernelType type) {
    switch (type) {
        case KernelType::BusyWait:
            return "busy_wait";
        case KernelType::Compute:
            return "compute";
        case KernelType::Memory:
            return "memory";
        case KernelType::Mixed:
            return "mixed";
    }
    return "unknown";
}

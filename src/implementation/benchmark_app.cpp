#include "benchmark_app.hpp"

#include "config.hpp"
#include "env_loader.hpp"
#include "experiment.hpp"
#include "timer.hpp"

#include <mpi.h>

#include <array>
#include <cstring>
#include <exception>
#include <iostream>
#include <string>

int run_benchmark_app(int argc, char **argv, int mpi_world_size, int mpi_rank) {
    try {
        const EnvMap env = load_env_file(".env");

        ExperimentConfig config;
        std::string parse_error;
        if (!parse_command_line(argc, argv, env, config, parse_error)) {
            if (mpi_rank == 0) {
                if (!parse_error.empty()) {
                    std::cerr << "Error: " << parse_error << "\n\n";
                }
                std::cerr << usage(argv[0]);
            }
            return 1;
        }

        if (config.help_requested) {
            if (mpi_rank == 0) {
                std::cout << usage(argv[0]);
            }
            return 0;
        }

        std::array<char, 32> timestamp_buffer{};
        if (mpi_rank == 0) {
            const std::string timestamp = Timer::timestamp_yyyymmdd_hhmmss();
            std::strncpy(timestamp_buffer.data(), timestamp.c_str(), timestamp_buffer.size() - 1);
        }
        MPI_Bcast(timestamp_buffer.data(), static_cast<int>(timestamp_buffer.size()), MPI_CHAR, 0, MPI_COMM_WORLD);

        run_experiment(config, mpi_world_size, mpi_rank, std::string(timestamp_buffer.data()));
    } catch (const std::exception &ex) {
        std::cerr << "[rank " << mpi_rank << "] Fatal error: " << ex.what() << '\n';
        return 1;
    } catch (...) {
        std::cerr << "[rank " << mpi_rank << "] Fatal unknown error\n";
        return 1;
    }

    return 0;
}

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

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int mpi_rank = 0;
    int mpi_world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_world_size);

    int exit_code = 0;

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
            MPI_Finalize();
            return 1;
        }

        if (config.help_requested) {
            if (mpi_rank == 0) {
                std::cout << usage(argv[0]);
            }
            MPI_Finalize();
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
        exit_code = 1;
    } catch (...) {
        std::cerr << "[rank " << mpi_rank << "] Fatal unknown error\n";
        exit_code = 1;
    }

    int global_exit_code = 0;
    MPI_Allreduce(&exit_code, &global_exit_code, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

    MPI_Finalize();
    return global_exit_code;
}

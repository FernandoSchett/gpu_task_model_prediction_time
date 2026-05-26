#include "benchmark_app.hpp"

#include <mpi.h>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int mpi_rank = 0;
    int mpi_world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_world_size);

    const int exit_code = run_benchmark_app(argc, argv, mpi_world_size, mpi_rank);

    int global_exit_code = 0;
    MPI_Allreduce(&exit_code, &global_exit_code, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

    MPI_Finalize();
    return global_exit_code;
}

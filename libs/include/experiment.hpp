#ifndef EXPERIMENT_HPP
#define EXPERIMENT_HPP

#include "config.hpp"

#include <string>

void run_experiment(const ExperimentConfig &config,
                    int mpi_world_size,
                    int mpi_rank,
                    const std::string &run_timestamp);

#endif

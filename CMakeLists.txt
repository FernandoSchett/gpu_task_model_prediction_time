cmake_minimum_required(VERSION 3.24)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA architectures")
endif()

project(gpu_task_model_prediction_time LANGUAGES CXX CUDA)

find_package(MPI COMPONENTS CXX QUIET)
find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

add_executable(main
    main.cpp
    src/implementation/benchmark_app.cpp
    libs/src/experiment.cpp
    libs/src/csv_writer.cpp
    libs/src/config.cpp
    libs/src/env_loader.cpp
    libs/src/gpu_telemetry.cpp
    libs/src/timer.cpp
    libs/src/cuda_kernels.cu
)

target_compile_features(main PRIVATE cxx_std_17 cuda_std_17)
target_include_directories(main PRIVATE
    src/include
    libs/include
)
target_link_libraries(main PRIVATE CUDA::cudart Threads::Threads)

if(MPI_CXX_FOUND)
    target_link_libraries(main PRIVATE MPI::MPI_CXX)
else()
    find_program(MPI_CXX_WRAPPER mpicxx REQUIRED)
    execute_process(
        COMMAND ${MPI_CXX_WRAPPER} --showme:compile
        OUTPUT_VARIABLE MPI_WRAPPER_COMPILE_FLAGS
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    execute_process(
        COMMAND ${MPI_CXX_WRAPPER} --showme:link
        OUTPUT_VARIABLE MPI_WRAPPER_LINK_FLAGS
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    separate_arguments(MPI_WRAPPER_COMPILE_FLAGS)
    separate_arguments(MPI_WRAPPER_LINK_FLAGS)
    target_compile_options(main PRIVATE ${MPI_WRAPPER_COMPILE_FLAGS})
    target_link_libraries(main PRIVATE ${MPI_WRAPPER_LINK_FLAGS})
endif()

set_target_properties(main PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_RUNTIME_LIBRARY Shared
)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

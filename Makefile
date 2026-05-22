CXX := mpicxx
NVCC := nvcc

TARGET := main
BUILD_DIR := build
SRC_DIR := src
INCLUDE_DIRS := \
	$(SRC_DIR)/config \
	$(SRC_DIR)/cuda \
	$(SRC_DIR)/experiment \
	$(SRC_DIR)/io \
	$(SRC_DIR)/telemetry \
	$(SRC_DIR)/timing

CUDA_HOME ?= /usr/local/cuda
CUDA_INC ?= $(CUDA_HOME)/include
CUDA_LIBDIR ?= $(CUDA_HOME)/lib64

CPP_SRCS := main.cpp \
	src/experiment/experiment.cpp \
	src/io/csv_writer.cpp \
	src/config/config.cpp \
	src/config/env_loader.cpp \
	src/telemetry/gpu_telemetry.cpp \
	src/timing/timer.cpp

CU_SRCS := src/cuda/cuda_kernels.cu

CPP_OBJS := $(patsubst %.cpp,$(BUILD_DIR)/%.o,$(CPP_SRCS))
CU_OBJS := $(patsubst %.cu,$(BUILD_DIR)/%.o,$(CU_SRCS))
DEPS := $(CPP_OBJS:.o=.d)

INCLUDES := $(addprefix -I,$(INCLUDE_DIRS)) -I$(CUDA_INC)

CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra -Wpedantic -pthread $(INCLUDES)
NVCCFLAGS ?= -std=c++17 -O2 $(INCLUDES)
LDFLAGS ?= -pthread
CUDA_LIBS ?= -L$(CUDA_LIBDIR) -lcudart

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(CPP_OBJS) $(CU_OBJS)
	$(CXX) $(LDFLAGS) -o $@ $^ $(CUDA_LIBS)

$(BUILD_DIR)/%.o: %.cpp
	mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -MMD -MP -c $< -o $@

$(BUILD_DIR)/%.o: %.cu
	mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

run: $(TARGET)
	mkdir -p resultados
	mpirun -np 1 ./$(TARGET) \
		--threads-per-process 2 \
		--kernels-per-thread 10 \
		--warmup-kernels 20 \
		--flush-every 1000 \
		--gpu-telemetry on \
		--telemetry-interval-ms 1000 \
		--arrival-min-ms 1 \
		--arrival-max-ms 5 \
		--kernel-min-us 100 \
		--kernel-max-us 500 \
		--blocks-x 16 \
		--threads-per-block 128 \
		--grid-z 1 \
		--kernel-type busy_wait \
		--experiment-name make_run

clean:
	rm -rf $(BUILD_DIR) $(TARGET)

-include $(DEPS)

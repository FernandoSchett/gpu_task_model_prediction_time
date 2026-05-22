#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

OUTPUT_DIR="${OUTPUT_DIR:-resultados}"
DEFAULT_DEVICE="${DEFAULT_DEVICE:-0}"
SYNC_MODE="${SYNC_MODE:-blocking}"
WARMUP_KERNELS="${WARMUP_KERNELS:-20}"
FLUSH_EVERY="${FLUSH_EVERY:-1000}"
GPU_TELEMETRY="${GPU_TELEMETRY:-on}"
TELEMETRY_INTERVAL_MS="${TELEMETRY_INTERVAL_MS:-1000}"

mkdir -p "${OUTPUT_DIR}"
make

MPI_RANKS=(1 2 4)

SEEDS=(42)

THREADS_PER_PROCESS=(1 2 4 8)

KERNELS_PER_THREAD=300

BLOCKS_X=(1 4 8 16 32 64 128 256)

THREADS_PER_BLOCK=(32 64 128 256 512)

GRID_Z=1

KERNEL_RANGES=(
  "100:500"
)

ARRIVAL_RANGES=(
  "1:5"
)

KERNEL_TYPES=(
  "busy_wait"
  "compute"
  "memory"
  "mixed"
)

for ranks in "${MPI_RANKS[@]}"; do
  for threads in "${THREADS_PER_PROCESS[@]}"; do
    for blocks_x in "${BLOCKS_X[@]}"; do
      for threads_per_block in "${THREADS_PER_BLOCK[@]}"; do
        for kernel_type in "${KERNEL_TYPES[@]}"; do
          for kernel_range in "${KERNEL_RANGES[@]}"; do
            IFS=":" read -r kernel_min_us kernel_max_us <<< "${kernel_range}"
            for arrival_range in "${ARRIVAL_RANGES[@]}"; do
              IFS=":" read -r arrival_min_ms arrival_max_ms <<< "${arrival_range}"
              for seed in "${SEEDS[@]}"; do

                experiment_name="s${seed}_r${ranks}_t${threads}_k${KERNELS_PER_THREAD}_w${WARMUP_KERNELS}_kt${kernel_type}_bx${blocks_x}_tpb${threads_per_block}_gz${GRID_Z}_ku${kernel_min_us}-${kernel_max_us}_am${arrival_min_ms}-${arrival_max_ms}"

                echo "Running ${experiment_name}"
                mpirun -np "${ranks}" ./main \
                  --threads-per-process "${threads}" \
                  --kernels-per-thread "${KERNELS_PER_THREAD}" \
                  --warmup-kernels "${WARMUP_KERNELS}" \
                  --flush-every "${FLUSH_EVERY}" \
                  --gpu-telemetry "${GPU_TELEMETRY}" \
                  --telemetry-interval-ms "${TELEMETRY_INTERVAL_MS}" \
                  --arrival-min-ms "${arrival_min_ms}" \
                  --arrival-max-ms "${arrival_max_ms}" \
                  --kernel-min-us "${kernel_min_us}" \
                  --kernel-max-us "${kernel_max_us}" \
                  --blocks-x "${blocks_x}" \
                  --threads-per-block "${threads_per_block}" \
                  --grid-z "${GRID_Z}" \
                  --seed "${seed}" \
                  --experiment-name "${experiment_name}" \
                  --output-dir "${OUTPUT_DIR}" \
                  --device "${DEFAULT_DEVICE}" \
                  --sync-mode "${SYNC_MODE}" \
                  --kernel-type "${kernel_type}"
              done
            done
          done
        done
      done
    done
  done
done

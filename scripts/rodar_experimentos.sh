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
GPU_TELEMETRY_DURING="${GPU_TELEMETRY_DURING:-off}"
TELEMETRY_INTERVAL_MS="${TELEMETRY_INTERVAL_MS:-1000}"
EXPERIMENT_CONFIG="${EXPERIMENT_CONFIG:-sweep_padrao}"
EXPERIMENT_CONFIG_PATH="${EXPERIMENT_CONFIG_PATH:-experimentos/${EXPERIMENT_CONFIG}.json}"

if [[ ! -f "${EXPERIMENT_CONFIG_PATH}" ]]; then
  echo "Arquivo de configuracao de experimento nao encontrado: ${EXPERIMENT_CONFIG_PATH}" >&2
  exit 1
fi

eval "$(
  python3 - "${EXPERIMENT_CONFIG_PATH}" <<'PY'
import json
import shlex
import sys

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as file:
    config = json.load(file)


def require(name):
    if name not in config:
        raise SystemExit(f"Campo obrigatorio ausente no JSON: {name}")
    return config[name]


def bash_array(name, values):
    print(f"{name}=({' '.join(shlex.quote(str(value)) for value in values)})")


def bash_scalar(name, value):
    print(f"{name}={shlex.quote(str(value))}")


def optional_bash_scalar(name, value):
    if value is not None:
        bash_scalar(name, value)


def range_value(item, min_key, max_key):
    if isinstance(item, str):
        return item
    return f"{item[min_key]}:{item[max_key]}"


def kernel_range_value(item):
    if isinstance(item, str):
        return item
    if "min_us" in item and "max_us" in item:
        return f"{item['min_us']}:{item['max_us']}"
    if "min_s" in item and "max_s" in item:
        return f"{int(float(item['min_s']) * 1_000_000)}:{int(float(item['max_s']) * 1_000_000)}"
    raise SystemExit("kernel_ranges deve usar min_us/max_us ou min_s/max_s")


bash_array("MPI_RANKS", require("mpi_ranks"))
bash_array("SEEDS", require("seeds"))
bash_array("THREADS_PER_PROCESS", require("threads_per_process"))
bash_scalar("KERNELS_PER_THREAD", require("kernels_per_thread"))
optional_bash_scalar("WARMUP_KERNELS", config.get("warmup_kernels"))
bash_array("BLOCKS_X", require("blocks_x"))
bash_array("THREADS_PER_BLOCK", require("threads_per_block"))
bash_scalar("GRID_Z", require("grid_z"))
bash_array(
    "KERNEL_RANGES",
    [kernel_range_value(item) for item in require("kernel_ranges")],
)
bash_array(
    "ARRIVAL_RANGES",
    [range_value(item, "min_ms", "max_ms") for item in require("arrival_ranges")],
)
bash_array("KERNEL_TYPES", require("kernel_types"))
PY
)"

mkdir -p "${OUTPUT_DIR}"
make

echo "Usando configuracao de experimento: ${EXPERIMENT_CONFIG_PATH}"

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
                experiment_output_dir="${OUTPUT_DIR}/${experiment_name}"
                mkdir -p "${experiment_output_dir}"

                echo "Running ${experiment_name} -> ${experiment_output_dir}"
                mpirun -np "${ranks}" ./main \
                  --threads-per-process "${threads}" \
                  --kernels-per-thread "${KERNELS_PER_THREAD}" \
                  --warmup-kernels "${WARMUP_KERNELS}" \
                  --flush-every "${FLUSH_EVERY}" \
                  --gpu-telemetry "${GPU_TELEMETRY}" \
                  --gpu-telemetry-during "${GPU_TELEMETRY_DURING}" \
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
                  --kernel-type "${kernel_type}" \
                  < /dev/null
              done
            done
          done
        done
      done
    done
  done
done

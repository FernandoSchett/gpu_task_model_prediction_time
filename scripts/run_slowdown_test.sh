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

SLOWDOWN_EXPERIMENT_CONFIG="${SLOWDOWN_EXPERIMENT_CONFIG:-slowdown_test}"
SLOWDOWN_EXPERIMENT_CONFIG_PATH="${SLOWDOWN_EXPERIMENT_CONFIG_PATH:-experimentos/${SLOWDOWN_EXPERIMENT_CONFIG}.json}"
GPU_TELEMETRY="${GPU_TELEMETRY:-}"
GPU_TELEMETRY_DURING="${GPU_TELEMETRY_DURING:-}"
TELEMETRY_INTERVAL_MS="${TELEMETRY_INTERVAL_MS:-}"
BLOCK_ID="${BLOCK_ID:-0}"
REPETITION_ID="${REPETITION_ID:-}"

if [[ ! -f "${SLOWDOWN_EXPERIMENT_CONFIG_PATH}" ]]; then
  echo "Arquivo de configuracao de slowdown nao encontrado: ${SLOWDOWN_EXPERIMENT_CONFIG_PATH}" >&2
  exit 1
fi

eval "$(
  python3 - "${SLOWDOWN_EXPERIMENT_CONFIG_PATH}" <<'PY'
import json
import shlex
import sys

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as file:
    config = json.load(file)


def optional_bash_scalar(name, value):
    if value is not None:
        print(f"{name}={shlex.quote(str(value))}")


optional_bash_scalar("JSON_GPU_TELEMETRY", config.get("gpu_telemetry"))
optional_bash_scalar("JSON_GPU_TELEMETRY_DURING", config.get("gpu_telemetry_during"))
optional_bash_scalar("JSON_TELEMETRY_INTERVAL_MS", config.get("telemetry_interval_ms"))
PY
)"

GPU_TELEMETRY="${JSON_GPU_TELEMETRY:-${GPU_TELEMETRY:-on}}"
GPU_TELEMETRY_DURING="${JSON_GPU_TELEMETRY_DURING:-${GPU_TELEMETRY_DURING:-off}}"
TELEMETRY_INTERVAL_MS="${JSON_TELEMETRY_INTERVAL_MS:-${TELEMETRY_INTERVAL_MS:-1000}}"

make
mkdir -p resultados

echo "Usando configuracao de slowdown: ${SLOWDOWN_EXPERIMENT_CONFIG_PATH}"
echo "Telemetria GPU: gpu_telemetry=${GPU_TELEMETRY}, gpu_telemetry_during=${GPU_TELEMETRY_DURING}, telemetry_interval_ms=${TELEMETRY_INTERVAL_MS}"

BLOCK_ID_COUNTER="${BLOCK_ID}"

while IFS=$'\t' read -r \
  experiment_name \
  mpi_ranks \
  threads_per_process \
  kernels_per_thread \
  arrival_min_ms \
  arrival_max_ms \
  kernel_min_us \
  kernel_max_us \
  blocks_x \
  threads_per_block \
  grid_z \
  seed \
  kernel_type; do

  block_id="${BLOCK_ID_COUNTER}"
  BLOCK_ID_COUNTER=$((BLOCK_ID_COUNTER + 1))
  repetition_id="${REPETITION_ID:-${seed}}"

  mkdir -p "resultados/${experiment_name}"
  echo "Running ${experiment_name} -> resultados/${experiment_name} (block_id=${block_id}, repetition_id=${repetition_id})"
  mpirun -np "${mpi_ranks}" ./main \
    --threads-per-process "${threads_per_process}" \
    --kernels-per-thread "${kernels_per_thread}" \
    --arrival-min-ms "${arrival_min_ms}" \
    --arrival-max-ms "${arrival_max_ms}" \
    --kernel-min-us "${kernel_min_us}" \
    --kernel-max-us "${kernel_max_us}" \
    --blocks-x "${blocks_x}" \
    --threads-per-block "${threads_per_block}" \
    --grid-z "${grid_z}" \
    --seed "${seed}" \
    --repetition-id "${repetition_id}" \
    --block-id "${block_id}" \
    --experiment-name "${experiment_name}" \
    --gpu-telemetry "${GPU_TELEMETRY}" \
    --gpu-telemetry-during "${GPU_TELEMETRY_DURING}" \
    --telemetry-interval-ms "${TELEMETRY_INTERVAL_MS}" \
    --kernel-type "${kernel_type}" \
    < /dev/null
done < <(
  python3 - "${SLOWDOWN_EXPERIMENT_CONFIG_PATH}" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as file:
    config = json.load(file)

fields = [
    "experiment_name",
    "mpi_ranks",
    "threads_per_process",
    "kernels_per_thread",
    "arrival_min_ms",
    "arrival_max_ms",
    "kernel_min_us",
    "kernel_max_us",
    "blocks_x",
    "threads_per_block",
    "grid_z",
    "seed",
]

for run in config.get("runs", []):
    values = []
    for field in fields:
        if field not in run:
            raise SystemExit(f"Campo obrigatorio ausente no run: {field}")
        values.append(str(run[field]))
    values.append(str(run.get("kernel_type", "busy_wait")))
    print("\t".join(values))
PY
)

python3 scripts/py_outros/A0_analisar_slowdown.py

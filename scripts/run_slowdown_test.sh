#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PRESERVED_ENV_VARS=(
  SLOWDOWN_EXPERIMENT_CONFIG_PATH
  BLOCK_ID
  REPETITION_ID
  PYTHON_BIN
)
for var_name in "${PRESERVED_ENV_VARS[@]}"; do
  if [[ -v "${var_name}" ]]; then
    printf -v "PRESERVED_${var_name}" "%s" "${!var_name}"
    printf -v "PRESERVED_${var_name}_SET" "%s" "1"
  else
    printf -v "PRESERVED_${var_name}_SET" "%s" ""
  fi
done

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

for var_name in "${PRESERVED_ENV_VARS[@]}"; do
  set_var_name="PRESERVED_${var_name}_SET"
  if [[ -n "${!set_var_name}" ]]; then
    value_var_name="PRESERVED_${var_name}"
    printf -v "${var_name}" "%s" "${!value_var_name}"
    export "${var_name}"
  fi
done

SLOWDOWN_EXPERIMENT_CONFIG_PATH="${SLOWDOWN_EXPERIMENT_CONFIG_PATH:-experimentos/slowdown_test.json}"
BLOCK_ID="${BLOCK_ID:-0}"
REPETITION_ID="${REPETITION_ID:-}"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
    PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi

if [[ ! -f "${SLOWDOWN_EXPERIMENT_CONFIG_PATH}" ]]; then
  echo "Arquivo de configuracao de slowdown nao encontrado: ${SLOWDOWN_EXPERIMENT_CONFIG_PATH}" >&2
  exit 1
fi

eval "$(
  "${PYTHON_BIN}" - "${SLOWDOWN_EXPERIMENT_CONFIG_PATH}" <<'PY'
import json
import shlex
import sys

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as file:
    config = json.load(file)


def optional_bash_scalar(name, value):
    if value is not None:
        print(f"{name}={shlex.quote(str(value))}")

def require(name):
    if name not in config:
        raise SystemExit(f"Campo obrigatorio ausente no JSON: {name}")
    return config[name]


optional_bash_scalar("OUTPUT_DIR", require("output_dir"))
optional_bash_scalar("DEFAULT_DEVICE", require("default_device"))
optional_bash_scalar("SYNC_MODE", require("sync_mode"))
optional_bash_scalar("WARMUP_KERNELS", require("warmup_kernels"))
optional_bash_scalar("FLUSH_EVERY", require("flush_every"))
optional_bash_scalar("GPU_TELEMETRY", require("gpu_telemetry"))
optional_bash_scalar("GPU_TELEMETRY_DURING", require("gpu_telemetry_during"))
optional_bash_scalar("TELEMETRY_INTERVAL_MS", require("telemetry_interval_ms"))
PY
)"

make
mkdir -p "${OUTPUT_DIR}"

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

  mkdir -p "${OUTPUT_DIR}/${experiment_name}"
  echo "Running ${experiment_name} -> ${OUTPUT_DIR}/${experiment_name} (block_id=${block_id}, repetition_id=${repetition_id})"
  mpirun -np "${mpi_ranks}" ./main \
    --threads-per-process "${threads_per_process}" \
    --kernels-per-thread "${kernels_per_thread}" \
    --warmup-kernels "${WARMUP_KERNELS}" \
    --flush-every "${FLUSH_EVERY}" \
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
    --output-dir "${OUTPUT_DIR}" \
    --device "${DEFAULT_DEVICE}" \
    --sync-mode "${SYNC_MODE}" \
    --gpu-telemetry "${GPU_TELEMETRY}" \
    --gpu-telemetry-during "${GPU_TELEMETRY_DURING}" \
    --telemetry-interval-ms "${TELEMETRY_INTERVAL_MS}" \
    --kernel-type "${kernel_type}" \
    < /dev/null
done < <(
  "${PYTHON_BIN}" - "${SLOWDOWN_EXPERIMENT_CONFIG_PATH}" <<'PY'
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

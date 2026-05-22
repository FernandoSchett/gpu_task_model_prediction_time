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

if [[ ! -f "${SLOWDOWN_EXPERIMENT_CONFIG_PATH}" ]]; then
  echo "Arquivo de configuracao de slowdown nao encontrado: ${SLOWDOWN_EXPERIMENT_CONFIG_PATH}" >&2
  exit 1
fi

make
mkdir -p resultados

echo "Usando configuracao de slowdown: ${SLOWDOWN_EXPERIMENT_CONFIG_PATH}"

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

  echo "Running ${experiment_name}"
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
    --experiment-name "${experiment_name}" \
    --kernel-type "${kernel_type}"
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

python3 scripts/analyze_slowdown.py

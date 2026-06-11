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
EXPERIMENT_CONFIG="${EXPERIMENT_CONFIG:-sweep_padrao}"
EXPERIMENT_CONFIG_PATH="${EXPERIMENT_CONFIG_PATH:-experimentos/${EXPERIMENT_CONFIG}.json}"
TELEMETRY_INTERVAL_MS="${TELEMETRY_INTERVAL_MS:-200}"
RUN_TIMESTAMP_BASE="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
SEEDS="${SEEDS:-67 42}"
RUN_PIPELINES="${RUN_PIPELINES:-true}"

if [[ ! -f "${EXPERIMENT_CONFIG_PATH}" ]]; then
  echo "Arquivo de configuracao de experimento nao encontrado: ${EXPERIMENT_CONFIG_PATH}" >&2
  exit 1
fi

TEMP_DIR="$(mktemp -d /tmp/sweeps_2seeds_XXXXXX)"
trap 'rm -rf "${TEMP_DIR}"' EXIT

NORMAL_RESULTS_DIRS_LIST=()
TELEMETRY_RESULTS_DIRS_LIST=()

make_seed_config() {
  local seed="$1"
  local telemetry="$2"
  local output_path="$3"

  python3 - "${EXPERIMENT_CONFIG_PATH}" "${output_path}" "${seed}" "${telemetry}" "${TELEMETRY_INTERVAL_MS}" <<'PY'
import json
import sys

source_path, target_path, seed, telemetry, telemetry_interval_ms = sys.argv[1:6]
with open(source_path, "r", encoding="utf-8") as file:
    config = json.load(file)

config["seeds"] = [int(seed)]
if telemetry == "on":
    config["name"] = f"{config.get('name', 'sweep')}_telemetry"
    config["gpu_telemetry"] = "on"
    config["gpu_telemetry_during"] = "on"
    config["telemetry_interval_ms"] = int(float(telemetry_interval_ms))
else:
    config["gpu_telemetry"] = "off"
    config["gpu_telemetry_during"] = "off"

with open(target_path, "w", encoding="utf-8") as file:
    json.dump(config, file, indent=2)
    file.write("\n")
PY
}

for seed in ${SEEDS}; do
  normal_timestamp="${RUN_TIMESTAMP_BASE}_seed${seed}"
  telemetry_timestamp="${RUN_TIMESTAMP_BASE}_seed${seed}_telemetry"
  normal_config="${TEMP_DIR}/sweep_seed_${seed}.json"
  telemetry_config="${TEMP_DIR}/sweep_seed_${seed}_telemetry.json"

  make_seed_config "${seed}" "off" "${normal_config}"
  make_seed_config "${seed}" "on" "${telemetry_config}"

  echo "Rodando sweep SEM telemetria com seed=${seed}..."
  RUN_TIMESTAMP="${normal_timestamp}" \
  OUTPUT_DIR="${OUTPUT_DIR}" \
  EXPERIMENT_CONFIG_PATH="${normal_config}" \
  bash scripts/rodar_experimentos.sh

  NORMAL_RESULTS_DIRS_LIST+=("${OUTPUT_DIR}/sweep_moderado_sem_estimativas_${normal_timestamp}")

  echo "Rodando sweep COM telemetria com seed=${seed}..."
  RUN_TIMESTAMP="${telemetry_timestamp}" \
  OUTPUT_DIR="${OUTPUT_DIR}" \
  EXPERIMENT_CONFIG_PATH="${telemetry_config}" \
  TELEMETRY_INTERVAL_MS="${TELEMETRY_INTERVAL_MS}" \
  bash scripts/rodar_experimentos.sh

  TELEMETRY_RESULTS_DIRS_LIST+=("${OUTPUT_DIR}/sweep_moderado_sem_estimativas_telemetry_${telemetry_timestamp}")
done

NORMAL_RESULTS_DIRS="$(printf '%s ' "${NORMAL_RESULTS_DIRS_LIST[@]}")"
TELEMETRY_RESULTS_DIRS="$(printf '%s ' "${TELEMETRY_RESULTS_DIRS_LIST[@]}")"
NORMAL_RESULTS_DIRS="${NORMAL_RESULTS_DIRS% }"
TELEMETRY_RESULTS_DIRS="${TELEMETRY_RESULTS_DIRS% }"

echo "Pastas sem telemetria: ${NORMAL_RESULTS_DIRS}"
echo "Pastas com telemetria: ${TELEMETRY_RESULTS_DIRS}"

if [[ "${RUN_PIPELINES}" == "true" ]] || [[ "${RUN_PIPELINES}" == "1" ]]; then
  echo "Rodando pipelines A e B..."
  NORMAL_RESULTS_DIRS="${NORMAL_RESULTS_DIRS}" \
  TELEMETRY_RESULTS_DIRS="${TELEMETRY_RESULTS_DIRS}" \
  bash scripts/rodar_todas_pipelines.sh
fi

echo "Experimentos e pipelines concluidos."

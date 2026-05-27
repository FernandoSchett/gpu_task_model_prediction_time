#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

EXPERIMENT_CONFIG="${EXPERIMENT_CONFIG:-sweep_padrao}"
EXPERIMENT_CONFIG_PATH="${EXPERIMENT_CONFIG_PATH:-experimentos/${EXPERIMENT_CONFIG}.json}"
TELEMETRY_INTERVAL_MS="${TELEMETRY_INTERVAL_MS:-200}"

if [[ ! -f "${EXPERIMENT_CONFIG_PATH}" ]]; then
  echo "Arquivo de configuracao de experimento nao encontrado: ${EXPERIMENT_CONFIG_PATH}" >&2
  exit 1
fi

TEMP_CONFIG="$(mktemp /tmp/sweep_telemetry_XXXXXX.json)"
trap 'rm -f "${TEMP_CONFIG}"' EXIT

python3 - "${EXPERIMENT_CONFIG_PATH}" "${TEMP_CONFIG}" "${TELEMETRY_INTERVAL_MS}" <<'PY'
import json
import sys

source_path, target_path, telemetry_interval_ms = sys.argv[1:4]
with open(source_path, "r", encoding="utf-8") as file:
    config = json.load(file)

config["name"] = f"{config.get('name', 'sweep')}_telemetry"
config["gpu_telemetry"] = "on"
config["gpu_telemetry_during"] = "on"
config["telemetry_interval_ms"] = int(float(telemetry_interval_ms))

with open(target_path, "w", encoding="utf-8") as file:
    json.dump(config, file, indent=2)
    file.write("\n")
PY

EXPERIMENT_CONFIG_PATH="${TEMP_CONFIG}" bash scripts/rodar_experimentos.sh

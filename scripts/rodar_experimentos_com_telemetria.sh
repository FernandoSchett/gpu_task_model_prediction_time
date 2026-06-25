#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PRESERVED_ENV_VARS=(
  EXPERIMENT_CONFIG_PATH
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

EXPERIMENT_CONFIG_PATH="${EXPERIMENT_CONFIG_PATH:-experimentos/sweep_padrao.json}"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
    PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi

if [[ ! -f "${EXPERIMENT_CONFIG_PATH}" ]]; then
  echo "Arquivo de configuracao de experimento nao encontrado: ${EXPERIMENT_CONFIG_PATH}" >&2
  exit 1
fi

TEMP_CONFIG="$(mktemp /tmp/sweep_telemetry_XXXXXX.json)"
trap 'rm -f "${TEMP_CONFIG}"' EXIT

"${PYTHON_BIN}" - "${EXPERIMENT_CONFIG_PATH}" "${TEMP_CONFIG}" <<'PY'
import json
import sys

source_path, target_path = sys.argv[1:3]
with open(source_path, "r", encoding="utf-8") as file:
    config = json.load(file)

config["name"] = f"{config.get('name', 'sweep')}_telemetry"
config["gpu_telemetry"] = "on"
config["gpu_telemetry_during"] = "on"

with open(target_path, "w", encoding="utf-8") as file:
    json.dump(config, file, indent=2)
    file.write("\n")
PY

EXPERIMENT_CONFIG_PATH="${TEMP_CONFIG}" bash scripts/rodar_experimentos.sh

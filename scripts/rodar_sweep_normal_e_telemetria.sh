#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PRESERVED_ENV_VARS=(
  RUN_TIMESTAMP
  EXPERIMENT_CONFIG_PATH
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

RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
EXPERIMENT_CONFIG_PATH="${EXPERIMENT_CONFIG_PATH:-experimentos/sweep_padrao.json}"

echo "Rodando sweep normal..."
RUN_TIMESTAMP="${RUN_TIMESTAMP}" \
EXPERIMENT_CONFIG_PATH="${EXPERIMENT_CONFIG_PATH}" \
bash scripts/rodar_experimentos.sh

echo "Rodando sweep com telemetria..."
RUN_TIMESTAMP="${RUN_TIMESTAMP}_telemetry" \
EXPERIMENT_CONFIG_PATH="${EXPERIMENT_CONFIG_PATH}" \
bash scripts/rodar_experimentos_com_telemetria.sh

echo "Sweeps concluidos."

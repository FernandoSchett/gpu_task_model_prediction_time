#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-resultados}"
EXPERIMENT_CONFIG="${EXPERIMENT_CONFIG:-sweep_padrao}"
EXPERIMENT_CONFIG_PATH="${EXPERIMENT_CONFIG_PATH:-experimentos/${EXPERIMENT_CONFIG}.json}"

echo "Rodando sweep normal..."
RUN_TIMESTAMP="${RUN_TIMESTAMP}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
EXPERIMENT_CONFIG_PATH="${EXPERIMENT_CONFIG_PATH}" \
bash scripts/rodar_experimentos.sh

echo "Rodando sweep com telemetria..."
RUN_TIMESTAMP="${RUN_TIMESTAMP}_telemetry" \
OUTPUT_DIR="${OUTPUT_DIR}" \
EXPERIMENT_CONFIG_PATH="${EXPERIMENT_CONFIG_PATH}" \
bash scripts/rodar_experimentos_com_telemetria.sh

echo "Sweeps concluidos."

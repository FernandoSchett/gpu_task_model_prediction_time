#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

RESULTS_DIR="${RESULTS_DIR:-}"
ANALYSIS_DIR="${ANALYSIS_DIR:-}"
MAX_ROWS="${MAX_ROWS:-120000}"
TARGETS="${TARGETS:-response_time_us queueing_delay_us slowdown}"
GPU_TARGETS="${GPU_TARGETS:-10 50 100 120}"
MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-gpu-task-model}"
export MPLCONFIGDIR

if [[ -z "${RESULTS_DIR}" ]]; then
  RESULTS_DIR="$(
    find resultados -mindepth 1 -maxdepth 1 -type d \
      \( -name 'sweep_*' -o -name '*sweep*' \) \
      -printf '%T@ %p\n' 2>/dev/null \
      | sort -nr \
      | awk 'NR == 1 {print $2}' \
      || true
  )"
fi

if [[ -z "${RESULTS_DIR}" || ! -d "${RESULTS_DIR}" ]]; then
  echo "Pasta de resultados do sweep nao encontrada. Use RESULTS_DIR=/caminho/do/sweep." >&2
  exit 1
fi

if [[ -z "${ANALYSIS_DIR}" ]]; then
  ANALYSIS_DIR="resultados/analises_regressao/$(basename "${RESULTS_DIR}")"
fi

python3 scripts/gerar_resultados_sweep.py \
  --results-dir "${RESULTS_DIR}" \
  --analysis-dir "${ANALYSIS_DIR}" \
  --targets ${TARGETS} \
  --gpu-targets ${GPU_TARGETS}

python3 scripts/treinar_modelos_sweep.py \
  --results-dir "${RESULTS_DIR}" \
  --analysis-dir "${ANALYSIS_DIR}" \
  --max-rows "${MAX_ROWS}"

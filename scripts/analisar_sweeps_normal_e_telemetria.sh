#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

NORMAL_RESULTS_DIRS="${NORMAL_RESULTS_DIRS:-${NORMAL_RESULTS_DIR:-}}"
TELEMETRY_RESULTS_DIRS="${TELEMETRY_RESULTS_DIRS:-${TELEMETRY_RESULTS_DIR:-}}"
ANALYSIS_ROOT="${ANALYSIS_ROOT:-resultados/analises_regressao}"
MAX_ROWS="${MAX_ROWS:-120000}"
TARGETS="${TARGETS:-response_time_us queueing_delay_us slowdown}"
GPU_TARGETS="${GPU_TARGETS:-10 50 100 120}"
CV_FOLDS="${CV_FOLDS:-5}"
OPTIMIZE_HYPERPARAMS="${OPTIMIZE_HYPERPARAMS:-false}"
OPTUNA_TRIALS="${OPTUNA_TRIALS:-20}"

matching_dirs() {
  local pattern="$1"
  find resultados -mindepth 1 -maxdepth 1 -type d -name "${pattern}" -printf '%T@ %p\n' \
    | sort -n \
    | awk '{print $2}'
}

if [[ -z "${NORMAL_RESULTS_DIRS}" ]]; then
  NORMAL_RESULTS_DIRS="$(matching_dirs 'sweep_moderado_sem_estimativas_[0-9]*' || true)"
fi

if [[ -z "${TELEMETRY_RESULTS_DIRS}" ]]; then
  TELEMETRY_RESULTS_DIRS="$(matching_dirs 'sweep_moderado_sem_estimativas_telemetry_*' || true)"
fi

run_one() {
  local label="$1"
  local results_dirs="$2"
  if [[ -z "${results_dirs}" ]]; then
    echo "Pastas do sweep ${label} nao encontradas." >&2
    exit 1
  fi

  local analysis_dir="${ANALYSIS_ROOT}/${label}_sweep_moderado_sem_estimativas_agrupado"
  echo "Analise ${label}: ${results_dirs} -> ${analysis_dir}"
  RESULTS_DIRS="${results_dirs}" \
  ANALYSIS_DIR="${analysis_dir}" \
  MAX_ROWS="${MAX_ROWS}" \
  TARGETS="${TARGETS}" \
  GPU_TARGETS="${GPU_TARGETS}" \
  CV_FOLDS="${CV_FOLDS}" \
  OPTIMIZE_HYPERPARAMS="${OPTIMIZE_HYPERPARAMS}" \
  OPTUNA_TRIALS="${OPTUNA_TRIALS}" \
  bash scripts/analisar_regressoes_sweep.sh
}

run_one "sem_telemetria" "${NORMAL_RESULTS_DIRS}"
run_one "com_telemetria" "${TELEMETRY_RESULTS_DIRS}"

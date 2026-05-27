#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

NORMAL_RESULTS_DIRS="${NORMAL_RESULTS_DIRS:-${NORMAL_RESULTS_DIR:-}}"
TELEMETRY_RESULTS_DIRS="${TELEMETRY_RESULTS_DIRS:-${TELEMETRY_RESULTS_DIR:-}}"

ANALYSIS_ROOT="${ANALYSIS_ROOT:-resultados/analises_regressao}"
MAX_ROWS="${MAX_ROWS:-0}"
TARGETS="${TARGETS:-response_time_us queueing_delay_us slowdown}"
GPU_TARGETS="${GPU_TARGETS:-10 50 100 120}"
CV_FOLDS="${CV_FOLDS:-5}"
OPTIMIZE_HYPERPARAMS="${OPTIMIZE_HYPERPARAMS:-false}"
OPTUNA_TRIALS="${OPTUNA_TRIALS:-20}"

if [[ ! -f ".env" ]]; then
  echo "Erro: arquivo .env nao encontrado. Crie um .env com SEED=42." >&2
  exit 1
fi

if ! grep -qE '^SEED=' ".env"; then
  echo "Erro: .env existe, mas nao contem SEED=..." >&2
  exit 1
fi

if [[ ! -f "scripts/03_regressor_analysis.py" ]]; then
  echo "Erro: scripts/03_regressor_analysis.py nao encontrado." >&2
  exit 1
fi

if [[ ! -f "scripts/analisar_regressoes_sweep.sh" ]]; then
  echo "Erro: scripts/analisar_regressoes_sweep.sh nao encontrado." >&2
  exit 1
fi

matching_dirs() {
  local pattern="$1"

  if [[ ! -d "resultados" ]]; then
    return 0
  fi

  find resultados -mindepth 1 -maxdepth 1 -type d -name "${pattern}" -print0 \
    | xargs -0 -r stat -c '%Y %n' \
    | sort -n \
    | cut -d' ' -f2- \
    | paste -sd ' ' -
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

  echo "Analise ${label}:"
  echo "  results_dirs=${results_dirs}"
  echo "  analysis_dir=${analysis_dir}"
  echo "  targets=${TARGETS}"
  echo "  max_rows=${MAX_ROWS}"
  echo "  cv_folds=${CV_FOLDS}"
  echo "  optimize_hyperparams=${OPTIMIZE_HYPERPARAMS}"
  echo "  optuna_trials=${OPTUNA_TRIALS}"

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
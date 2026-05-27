#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

RESULTS_DIR="${RESULTS_DIR:-}"
RESULTS_DIRS="${RESULTS_DIRS:-${RESULTS_DIR}}"
ANALYSIS_DIR="${ANALYSIS_DIR:-}"
MAX_ROWS="${MAX_ROWS:-120000}"
TARGETS="${TARGETS:-response_time_us queueing_delay_us slowdown}"
GPU_TARGETS="${GPU_TARGETS:-10 50 100 120}"
CV_FOLDS="${CV_FOLDS:-5}"
OPTIMIZE_HYPERPARAMS="${OPTIMIZE_HYPERPARAMS:-false}"
OPTUNA_TRIALS="${OPTUNA_TRIALS:-20}"
MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-gpu-task-model}"
export MPLCONFIGDIR

if [[ -z "${RESULTS_DIRS}" ]]; then
  RESULTS_DIRS="$(
    find resultados -mindepth 1 -maxdepth 1 -type d \
      \( -name 'sweep_*' -o -name '*sweep*' \) \
      -printf '%T@ %p\n' 2>/dev/null \
      | sort -nr \
      | awk 'NR == 1 {print $2}' \
      || true
  )"
fi

if [[ -z "${RESULTS_DIRS}" ]]; then
  echo "Pasta de resultados do sweep nao encontrada. Use RESULTS_DIRS='dir1 dir2 ...'." >&2
  exit 1
fi

if [[ -z "${ANALYSIS_DIR}" ]]; then
  first_results_dir="${RESULTS_DIRS%% *}"
  ANALYSIS_DIR="resultados/analises_regressao/$(basename "${first_results_dir}")"
fi

python3 scripts/gerar_resultados_sweep.py \
  --results-dir ${RESULTS_DIRS} \
  --analysis-dir "${ANALYSIS_DIR}" \
  --targets ${TARGETS} \
  --gpu-targets ${GPU_TARGETS}

COMPARE_ARGS=(
  compare
  --results-dir ${RESULTS_DIRS}
  --analysis-dir "${ANALYSIS_DIR}"
  --jobs-file "${ANALYSIS_DIR}/analysis_jobs.csv"
  --first-sweep
  --max-rows "${MAX_ROWS}"
  --cv-folds "${CV_FOLDS}"
  --optuna-trials "${OPTUNA_TRIALS}"
)

if [[ "${OPTIMIZE_HYPERPARAMS}" == "true" ]] || [[ "${OPTIMIZE_HYPERPARAMS}" == "1" ]]; then
  COMPARE_ARGS+=(--optimize-hyperparams)
fi

python3 scripts/regressor_analysis.py "${COMPARE_ARGS[@]}"

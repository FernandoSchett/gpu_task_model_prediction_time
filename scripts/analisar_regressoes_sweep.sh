#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PRESERVED_ENV_VARS=(
  RESULTS_DIR
  RESULTS_DIRS
  ANALYSIS_DIR
  TARGETS
  GPU_TARGETS
  CV_FOLDS
  DEPENDENCY_ONLY
  DEPENDENCY_CACHE
  MPLCONFIGDIR
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

RESULTS_DIR="${RESULTS_DIR:-}"
RESULTS_DIRS="${RESULTS_DIRS:-${RESULTS_DIR}}"
ANALYSIS_DIR="${ANALYSIS_DIR:-}"
TARGETS="${TARGETS:-response_time_us queueing_delay_us slowdown}"
GPU_TARGETS="${GPU_TARGETS:-10 50 100 120}"
CV_FOLDS="${CV_FOLDS:-5}"
DEPENDENCY_ONLY="${DEPENDENCY_ONLY:-false}"
DEPENDENCY_CACHE="${DEPENDENCY_CACHE:-true}"
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

python3 scripts/02_gerar_resultados_sweep.py \
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
  --cv-folds "${CV_FOLDS}"
)

if [[ "${DEPENDENCY_ONLY}" == "true" ]] || [[ "${DEPENDENCY_ONLY}" == "1" ]]; then
  COMPARE_ARGS+=(--dependency-only)
fi

if [[ "${DEPENDENCY_CACHE}" == "false" ]] || [[ "${DEPENDENCY_CACHE}" == "0" ]]; then
  COMPARE_ARGS+=(--no-dependency-cache)
fi

python3 scripts/03_regressor_analysis.py "${COMPARE_ARGS[@]}"

if [[ -f "scripts/05_plot_best_model_rankings.py" ]]; then
  python3 scripts/05_plot_best_model_rankings.py --analysis-root "$(dirname "${ANALYSIS_DIR}")"
fi

if [[ "${DEPENDENCY_ONLY}" == "true" || "${DEPENDENCY_ONLY}" == "1" ]] && [[ -f "scripts/06_plot_dependency_rankings.py" ]]; then
  python3 scripts/06_plot_dependency_rankings.py --analysis-root "$(dirname "${ANALYSIS_DIR}")"
fi

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PRESERVED_ENV_VARS=(
  NORMAL_RESULTS_DIR
  NORMAL_RESULTS_DIRS
  TELEMETRY_RESULTS_DIR
  TELEMETRY_RESULTS_DIRS
  ANALYSIS_ROOT
  TARGETS
  GPU_TARGETS
  CV_FOLDS
  DEPENDENCY_ONLY
  DEPENDENCY_CACHE
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

NORMAL_RESULTS_DIRS="${NORMAL_RESULTS_DIRS:-${NORMAL_RESULTS_DIR:-}}"
TELEMETRY_RESULTS_DIRS="${TELEMETRY_RESULTS_DIRS:-${TELEMETRY_RESULTS_DIR:-}}"

ANALYSIS_ROOT="${ANALYSIS_ROOT:-resultados/analises_regressao}"
TARGETS="${TARGETS:-response_time_us queueing_delay_us slowdown}"
GPU_TARGETS="${GPU_TARGETS:-10 50 100 120}"
CV_FOLDS="${CV_FOLDS:-5}"
DEPENDENCY_ONLY="${DEPENDENCY_ONLY:-false}"
DEPENDENCY_CACHE="${DEPENDENCY_CACHE:-true}"

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
  echo "  cv_folds=${CV_FOLDS}"
  echo "  dependency_only=${DEPENDENCY_ONLY}"
  echo "  dependency_cache=${DEPENDENCY_CACHE}"

  RESULTS_DIRS="${results_dirs}" \
  ANALYSIS_DIR="${analysis_dir}" \
  TARGETS="${TARGETS}" \
  GPU_TARGETS="${GPU_TARGETS}" \
  CV_FOLDS="${CV_FOLDS}" \
  DEPENDENCY_ONLY="${DEPENDENCY_ONLY}" \
  DEPENDENCY_CACHE="${DEPENDENCY_CACHE}" \
  bash scripts/analisar_regressoes_sweep.sh
}

run_one "sem_telemetria" "${NORMAL_RESULTS_DIRS}"
run_one "com_telemetria" "${TELEMETRY_RESULTS_DIRS}"

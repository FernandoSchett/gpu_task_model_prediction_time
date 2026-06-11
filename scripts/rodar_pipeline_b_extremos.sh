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
  EVT_BLOCK_SIZE
  EVT_THRESHOLD_QUANTILE
  EVT_DECLUSTER_RUN_LENGTH
  EVT_RETURN_QUANTILES
  EVT_CACHE
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

ANALYSIS_ROOT="${ANALYSIS_ROOT:-resultados/analises_regressao}"
TARGETS="${TARGETS:-response_time_us queueing_delay_us slowdown}"
GPU_TARGETS="${GPU_TARGETS:-10 50 100 120}"
EVT_BLOCK_SIZE="${EVT_BLOCK_SIZE:-1024}"
EVT_THRESHOLD_QUANTILE="${EVT_THRESHOLD_QUANTILE:-0.95}"
EVT_DECLUSTER_RUN_LENGTH="${EVT_DECLUSTER_RUN_LENGTH:-50}"
EVT_RETURN_QUANTILES="${EVT_RETURN_QUANTILES:-0.95 0.99 0.999}"
EVT_CACHE="${EVT_CACHE:-true}"
EVT_CACHE_ARGS=()
if [[ "${EVT_CACHE}" == "false" ]] || [[ "${EVT_CACHE}" == "0" ]]; then
  EVT_CACHE_ARGS+=(--no-cache)
fi

matching_dirs() {
  local pattern="$1"
  find resultados -mindepth 1 -maxdepth 1 -type d -name "${pattern}" -print0 2>/dev/null \
    | xargs -0 -r stat -c '%Y %n' \
    | sort -n \
    | cut -d' ' -f2- \
    | paste -sd ' ' -
}

NORMAL_RESULTS_DIRS="${NORMAL_RESULTS_DIRS:-${NORMAL_RESULTS_DIR:-}}"
TELEMETRY_RESULTS_DIRS="${TELEMETRY_RESULTS_DIRS:-${TELEMETRY_RESULTS_DIR:-}}"

if [[ -z "${NORMAL_RESULTS_DIRS}" ]]; then
  NORMAL_RESULTS_DIRS="$(matching_dirs 'sweep_moderado_sem_estimativas_[0-9]*' || true)"
fi
if [[ -z "${TELEMETRY_RESULTS_DIRS}" ]]; then
  TELEMETRY_RESULTS_DIRS="$(matching_dirs 'sweep_moderado_sem_estimativas_telemetry_*' || true)"
fi

run_one() {
  local label="$1"
  local results_dirs="$2"
  local analysis_dir="${ANALYSIS_ROOT}/${label}_sweep_moderado_sem_estimativas_agrupado"

  if [[ -z "${results_dirs}" ]]; then
    echo "Pastas do sweep ${label} nao encontradas." >&2
    exit 1
  fi

  python3 scripts/py_pipeline_A/A1_gerar_manifesto_analise.py \
    --results-dir ${results_dirs} \
    --analysis-dir "${analysis_dir}" \
    --targets ${TARGETS} \
    --gpu-targets ${GPU_TARGETS}

  EVT_BLOCK_SIZE="${EVT_BLOCK_SIZE}" \
  EVT_THRESHOLD_QUANTILE="${EVT_THRESHOLD_QUANTILE}" \
  EVT_DECLUSTER_RUN_LENGTH="${EVT_DECLUSTER_RUN_LENGTH}" \
  python3 scripts/py_pipeline_B/B1_valores_extremos.py \
    --results-dir ${results_dirs} \
    --analysis-dir "${analysis_dir}" \
    --jobs-file "${analysis_dir}/analysis_jobs.csv" \
    --first-sweep \
    --block-size "${EVT_BLOCK_SIZE}" \
    --threshold-quantile "${EVT_THRESHOLD_QUANTILE}" \
    --decluster-run-length "${EVT_DECLUSTER_RUN_LENGTH}" \
    --return-quantiles ${EVT_RETURN_QUANTILES} \
    "${EVT_CACHE_ARGS[@]}"
}

run_one "sem_telemetria" "${NORMAL_RESULTS_DIRS}"
run_one "com_telemetria" "${TELEMETRY_RESULTS_DIRS}"

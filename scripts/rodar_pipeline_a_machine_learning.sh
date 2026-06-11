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
  PIPELINE_A_CLASSICAL
  PIPELINE_A_SEQUENTIAL
  SEQUENCE_LENGTH
  SEQUENCE_STRIDE
  SEQUENCE_MAX_SEQUENCES
  SEQUENCE_EPOCHS
  SEQUENCE_BATCH_SIZE
  SEQUENCE_CACHE
  SEED
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
CV_FOLDS="${CV_FOLDS:-5}"
DEPENDENCY_ONLY="${DEPENDENCY_ONLY:-false}"
DEPENDENCY_CACHE="${DEPENDENCY_CACHE:-true}"
PIPELINE_A_CLASSICAL="${PIPELINE_A_CLASSICAL:-true}"
PIPELINE_A_SEQUENTIAL="${PIPELINE_A_SEQUENTIAL:-true}"
SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-16}"
SEQUENCE_STRIDE="${SEQUENCE_STRIDE:-4}"
SEQUENCE_MAX_SEQUENCES="${SEQUENCE_MAX_SEQUENCES:-200000}"
SEQUENCE_EPOCHS="${SEQUENCE_EPOCHS:-5}"
SEQUENCE_BATCH_SIZE="${SEQUENCE_BATCH_SIZE:-256}"
SEQUENCE_CACHE="${SEQUENCE_CACHE:-true}"
SEED="${SEED:-42}"
SEQUENCE_CACHE_ARGS=()
if [[ "${SEQUENCE_CACHE}" == "false" ]] || [[ "${SEQUENCE_CACHE}" == "0" ]]; then
  SEQUENCE_CACHE_ARGS+=(--no-cache)
fi

CLASSICAL_COMPARE_ARGS=()
if [[ "${DEPENDENCY_ONLY}" == "true" ]] || [[ "${DEPENDENCY_ONLY}" == "1" ]]; then
  CLASSICAL_COMPARE_ARGS+=(--dependency-only)
fi
if [[ "${DEPENDENCY_CACHE}" == "false" ]] || [[ "${DEPENDENCY_CACHE}" == "0" ]]; then
  CLASSICAL_COMPARE_ARGS+=(--no-dependency-cache)
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

  if [[ "${PIPELINE_A_CLASSICAL}" == "true" ]] || [[ "${PIPELINE_A_CLASSICAL}" == "1" ]]; then
    python3 scripts/py_pipeline_A/A2_regressores_classicos.py \
      compare \
      --results-dir ${results_dirs} \
      --analysis-dir "${analysis_dir}" \
      --jobs-file "${analysis_dir}/analysis_jobs.csv" \
      --first-sweep \
      --cv-folds "${CV_FOLDS}" \
      "${CLASSICAL_COMPARE_ARGS[@]}"

    if [[ -f "scripts/py_pipeline_A/A3_rankings_regressores.py" ]]; then
      python3 scripts/py_pipeline_A/A3_rankings_regressores.py --analysis-root "$(dirname "${analysis_dir}")"
    fi

    if [[ ("${DEPENDENCY_ONLY}" == "true" || "${DEPENDENCY_ONLY}" == "1") && -f "scripts/py_pipeline_A/A4_rankings_dependencia.py" ]]; then
      python3 scripts/py_pipeline_A/A4_rankings_dependencia.py --analysis-root "$(dirname "${analysis_dir}")"
    fi
  fi

  if [[ "${PIPELINE_A_SEQUENTIAL}" == "true" ]] || [[ "${PIPELINE_A_SEQUENTIAL}" == "1" ]]; then
    SEQUENCE_LENGTH="${SEQUENCE_LENGTH}" \
    SEQUENCE_STRIDE="${SEQUENCE_STRIDE}" \
    SEQUENCE_MAX_SEQUENCES="${SEQUENCE_MAX_SEQUENCES}" \
    SEQUENCE_EPOCHS="${SEQUENCE_EPOCHS}" \
    SEQUENCE_BATCH_SIZE="${SEQUENCE_BATCH_SIZE}" \
    python3 scripts/py_pipeline_A/A5_modelos_sequenciais.py \
      --results-dir ${results_dirs} \
      --analysis-dir "${analysis_dir}" \
      --jobs-file "${analysis_dir}/analysis_jobs.csv" \
      --first-sweep \
      --seed "${SEED}" \
      "${SEQUENCE_CACHE_ARGS[@]}"
  fi
}

run_one "sem_telemetria" "${NORMAL_RESULTS_DIRS}"
run_one "com_telemetria" "${TELEMETRY_RESULTS_DIRS}"

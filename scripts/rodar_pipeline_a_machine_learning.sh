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
  RUN_CONDITIONS
  TARGETS
  GPU_TARGETS
  CV_FOLDS
  MODEL_N_JOBS
  CLASSICAL_MODEL_ONLY
  CLASSICAL_FORCE_MODEL
  CLASSICAL_CACHE
  CLASSICAL_PARALLEL_JOBS
  DEPENDENCY_ONLY
  DEPENDENCY_CACHE
  PIPELINE_A_CLASSICAL
  PIPELINE_A_SEQUENTIAL
  PYTHON_BIN
  SEQUENCE_LENGTH
  SEQUENCE_STRIDE
  SEQUENCE_MAX_SEQUENCES
  SEQUENCE_EPOCHS
  SEQUENCE_BATCH_SIZE
  SEQUENCE_CNN_MAX_TRIALS
  SEQUENCE_SPLIT_MODE
  SEQUENCE_SAMPLE_MODE
  SEQUENCE_TF_DEVICE
  SEQUENCE_REQUIRE_GPU
  SEQUENCE_MODEL_ONLY
  SEQUENCE_FORCE_MODEL
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
RUN_CONDITIONS="${RUN_CONDITIONS:-sem_telemetria com_telemetria}"
TARGETS="${TARGETS:-response_time_us}"
GPU_TARGETS="${GPU_TARGETS:-10 50 100 120}"
CV_FOLDS="${CV_FOLDS:-5}"
MODEL_N_JOBS="${MODEL_N_JOBS:--1}"
export MODEL_N_JOBS
CLASSICAL_MODEL_ONLY="${CLASSICAL_MODEL_ONLY:-}"
CLASSICAL_FORCE_MODEL="${CLASSICAL_FORCE_MODEL:-false}"
CLASSICAL_CACHE="${CLASSICAL_CACHE:-true}"
CLASSICAL_PARALLEL_JOBS="${CLASSICAL_PARALLEL_JOBS:-1}"
DEPENDENCY_ONLY="${DEPENDENCY_ONLY:-false}"
DEPENDENCY_CACHE="${DEPENDENCY_CACHE:-true}"
PIPELINE_A_CLASSICAL="${PIPELINE_A_CLASSICAL:-true}"
PIPELINE_A_SEQUENTIAL="${PIPELINE_A_SEQUENTIAL:-true}"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
    PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi
if [[ "${PYTHON_BIN}" == "${REPO_ROOT}/.venv/bin/python" ]]; then
  CUDA_LIB_PATH="$("${PYTHON_BIN}" - <<'PY'
from pathlib import Path
import site

paths = []
for site_dir in [Path(path) for path in site.getsitepackages()]:
    nvidia_dir = site_dir / "nvidia"
    if nvidia_dir.exists():
        paths.extend(str(path) for path in sorted(nvidia_dir.glob("*/lib")))
print(":".join(paths))
PY
)"
  export LD_LIBRARY_PATH="${CUDA_LIB_PATH}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi
SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-16}"
SEQUENCE_STRIDE="${SEQUENCE_STRIDE:-1}"
SEQUENCE_MAX_SEQUENCES="${SEQUENCE_MAX_SEQUENCES:-200000}"
SEQUENCE_EPOCHS="${SEQUENCE_EPOCHS:-5}"
SEQUENCE_BATCH_SIZE="${SEQUENCE_BATCH_SIZE:-256}"
SEQUENCE_CNN_MAX_TRIALS="${SEQUENCE_CNN_MAX_TRIALS:-6}"
SEQUENCE_SPLIT_MODE="${SEQUENCE_SPLIT_MODE:-random}"
SEQUENCE_SAMPLE_MODE="${SEQUENCE_SAMPLE_MODE:-random}"
SEQUENCE_TF_DEVICE="${SEQUENCE_TF_DEVICE:-auto}"
SEQUENCE_REQUIRE_GPU="${SEQUENCE_REQUIRE_GPU:-true}"
SEQUENCE_MODEL_ONLY="${SEQUENCE_MODEL_ONLY:-}"
SEQUENCE_FORCE_MODEL="${SEQUENCE_FORCE_MODEL:-false}"
SEQUENCE_CACHE="${SEQUENCE_CACHE:-true}"
SEED="${SEED:-42}"
SEQUENCE_CACHE_ARGS=()
if [[ "${SEQUENCE_CACHE}" == "false" ]] || [[ "${SEQUENCE_CACHE}" == "0" ]]; then
  SEQUENCE_CACHE_ARGS+=(--no-cache)
fi
SEQUENCE_MODEL_ARGS=()
if [[ -n "${SEQUENCE_MODEL_ONLY}" ]]; then
  SEQUENCE_MODEL_ARGS+=(--only-model "${SEQUENCE_MODEL_ONLY}")
fi
if [[ "${SEQUENCE_FORCE_MODEL}" == "true" ]] || [[ "${SEQUENCE_FORCE_MODEL}" == "1" ]]; then
  SEQUENCE_MODEL_ARGS+=(--force-model)
fi

CLASSICAL_COMPARE_ARGS=()
if [[ -n "${CLASSICAL_MODEL_ONLY}" ]]; then
  CLASSICAL_COMPARE_ARGS+=(--only-model "${CLASSICAL_MODEL_ONLY}")
fi
if [[ "${CLASSICAL_FORCE_MODEL}" == "true" ]] || [[ "${CLASSICAL_FORCE_MODEL}" == "1" ]]; then
  CLASSICAL_COMPARE_ARGS+=(--force-model)
fi
if [[ "${CLASSICAL_CACHE}" == "false" ]] || [[ "${CLASSICAL_CACHE}" == "0" ]]; then
  CLASSICAL_COMPARE_ARGS+=(--no-cache)
fi
CLASSICAL_COMPARE_ARGS+=(--parallel-jobs "${CLASSICAL_PARALLEL_JOBS}")
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
  local analysis_dir="${ANALYSIS_ROOT}/pipeline_A/${label}"

  if [[ -z "${results_dirs}" ]]; then
    echo "Pastas do sweep ${label} nao encontradas." >&2
    exit 1
  fi

  "${PYTHON_BIN}" scripts/py_pipeline_A/A1_gerar_manifesto_analise.py \
    --results-dir ${results_dirs} \
    --analysis-dir "${analysis_dir}" \
    --targets ${TARGETS} \
    --gpu-targets ${GPU_TARGETS}

  if [[ "${PIPELINE_A_CLASSICAL}" == "true" ]] || [[ "${PIPELINE_A_CLASSICAL}" == "1" ]]; then
    "${PYTHON_BIN}" scripts/py_pipeline_A/A2_regressores_classicos.py \
      compare \
      --results-dir ${results_dirs} \
      --analysis-dir "${analysis_dir}" \
      --jobs-file "${analysis_dir}/analysis_jobs.csv" \
      --first-sweep \
      --cv-folds "${CV_FOLDS}" \
      "${CLASSICAL_COMPARE_ARGS[@]}"

    if [[ -f "scripts/py_outros/comparar_modelos_pipelines.py" ]]; then
      "${PYTHON_BIN}" scripts/py_outros/comparar_modelos_pipelines.py --analysis-root "${ANALYSIS_ROOT}"
    fi

    if [[ ("${DEPENDENCY_ONLY}" == "true" || "${DEPENDENCY_ONLY}" == "1") && -f "scripts/py_pipeline_A/A4_rankings_dependencia.py" ]]; then
      "${PYTHON_BIN}" scripts/py_pipeline_A/A4_rankings_dependencia.py --analysis-root "${ANALYSIS_ROOT}"
    fi
  fi

  if [[ "${PIPELINE_A_SEQUENTIAL}" == "true" ]] || [[ "${PIPELINE_A_SEQUENTIAL}" == "1" ]]; then
    SEQUENCE_LENGTH="${SEQUENCE_LENGTH}" \
    SEQUENCE_STRIDE="${SEQUENCE_STRIDE}" \
    SEQUENCE_MAX_SEQUENCES="${SEQUENCE_MAX_SEQUENCES}" \
    SEQUENCE_EPOCHS="${SEQUENCE_EPOCHS}" \
    SEQUENCE_BATCH_SIZE="${SEQUENCE_BATCH_SIZE}" \
    SEQUENCE_CNN_MAX_TRIALS="${SEQUENCE_CNN_MAX_TRIALS}" \
    SEQUENCE_SPLIT_MODE="${SEQUENCE_SPLIT_MODE}" \
    SEQUENCE_SAMPLE_MODE="${SEQUENCE_SAMPLE_MODE}" \
    SEQUENCE_TF_DEVICE="${SEQUENCE_TF_DEVICE}" \
    SEQUENCE_REQUIRE_GPU="${SEQUENCE_REQUIRE_GPU}" \
    SEQUENCE_MODEL_ONLY="${SEQUENCE_MODEL_ONLY}" \
    SEQUENCE_FORCE_MODEL="${SEQUENCE_FORCE_MODEL}" \
    "${PYTHON_BIN}" scripts/py_pipeline_A/A5_modelos_sequenciais.py \
      --results-dir ${results_dirs} \
      --analysis-dir "${analysis_dir}" \
      --jobs-file "${analysis_dir}/analysis_jobs.csv" \
      --first-sweep \
      --seed "${SEED}" \
      --split-mode "${SEQUENCE_SPLIT_MODE}" \
      --sample-mode "${SEQUENCE_SAMPLE_MODE}" \
      "${SEQUENCE_MODEL_ARGS[@]}" \
      "${SEQUENCE_CACHE_ARGS[@]}"
  fi
}

should_run_condition() {
  local label="$1"
  [[ " ${RUN_CONDITIONS} " == *" ${label} "* ]]
}

if should_run_condition "sem_telemetria"; then
  run_one "sem_telemetria" "${NORMAL_RESULTS_DIRS}"
fi
if should_run_condition "com_telemetria"; then
  run_one "com_telemetria" "${TELEMETRY_RESULTS_DIRS}"
fi

if [[ -f "scripts/py_outros/comparar_modelos_pipelines.py" ]]; then
  "${PYTHON_BIN}" scripts/py_outros/comparar_modelos_pipelines.py --analysis-root "${ANALYSIS_ROOT}"
fi

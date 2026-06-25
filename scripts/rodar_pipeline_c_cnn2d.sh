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
  PYTHON_BIN
  TARGETS
  GPU_TARGETS
  CNN2D_WINDOW_SIZE
  CNN2D_STRIDE
  CNN2D_MAX_SAMPLES
  CNN2D_MAX_SOURCE_ROWS
  CNN2D_MAX_TENSOR_GB
  CNN2D_SAMPLE_MODE
  CNN2D_PREPROCESS_PARALLEL_JOBS
  CNN2D_EPOCHS
  CNN2D_BATCH_SIZE
  CNN2D_TEST_FRACTION
  CNN2D_MAX_ARCHITECTURES
  CNN2D_TF_DEVICE
  CNN2D_REQUIRE_GPU
  CNN2D_CACHE
  CNN2D_MODEL_ONLY
  CNN2D_FORCE_MODEL
  CNN2D_PLOTS_ONLY
  CNN2D_TRAIN_PARALLEL_JOBS
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
CNN2D_WINDOW_SIZE="${CNN2D_WINDOW_SIZE:-32}"
CNN2D_STRIDE="${CNN2D_STRIDE:-1}"
CNN2D_MAX_SAMPLES="${CNN2D_MAX_SAMPLES:-0}"
CNN2D_MAX_SOURCE_ROWS="${CNN2D_MAX_SOURCE_ROWS:-500000}"
CNN2D_MAX_TENSOR_GB="${CNN2D_MAX_TENSOR_GB:-4.0}"
CNN2D_SAMPLE_MODE="${CNN2D_SAMPLE_MODE:-random}"
CNN2D_PREPROCESS_PARALLEL_JOBS="${CNN2D_PREPROCESS_PARALLEL_JOBS:-1}"
CNN2D_EPOCHS="${CNN2D_EPOCHS:-8}"
CNN2D_BATCH_SIZE="${CNN2D_BATCH_SIZE:-128}"
CNN2D_TEST_FRACTION="${CNN2D_TEST_FRACTION:-0.25}"
CNN2D_MAX_ARCHITECTURES="${CNN2D_MAX_ARCHITECTURES:-8}"
CNN2D_TF_DEVICE="${CNN2D_TF_DEVICE:-auto}"
CNN2D_REQUIRE_GPU="${CNN2D_REQUIRE_GPU:-true}"
CNN2D_CACHE="${CNN2D_CACHE:-true}"
CNN2D_MODEL_ONLY="${CNN2D_MODEL_ONLY:-}"
CNN2D_FORCE_MODEL="${CNN2D_FORCE_MODEL:-false}"
CNN2D_PLOTS_ONLY="${CNN2D_PLOTS_ONLY:-false}"
CNN2D_TRAIN_PARALLEL_JOBS="${CNN2D_TRAIN_PARALLEL_JOBS:-1}"
SEED="${SEED:-42}"

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

CNN2D_CACHE_ARGS=()
if [[ "${CNN2D_CACHE}" == "false" ]] || [[ "${CNN2D_CACHE}" == "0" ]]; then
  CNN2D_CACHE_ARGS+=(--no-cache)
fi
CNN2D_MODEL_ARGS=()
if [[ -n "${CNN2D_MODEL_ONLY}" ]]; then
  CNN2D_MODEL_ARGS+=(--only-model "${CNN2D_MODEL_ONLY}")
fi
if [[ "${CNN2D_FORCE_MODEL}" == "true" ]] || [[ "${CNN2D_FORCE_MODEL}" == "1" ]]; then
  CNN2D_MODEL_ARGS+=(--force-model)
fi
if [[ "${CNN2D_PLOTS_ONLY}" == "true" ]] || [[ "${CNN2D_PLOTS_ONLY}" == "1" ]]; then
  CNN2D_MODEL_ARGS+=(--plots-only)
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
  local analysis_dir="${ANALYSIS_ROOT}/pipeline_C/${label}"

  if [[ -z "${results_dirs}" ]]; then
    echo "Pastas do sweep ${label} nao encontradas." >&2
    exit 1
  fi

  "${PYTHON_BIN}" scripts/py_pipeline_A/A1_gerar_manifesto_analise.py \
    --results-dir ${results_dirs} \
    --analysis-dir "${analysis_dir}" \
    --targets ${TARGETS} \
    --gpu-targets ${GPU_TARGETS}

  "${PYTHON_BIN}" scripts/py_pipeline_C/C1_preprocessamento_2d.py \
    --results-dir ${results_dirs} \
    --analysis-dir "${analysis_dir}" \
    --jobs-file "${analysis_dir}/analysis_jobs.csv" \
    --window-size "${CNN2D_WINDOW_SIZE}" \
    --stride "${CNN2D_STRIDE}" \
    --max-samples "${CNN2D_MAX_SAMPLES}" \
    --max-source-rows "${CNN2D_MAX_SOURCE_ROWS}" \
    --max-tensor-gb "${CNN2D_MAX_TENSOR_GB}" \
    --sample-mode "${CNN2D_SAMPLE_MODE}" \
    --seed "${SEED}" \
    --parallel-jobs "${CNN2D_PREPROCESS_PARALLEL_JOBS}" \
    "${CNN2D_CACHE_ARGS[@]}"

  "${PYTHON_BIN}" scripts/py_pipeline_C/C2_treinar_cnn_2d.py \
    --analysis-dir "${analysis_dir}" \
    --preprocess-summary "${analysis_dir}/pipeline_c_preprocess_summary.csv" \
    --epochs "${CNN2D_EPOCHS}" \
    --batch-size "${CNN2D_BATCH_SIZE}" \
    --test-fraction "${CNN2D_TEST_FRACTION}" \
    --seed "${SEED}" \
    --max-architectures "${CNN2D_MAX_ARCHITECTURES}" \
    --tf-device "${CNN2D_TF_DEVICE}" \
    --parallel-jobs "${CNN2D_TRAIN_PARALLEL_JOBS}" \
    "${CNN2D_MODEL_ARGS[@]}" \
    "${CNN2D_CACHE_ARGS[@]}"
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

"${PYTHON_BIN}" scripts/py_pipeline_C/C3_rankings_e_graficos.py --analysis-root "${ANALYSIS_ROOT}"

"${PYTHON_BIN}" scripts/py_outros/comparar_modelos_pipelines.py --analysis-root "${ANALYSIS_ROOT}"

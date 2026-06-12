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
  PYTHON_BIN
  TARGETS
  GPU_TARGETS
  CNN2D_WINDOW_SIZE
  CNN2D_STRIDE
  CNN2D_MAX_SAMPLES
  CNN2D_SAMPLE_MODE
  CNN2D_EPOCHS
  CNN2D_BATCH_SIZE
  CNN2D_TEST_FRACTION
  CNN2D_MAX_ARCHITECTURES
  CNN2D_TF_DEVICE
  CNN2D_REQUIRE_GPU
  CNN2D_CACHE
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
CNN2D_WINDOW_SIZE="${CNN2D_WINDOW_SIZE:-32}"
CNN2D_STRIDE="${CNN2D_STRIDE:-1}"
CNN2D_MAX_SAMPLES="${CNN2D_MAX_SAMPLES:-120000}"
CNN2D_SAMPLE_MODE="${CNN2D_SAMPLE_MODE:-random}"
CNN2D_EPOCHS="${CNN2D_EPOCHS:-8}"
CNN2D_BATCH_SIZE="${CNN2D_BATCH_SIZE:-128}"
CNN2D_TEST_FRACTION="${CNN2D_TEST_FRACTION:-0.25}"
CNN2D_MAX_ARCHITECTURES="${CNN2D_MAX_ARCHITECTURES:-8}"
CNN2D_TF_DEVICE="${CNN2D_TF_DEVICE:-auto}"
CNN2D_REQUIRE_GPU="${CNN2D_REQUIRE_GPU:-true}"
CNN2D_CACHE="${CNN2D_CACHE:-true}"
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
    --sample-mode "${CNN2D_SAMPLE_MODE}" \
    --seed "${SEED}" \
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
    "${CNN2D_CACHE_ARGS[@]}"
}

run_one "sem_telemetria" "${NORMAL_RESULTS_DIRS}"
run_one "com_telemetria" "${TELEMETRY_RESULTS_DIRS}"

"${PYTHON_BIN}" scripts/py_pipeline_C/C3_rankings_e_graficos.py --analysis-root "${ANALYSIS_ROOT}"

"${PYTHON_BIN}" scripts/py_outros/comparar_modelos_pipelines.py --analysis-root "${ANALYSIS_ROOT}"

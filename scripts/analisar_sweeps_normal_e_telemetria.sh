#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

NORMAL_RESULTS_DIR="${NORMAL_RESULTS_DIR:-}"
TELEMETRY_RESULTS_DIR="${TELEMETRY_RESULTS_DIR:-}"
ANALYSIS_ROOT="${ANALYSIS_ROOT:-resultados/analises_regressao}"
MAX_ROWS="${MAX_ROWS:-120000}"
TARGETS="${TARGETS:-response_time_us queueing_delay_us slowdown}"
GPU_TARGETS="${GPU_TARGETS:-10 50 100 120}"

latest_dir() {
  local pattern="$1"
  find resultados -mindepth 1 -maxdepth 1 -type d -name "${pattern}" -printf '%T@ %p\n' \
    | sort -nr \
    | awk 'NR == 1 {print $2}'
}

if [[ -z "${NORMAL_RESULTS_DIR}" ]]; then
  NORMAL_RESULTS_DIR="$(latest_dir 'sweep_moderado_sem_estimativas_[0-9]*' || true)"
fi

if [[ -z "${TELEMETRY_RESULTS_DIR}" ]]; then
  TELEMETRY_RESULTS_DIR="$(latest_dir 'sweep_moderado_sem_estimativas_telemetry_*' || true)"
fi

run_one() {
  local label="$1"
  local results_dir="$2"
  if [[ -z "${results_dir}" || ! -d "${results_dir}" ]]; then
    echo "Pasta do sweep ${label} nao encontrada: ${results_dir}" >&2
    exit 1
  fi

  local analysis_dir="${ANALYSIS_ROOT}/${label}_$(basename "${results_dir}")"
  echo "Analise ${label}: ${results_dir} -> ${analysis_dir}"
  RESULTS_DIR="${results_dir}" \
  ANALYSIS_DIR="${analysis_dir}" \
  MAX_ROWS="${MAX_ROWS}" \
  TARGETS="${TARGETS}" \
  GPU_TARGETS="${GPU_TARGETS}" \
  bash scripts/analisar_regressoes_sweep.sh
}

run_one "sem_telemetria" "${NORMAL_RESULTS_DIR}"
run_one "com_telemetria" "${TELEMETRY_RESULTS_DIR}"

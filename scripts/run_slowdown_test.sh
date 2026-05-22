#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

make
mkdir -p resultados

mpirun -np 1 ./main \
  --threads-per-process 1 \
  --kernels-per-thread 300 \
  --arrival-min-ms 1 \
  --arrival-max-ms 5 \
  --kernel-min-us 100 \
  --kernel-max-us 500 \
  --blocks-x 16 \
  --threads-per-block 128 \
  --grid-z 1 \
  --seed 42 \
  --experiment-name baseline

mpirun -np 4 ./main \
  --threads-per-process 8 \
  --kernels-per-thread 300 \
  --arrival-min-ms 0 \
  --arrival-max-ms 0 \
  --kernel-min-us 500 \
  --kernel-max-us 2000 \
  --blocks-x 64 \
  --threads-per-block 256 \
  --grid-z 1 \
  --seed 42 \
  --experiment-name stress

python3 analyze_slowdown.py

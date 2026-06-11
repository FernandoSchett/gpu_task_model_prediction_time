#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

VENV_DIR="${VENV_DIR:-${REPO_ROOT}/.venv}"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    echo "Criando ambiente virtual em ${VENV_DIR}..."
    python3 -m venv "${VENV_DIR}"
  fi
  PYTHON_BIN="${VENV_DIR}/bin/python"
fi

echo "Verificando driver NVIDIA..."
nvidia-smi

echo "Atualizando pip e instalando dependencias do projeto..."
"${PYTHON_BIN}" -m pip install -U pip
"${PYTHON_BIN}" -m pip install -r requirements.txt

echo "Criando links das bibliotecas CUDA/cuDNN instaladas pelo pip para o TensorFlow..."
"${PYTHON_BIN}" - <<'PY'
from __future__ import annotations

import pathlib
import site
import subprocess
import sys

import tensorflow as tf

tf_dir = pathlib.Path(tf.__file__).resolve().parent
site_dirs = [pathlib.Path(path) for path in site.getsitepackages()]
user_site = site.getusersitepackages()
if user_site:
    site_dirs.append(pathlib.Path(user_site))

linked = 0
for site_dir in site_dirs:
    nvidia_dir = site_dir / "nvidia"
    if not nvidia_dir.exists():
        continue
    for lib in nvidia_dir.glob("*/lib/*.so*"):
        destination = tf_dir / lib.name
        if destination.exists() or destination.is_symlink():
            destination.unlink()
        destination.symlink_to(lib)
        linked += 1

print(f"TensorFlow instalado em: {tf_dir}")
print(f"Links criados: {linked}")

try:
    import nvidia.cuda_nvcc as cuda_nvcc
except Exception:
    cuda_nvcc = None

venv_bin = pathlib.Path(sys.prefix) / "bin"
if cuda_nvcc is not None and venv_bin.exists():
    nvcc_root = pathlib.Path(cuda_nvcc.__file__).resolve().parents[1]
    candidates = list(nvcc_root.glob("*/bin/ptxas"))
    if candidates:
        target = venv_bin / "ptxas"
        if target.exists() or target.is_symlink():
            target.unlink()
        target.symlink_to(candidates[0])
        print(f"Link ptxas criado: {target} -> {candidates[0]}")
PY

echo "Verificando GPUs visiveis no TensorFlow..."
"${PYTHON_BIN}" - <<'PY'
import tensorflow as tf

gpus = tf.config.list_physical_devices("GPU")
print(gpus)
if not gpus:
    raise SystemExit("TensorFlow ainda nao encontrou GPU.")
PY

echo "TensorFlow com GPU pronto."

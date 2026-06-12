#!/usr/bin/env python3
"""Compatibility wrapper for the shared model comparison script."""

from __future__ import annotations

import runpy
from pathlib import Path


SCRIPT = Path(__file__).resolve().parent.parent / "py_outros" / "comparar_modelos_pipelines.py"


if __name__ == "__main__":
    runpy.run_path(str(SCRIPT), run_name="__main__")

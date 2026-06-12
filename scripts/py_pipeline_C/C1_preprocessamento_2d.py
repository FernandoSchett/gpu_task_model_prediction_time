#!/usr/bin/env python3
"""Build 2D worker-time tensors for Pipeline C CNN models."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import re
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
REGRESSOR_SCRIPT = REPO_ROOT / "scripts" / "py_pipeline_A" / "A2_regressores_classicos.py"
DEFAULT_ANALYSIS_ROOT = REPO_ROOT / "resultados" / "analises_regressao"


def load_regressor_module():
    spec = importlib.util.spec_from_file_location("regressor_analysis", REGRESSOR_SCRIPT)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Nao foi possivel carregar {REGRESSOR_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


REG = load_regressor_module()
FEATURES = list(REG.BASE_FEATURES) + [f"kernel_type_{name}" for name in REG.KERNEL_TYPES]


def env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on", "sim"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, nargs="+", required=True)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--jobs-file", type=Path, required=True)
    parser.add_argument("--window-size", type=int, default=int(os.getenv("CNN2D_WINDOW_SIZE", "32")))
    parser.add_argument("--stride", type=int, default=int(os.getenv("CNN2D_STRIDE", "1")))
    parser.add_argument("--max-samples", type=int, default=int(os.getenv("CNN2D_MAX_SAMPLES", "120000")))
    parser.add_argument(
        "--sample-mode",
        choices=("random", "linspace"),
        default=os.getenv("CNN2D_SAMPLE_MODE", "random"),
    )
    parser.add_argument("--seed", type=int, default=int(os.getenv("SEED", "42") or "42"))
    parser.add_argument("--no-cache", action="store_true")
    return parser.parse_args()


def load_jobs(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def result_paths(results_dirs: list[Path], include_regex: str) -> list[Path]:
    pattern = re.compile(include_regex)
    return sorted(
        path
        for results_dir in results_dirs
        for path in results_dir.rglob("resultados_experimentos_*.csv")
        if pattern.search(str(path))
    )


def signature(args: argparse.Namespace, job: dict[str, str], paths: list[Path]) -> dict[str, object]:
    return {
        "target": job["target"],
        "label": job["label"],
        "source_paths": [str(path) for path in paths],
        "window_size": args.window_size,
        "stride": args.stride,
        "max_samples": args.max_samples,
        "sample_mode": args.sample_mode,
        "features": FEATURES,
        "target_alignment": "next_kernel",
    }


def hash_signature(payload: dict[str, object]) -> str:
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def worker_id(row: dict[str, str]) -> int:
    rank = int(REG.to_float(row, "mpi_rank", 0.0))
    threads = int(REG.to_float(row, "threads_per_process", 1.0))
    host_thread = int(REG.to_float(row, "host_thread_id", 0.0))
    return rank * max(1, threads) + host_thread


def order_key(row: dict[str, str], fallback_index: int) -> tuple[float, float, int]:
    submit_time = REG.to_float(row, "submit_time_ns", math.nan)
    if not math.isfinite(submit_time):
        submit_time = REG.to_float(row, "time_since_experiment_start_us", float(fallback_index))
    execution_order = REG.to_float(row, "execution_order", REG.to_float(row, "rank_local_submitted_count", fallback_index))
    return submit_time, execution_order, fallback_index


def load_events(paths: list[Path], target: str) -> tuple[list[dict[str, object]], int, int]:
    events: list[dict[str, object]] = []
    rows_loaded = 0
    max_worker = -1
    for path in paths:
        with path.open("r", encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            if reader.fieldnames is None or "response_time_us" not in reader.fieldnames:
                continue
            for row_index, row in enumerate(reader):
                rows_loaded += 1
                if REG.to_float(row, "cuda_error_code", 0.0) != 0.0:
                    continue
                values = REG.row_features_compare(row)
                y = values[target] if target in values else REG.to_float(row, target)
                x = [values[name] for name in FEATURES]
                wid = worker_id(row)
                if not math.isfinite(y) or any(not math.isfinite(value) for value in x) or wid < 0:
                    continue
                max_worker = max(max_worker, wid)
                events.append(
                    {
                        "worker": wid,
                        "order": order_key(row, len(events)),
                        "x": np.asarray(x, dtype=np.float32),
                        "y": float(y),
                    }
                )
    events.sort(key=lambda item: item["order"])
    return events, max_worker + 1, rows_loaded


def choose_starts(count: int, stride: int, max_samples: int, seed: int, sample_mode: str) -> np.ndarray:
    starts = np.arange(0, count * stride, stride, dtype=int)
    if max_samples > 0 and len(starts) > max_samples:
        if sample_mode == "random":
            rng = np.random.default_rng(seed)
            starts = np.sort(rng.choice(starts, size=max_samples, replace=False))
        else:
            starts = np.linspace(0, starts[-1], num=max_samples, dtype=int)
    return starts


def build_tensor_dataset(
    events: list[dict[str, object]],
    workers: int,
    window_size: int,
    stride: int,
    max_samples: int,
    seed: int,
    sample_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    if window_size < 2:
        raise SystemExit("--window-size precisa ser >= 2.")
    if stride < 1:
        raise SystemExit("--stride precisa ser >= 1.")
    count = max(0, (len(events) - window_size - 1) // stride + 1)
    if count <= 0:
        return (
            np.empty((0, workers, window_size, len(FEATURES)), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        )
    starts = choose_starts(count, stride, max_samples, seed, sample_mode)
    x = np.zeros((len(starts), workers, window_size, len(FEATURES)), dtype=np.float32)
    y = np.empty((len(starts),), dtype=np.float32)
    for out_index, start in enumerate(starts):
        end = start + window_size
        for temporal_index, event in enumerate(events[start:end]):
            x[out_index, int(event["worker"]), temporal_index, :] = event["x"]
        y[out_index] = float(events[end]["y"])
    return x, y


def write_summary(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "label", "target", "source_files", "rows_loaded", "events_used", "workers",
        "samples", "window_size", "stride", "features", "tensor_path", "metadata_path", "cached",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_job(args: argparse.Namespace, job: dict[str, str]) -> dict[str, str]:
    paths = result_paths(args.results_dir, job["include_regex"])
    if not paths:
        raise SystemExit(f"Nenhum CSV encontrado para {job['label']} {job['target']}")
    output_dir = Path(job["output_dir"]) / "2d_models"
    output_dir.mkdir(parents=True, exist_ok=True)
    tensor_path = output_dir / "cnn2d_dataset.npz"
    metadata_path = output_dir / "cnn2d_dataset_metadata.json"
    expected = signature(args, job, paths)
    expected_hash = hash_signature(expected)

    if not args.no_cache and tensor_path.exists() and metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("signature_hash") == expected_hash:
                return {
                    "label": job["label"],
                    "target": job["target"],
                    "source_files": str(metadata.get("source_files", len(paths))),
                    "rows_loaded": str(metadata.get("rows_loaded", "")),
                    "events_used": str(metadata.get("events_used", "")),
                    "workers": str(metadata.get("workers", "")),
                    "samples": str(metadata.get("samples", "")),
                    "window_size": str(args.window_size),
                    "stride": str(args.stride),
                    "features": str(len(FEATURES)),
                    "tensor_path": str(tensor_path),
                    "metadata_path": str(metadata_path),
                    "cached": "true",
                }
        except (OSError, json.JSONDecodeError):
            pass

    events, workers, rows_loaded = load_events(paths, job["target"])
    if not events:
        raise SystemExit(f"Nenhum evento valido para {job['label']} {job['target']}")
    x, y = build_tensor_dataset(
        events,
        workers,
        args.window_size,
        args.stride,
        args.max_samples,
        args.seed,
        args.sample_mode,
    )
    if len(y) == 0:
        raise SystemExit(f"Nenhuma janela valida para {job['label']} {job['target']}")
    np.savez_compressed(tensor_path, x=x, y=y)
    metadata = {
        **expected,
        "signature_hash": expected_hash,
        "source_files": len(paths),
        "rows_loaded": rows_loaded,
        "events_used": len(events),
        "workers": workers,
        "samples": int(len(y)),
        "feature_count": len(FEATURES),
        "tensor_shape": list(x.shape),
        "tensor_path": str(tensor_path),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {
        "label": job["label"],
        "target": job["target"],
        "source_files": str(len(paths)),
        "rows_loaded": str(rows_loaded),
        "events_used": str(len(events)),
        "workers": str(workers),
        "samples": str(len(y)),
        "window_size": str(args.window_size),
        "stride": str(args.stride),
        "features": str(len(FEATURES)),
        "tensor_path": str(tensor_path),
        "metadata_path": str(metadata_path),
        "cached": "false",
    }


def main() -> int:
    args = parse_args()
    rows = []
    for job in load_jobs(args.jobs_file):
        row = run_job(args, job)
        rows.append(row)
        cached = " cached" if row["cached"] == "true" else ""
        print(
            f"cnn2d_preprocess {row['label']} {row['target']}:{cached} "
            f"samples={row['samples']} workers={row['workers']} tensor={row['tensor_path']}"
        )
    summary_path = args.analysis_dir / "pipeline_c_preprocess_summary.csv"
    write_summary(summary_path, rows)
    print(f"pipeline_c_preprocess_summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

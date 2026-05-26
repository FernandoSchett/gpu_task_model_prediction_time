#!/usr/bin/env python3
"""Train regression baselines for CUDA timing CSVs.

Linear regression is intentionally treated as a baseline, not a claim that CUDA timings are linear.
Occupancy, SM waves, warp scheduling, launch overhead, cache/memory effects,
cross-process interference, and internal GPU scheduling can all introduce
non-linear behavior. The extra models here are lightweight baselines that avoid
an external scikit-learn dependency.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Iterable

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_RESULTS_DIR = REPO_ROOT / "resultados"
TARGETS = ("response_time_us", "queueing_delay_us", "slowdown")
MODELS = ("linear", "ridge", "quadratic_ridge", "knn")
KERNEL_TYPES = ("busy_wait", "compute", "memory", "mixed")
DEFAULT_FEATURES = (
    "requested_busy_wait_us",
    "mpi_world_size",
    "threads_per_process",
    "blocks_x",
    "threads_per_block",
    "grid_z",
    "total_blocks",
    "total_cuda_threads",
    "total_warps",
    "warps_per_block",
    "estimated_waves",
    "active_kernels_estimate",
    "inflight_kernels_estimate",
    "concurrent_kernels_estimate",
    "arrival_wait_ms",
    "launch_overhead_us",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory containing experiment result CSVs.",
    )
    parser.add_argument(
        "--target",
        choices=TARGETS,
        default="response_time_us",
        help="Prediction target.",
    )
    parser.add_argument(
        "--model",
        choices=MODELS,
        default="linear",
        help="Regression model to train.",
    )
    parser.add_argument(
        "--ridge-alpha",
        type=float,
        default=1.0,
        help="L2 regularization strength for ridge-based models.",
    )
    parser.add_argument(
        "--knn-k",
        type=int,
        default=15,
        help="Number of neighbors for knn regression.",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.25,
        help="Fraction of rows reserved for deterministic hold-out evaluation.",
    )
    return parser.parse_args()


def to_float(row: dict[str, str], name: str, default: float = math.nan) -> float:
    value = row.get(name, "")
    if value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def load_rows(results_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for csv_path in sorted(results_dir.glob("resultados_experimentos_*.csv")):
        with csv_path.open("r", encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            if reader.fieldnames is None or "response_time_us" not in reader.fieldnames:
                continue
            for row in reader:
                row["source_file"] = csv_path.name
                rows.append(row)
    return rows


def add_derived_features(row: dict[str, str]) -> dict[str, float]:
    blocks_x = to_float(row, "blocks_x", 0.0)
    grid_z = to_float(row, "grid_z", 1.0)
    threads_per_block = to_float(row, "threads_per_block", 0.0)
    requested_us = to_float(row, "requested_busy_wait_us", 0.0)
    response_us = to_float(row, "response_time_us", math.nan)
    sm_count = to_float(row, "sm_count", 0.0)
    warps_per_block = math.ceil(threads_per_block / 32.0) if threads_per_block > 0 else 0.0
    total_blocks = to_float(row, "total_blocks", blocks_x * grid_z)
    total_threads = to_float(row, "total_cuda_threads", total_blocks * threads_per_block)
    total_warps = to_float(row, "total_warps", total_blocks * warps_per_block)
    queueing_delay_us = to_float(row, "queueing_delay_us", response_us - requested_us)
    slowdown = to_float(
        row,
        "slowdown",
        response_us / requested_us if requested_us > 0 else math.nan,
    )

    values = {name: to_float(row, name) for name in DEFAULT_FEATURES}
    values.update(
        {
            "threads_per_process": to_float(row, "threads_per_process", 1.0),
            "total_blocks": total_blocks,
            "total_cuda_threads": total_threads,
            "total_warps": total_warps,
            "warps_per_block": warps_per_block,
            "estimated_waves": to_float(
                row,
                "estimated_waves",
                total_blocks / sm_count if sm_count > 0 else 0.0,
            ),
            "concurrent_kernels_estimate": to_float(
                row,
                "concurrent_kernels_estimate",
                to_float(row, "active_kernels_estimate", 0.0),
            ),
            "queueing_delay_us": queueing_delay_us,
            "slowdown": slowdown,
        }
    )
    kernel_type = row.get("kernel_type", "")
    for name in KERNEL_TYPES:
        values[f"kernel_type_{name}"] = 1.0 if kernel_type == name else 0.0
    return values


def build_matrix(rows: Iterable[dict[str, str]], target: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    x_rows: list[list[float]] = []
    y_values: list[float] = []
    features = list(DEFAULT_FEATURES) + [f"kernel_type_{name}" for name in KERNEL_TYPES]

    for row in rows:
        if to_float(row, "cuda_error_code", 0.0) != 0.0:
            continue
        derived = add_derived_features(row)
        target_value = derived[target] if target in derived else to_float(row, target)
        feature_values = [derived[name] for name in features]
        if not math.isfinite(target_value) or any(not math.isfinite(value) for value in feature_values):
            continue
        x_rows.append(feature_values)
        y_values.append(target_value)

    if not x_rows:
        raise SystemExit("No valid rows found after filtering cuda_error_code == 0.")

    return np.asarray(x_rows, dtype=float), np.asarray(y_values, dtype=float), features


def train_test_split(
    x: np.ndarray,
    y: np.ndarray,
    test_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not 0.0 < test_fraction < 1.0:
        raise SystemExit("--test-fraction must be between 0 and 1.")

    indices = np.arange(len(y))
    rng = np.random.default_rng(42)
    rng.shuffle(indices)
    test_size = max(1, int(round(len(indices) * test_fraction)))
    test_idx = indices[:test_size]
    train_idx = indices[test_size:]
    if len(train_idx) == 0:
        raise SystemExit("Not enough rows to create a train/test split.")
    return x[train_idx], x[test_idx], y[train_idx], y[test_idx]


def standardize_train_test(
    train_x: np.ndarray,
    test_x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0)
    scale = train_x.std(axis=0)
    scale[scale == 0.0] = 1.0
    return (train_x - mean) / scale, (test_x - mean) / scale, mean, scale


def fit_linear_regression(train_x: np.ndarray, train_y: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(train_x.shape[0]), train_x])
    coef, *_ = np.linalg.lstsq(design, train_y, rcond=None)
    return coef


def fit_ridge_regression(train_x: np.ndarray, train_y: np.ndarray, alpha: float) -> np.ndarray:
    design = np.column_stack([np.ones(train_x.shape[0]), train_x])
    penalty = np.eye(design.shape[1]) * alpha
    penalty[0, 0] = 0.0
    return np.linalg.solve(design.T @ design + penalty, design.T @ train_y)


def quadratic_features(x: np.ndarray) -> np.ndarray:
    squared = x * x
    return np.column_stack([x, squared])


def predict(x: np.ndarray, coef: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(x.shape[0]), x])
    return design @ coef


def predict_knn(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, k: int) -> np.ndarray:
    k = max(1, min(k, train_x.shape[0]))
    predictions = []
    for row in test_x:
        distances = np.sum((train_x - row) ** 2, axis=1)
        nearest = np.argpartition(distances, k - 1)[:k]
        predictions.append(float(np.mean(train_y[nearest])))
    return np.asarray(predictions, dtype=float)


def print_metrics(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    residual = y_true - y_pred
    mae = np.mean(np.abs(residual))
    rmse = math.sqrt(float(np.mean(residual * residual)))
    denominator = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1.0 - float(np.sum(residual * residual) / denominator) if denominator > 0 else math.nan
    print(f"{name}_rows: {len(y_true)}")
    print(f"{name}_mae: {mae:.6f}")
    print(f"{name}_rmse: {rmse:.6f}")
    print(f"{name}_r2: {r2:.6f}")


def main() -> int:
    args = parse_args()
    rows = load_rows(args.results_dir)
    if not rows:
        print(f"No result CSVs found in {args.results_dir}.")
        return 1

    x, y, features = build_matrix(rows, args.target)
    train_x, test_x, train_y, test_y = train_test_split(x, y, args.test_fraction)
    train_x_std, test_x_std, _, _ = standardize_train_test(train_x, test_x)

    print(f"target: {args.target}")
    print(f"model: {args.model}")

    if args.model == "linear":
        coef = fit_linear_regression(train_x_std, train_y)
        print_metrics("train", train_y, predict(train_x_std, coef))
        print_metrics("test", test_y, predict(test_x_std, coef))

        print("\nstandardized_coefficients:")
        ranked = sorted(zip(features, coef[1:]), key=lambda item: abs(item[1]), reverse=True)
        for feature, value in ranked:
            print(f"{feature}: {value:.6f}")
        print(f"intercept: {coef[0]:.6f}")
    elif args.model == "ridge":
        coef = fit_ridge_regression(train_x_std, train_y, args.ridge_alpha)
        print_metrics("train", train_y, predict(train_x_std, coef))
        print_metrics("test", test_y, predict(test_x_std, coef))

        print("\nstandardized_coefficients:")
        ranked = sorted(zip(features, coef[1:]), key=lambda item: abs(item[1]), reverse=True)
        for feature, value in ranked:
            print(f"{feature}: {value:.6f}")
        print(f"intercept: {coef[0]:.6f}")
    elif args.model == "quadratic_ridge":
        train_quad = quadratic_features(train_x_std)
        test_quad = quadratic_features(test_x_std)
        quad_features = features + [f"{feature}^2" for feature in features]
        coef = fit_ridge_regression(train_quad, train_y, args.ridge_alpha)
        print_metrics("train", train_y, predict(train_quad, coef))
        print_metrics("test", test_y, predict(test_quad, coef))

        print("\nstandardized_coefficients:")
        ranked = sorted(zip(quad_features, coef[1:]), key=lambda item: abs(item[1]), reverse=True)
        for feature, value in ranked[:30]:
            print(f"{feature}: {value:.6f}")
        print(f"intercept: {coef[0]:.6f}")
    elif args.model == "knn":
        print_metrics("train", train_y, predict_knn(train_x_std, train_y, train_x_std, args.knn_k))
        print_metrics("test", test_y, predict_knn(train_x_std, train_y, test_x_std, args.knn_k))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

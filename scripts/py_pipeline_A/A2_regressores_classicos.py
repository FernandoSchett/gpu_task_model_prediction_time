#!/usr/bin/env python3
"""Lightweight regression analysis for CUDA timing CSVs using scikit-learn.

Supports two modes:
1. compare: Compare multiple regression models
2. baseline: Train linear/ridge/quadratic regression models with cross-validation

Telemetry/dependency analysis is available as a separate compare-mode path so
it can be generated without retraining models.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pickle
import random
import re
from pathlib import Path
from statistics import NormalDist
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_RESULTS_DIR = REPO_ROOT / "resultados"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "resultados" / "model_comparison"
DEFAULT_KNN_TRAIN_LIMIT = 8000
DEPENDENCY_MAX_LAG = 50
DIAGNOSTIC_PLOT_SAMPLE_LIMIT = 50000
TARGETS = ("response_time_us", "queueing_delay_us", "slowdown")
KERNEL_TYPES = ("busy_wait", "compute", "memory", "mixed")
BASE_FEATURES = (
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
    "blocks_per_sm",
    "total_blocks_per_sm",
    "arrival_wait_ms",
    "launch_overhead_us",
    "effective_workers",
    "workers_x_requested_busy_wait_us",
    "workers_x_blocks_per_sm",
    "workers_x_total_warps",
    "requested_busy_wait_us_per_arrival_ms",
    "target_gpu_demand_percent",
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", help="Analysis mode")
    
    # Common arguments
    compare_parser = subparsers.add_parser("compare", help="Compare multiple regression models")
    compare_parser.add_argument("--results-dir", type=Path, nargs="+", default=[DEFAULT_RESULTS_DIR])
    compare_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    compare_parser.add_argument("--analysis-dir", type=Path, default=None)
    compare_parser.add_argument("--jobs-file", type=Path, default=None)
    compare_parser.add_argument("--target", choices=TARGETS, default="response_time_us")
    compare_parser.add_argument("--first-sweep", action="store_true", help="Use first occurrence of each sweep config/rank.")
    compare_parser.add_argument("--include-regex", default="", help="Only use result CSV paths matching this regular expression.")
    compare_parser.add_argument("--test-fraction", type=float, default=0.25)
    compare_parser.add_argument("--seed", type=int, default=42, help="Fallback seed if SEED is not defined in .env")
    compare_parser.add_argument("--knn-k", type=int, default=15)
    compare_parser.add_argument("--knn-train-limit", type=int, default=DEFAULT_KNN_TRAIN_LIMIT)
    compare_parser.add_argument("--cv-folds", type=int, default=5, help="Number of cross-validation folds.")
    compare_parser.add_argument("--dependency-only", "--skip-training", action="store_true", help="Generate correlation/dependency metrics without training models.")
    compare_parser.add_argument("--no-dependency-cache", action="store_true", help="Recompute dependency metrics even when dependency_metrics.csv already exists.")
    
    baseline_parser = subparsers.add_parser("baseline", help="Train linear/ridge/quadratic regression models")
    baseline_parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR, help="Directory containing experiment result CSVs.")
    baseline_parser.add_argument("--target", choices=TARGETS, default="response_time_us", help="Prediction target.")
    baseline_parser.add_argument("--model", choices=("linear", "ridge", "quadratic_ridge"), default="linear", help="Regression model to train.")
    baseline_parser.add_argument("--ridge-alpha", type=float, default=1.0, help="L2 regularization strength for ridge-based models.")
    baseline_parser.add_argument("--cv-folds", type=int, default=5, help="Number of cross-validation folds.")
    baseline_parser.add_argument("--test-fraction", type=float, default=0.25, help="Fraction of rows reserved for deterministic hold-out evaluation.")
    baseline_parser.add_argument("--seed", type=int, default=42, help="Fallback seed if SEED is not defined in .env")
    
    return parser.parse_args()


# ============================================================================
# SHARED UTILITIES
# ============================================================================

def load_env_file(path: Path) -> dict[str, str]:
    """Parse a .env file in the same style as the C++ env_loader.

    Supports:
    - comments (#...)
    - optional 'export ' prefix
    - optional single/double quotes around values
    """
    values: dict[str, str] = {}
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return values

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and ((value[0] == value[-1] == '"') or (value[0] == value[-1] == "'")):
            value = value[1:-1]
        if key:
            values[key] = value
    return values


def seed_from_dotenv(repo_root: Path, fallback: int = 42) -> int:
    """Return the global seed from SEED in environment/.env, falling back to fallback."""
    env_seed = os.getenv("SEED")
    if env_seed is None or env_seed.strip() == "":
        env = load_env_file(repo_root / ".env")
        env_seed = env.get("SEED", "")
    if env_seed is None or str(env_seed).strip() == "":
        print(f"Warning: SEED not found in .env. Using fallback seed={fallback}.")
        return fallback
    try:
        seed = int(str(env_seed).strip())
    except ValueError as exc:
        raise SystemExit("Invalid SEED value in .env (expected a non-negative integer).") from exc
    if seed < 0:
        raise SystemExit("Invalid SEED value in .env (expected a non-negative integer).")
    return seed


def set_global_seed(seed: int) -> None:
    """Seed Python and NumPy RNGs used by this script."""
    random.seed(seed)
    np.random.seed(seed)

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
    for csv_path in sorted(results_dir.rglob("resultados_experimentos_*.csv")):
        with csv_path.open("r", encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            if reader.fieldnames is None or "response_time_us" not in reader.fieldnames:
                continue
            for row in reader:
                row["source_file"] = csv_path.name
                rows.append(row)
    return rows


def target_gpu_demand_percent(row: dict[str, str]) -> float:
    match = re.search(r"_gputarget([0-9]+(?:p[0-9]+)?)_", row.get("experiment_name", ""))
    if not match:
        return 0.0
    return float(match.group(1).replace("p", "."))


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
    mpi_world_size = to_float(row, "mpi_world_size", 1.0)
    threads_per_process = to_float(row, "threads_per_process", 1.0)
    effective_workers = to_float(row, "effective_workers", mpi_world_size * threads_per_process)
    blocks_per_sm = to_float(row, "blocks_per_sm", total_blocks / sm_count if sm_count > 0 else 0.0)
    arrival_wait_ms = to_float(row, "arrival_wait_ms", 0.0)
    queueing_delay_us = to_float(row, "queueing_delay_us", response_us - requested_us)
    slowdown = to_float(row, "slowdown", response_us / requested_us if requested_us > 0 else math.nan)

    values = {name: to_float(row, name) for name in BASE_FEATURES}
    values.update({
        "threads_per_process": to_float(row, "threads_per_process", 1.0),
        "total_blocks": total_blocks,
        "total_cuda_threads": total_threads,
        "total_warps": total_warps,
        "warps_per_block": warps_per_block,
        "blocks_per_sm": blocks_per_sm,
        "total_blocks_per_sm": to_float(row, "total_blocks_per_sm", blocks_per_sm),
        "effective_workers": effective_workers,
        "workers_x_requested_busy_wait_us": to_float(row, "workers_x_requested_busy_wait_us", effective_workers * requested_us),
        "workers_x_total_warps": to_float(row, "workers_x_total_warps", effective_workers * total_warps),
        "workers_x_blocks_per_sm": to_float(row, "workers_x_blocks_per_sm", effective_workers * blocks_per_sm),
        "requested_busy_wait_us_per_arrival_ms": to_float(
            row, "requested_busy_wait_us_per_arrival_ms",
            requested_us / arrival_wait_ms if arrival_wait_ms > 0 else 0.0,
        ),
        "target_gpu_demand_percent": target_gpu_demand_percent(row),
        "queueing_delay_us": queueing_delay_us,
        "slowdown": slowdown,
    })
    kernel_type = row.get("kernel_type", "")
    for name in KERNEL_TYPES:
        values[f"kernel_type_{name}"] = 1.0 if kernel_type == name else 0.0
    return values


def build_matrix(rows: Iterable[dict[str, str]], target: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    x_rows: list[list[float]] = []
    y_values: list[float] = []
    features = list(BASE_FEATURES) + [f"kernel_type_{name}" for name in KERNEL_TYPES]

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
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not 0.0 < test_fraction < 1.0:
        raise SystemExit("--test-fraction must be between 0 and 1.")
    indices = np.arange(len(y))
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    test_size = max(1, int(round(len(indices) * test_fraction)))
    test_idx = indices[:test_size]
    train_idx = indices[test_size:]
    if len(train_idx) == 0:
        raise SystemExit("Not enough rows to create a train/test split.")
    return x[train_idx], x[test_idx], y[train_idx], y[test_idx]


# ============================================================================
# COMPARE MODE FUNCTIONS
# ============================================================================

def row_features_compare(row: dict[str, str]) -> dict[str, float]:
    blocks_x = to_float(row, "blocks_x", 0.0)
    grid_z = to_float(row, "grid_z", 1.0)
    threads_per_block = to_float(row, "threads_per_block", 0.0)
    total_blocks = to_float(row, "total_blocks", blocks_x * grid_z)
    warps_per_block = math.ceil(threads_per_block / 32.0) if threads_per_block > 0 else 0.0
    total_threads = to_float(row, "total_cuda_threads", total_blocks * threads_per_block)
    total_warps = to_float(row, "total_warps", total_blocks * warps_per_block)
    mpi_world_size = to_float(row, "mpi_world_size", 1.0)
    threads_per_process = to_float(row, "threads_per_process", 1.0)
    sm_count = to_float(row, "sm_count", 0.0)
    requested_us = to_float(row, "requested_busy_wait_us", 0.0)
    response_us = to_float(row, "response_time_us", math.nan)
    effective_workers = mpi_world_size * threads_per_process
    blocks_per_sm = to_float(row, "blocks_per_sm", total_blocks / sm_count if sm_count > 0 else 0.0)
    arrival_wait_ms = to_float(row, "arrival_wait_ms", 0.0)
    target_match = re.search(r"_gputarget([0-9]+(?:p[0-9]+)?)_", row.get("experiment_name", ""))
    target_gpu_demand_percent = float(target_match.group(1).replace("p", ".")) if target_match is not None else 0.0

    values = {
        "requested_busy_wait_us": requested_us,
        "mpi_world_size": mpi_world_size,
        "threads_per_process": threads_per_process,
        "blocks_x": blocks_x,
        "threads_per_block": threads_per_block,
        "grid_z": grid_z,
        "total_blocks": total_blocks,
        "total_cuda_threads": total_threads,
        "total_warps": total_warps,
        "warps_per_block": warps_per_block,
        "blocks_per_sm": blocks_per_sm,
        "total_blocks_per_sm": to_float(row, "total_blocks_per_sm", blocks_per_sm),
        "arrival_wait_ms": arrival_wait_ms,
        "launch_overhead_us": to_float(row, "launch_overhead_us", 0.0),
        "effective_workers": to_float(row, "effective_workers", effective_workers),
        "workers_x_requested_busy_wait_us": to_float(row, "workers_x_requested_busy_wait_us", requested_us * effective_workers),
        "workers_x_blocks_per_sm": to_float(row, "workers_x_blocks_per_sm", blocks_per_sm * effective_workers),
        "workers_x_total_warps": to_float(row, "workers_x_total_warps", total_warps * effective_workers),
        "requested_busy_wait_us_per_arrival_ms": to_float(
            row, "requested_busy_wait_us_per_arrival_ms",
            requested_us / arrival_wait_ms if arrival_wait_ms > 0 else 0.0,
        ),
        "target_gpu_demand_percent": target_gpu_demand_percent,
        "queueing_delay_us": to_float(row, "queueing_delay_us", response_us - requested_us),
        "slowdown": to_float(row, "slowdown", response_us / requested_us if requested_us > 0 else math.nan),
    }
    kernel_type = row.get("kernel_type", "")
    for name in KERNEL_TYPES:
        values[f"kernel_type_{name}"] = 1.0 if kernel_type == name else 0.0
    return values


def load_matrix_compare(paths: list[Path], target: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    features = list(BASE_FEATURES) + [f"kernel_type_{name}" for name in KERNEL_TYPES]
    x_rows = []
    y_values = []
    for path in paths:
        with path.open("r", encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            if reader.fieldnames is None or "response_time_us" not in reader.fieldnames:
                continue
            for row in reader:
                if to_float(row, "cuda_error_code", 0.0) != 0.0:
                    continue
                values = row_features_compare(row)
                y = values[target] if target in values else to_float(row, target)
                x = [values[name] for name in features]
                if math.isfinite(y) and all(math.isfinite(value) for value in x):
                    x_rows.append(x)
                    y_values.append(y)
    return np.asarray(x_rows, dtype=float), np.asarray(y_values, dtype=float), features


def result_paths(results_dirs: Path | list[Path], first_sweep: bool, include_regex: str = "") -> list[Path]:
    if isinstance(results_dirs, Path):
        results_dirs = [results_dirs]
    paths = sorted(
        path for results_dir in results_dirs
        for path in results_dir.rglob("resultados_experimentos_*.csv")
    )
    if include_regex:
        pattern_filter = re.compile(include_regex)
        paths = [path for path in paths if pattern_filter.search(str(path))]
    if not first_sweep:
        return paths
    pattern = re.compile(r"resultados_experimentos_(.+)_seed_(\d+)_(\d{8}_\d{6})_rank_(\d+)\.csv$")
    chosen: dict[tuple[str, str, str], tuple[str, Path]] = {}
    for path in paths:
        match = pattern.match(path.name)
        if not match:
            continue
        experiment_name, seed, timestamp, rank = match.groups()
        key = (experiment_name, seed, rank)
        if key not in chosen or timestamp < chosen[key][0]:
            chosen[key] = (timestamp, path)
    return sorted(path for _, path in chosen.values())


def standardize(train_x: np.ndarray, test_x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0)
    scale = train_x.std(axis=0)
    scale[scale == 0.0] = 1.0
    return (train_x - mean) / scale, (test_x - mean) / scale, mean, scale


def save_preprocessing(models_dir: Path, feature_names: list[str], mean: np.ndarray, scale: np.ndarray) -> Path:
    """Save standardization parameters required to reuse the saved models."""
    path = models_dir / "preprocessing.json"
    payload = {
        "features": feature_names,
        "standardize": {
            "mean": [float(v) for v in mean.tolist()],
            "scale": [float(v) for v in scale.tolist()],
        },
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    residual = y_true - y_pred
    mae = float(np.mean(np.abs(residual)))
    rmse = math.sqrt(float(np.mean(residual * residual)))
    denom = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - float(np.sum(residual * residual)) / denom if denom > 0 else math.nan
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


def fit_linear(train_x: np.ndarray, train_y: np.ndarray):
    model = LinearRegression()
    model.fit(train_x, train_y)
    return model


def fit_ridge(train_x: np.ndarray, train_y: np.ndarray, alpha: float = 1.0):
    model = Ridge(alpha=alpha)
    model.fit(train_x, train_y)
    return model


def predict_linear(model, x: np.ndarray) -> np.ndarray:
    return model.predict(x)


def quadratic_features(x: np.ndarray) -> np.ndarray:
    squared = x * x
    return np.column_stack([x, squared])


def feature_frame(x: np.ndarray, feature_names: list[str] | None):
    if not PANDAS_AVAILABLE or feature_names is None:
        return x
    return pd.DataFrame(x, columns=feature_names)


def train_knn(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, k: int, train_limit: int, params: dict | None = None) -> tuple:
    if len(train_y) > train_limit:
        train_x = train_x[:train_limit]
        train_y = train_y[:train_limit]
    params = dict(params or {})
    if "n_neighbors" not in params:
        params["n_neighbors"] = k
    params["n_neighbors"] = max(1, min(int(params["n_neighbors"]), len(train_y)))
    model = KNeighborsRegressor(**params)
    model.fit(train_x, train_y)
    return model, model.predict(test_x)


def train_lightgbm(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    seed: int = 42,
    objective: str = "regression",
    params: dict | None = None,
    feature_names: list[str] | None = None,
) -> tuple:
    """Train LightGBM model."""
    if not LIGHTGBM_AVAILABLE:
        return None, None
    final_params = {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": 7,
        "num_leaves": 31,
        "objective": objective,
        "random_state": seed,
        "n_jobs": 1,
        "verbose": -1,
    }
    if params:
        final_params.update(params)
    final_params.update({"objective": objective, "random_state": seed, "n_jobs": 1, "verbose": -1})
    model = lgb.LGBMRegressor(**final_params)
    train_input = feature_frame(train_x, feature_names)
    test_input = feature_frame(test_x, feature_names)
    model.fit(train_input, train_y)
    return model, model.predict(test_input)


def train_xgboost(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    seed: int = 42,
    objective: str = "reg:squarederror",
    params: dict | None = None,
    feature_names: list[str] | None = None,
) -> tuple:
    """Train XGBoost model."""
    if not XGBOOST_AVAILABLE:
        return None, None
    final_params = {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": 6,
        "objective": objective,
        "random_state": seed,
        "n_jobs": 1,
        "verbosity": 0,
    }
    if params:
        final_params.update(params)
    final_params.update({"objective": objective, "random_state": seed, "n_jobs": 1, "verbosity": 0})
    model = xgb.XGBRegressor(**final_params)
    train_input = feature_frame(train_x, feature_names)
    test_input = feature_frame(test_x, feature_names)
    model.fit(train_input, train_y)
    return model, model.predict(test_input)


def train_catboost(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    seed: int = 42,
    params: dict | None = None,
    feature_names: list[str] | None = None,
) -> tuple:
    """Train CatBoost model."""
    if not CATBOOST_AVAILABLE:
        return None, None
    final_params = {
        "iterations": 100,
        "learning_rate": 0.1,
        "depth": 6,
        "random_state": seed,
        "verbose": 0,
        "thread_count": 1,
        "allow_writing_files": False,
    }
    if params:
        final_params.update(params)
    final_params.update({"random_state": seed, "verbose": 0, "thread_count": 1, "allow_writing_files": False})
    model = CatBoostRegressor(**final_params)
    train_input = feature_frame(train_x, feature_names)
    test_input = feature_frame(test_x, feature_names)
    model.fit(train_input, train_y)
    return model, model.predict(test_input)


def train_lightgbm_quantile(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    quantile: float,
    seed: int = 42,
    params: dict | None = None,
    feature_names: list[str] | None = None,
) -> tuple:
    """Train LightGBM model with quantile regression objective."""
    if not LIGHTGBM_AVAILABLE:
        return None, None
    final_params = {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": 7,
        "num_leaves": 31,
        "objective": "quantile",
        "alpha": quantile,
        "random_state": seed,
        "n_jobs": 1,
        "verbose": -1,
    }
    if params:
        final_params.update(params)
    final_params.update({"objective": "quantile", "alpha": quantile, "random_state": seed, "n_jobs": 1, "verbose": -1})
    model = lgb.LGBMRegressor(**final_params)
    train_input = feature_frame(train_x, feature_names)
    test_input = feature_frame(test_x, feature_names)
    model.fit(train_input, train_y)
    return model, model.predict(test_input)


def train_xgboost_quantile(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    quantile: float,
    seed: int = 42,
    params: dict | None = None,
    feature_names: list[str] | None = None,
) -> tuple:
    """Train XGBoost model with quantile regression objective."""
    if not XGBOOST_AVAILABLE:
        return None, None
    final_params = {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": 6,
        "objective": "reg:quantileerror",
        "quantile_alpha": quantile,
        "random_state": seed,
        "n_jobs": 1,
        "verbosity": 0,
    }
    if params:
        final_params.update(params)
    final_params.update({"objective": "reg:quantileerror", "quantile_alpha": quantile, "random_state": seed, "n_jobs": 1, "verbosity": 0})
    model = xgb.XGBRegressor(**final_params)
    train_input = feature_frame(train_x, feature_names)
    test_input = feature_frame(test_x, feature_names)
    model.fit(train_input, train_y)
    return model, model.predict(test_input)


def save_metrics_csv(output_dir: Path, rows: list[dict[str, float | str]]) -> Path:
    path = output_dir / "regression_metrics.csv"
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["model", "MAE", "RMSE", "R2"])
        writer.writeheader()
        writer.writerows(rows)
    return path


def model_path(model_dir: Path, model_name: str) -> Path:
    return model_dir / f"{model_name}.pkl"


def model_metadata_path(model_dir: Path, model_name: str) -> Path:
    return model_dir / f"{model_name}.meta.json"


def save_model(model, model_dir: Path, model_name: str, metadata: dict[str, object] | None = None) -> Path:
    """Save trained model to disk using pickle or keras."""
    path = model_path(model_dir, model_name)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        with tmp_path.open("wb") as f:
            pickle.dump(model, f)
        tmp_path.replace(path)
        if metadata is not None:
            metadata_path = model_metadata_path(model_dir, model_name)
            metadata_tmp_path = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
            metadata_tmp_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            metadata_tmp_path.replace(metadata_path)
    except Exception as e:
        print(f"Warning: Could not save {model_name}: {e}")
    return path


def load_saved_model(model_dir: Path, model_name: str, expected_signature: dict[str, object]) -> object | None:
    path = model_path(model_dir, model_name)
    metadata_path = model_metadata_path(model_dir, model_name)
    if not path.exists() or not metadata_path.exists():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"Warning: Could not read metadata for {model_name}: {e}")
        return None
    if metadata.get("signature") != expected_signature:
        return None
    try:
        with path.open("rb") as file:
            return pickle.load(file)
    except Exception as e:
        print(f"Warning: Could not load {model_name}; retraining. Error: {e}")
        return None


def predict_model(model, x: np.ndarray, feature_names: list[str] | None = None) -> np.ndarray:
    model_input = feature_frame(x, feature_names)
    try:
        return model.predict(model_input)
    except Exception:
        if model_input is x:
            raise
        return model.predict(x)


def plot_metric(output_dir: Path, rows: list[dict[str, float | str]], metric: str) -> Path:
    path = output_dir / f"{metric.lower()}_comparison.png"
    names = [str(row["model"]) for row in rows]
    values = [float(row[metric]) for row in rows]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(names, values, color="#4c78a8")
    ax.set_title(f"{metric} por modelo")
    ax.set_ylabel(metric)
    ax.tick_params(axis="x", rotation=25)
    if metric == "R2":
        ax.axhline(0.0, color="black", linewidth=0.8)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def safe_plot_name(name: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", name.strip().lower()).strip("_")
    return normalized or "model"


def diagnostic_sample_indices(size: int, limit: int = DIAGNOSTIC_PLOT_SAMPLE_LIMIT) -> np.ndarray:
    if size <= limit:
        return np.arange(size)
    return np.linspace(0, size - 1, num=limit, dtype=int)


def residual_distribution_stats(model_name: str, residual: np.ndarray) -> dict[str, str]:
    residual = np.asarray(residual, dtype=float)
    residual = residual[np.isfinite(residual)]
    if len(residual) == 0:
        return {"model": model_name, "n": "0"}

    mean = float(np.mean(residual))
    std = float(np.std(residual, ddof=1)) if len(residual) > 1 else math.nan
    centered = residual - mean
    if len(residual) > 2 and math.isfinite(std) and std > 0:
        z = centered / std
        skewness = float(np.mean(z ** 3))
        excess_kurtosis = float(np.mean(z ** 4) - 3.0)
        jarque_bera = len(residual) / 6.0 * (skewness ** 2 + 0.25 * excess_kurtosis ** 2)
        jarque_bera_p_approx = math.exp(-0.5 * jarque_bera)
    else:
        skewness = math.nan
        excess_kurtosis = math.nan
        jarque_bera = math.nan
        jarque_bera_p_approx = math.nan

    quantiles = np.quantile(residual, [0.05, 0.25, 0.5, 0.75, 0.95])
    normal_like = (
        math.isfinite(skewness)
        and math.isfinite(excess_kurtosis)
        and abs(skewness) < 0.5
        and abs(excess_kurtosis) < 1.0
    )
    return {
        "model": model_name,
        "n": str(len(residual)),
        "mean_error": metric_value(mean),
        "std_error": metric_value(std),
        "median_error": metric_value(float(quantiles[2])),
        "q05_error": metric_value(float(quantiles[0])),
        "q25_error": metric_value(float(quantiles[1])),
        "q75_error": metric_value(float(quantiles[3])),
        "q95_error": metric_value(float(quantiles[4])),
        "skewness": metric_value(skewness),
        "excess_kurtosis": metric_value(excess_kurtosis),
        "jarque_bera": metric_value(jarque_bera),
        "jarque_bera_p_approx": metric_value(jarque_bera_p_approx),
        "normal_like_rule": "true" if normal_like else "false",
    }


def append_residual_diagnostics(output_dir: Path, stats_row: dict[str, str]) -> Path:
    diagnostics_dir = output_dir / "model_diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    path = diagnostics_dir / "residual_distribution_metrics.csv"
    fieldnames = [
        "model", "n", "mean_error", "std_error", "median_error",
        "q05_error", "q25_error", "q75_error", "q95_error",
        "skewness", "excess_kurtosis", "jarque_bera",
        "jarque_bera_p_approx", "normal_like_rule",
    ]
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(stats_row)
    return path


def plot_prediction_diagnostics(
    output_dir: Path,
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> list[Path]:
    diagnostics_dir = output_dir / "model_diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[valid]
    y_pred = y_pred[valid]
    if len(y_true) == 0:
        return []

    slug = safe_plot_name(model_name)
    residual = y_true - y_pred
    stats_row = residual_distribution_stats(model_name, residual)
    append_residual_diagnostics(output_dir, stats_row)

    error_path = diagnostics_dir / f"{slug}_error_distribution.png"
    fig, ax = plt.subplots(figsize=(8, 4.5))
    counts, bins, _patches = ax.hist(residual, bins=80, color="#4c78a8", alpha=0.85)
    mean = float(stats_row.get("mean_error", "") or "nan")
    std = float(stats_row.get("std_error", "") or "nan")
    if math.isfinite(mean) and math.isfinite(std) and std > 0 and len(bins) > 1:
        xs = np.linspace(float(bins[0]), float(bins[-1]), 300)
        bin_width = float(bins[1] - bins[0])
        ys = len(residual) * bin_width * (1.0 / (std * math.sqrt(2.0 * math.pi))) * np.exp(-0.5 * ((xs - mean) / std) ** 2)
        ax.plot(xs, ys, color="#f58518", linewidth=1.6, label="Normal de referencia")
        ax.legend(loc="best", fontsize=8)
    ax.axvline(0.0, color="black", linewidth=1.0)
    ax.set_title(f"Distribuicao do erro - {model_name}")
    ax.set_xlabel("Erro (real - predito)")
    ax.set_ylabel("Frequencia")
    fig.tight_layout()
    fig.savefig(error_path, dpi=150)
    plt.close(fig)

    qq_path = diagnostics_dir / f"{slug}_error_qq_normal.png"
    sample = diagnostic_sample_indices(len(residual))
    sample_residual = np.sort(residual[sample])
    if len(sample_residual) >= 3:
        probabilities = (np.arange(1, len(sample_residual) + 1) - 0.5) / len(sample_residual)
        normal = NormalDist()
        theoretical = np.asarray([normal.inv_cdf(float(p)) for p in probabilities], dtype=float)
        if math.isfinite(std) and std > 0:
            theoretical = mean + std * theoretical
        min_value = float(min(np.min(theoretical), np.min(sample_residual)))
        max_value = float(max(np.max(theoretical), np.max(sample_residual)))
        fig, ax = plt.subplots(figsize=(5.8, 5.8))
        ax.scatter(theoretical, sample_residual, s=7, alpha=0.18, color="#4c78a8", edgecolors="none")
        ax.plot([min_value, max_value], [min_value, max_value], color="black", linewidth=1.0)
        ax.set_title(f"QQ plot do erro - {model_name}")
        ax.set_xlabel("Quantis esperados se normal")
        ax.set_ylabel("Quantis observados do erro")
        fig.tight_layout()
        fig.savefig(qq_path, dpi=150)
        plt.close(fig)

    scatter_path = diagnostics_dir / f"{slug}_predicted_vs_actual.png"
    sample = diagnostic_sample_indices(len(y_true))
    sampled_true = y_true[sample]
    sampled_pred = y_pred[sample]
    min_value = float(min(np.min(sampled_true), np.min(sampled_pred)))
    max_value = float(max(np.max(sampled_true), np.max(sampled_pred)))

    fig, ax = plt.subplots(figsize=(5.8, 5.8))
    ax.scatter(sampled_true, sampled_pred, s=7, alpha=0.18, color="#2f7f7f", edgecolors="none")
    ax.plot([min_value, max_value], [min_value, max_value], color="black", linewidth=1.0)
    ax.set_title(f"Real x predito - {model_name}")
    ax.set_xlabel("Valor real")
    ax.set_ylabel("Valor predito")
    fig.tight_layout()
    fig.savefig(scatter_path, dpi=150)
    plt.close(fig)

    return [error_path, qq_path, scatter_path]


class CorrelationStats:
    def __init__(self) -> None:
        self.n = 0
        self.sum_x = 0.0
        self.sum_y = 0.0
        self.sum_x2 = 0.0
        self.sum_y2 = 0.0
        self.sum_xy = 0.0

    def add(self, x: float, y: float) -> None:
        if not math.isfinite(x) or not math.isfinite(y):
            return
        self.n += 1
        self.sum_x += x
        self.sum_y += y
        self.sum_x2 += x * x
        self.sum_y2 += y * y
        self.sum_xy += x * y

    def pearson(self) -> float:
        if self.n < 3:
            return math.nan
        numerator = self.n * self.sum_xy - self.sum_x * self.sum_y
        denom_x = self.n * self.sum_x2 - self.sum_x * self.sum_x
        denom_y = self.n * self.sum_y2 - self.sum_y * self.sum_y
        if denom_x <= 0.0 or denom_y <= 0.0:
            return math.nan
        return numerator / math.sqrt(denom_x * denom_y)


class SequenceDependencyStats:
    def __init__(self, max_lag: int = DEPENDENCY_MAX_LAG) -> None:
        self.n = 0
        self.sum_y = 0.0
        self.sum_y2 = 0.0
        self.diff2 = 0.0
        self.max_lag = max_lag
        self.lag_stats = {lag: CorrelationStats() for lag in range(1, max_lag + 1)}
        self._history: list[float] = []

    def add(self, value: float) -> None:
        if not math.isfinite(value):
            return
        self.n += 1
        self.sum_y += value
        self.sum_y2 += value * value
        if self._history:
            self.diff2 += (value - self._history[-1]) ** 2
        for lag in range(1, min(self.max_lag, len(self._history)) + 1):
            self.lag_stats[lag].add(self._history[-lag], value)
        self._history.append(value)
        if len(self._history) > self.max_lag:
            self._history.pop(0)

    def reset_group(self) -> None:
        self._history = []

    def lag1_acf(self) -> float:
        return self.lag_stats[1].pearson()

    def acf_rows(self) -> list[tuple[int, float, int]]:
        return [(lag, stats.pearson(), stats.n) for lag, stats in self.lag_stats.items()]

    def durbin_watson(self) -> float:
        if self.n < 3:
            return math.nan
        mean = self.sum_y / self.n
        centered_sum2 = self.sum_y2 - self.n * mean * mean
        if centered_sum2 <= 0.0:
            return math.nan
        return self.diff2 / centered_sum2

    def effective_sample_ratio(self) -> float:
        acf = self.lag1_acf()
        if not math.isfinite(acf):
            return math.nan
        positive_acf = max(0.0, acf)
        return 1.0 / (1.0 + 2.0 * positive_acf)


def metric_value(value: float) -> str:
    return f"{value:.12g}" if math.isfinite(value) else ""


def write_dependency_metrics(
    output_dir: Path,
    target: str,
    feature_stats: dict[str, CorrelationStats],
    sequence_stats: SequenceDependencyStats,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "dependency_metrics.csv"
    rows: list[dict[str, str]] = []
    for metric, value in (
        ("lag1_autocorrelation", sequence_stats.lag1_acf()),
        ("abs_lag1_autocorrelation", abs(sequence_stats.lag1_acf()) if math.isfinite(sequence_stats.lag1_acf()) else math.nan),
        ("durbin_watson", sequence_stats.durbin_watson()),
        ("effective_sample_size_ratio_lag1", sequence_stats.effective_sample_ratio()),
    ):
        rows.append({
            "category": "temporal",
            "metric": metric,
            "series": target,
            "feature": "",
            "lag": "1" if "lag1" in metric else "",
            "value": metric_value(value),
            "n": str(sequence_stats.n),
        })
    for lag, value, count in sequence_stats.acf_rows():
        rows.append({
            "category": "temporal",
            "metric": "autocorrelation",
            "series": target,
            "feature": "",
            "lag": str(lag),
            "value": metric_value(value),
            "n": str(count),
        })
    for feature, stats in feature_stats.items():
        value = stats.pearson()
        rows.append({
            "category": "feature_target",
            "metric": "pearson",
            "series": target,
            "feature": feature,
            "lag": "",
            "value": metric_value(value),
            "n": str(stats.n),
        })
    rows.sort(key=lambda row: (row["category"] != "temporal", -(abs(float(row["value"])) if row["value"] else -1.0)))
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["category", "metric", "series", "feature", "lag", "value", "n"])
        writer.writeheader()
        writer.writerows(rows)
    plot_dependency_feature_pearson(output_dir, target, rows)
    plot_dependency_acf(output_dir, target, rows)
    return path


def plot_dependency_feature_pearson(output_dir: Path, target: str, rows: list[dict[str, str]]) -> Path:
    path = output_dir / "dependency_feature_pearson.png"
    pairs = [
        (row["feature"], float(row["value"]))
        for row in rows
        if row.get("category") == "feature_target" and row.get("feature") and row.get("value")
    ]
    pairs = pairs[:20]
    if not pairs:
        return path
    labels = [name for name, _value in pairs]
    values = np.asarray([[value for _name, value in pairs]], dtype=float)
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.55), 2.6))
    image = ax.imshow(values, aspect="auto", cmap="coolwarm", vmin=-1.0, vmax=1.0)
    ax.set_title(f"Pearson feature x {target}")
    ax.set_yticks([0])
    ax.set_yticklabels([target])
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    for index, value in enumerate(values[0]):
        ax.text(index, 0, f"{value:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_dependency_acf(output_dir: Path, target: str, rows: list[dict[str, str]]) -> Path:
    path = output_dir / "dependency_acf.png"
    pairs: list[tuple[int, float]] = []
    for row in rows:
        if row.get("category") != "temporal" or row.get("metric") != "autocorrelation":
            continue
        try:
            lag = int(row.get("lag", ""))
            value = float(row.get("value", ""))
        except ValueError:
            continue
        if math.isfinite(value):
            pairs.append((lag, value))
    pairs.sort()
    if not pairs:
        return path

    lags = [lag for lag, _value in pairs]
    values = [value for _lag, value in pairs]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(lags, values, marker="o", linewidth=1.4, markersize=3.5, color="#4c78a8")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title(f"Autocorrelacao por lag - {target}")
    ax.set_xlabel("Lag")
    ax.set_ylabel("Autocorrelacao")
    ax.set_ylim(-1.05, 1.05)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def existing_dependency_result(paths: list[Path], target: str, output_dir: Path) -> dict[str, str] | None:
    metrics_path = output_dir / "dependency_metrics.csv"
    feature_plot = output_dir / "dependency_feature_pearson.png"
    acf_plot = output_dir / "dependency_acf.png"
    if not metrics_path.exists():
        return None
    values = {
        "dependency_lag1_autocorrelation": "",
        "dependency_abs_lag1_autocorrelation": "",
        "dependency_durbin_watson": "",
        "dependency_effective_sample_ratio_lag1": "",
    }
    rows_used = ""
    has_multilag_acf = False
    metrics_rows: list[dict[str, str]] = []
    try:
        with metrics_path.open("r", encoding="utf-8", newline="") as file:
            metrics_rows = list(csv.DictReader(file))
            for row in metrics_rows:
                if row.get("category") != "temporal" or row.get("series") != target:
                    continue
                metric = row.get("metric", "")
                if metric == "lag1_autocorrelation":
                    values["dependency_lag1_autocorrelation"] = row.get("value", "")
                    rows_used = row.get("n", "")
                elif metric == "abs_lag1_autocorrelation":
                    values["dependency_abs_lag1_autocorrelation"] = row.get("value", "")
                elif metric == "durbin_watson":
                    values["dependency_durbin_watson"] = row.get("value", "")
                elif metric == "effective_sample_size_ratio_lag1":
                    values["dependency_effective_sample_ratio_lag1"] = row.get("value", "")
                elif metric == "autocorrelation" and row.get("lag") not in ("", "1"):
                    has_multilag_acf = True
    except (OSError, csv.Error):
        return None
    if not all(values.values()) or not has_multilag_acf:
        return None
    try:
        plot_dependency_feature_pearson(output_dir, target, metrics_rows)
        plot_dependency_acf(output_dir, target, metrics_rows)
    except (OSError, csv.Error):
        pass
    return {
        "source_files": str(len(paths)),
        "rows_loaded": rows_used,
        "rows_used": rows_used,
        "train_rows": "",
        "test_rows": "",
        "best_model": "",
        "best_mae": "",
        "best_rmse": "",
        "best_r2": "",
        "metrics_csv": "",
        "mae_plot": "",
        "rmse_plot": "",
        "r2_plot": "",
        "models_dir": "",
        "dependency_metrics_csv": str(metrics_path),
        "dependency_feature_plot": str(feature_plot),
        "dependency_acf_plot": str(acf_plot),
        "dependency_cached": "true",
        **values,
    }


def dependency_only_result(paths: list[Path], target: str, output_dir: Path, use_cache: bool = True) -> dict[str, str]:
    if not paths:
        raise SystemExit("Nenhum CSV de resultados encontrado.")
    if use_cache:
        cached = existing_dependency_result(paths, target, output_dir)
        if cached is not None:
            return cached
    feature_names = list(BASE_FEATURES) + [f"kernel_type_{name}" for name in KERNEL_TYPES]
    feature_stats = {feature: CorrelationStats() for feature in feature_names}
    sequence_stats = SequenceDependencyStats()
    rows_loaded = 0
    rows_used = 0
    for path in paths:
        sequence_stats.reset_group()
        with path.open("r", encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            if reader.fieldnames is None or "response_time_us" not in reader.fieldnames:
                continue
            for row in reader:
                rows_loaded += 1
                if to_float(row, "cuda_error_code", 0.0) != 0.0:
                    continue
                values = row_features_compare(row)
                target_value = values[target] if target in values else to_float(row, target)
                if not math.isfinite(target_value):
                    continue
                rows_used += 1
                sequence_stats.add(target_value)
                for feature in feature_names:
                    feature_stats[feature].add(values[feature], target_value)
    metrics_path = write_dependency_metrics(output_dir, target, feature_stats, sequence_stats)
    lag1_acf = sequence_stats.lag1_acf()
    abs_lag1_acf = abs(lag1_acf) if math.isfinite(lag1_acf) else math.nan
    durbin_watson = sequence_stats.durbin_watson()
    ess_ratio = sequence_stats.effective_sample_ratio()
    return {
        "source_files": str(len(paths)),
        "rows_loaded": str(rows_loaded),
        "rows_used": str(rows_used),
        "train_rows": "",
        "test_rows": "",
        "best_model": "",
        "best_mae": "",
        "best_rmse": "",
        "best_r2": "",
        "metrics_csv": "",
        "mae_plot": "",
        "rmse_plot": "",
        "r2_plot": "",
        "models_dir": "",
        "dependency_metrics_csv": str(metrics_path),
        "dependency_feature_plot": str(output_dir / "dependency_feature_pearson.png"),
        "dependency_acf_plot": str(output_dir / "dependency_acf.png"),
        "dependency_cached": "false",
        "dependency_lag1_autocorrelation": metric_value(lag1_acf),
        "dependency_abs_lag1_autocorrelation": metric_value(abs_lag1_acf),
        "dependency_durbin_watson": metric_value(durbin_watson),
        "dependency_effective_sample_ratio_lag1": metric_value(ess_ratio),
    }


def train_and_plot(
    paths: list[Path], target: str, output_dir: Path, test_fraction: float,
    seed: int, knn_k: int, knn_train_limit: int, cv_folds: int = 5,
) -> dict[str, str]:
    if knn_train_limit <= 0:
        knn_train_limit = DEFAULT_KNN_TRAIN_LIMIT
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir = output_dir / "trained_models"
    models_dir.mkdir(parents=True, exist_ok=True)
    residual_metrics_path = output_dir / "model_diagnostics" / "residual_distribution_metrics.csv"
    if residual_metrics_path.exists():
        residual_metrics_path.unlink()

    if not paths:
        raise SystemExit("Nenhum CSV de resultados encontrado.")
    x, y, feature_names = load_matrix_compare(paths, target)
    if len(y) == 0:
        raise SystemExit("Nenhuma linha valida apos filtrar CSVs.")

    set_global_seed(seed)
    original_rows = len(y)
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_fraction, seed)
    train_x_std, test_x_std, mean, scale = standardize(train_x, test_x)

    model_signature: dict[str, object] = {
        "target": target,
        "seed": seed,
        "test_fraction": test_fraction,
        "cv_folds": cv_folds,
        "knn_k": knn_k,
        "knn_train_limit": knn_train_limit,
        "source_files": [str(path) for path in paths],
        "rows_loaded": original_rows,
        "rows_used": len(y),
        "train_rows": len(train_y),
        "test_rows": len(test_y),
        "features": feature_names,
    }
    save_preprocessing(models_dir, feature_names, mean, scale)
    (models_dir / "training_signature.json").write_text(
        json.dumps(model_signature, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    results: list[dict[str, float | str]] = []
    saved_model_paths: list[Path] = []
    resumed_model_names: list[str] = []
    trained_model_names: list[str] = []

    def fit_or_resume(
        model_name: str,
        train_fn,
        predict_x: np.ndarray,
        predict_feature_names: list[str] | None = None,
    ) -> tuple[object | None, np.ndarray | None]:
        model = load_saved_model(models_dir, model_name, model_signature)
        if model is not None:
            try:
                prediction = predict_model(model, predict_x, predict_feature_names)
                print(f"Resume: loaded {model_name}")
                saved_model_paths.append(model_path(models_dir, model_name))
                resumed_model_names.append(model_name)
                return model, prediction
            except Exception as e:
                print(f"Warning: Could not predict with saved {model_name}; retraining. Error: {e}")
        model, prediction = train_fn()
        if model is not None and prediction is not None:
            metadata = {"model_name": model_name, "signature": model_signature}
            path = save_model(model, models_dir, model_name, metadata)
            saved_model_paths.append(path)
            trained_model_names.append(model_name)
        return model, prediction

    def add_result(display_name: str, prediction: np.ndarray | None) -> None:
        if prediction is not None:
            results.append({"model": display_name, **metrics(test_y, prediction)})
            plot_prediction_diagnostics(output_dir, display_name, test_y, prediction)

    lr_model, lr_pred = fit_or_resume(
        "linear_regression",
        lambda: (lambda model: (model, predict_linear(model, test_x_std)))(fit_linear(train_x_std, train_y)),
        test_x_std,
    )
    add_result("Linear Regression", lr_pred)

    ridge_model, ridge_pred = fit_or_resume(
        "ridge_regression",
        lambda: (lambda model: (model, predict_linear(model, test_x_std)))(fit_ridge(train_x_std, train_y, alpha=1.0)),
        test_x_std,
    )
    add_result("Ridge Regression", ridge_pred)

    train_quad = quadratic_features(train_x_std)
    test_quad = quadratic_features(test_x_std)
    poly_model, poly_pred = fit_or_resume(
        "polynomial_ridge",
        lambda: (lambda model: (model, predict_linear(model, test_quad)))(fit_ridge(train_quad, train_y, alpha=1.0)),
        test_quad,
    )
    add_result("Polynomial Ridge", poly_pred)

    def train_decision_tree():
        model = DecisionTreeRegressor(max_depth=10, min_samples_leaf=100, random_state=seed)
        model.fit(train_x, train_y)
        return model, model.predict(test_x)

    tree_model, tree_pred = fit_or_resume("decision_tree", train_decision_tree, test_x)
    add_result("Decision Tree", tree_pred)

    def train_random_forest():
        model = RandomForestRegressor(n_estimators=60, max_depth=10, min_samples_leaf=100, random_state=seed, n_jobs=1)
        model.fit(train_x, train_y)
        return model, model.predict(test_x)

    forest_model, forest_pred = fit_or_resume("random_forest", train_random_forest, test_x)
    add_result("Random Forest", forest_pred)

    def train_gradient_boosting():
        model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=seed)
        model.fit(train_x, train_y)
        return model, model.predict(test_x)

    boosting_model, boosting_pred = fit_or_resume("gradient_boosting", train_gradient_boosting, test_x)
    add_result("Gradient Boosting", boosting_pred)

    knn_model, knn_pred = fit_or_resume(
        "knn_regression",
        lambda: train_knn(
            train_x_std,
            train_y,
            test_x_std,
            knn_k,
            knn_train_limit,
        ),
        test_x_std,
    )
    add_result("kNN Regression", knn_pred)

    if LIGHTGBM_AVAILABLE:
        try:
            lgb_model, lgb_pred = fit_or_resume(
                "lightgbm",
                lambda: train_lightgbm(
                    train_x,
                    train_y,
                    test_x,
                    seed,
                    feature_names=feature_names,
                ),
                test_x,
                feature_names,
            )
            add_result("LightGBM", lgb_pred)
        except Exception as e:
            print(f"Warning: LightGBM training failed: {e}")
    else:
        print("Warning: LightGBM not available")

    if XGBOOST_AVAILABLE:
        try:
            xgb_model, xgb_pred = fit_or_resume(
                "xgboost",
                lambda: train_xgboost(
                    train_x,
                    train_y,
                    test_x,
                    seed,
                    feature_names=feature_names,
                ),
                test_x,
                feature_names,
            )
            add_result("XGBoost", xgb_pred)
        except Exception as e:
            print(f"Warning: XGBoost training failed: {e}")
    else:
        print("Warning: XGBoost not available")

    if CATBOOST_AVAILABLE:
        try:
            cb_model, cb_pred = fit_or_resume(
                "catboost",
                lambda: train_catboost(
                    train_x,
                    train_y,
                    test_x,
                    seed,
                    feature_names=feature_names,
                ),
                test_x,
                feature_names,
            )
            add_result("CatBoost", cb_pred)
        except Exception as e:
            print(f"Warning: CatBoost training failed: {e}")
    else:
        print("Warning: CatBoost not available")

    if LIGHTGBM_AVAILABLE:
        quantile_preds = []
        for q in [0.90, 0.95, 0.99]:
            try:
                model_name = f"lightgbm_quantile_p{int(q * 100)}"
                q_model, q_pred = fit_or_resume(
                    model_name,
                    lambda q=q: train_lightgbm_quantile(train_x, train_y, test_x, q, seed, feature_names=feature_names),
                    test_x,
                    feature_names,
                )
                if q_pred is not None:
                    quantile_preds.append(q_pred)
            except Exception as e:
                print(f"Warning: LightGBM Quantile {q} training failed: {e}")
        if quantile_preds:
            lgb_q_pred = np.mean(quantile_preds, axis=0)
            add_result("LightGBM Quantile (p90/p95/p99)", lgb_q_pred)

    if XGBOOST_AVAILABLE:
        quantile_preds = []
        for q in [0.90, 0.95, 0.99]:
            try:
                model_name = f"xgboost_quantile_p{int(q * 100)}"
                q_model, q_pred = fit_or_resume(
                    model_name,
                    lambda q=q: train_xgboost_quantile(train_x, train_y, test_x, q, seed, feature_names=feature_names),
                    test_x,
                    feature_names,
                )
                if q_pred is not None:
                    quantile_preds.append(q_pred)
            except Exception as e:
                print(f"Warning: XGBoost Quantile {q} training failed: {e}")
        if quantile_preds:
            xgb_q_pred = np.mean(quantile_preds, axis=0)
            add_result("XGBoost Quantile (p90/p95/p99)", xgb_q_pred)


    saved_model_paths = sorted(set(saved_model_paths))

    metrics_path = save_metrics_csv(output_dir, results)
    plot_paths = [plot_metric(output_dir, results, metric) for metric in ("MAE", "RMSE", "R2")]
    best = min(results, key=lambda row: float(row["RMSE"]))

    models_info_path = models_dir / "models_info.txt"
    with models_info_path.open("w", encoding="utf-8") as f:
        f.write(f"Target: {target}\n")
        f.write(f"Seed: {seed}\n")
        f.write(f"Training set size: {len(train_y)}\n")
        f.write(f"Test set size: {len(test_y)}\n")
        f.write(f"Features: {train_x.shape[1]}\n")
        f.write(f"Models resumed: {len(resumed_model_names)}\n")
        for model_name in sorted(resumed_model_names):
            f.write(f"  - {model_name}\n")
        f.write(f"Models trained this run: {len(trained_model_names)}\n")
        for model_name in sorted(trained_model_names):
            f.write(f"  - {model_name}\n")
        f.write("Models saved:\n")
        for model_file in saved_model_paths:
            f.write(f"  - {model_file.name}\n")

    return {
        "source_files": str(len(paths)),
        "rows_loaded": str(original_rows),
        "rows_used": str(len(y)),
        "train_rows": str(len(train_y)),
        "test_rows": str(len(test_y)),
        "best_model": str(best["model"]),
        "best_mae": f"{float(best['MAE']):.6f}",
        "best_rmse": f"{float(best['RMSE']):.6f}",
        "best_r2": f"{float(best['R2']):.6f}",
        "metrics_csv": str(metrics_path),
        "mae_plot": str(plot_paths[0]),
        "rmse_plot": str(plot_paths[1]),
        "r2_plot": str(plot_paths[2]),
        "models_dir": str(models_dir),
        "model_diagnostics_dir": str(output_dir / "model_diagnostics"),
    }


def load_jobs(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def run_compare_jobs(args: argparse.Namespace, jobs_file: Path) -> int:
    if args.analysis_dir is None:
        raise SystemExit("--analysis-dir e obrigatorio quando --jobs-file e usado.")
    jobs = load_jobs(jobs_file)
    rows: list[dict[str, str]] = []
    for job in jobs:
        paths = result_paths(args.results_dir, args.first_sweep, job["include_regex"])
        if args.dependency_only:
            result = dependency_only_result(
                paths,
                job["target"],
                Path(job["output_dir"]),
                use_cache=not args.no_dependency_cache,
            )
        else:
            result = train_and_plot(
                paths, job["target"], Path(job["output_dir"]), args.test_fraction,
                args.seed, args.knn_k, args.knn_train_limit, args.cv_folds,
            )
        row = {"label": job["label"], "target": job["target"], **result}
        rows.append(row)
        if args.dependency_only:
            cached = " cached" if result.get("dependency_cached") == "true" else ""
            print(f"{job['label']} {job['target']}:{cached} files={result['source_files']} rows={result['rows_loaded']} used={result['rows_used']} dependency_metrics={result['dependency_metrics_csv']}")
        else:
            print(f"{job['label']} {job['target']}: files={result['source_files']} rows={result['rows_loaded']} used={result['rows_used']} best={result['best_model']} rmse={float(result['best_rmse']):.3f} r2={float(result['best_r2']):.3f} models_dir={result['models_dir']}")

    summary_path = args.analysis_dir / ("dependency_summary.csv" if args.dependency_only else "training_summary.csv")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "label", "target", "source_files", "rows_loaded", "rows_used", "train_rows", "test_rows",
        "best_model", "best_mae", "best_rmse", "best_r2", "metrics_csv", "mae_plot", "rmse_plot",
        "r2_plot", "models_dir", "model_diagnostics_dir", "dependency_metrics_csv", "dependency_feature_plot",
        "dependency_acf_plot",
        "dependency_cached",
        "dependency_lag1_autocorrelation", "dependency_abs_lag1_autocorrelation",
        "dependency_durbin_watson", "dependency_effective_sample_ratio_lag1",
    ]
    with summary_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"{'dependency_summary' if args.dependency_only else 'training_summary'}: {summary_path}")
    return 0


def mode_compare(args: argparse.Namespace) -> int:
    if args.jobs_file is not None:
        return run_compare_jobs(args, args.jobs_file)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = result_paths(args.results_dir, args.first_sweep, args.include_regex)
    if args.dependency_only:
        result = dependency_only_result(
            paths,
            args.target,
            args.output_dir,
            use_cache=not args.no_dependency_cache,
        )
    else:
        result = train_and_plot(
            paths, args.target, args.output_dir, args.test_fraction,
            args.seed, args.knn_k, args.knn_train_limit, args.cv_folds,
        )
    print(f"target: {args.target}")
    print(f"cv_folds: {args.cv_folds}")
    for key in (
        "source_files", "rows_loaded", "rows_used", "train_rows", "test_rows", "metrics_csv",
        "mae_plot", "rmse_plot", "r2_plot", "models_dir", "dependency_metrics_csv",
        "dependency_feature_plot", "best_model", "best_rmse", "best_r2",
    ):
        print(f"{key}: {result.get(key, '')}")
    return 0


# ============================================================================
# BASELINE MODE FUNCTIONS
# ============================================================================

def print_metrics_baseline(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    residual = y_true - y_pred
    mae = np.mean(np.abs(residual))
    rmse = math.sqrt(float(np.mean(residual * residual)))
    denominator = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1.0 - float(np.sum(residual * residual) / denominator) if denominator > 0 else math.nan
    print(f"{name}_rows: {len(y_true)}")
    print(f"{name}_mae: {mae:.6f}")
    print(f"{name}_rmse: {rmse:.6f}")
    print(f"{name}_r2: {r2:.6f}")


def mode_baseline(args: argparse.Namespace) -> int:
    rows = load_rows(args.results_dir)
    if not rows:
        print(f"No result CSVs found in {args.results_dir}.")
        return 1
    x, y, features = build_matrix(rows, args.target)
    train_x, test_x, train_y, test_y = train_test_split(x, y, args.test_fraction, args.seed)
    scaler = StandardScaler()
    train_x_std = scaler.fit_transform(train_x)
    test_x_std = scaler.transform(test_x)

    print(f"target: {args.target}")
    print(f"model: {args.model}")
    print(f"cv_folds: {args.cv_folds}")

    if args.model == "linear":
        model = fit_linear(train_x_std, train_y)
        print_metrics_baseline("train", train_y, model.predict(train_x_std))
        print_metrics_baseline("test", test_y, model.predict(test_x_std))
        kf = KFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)
        cv_scores = cross_val_score(model, train_x_std, train_y, cv=kf, scoring="neg_mean_squared_error")
        print(f"\ncross_validation_rmse_scores: {[math.sqrt(-s) for s in cv_scores]}")
        print(f"cross_validation_rmse_mean: {math.sqrt(-cv_scores.mean()):.6f}")
        print(f"cross_validation_rmse_std: {math.sqrt(cv_scores.std()):.6f}")
        print("\nstandardized_coefficients:")
        ranked = sorted(zip(features, model.coef_), key=lambda item: abs(item[1]), reverse=True)
        for feature, value in ranked:
            print(f"{feature}: {value:.6f}")
        print(f"intercept: {model.intercept_:.6f}")
    elif args.model == "ridge":
        model = fit_ridge(train_x_std, train_y, args.ridge_alpha)
        print_metrics_baseline("train", train_y, model.predict(train_x_std))
        print_metrics_baseline("test", test_y, model.predict(test_x_std))
        kf = KFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)
        cv_scores = cross_val_score(model, train_x_std, train_y, cv=kf, scoring="neg_mean_squared_error")
        print(f"\ncross_validation_rmse_scores: {[math.sqrt(-s) for s in cv_scores]}")
        print(f"cross_validation_rmse_mean: {math.sqrt(-cv_scores.mean()):.6f}")
        print(f"cross_validation_rmse_std: {math.sqrt(cv_scores.std()):.6f}")
        print("\nstandardized_coefficients:")
        ranked = sorted(zip(features, model.coef_), key=lambda item: abs(item[1]), reverse=True)
        for feature, value in ranked:
            print(f"{feature}: {value:.6f}")
        print(f"intercept: {model.intercept_:.6f}")
    elif args.model == "quadratic_ridge":
        train_quad = quadratic_features(train_x_std)
        test_quad = quadratic_features(test_x_std)
        quad_features = features + [f"{feature}^2" for feature in features]
        model = fit_ridge(train_quad, train_y, args.ridge_alpha)
        print_metrics_baseline("train", train_y, model.predict(train_quad))
        print_metrics_baseline("test", test_y, model.predict(test_quad))
        kf = KFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)
        cv_scores = cross_val_score(model, train_quad, train_y, cv=kf, scoring="neg_mean_squared_error")
        print(f"\ncross_validation_rmse_scores: {[math.sqrt(-s) for s in cv_scores]}")
        print(f"cross_validation_rmse_mean: {math.sqrt(-cv_scores.mean()):.6f}")
        print(f"cross_validation_rmse_std: {math.sqrt(cv_scores.std()):.6f}")
        print("\nstandardized_coefficients:")
        ranked = sorted(zip(quad_features, model.coef_), key=lambda item: abs(item[1]), reverse=True)
        for feature, value in ranked[:30]:
            print(f"{feature}: {value:.6f}")
        print(f"intercept: {model.intercept_:.6f}")
    return 0


def main() -> int:
    args = parse_args()
    args.seed = seed_from_dotenv(REPO_ROOT, fallback=args.seed)
    set_global_seed(args.seed)
    print(f"seed: {args.seed}")
    if args.mode == "compare":
        return mode_compare(args)
    elif args.mode == "baseline":
        return mode_baseline(args)
    else:
        print("Usage: regressor_analysis.py {compare|baseline} [options]")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Unified regression analysis for CUDA timing CSVs using scikit-learn.

Supports two modes:
1. compare: Compare multiple regression models
2. baseline: Train linear/ridge/quadratic regression models with cross-validation
"""

from __future__ import annotations

import argparse
import csv
import heapq
import inspect
import json
import math
import os
import pickle
import random
import re
from pathlib import Path
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

try:
    from tabpfn import TabPFNRegressor  # type: ignore
    TABPFN_AVAILABLE = True
except ImportError:
    try:
        from tab_pfn import TabPFNRegressor  # type: ignore
        TABPFN_AVAILABLE = True
    except ImportError:
        TABPFN_AVAILABLE = False

try:
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_RESULTS_DIR = REPO_ROOT / "resultados"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "resultados" / "model_comparison"
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
DEPENDENCE_MAX_LAG = 50
DEPENDENCE_MI_BINS = 32
DEPENDENCE_STATIC_FEATURES = tuple(
    name for name in BASE_FEATURES
    if name not in {"launch_overhead_us"}
) + tuple(f"kernel_type_{name}" for name in KERNEL_TYPES)
DEPENDENCE_PRESSURE_FEATURES = (
    "pending_kernels_at_launch",
    "in_flight_kernels_at_launch",
    "effective_workers",
    "workers_x_total_warps",
    "workers_x_blocks_per_sm",
    "target_gpu_demand_percent",
    "requested_busy_wait_us_per_arrival_ms",
    "arrival_wait_ms",
    "launch_overhead_us",
)
DEPENDENCE_HEATMAP_FEATURES = (
    "requested_busy_wait_us",
    "arrival_wait_ms",
    "mpi_world_size",
    "threads_per_process",
    "blocks_x",
    "threads_per_block",
    "total_warps",
    "blocks_per_sm",
    "effective_workers",
    "workers_x_total_warps",
    "workers_x_blocks_per_sm",
    "target_gpu_demand_percent",
    "pending_kernels_at_launch",
    "in_flight_kernels_at_launch",
    "launch_overhead_us",
) + tuple(f"kernel_type_{name}" for name in KERNEL_TYPES)


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
    compare_parser.add_argument("--max-rows", type=int, default=120000, help="Deterministic sample size for model comparison.")
    compare_parser.add_argument("--test-fraction", type=float, default=0.25)
    compare_parser.add_argument("--seed", type=int, default=42, help="Fallback seed if SEED is not defined in .env")
    compare_parser.add_argument("--knn-k", type=int, default=15)
    compare_parser.add_argument("--knn-train-limit", type=int, default=12000)
    compare_parser.add_argument("--cv-folds", type=int, default=5, help="Number of cross-validation folds.")
    compare_parser.add_argument("--optimize-hyperparams", action="store_true", help="Use Optuna to optimize hyperparameters.")
    compare_parser.add_argument("--optuna-trials", type=int, default=20, help="Number of Optuna trials for hyperparameter optimization.")
    compare_parser.add_argument("--dependency-only", "--skip-training", action="store_true", help="Generate dependency/independence CSVs and plots without training models.")
    compare_parser.add_argument("--enable-tabpfn", action="store_true", help="Train TabPFN. Disabled by default because large CPU datasets are too slow.")
    compare_parser.add_argument("--tabpfn-device", choices=("auto", "cpu", "cuda"), default="auto", help="Device used by TabPFN. auto uses CUDA when available, otherwise CPU.")
    compare_parser.add_argument("--tabpfn-train-limit", type=int, default=1000, help="Maximum CPU rows used by TabPFN when --enable-tabpfn is set. Ignored on CUDA.")
    
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


def deterministic_sample(x: np.ndarray, y: np.ndarray, max_rows: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if max_rows <= 0 or len(y) <= max_rows:
        return x, y
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(y), max_rows, replace=False)
    return x[idx], y[idx]


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


def optimize_tree_params(x: np.ndarray, y: np.ndarray, cv_folds: int, seed: int, n_trials: int = 20):
    if not OPTUNA_AVAILABLE:
        return {"max_depth": 10, "min_samples_leaf": 100}
    def objective(trial: optuna.Trial) -> float:
        max_depth = trial.suggest_int("max_depth", 3, 20)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 10, 200)
        model = DecisionTreeRegressor(max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=seed)
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)
        scores = cross_val_score(model, x, y, cv=kf, scoring="neg_mean_squared_error", n_jobs=-1)
        return float(np.mean(-scores))
    study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=seed), pruner=MedianPruner())
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def optimize_forest_params(x: np.ndarray, y: np.ndarray, cv_folds: int, seed: int, n_trials: int = 20):
    if not OPTUNA_AVAILABLE:
        return {"n_estimators": 100, "max_depth": 10, "min_samples_leaf": 100}
    def objective(trial: optuna.Trial) -> float:
        n_estimators = trial.suggest_int("n_estimators", 10, 300)
        max_depth = trial.suggest_int("max_depth", 5, 25)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 10, 200)
        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, 
                                     min_samples_leaf=min_samples_leaf, random_state=seed, n_jobs=-1)
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)
        scores = cross_val_score(model, x, y, cv=kf, scoring="neg_mean_squared_error", n_jobs=-1)
        return float(np.mean(-scores))
    study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=seed), pruner=MedianPruner())
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def optimize_boosting_params(x: np.ndarray, y: np.ndarray, cv_folds: int, seed: int, n_trials: int = 20):
    if not OPTUNA_AVAILABLE:
        return {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3}
    def objective(trial: optuna.Trial) -> float:
        n_estimators = trial.suggest_int("n_estimators", 50, 300)
        learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
        max_depth = trial.suggest_int("max_depth", 2, 8)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 20, 200)
        model = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate, 
                                         max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=seed)
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)
        scores = cross_val_score(model, x, y, cv=kf, scoring="neg_mean_squared_error", n_jobs=-1)
        return float(np.mean(-scores))
    study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=seed), pruner=MedianPruner())
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params



def safe_cv_folds(cv_folds: int, n_rows: int) -> int:
    return max(2, min(cv_folds, n_rows))


def optimize_knn_params(x: np.ndarray, y: np.ndarray, cv_folds: int, seed: int, n_trials: int = 20):
    if not OPTUNA_AVAILABLE:
        return {"n_neighbors": 15, "weights": "distance"}
    upper_k = max(1, min(100, len(y) - 1))
    def objective(trial: optuna.Trial) -> float:
        k = trial.suggest_int("n_neighbors", 1, upper_k)
        weights = trial.suggest_categorical("weights", ["uniform", "distance"])
        model = KNeighborsRegressor(n_neighbors=k, weights=weights)
        kf = KFold(n_splits=safe_cv_folds(cv_folds, len(y)), shuffle=True, random_state=seed)
        scores = cross_val_score(model, x, y, cv=kf, scoring="neg_mean_squared_error", n_jobs=-1)
        return float(np.mean(-scores))
    study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=seed), pruner=MedianPruner())
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def optimize_lightgbm_params(x: np.ndarray, y: np.ndarray, cv_folds: int, seed: int, n_trials: int = 20, objective: str = "regression"):
    if not OPTUNA_AVAILABLE or not LIGHTGBM_AVAILABLE:
        return {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 7, "num_leaves": 31}
    def objective_fn(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 400),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "num_leaves": trial.suggest_int("num_leaves", 8, 128),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "objective": objective,
            "random_state": seed,
            "n_jobs": -1,
            "verbose": -1,
        }
        model = lgb.LGBMRegressor(**params)
        kf = KFold(n_splits=safe_cv_folds(cv_folds, len(y)), shuffle=True, random_state=seed)
        scores = cross_val_score(model, x, y, cv=kf, scoring="neg_mean_squared_error", n_jobs=-1)
        return float(np.mean(-scores))
    study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=seed), pruner=MedianPruner())
    study.optimize(objective_fn, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def optimize_xgboost_params(x: np.ndarray, y: np.ndarray, cv_folds: int, seed: int, n_trials: int = 20, objective: str = "reg:squarederror"):
    if not OPTUNA_AVAILABLE or not XGBOOST_AVAILABLE:
        return {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 6}
    def objective_fn(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 400),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "objective": objective,
            **({"quantile_alpha": 0.95} if objective == "reg:quantileerror" else {}),
            "random_state": seed,
            "n_jobs": -1,
            "verbosity": 0,
        }
        model = xgb.XGBRegressor(**params)
        kf = KFold(n_splits=safe_cv_folds(cv_folds, len(y)), shuffle=True, random_state=seed)
        scores = cross_val_score(model, x, y, cv=kf, scoring="neg_mean_squared_error", n_jobs=-1)
        return float(np.mean(-scores))
    study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=seed), pruner=MedianPruner())
    study.optimize(objective_fn, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def optimize_catboost_params(x: np.ndarray, y: np.ndarray, cv_folds: int, seed: int, n_trials: int = 20):
    if not OPTUNA_AVAILABLE or not CATBOOST_AVAILABLE:
        return {"iterations": 100, "learning_rate": 0.1, "depth": 6}
    def objective(trial: optuna.Trial) -> float:
        params = {
            "iterations": trial.suggest_int("iterations", 50, 400),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "depth": trial.suggest_int("depth", 3, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
            "random_state": seed,
            "verbose": 0,
            "thread_count": -1,
            "allow_writing_files": False,
        }
        model = CatBoostRegressor(**params)
        kf = KFold(n_splits=safe_cv_folds(cv_folds, len(y)), shuffle=True, random_state=seed)
        scores = cross_val_score(model, x, y, cv=kf, scoring="neg_mean_squared_error", n_jobs=1)
        return float(np.mean(-scores))
    study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=seed), pruner=MedianPruner())
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def optimize_tabpfn_params(seed: int, n_trials: int = 20):
    if not OPTUNA_AVAILABLE or not TABPFN_AVAILABLE:
        return {"n_ensemble": 10}
    # TabPFN is expensive and version-dependent. Keep the search small and safe.
    study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=seed), pruner=MedianPruner())
    def objective(trial: optuna.Trial) -> float:
        # Used only to select a deterministic ensemble size; real fitting happens once later.
        return float(trial.suggest_int("n_ensemble", 4, 16))
    study.optimize(objective, n_trials=min(n_trials, 5), show_progress_bar=False)
    return study.best_params


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
        "n_jobs": -1,
        "verbose": -1,
    }
    if params:
        final_params.update(params)
    final_params.update({"objective": objective, "random_state": seed, "n_jobs": -1, "verbose": -1})
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
        "n_jobs": -1,
        "verbosity": 0,
    }
    if params:
        final_params.update(params)
    final_params.update({"objective": objective, "random_state": seed, "n_jobs": -1, "verbosity": 0})
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
        "thread_count": -1,
        "allow_writing_files": False,
    }
    if params:
        final_params.update(params)
    final_params.update({"random_state": seed, "verbose": 0, "thread_count": -1, "allow_writing_files": False})
    model = CatBoostRegressor(**final_params)
    train_input = feature_frame(train_x, feature_names)
    test_input = feature_frame(test_x, feature_names)
    model.fit(train_input, train_y)
    return model, model.predict(test_input)


def tabpfn_cuda_available() -> bool:
    try:
        import torch  # type: ignore
    except ImportError:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def resolve_tabpfn_device(requested_device: str) -> str:
    if requested_device == "auto":
        return "cuda" if tabpfn_cuda_available() else "cpu"
    return requested_device


def _make_tabpfn(seed: int, n_ensemble: int = 10, device: str = "cpu"):
    signature = inspect.signature(TabPFNRegressor)
    kwargs = {}
    if "device" in signature.parameters:
        kwargs["device"] = device
    elif device == "cuda":
        print("Warning: TabPFNRegressor has no device parameter; cannot force CUDA for this installed version.")
    if "n_ensemble" in signature.parameters:
        kwargs["n_ensemble"] = n_ensemble
    elif "N_ensemble_configurations" in signature.parameters:
        kwargs["N_ensemble_configurations"] = n_ensemble
    if "seed" in signature.parameters:
        kwargs["seed"] = seed
    elif "random_state" in signature.parameters:
        kwargs["random_state"] = seed
    return TabPFNRegressor(**kwargs)


def train_tabpfn(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    seed: int = 42,
    params: dict | None = None,
    train_limit: int = 1000,
    requested_device: str = "auto",
) -> tuple:
    """Train TabPFN model."""
    if not TABPFN_AVAILABLE:
        return None, None
    try:
        n_ensemble = int((params or {}).get("n_ensemble", 10))
        device = resolve_tabpfn_device(requested_device)
        if device == "cuda" and not tabpfn_cuda_available():
            print("Warning: TabPFN CUDA requested, but torch.cuda.is_available() is false. Skipping TabPFN.")
            return None, None
        effective_train_limit = train_limit if device == "cpu" else 0
        if effective_train_limit > 0 and len(train_y) > effective_train_limit:
            rng = np.random.default_rng(seed)
            indices = rng.choice(len(train_y), effective_train_limit, replace=False)
            train_x = train_x[indices]
            train_y = train_y[indices]
        print(f"TabPFN device: {device}; train_rows: {len(train_y)}")
        model = _make_tabpfn(seed, n_ensemble=n_ensemble, device=device)
        model.fit(train_x, train_y)
        return model, model.predict(test_x)
    except Exception as e:
        print(f"Warning: TabPFN training failed: {e}")
        return None, None


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
        "n_jobs": -1,
        "verbose": -1,
    }
    if params:
        final_params.update(params)
    final_params.update({"objective": "quantile", "alpha": quantile, "random_state": seed, "n_jobs": -1, "verbose": -1})
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
        "n_jobs": -1,
        "verbosity": 0,
    }
    if params:
        final_params.update(params)
    final_params.update({"objective": "reg:quantileerror", "quantile_alpha": quantile, "random_state": seed, "n_jobs": -1, "verbosity": 0})
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


def run_id_from_path(path: Path) -> str:
    match = re.match(r"resultados_experimentos_(.+)_seed_(\d+)_(\d{8}_\d{6})_rank_(\d+)\.csv$", path.name)
    if not match:
        return f"{path.parent.name}:{path.stem}"
    experiment_name, seed, timestamp, _rank = match.groups()
    return f"{path.parent.name}:{experiment_name}:seed{seed}:{timestamp}"


def finite_float(value: object, default: float = math.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def finite_values(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(list(values), dtype=float)
    return array[np.isfinite(array)]


def pearson_corr(x_values: Iterable[float], y_values: Iterable[float]) -> float:
    x = np.asarray(list(x_values), dtype=float)
    y = np.asarray(list(y_values), dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 3:
        return math.nan
    x = x[mask]
    y = y[mask]
    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return math.nan
    return float(np.corrcoef(x, y)[0, 1])


def rank_average(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        rank = (start + end - 1) / 2.0 + 1.0
        ranks[order[start:end]] = rank
        start = end
    return ranks


def spearman_corr(x_values: Iterable[float], y_values: Iterable[float]) -> float:
    x = np.asarray(list(x_values), dtype=float)
    y = np.asarray(list(y_values), dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 3:
        return math.nan
    x = x[mask]
    y = y[mask]
    if len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
        return math.nan
    return pearson_corr(rank_average(x), rank_average(y))


def normalized_mutual_information(x_values: Iterable[float], y_values: Iterable[float], bins: int = DEPENDENCE_MI_BINS) -> float:
    x = np.asarray(list(x_values), dtype=float)
    y = np.asarray(list(y_values), dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 8:
        return math.nan
    x = x[mask]
    y = y[mask]
    if len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
        return math.nan
    adjusted_bins = max(4, min(bins, int(math.sqrt(len(x)))))
    hist, _, _ = np.histogram2d(x, y, bins=adjusted_bins)
    total = float(hist.sum())
    if total <= 0.0:
        return math.nan
    pxy = hist / total
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)
    nz = pxy > 0
    px_py = px[:, None] * py[None, :]
    mi = float(np.sum(pxy[nz] * np.log(pxy[nz] / px_py[nz])))
    hx = float(-np.sum(px[px > 0] * np.log(px[px > 0])))
    hy = float(-np.sum(py[py > 0] * np.log(py[py > 0])))
    if hx <= 0.0 or hy <= 0.0:
        return math.nan
    return mi / math.sqrt(hx * hy)


def group_observations(observations: list[dict[str, float | str]]) -> dict[str, list[dict[str, float | str]]]:
    groups: dict[str, list[dict[str, float | str]]] = {}
    for observation in observations:
        groups.setdefault(str(observation["run_id"]), []).append(observation)
    for group in groups.values():
        group.sort(key=lambda row: (finite_float(row.get("order_ns"), math.inf), str(row.get("source_file", "")), finite_float(row.get("row_index"), 0.0)))
    return groups


def grouped_lag_pairs(groups: dict[str, list[dict[str, float | str]]], series: str, lag: int) -> tuple[list[float], list[float]]:
    previous: list[float] = []
    current: list[float] = []
    for group in groups.values():
        values = [finite_float(row.get(series)) for row in group]
        if len(values) <= lag:
            continue
        for index in range(lag, len(values)):
            left = values[index - lag]
            right = values[index]
            if math.isfinite(left) and math.isfinite(right):
                previous.append(left)
                current.append(right)
    return previous, current


def grouped_acf(groups: dict[str, list[dict[str, float | str]]], series: str, max_lag: int) -> list[tuple[int, float, int]]:
    rows: list[tuple[int, float, int]] = []
    for lag in range(1, max_lag + 1):
        previous, current = grouped_lag_pairs(groups, series, lag)
        rows.append((lag, pearson_corr(previous, current), len(previous)))
    return rows


def grouped_lag_mi(groups: dict[str, list[dict[str, float | str]]], series: str, max_lag: int) -> list[tuple[int, float, int]]:
    rows: list[tuple[int, float, int]] = []
    for lag in range(1, max_lag + 1):
        previous, current = grouped_lag_pairs(groups, series, lag)
        rows.append((lag, normalized_mutual_information(previous, current), len(previous)))
    return rows


def durbin_watson_grouped(groups: dict[str, list[dict[str, float | str]]], series: str) -> float:
    all_values = finite_values(finite_float(row.get(series)) for group in groups.values() for row in group)
    if len(all_values) < 3:
        return math.nan
    mean = float(np.mean(all_values))
    numerator = 0.0
    denominator = 0.0
    for group in groups.values():
        values = [finite_float(row.get(series)) for row in group]
        centered = [value - mean for value in values if math.isfinite(value)]
        if len(centered) < 2:
            continue
        array = np.asarray(centered, dtype=float)
        numerator += float(np.sum(np.diff(array) ** 2))
        denominator += float(np.sum(array ** 2))
    return numerator / denominator if denominator > 0.0 else math.nan


def effective_sample_size(n_rows: int, acf_rows: list[tuple[int, float, int]]) -> float:
    positive_sum = 0.0
    for _lag, value, _pairs in acf_rows:
        if not math.isfinite(value):
            continue
        if value <= 0.0:
            continue
        positive_sum += value
    denominator = 1.0 + 2.0 * positive_sum
    return n_rows / denominator if denominator > 0.0 else math.nan


def temporal_sample_observations(
    observations: list[dict[str, float | str]], max_rows: int
) -> list[dict[str, float | str]]:
    if max_rows <= 0 or len(observations) <= max_rows:
        return observations
    ordered = sorted(
        observations,
        key=lambda row: (str(row.get("run_id", "")), finite_float(row.get("order_ns"), math.inf), str(row.get("source_file", "")), finite_float(row.get("row_index"), 0.0)),
    )
    indices = np.linspace(0, len(ordered) - 1, max_rows, dtype=int)
    return [ordered[int(index)] for index in indices]


def load_dependency_observations(paths: list[Path], target: str, max_rows: int) -> tuple[list[dict[str, float | str]], int]:
    observations: list[dict[str, float | str]] = []
    for path in paths:
        run_id = run_id_from_path(path)
        with path.open("r", encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            if reader.fieldnames is None or "response_time_us" not in reader.fieldnames:
                continue
            for row_index, row in enumerate(reader):
                if to_float(row, "cuda_error_code", 0.0) != 0.0:
                    continue
                values = row_features_compare(row)
                target_value = values[target] if target in values else to_float(row, target)
                if not math.isfinite(target_value):
                    continue
                submit_ns = to_float(row, "submit_time_ns", math.nan)
                completion_ns = to_float(row, "completion_time_ns", math.nan)
                measurement_start_ns = to_float(row, "measurement_start_time_ns", math.nan)
                time_since_us = to_float(row, "time_since_experiment_start_us", math.nan)
                fallback_ns = measurement_start_ns + time_since_us * 1000.0 if math.isfinite(measurement_start_ns) and math.isfinite(time_since_us) else math.nan
                order_ns = submit_ns if math.isfinite(submit_ns) else fallback_ns
                submitted_count = to_float(row, "rank_local_submitted_count", math.nan)
                completed_count = to_float(row, "rank_local_completed_count", math.nan)
                pending = submitted_count - completed_count if math.isfinite(submitted_count) and math.isfinite(completed_count) else math.nan
                observation: dict[str, float | str] = {
                    **values,
                    "target_value": target_value,
                    "target_residual": math.nan,
                    "run_id": run_id,
                    "source_file": path.name,
                    "row_index": float(row_index),
                    "submit_time_ns": submit_ns,
                    "completion_time_ns": completion_ns,
                    "order_ns": order_ns,
                    "pending_kernels_at_launch": max(0.0, pending) if math.isfinite(pending) else math.nan,
                    "in_flight_kernels_at_launch": math.nan,
                }
                observations.append(observation)

    original_rows = len(observations)
    observations = temporal_sample_observations(observations, max_rows)
    compute_in_flight_counts(observations)
    add_linear_residuals(observations)
    return observations, original_rows


def compute_in_flight_counts(observations: list[dict[str, float | str]]) -> None:
    groups = group_observations(observations)
    for group in groups.values():
        heap: list[float] = []
        for observation in group:
            submit_ns = finite_float(observation.get("submit_time_ns"))
            completion_ns = finite_float(observation.get("completion_time_ns"))
            if not math.isfinite(submit_ns):
                observation["in_flight_kernels_at_launch"] = math.nan
                continue
            while heap and heap[0] <= submit_ns:
                heapq.heappop(heap)
            observation["in_flight_kernels_at_launch"] = float(len(heap))
            if math.isfinite(completion_ns) and completion_ns > submit_ns:
                heapq.heappush(heap, completion_ns)


def add_linear_residuals(observations: list[dict[str, float | str]]) -> None:
    if len(observations) < 3:
        return
    features = [
        name for name in DEPENDENCE_STATIC_FEATURES
        if any(math.isfinite(finite_float(row.get(name))) for row in observations)
    ]
    x_rows: list[list[float]] = []
    y_values: list[float] = []
    row_indices: list[int] = []
    for index, observation in enumerate(observations):
        y = finite_float(observation.get("target_value"))
        x = [finite_float(observation.get(name)) for name in features]
        if math.isfinite(y) and all(math.isfinite(value) for value in x):
            x_rows.append(x)
            y_values.append(y)
            row_indices.append(index)
    if len(y_values) < 3:
        return
    y = np.asarray(y_values, dtype=float)
    x = np.asarray(x_rows, dtype=float)
    if x.size == 0:
        residual = y - float(np.mean(y))
    else:
        scale = x.std(axis=0)
        keep = scale > 0.0
        if int(keep.sum()) == 0:
            residual = y - float(np.mean(y))
        else:
            x = x[:, keep]
            x = (x - x.mean(axis=0)) / x.std(axis=0)
            design = np.column_stack([np.ones(len(y)), x])
            coef, *_ = np.linalg.lstsq(design, y, rcond=None)
            residual = y - design @ coef
    for index, value in zip(row_indices, residual):
        observations[index]["target_residual"] = float(value)


def safe_metric_value(value: float) -> str:
    return f"{value:.12g}" if math.isfinite(value) else ""


def write_dependency_metrics(
    path: Path,
    rows: list[dict[str, str]],
) -> Path:
    fieldnames = ["category", "metric", "series", "feature", "lag", "value", "n"]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def plot_dependency_acf(output_dir: Path, target: str, acf_target: list[tuple[int, float, int]], acf_residual: list[tuple[int, float, int]]) -> Path:
    path = output_dir / "dependency_acf.png"
    lags = [lag for lag, _value, _pairs in acf_target]
    target_values = [value for _lag, value, _pairs in acf_target]
    residual_values = [value for _lag, value, _pairs in acf_residual]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.plot(lags, target_values, marker="o", linewidth=1.5, label=target)
    ax.plot(lags, residual_values, marker="s", linewidth=1.5, label="residuo_linear")
    ax.set_title(f"Autocorrelacao por lag - {target}")
    ax.set_xlabel("Lag")
    ax.set_ylabel("ACF")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_dependency_lag_mi(output_dir: Path, target: str, mi_target: list[tuple[int, float, int]], mi_residual: list[tuple[int, float, int]]) -> Path:
    path = output_dir / "dependency_lag_mi.png"
    lags = [lag for lag, _value, _pairs in mi_target]
    target_values = [value for _lag, value, _pairs in mi_target]
    residual_values = [value for _lag, value, _pairs in mi_residual]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(lags, target_values, marker="o", linewidth=1.5, label=target)
    ax.plot(lags, residual_values, marker="s", linewidth=1.5, label="residuo_linear")
    ax.set_title(f"Informacao mutua normalizada por lag - {target}")
    ax.set_xlabel("Lag")
    ax.set_ylabel("NMI")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_dependency_pressure(
    output_dir: Path,
    target: str,
    pressure_rows: list[dict[str, str]],
) -> Path:
    path = output_dir / "dependency_pressure_features.png"
    rows = [row for row in pressure_rows if row["metric"] in {"spearman", "normalized_mutual_information"} and row["value"] != ""]
    features = list(dict.fromkeys(row["feature"] for row in rows))
    spearman_values = [float(next((row["value"] for row in rows if row["feature"] == feature and row["metric"] == "spearman"), "nan")) for feature in features]
    mi_values = [float(next((row["value"] for row in rows if row["feature"] == feature and row["metric"] == "normalized_mutual_information"), "nan")) for feature in features]
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(features))
    width = 0.38
    ax.bar(x - width / 2, spearman_values, width=width, label="Spearman", color="#4c78a8")
    ax.bar(x + width / 2, mi_values, width=width, label="NMI", color="#f58518")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title(f"Dependencia com pressao GPU - {target}")
    ax.set_ylabel("Valor")
    ax.set_xticks(x)
    ax.set_xticklabels(features, rotation=30, ha="right")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_dependency_feature_heatmap(
    output_dir: Path,
    target: str,
    feature_rows: list[dict[str, str]],
) -> Path:
    path = output_dir / "dependency_feature_spearman.png"
    pairs = [
        (row["feature"], float(row["value"]))
        for row in feature_rows
        if row["metric"] == "spearman" and row["value"] != ""
    ]
    pairs.sort(key=lambda item: abs(item[1]), reverse=True)
    pairs = pairs[:20]
    if not pairs:
        pairs = [("sem_dados", 0.0)]
    labels = [name for name, _value in pairs]
    values = np.asarray([[value for _name, value in pairs]], dtype=float)
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.55), 2.6))
    image = ax.imshow(values, aspect="auto", cmap="coolwarm", vmin=-1.0, vmax=1.0)
    ax.set_title(f"Spearman feature x {target}")
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


def plot_dependency_rolling(output_dir: Path, target: str, observations: list[dict[str, float | str]]) -> Path:
    path = output_dir / "dependency_rolling_windows.png"
    ordered = sorted(
        observations,
        key=lambda row: (str(row.get("run_id", "")), finite_float(row.get("order_ns"), math.inf), str(row.get("source_file", "")), finite_float(row.get("row_index"), 0.0)),
    )
    n_rows = len(ordered)
    if n_rows == 0:
        return path
    window_count = max(1, min(160, n_rows // 20 if n_rows >= 20 else n_rows))
    chunks = np.array_split(np.arange(n_rows), window_count)
    x_values: list[float] = []
    mean_values: list[float] = []
    p95_values: list[float] = []
    std_values: list[float] = []
    in_flight_values: list[float] = []
    pending_values: list[float] = []
    for chunk in chunks:
        rows = [ordered[int(index)] for index in chunk]
        target_values = finite_values(finite_float(row.get("target_value")) for row in rows)
        if len(target_values) == 0:
            continue
        x_values.append(float(chunk.mean()))
        mean_values.append(float(np.mean(target_values)))
        p95_values.append(float(np.percentile(target_values, 95)))
        std_values.append(float(np.std(target_values)))
        in_flight = finite_values(finite_float(row.get("in_flight_kernels_at_launch")) for row in rows)
        pending = finite_values(finite_float(row.get("pending_kernels_at_launch")) for row in rows)
        in_flight_values.append(float(np.mean(in_flight)) if len(in_flight) else math.nan)
        pending_values.append(float(np.mean(pending)) if len(pending) else math.nan)

    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    ax_top.plot(x_values, mean_values, linewidth=1.6, label="media", color="#4c78a8")
    ax_top.plot(x_values, p95_values, linewidth=1.4, label="p95", color="#e45756")
    ax_top.fill_between(x_values, mean_values, p95_values, color="#e45756", alpha=0.12)
    ax_top.set_title(f"Janelas temporais - {target}")
    ax_top.set_ylabel(target)
    ax_top.legend()
    ax_top.grid(True, alpha=0.25)

    ax_bottom.plot(x_values, std_values, linewidth=1.4, label="std_target", color="#72b7b2")
    ax_bottom.plot(x_values, in_flight_values, linewidth=1.4, label="in_flight_medio", color="#54a24b")
    ax_bottom.plot(x_values, pending_values, linewidth=1.4, label="pending_medio", color="#f58518")
    ax_bottom.set_xlabel("Ordem temporal agrupada")
    ax_bottom.set_ylabel("Valor por janela")
    ax_bottom.legend()
    ax_bottom.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def save_dependency_analysis(
    paths: list[Path],
    target: str,
    output_dir: Path,
    max_rows: int,
) -> dict[str, str]:
    observations, original_rows = load_dependency_observations(paths, target, max_rows)
    metrics_path = output_dir / "dependency_metrics.csv"
    if len(observations) < 3:
        write_dependency_metrics(
            metrics_path,
            [
                {
                    "category": "summary",
                    "metric": "rows_loaded",
                    "series": target,
                    "feature": "",
                    "lag": "",
                    "value": str(original_rows),
                    "n": str(original_rows),
                },
                {
                    "category": "summary",
                    "metric": "rows_used",
                    "series": target,
                    "feature": "",
                    "lag": "",
                    "value": str(len(observations)),
                    "n": str(original_rows),
                },
            ],
        )
        return {
            "dependency_metrics_csv": str(metrics_path),
            "dependency_acf_plot": "",
            "dependency_lag_mi_plot": "",
            "dependency_pressure_plot": "",
            "dependency_rolling_plot": "",
            "dependency_feature_heatmap": "",
            "dependency_dw_target": "",
            "dependency_dw_residual": "",
            "dependency_ess_ratio_target": "",
            "dependency_ess_ratio_residual": "",
            "dependency_max_abs_acf10_target": "",
            "dependency_max_abs_acf10_residual": "",
        }

    groups = group_observations(observations)
    max_lag = max(1, min(DEPENDENCE_MAX_LAG, min((len(group) for group in groups.values()), default=2) - 1))
    acf_target = grouped_acf(groups, "target_value", max_lag)
    acf_residual = grouped_acf(groups, "target_residual", max_lag)
    mi_target = grouped_lag_mi(groups, "target_value", max_lag)
    mi_residual = grouped_lag_mi(groups, "target_residual", max_lag)
    dw_target = durbin_watson_grouped(groups, "target_value")
    dw_residual = durbin_watson_grouped(groups, "target_residual")
    ess_target = effective_sample_size(len(observations), acf_target)
    ess_residual = effective_sample_size(len(observations), acf_residual)
    ess_ratio_target = ess_target / len(observations) if len(observations) else math.nan
    ess_ratio_residual = ess_residual / len(observations) if len(observations) else math.nan
    max_abs_acf10_target = max((abs(value) for lag, value, _pairs in acf_target if lag <= 10 and math.isfinite(value)), default=math.nan)
    max_abs_acf10_residual = max((abs(value) for lag, value, _pairs in acf_residual if lag <= 10 and math.isfinite(value)), default=math.nan)

    metric_rows: list[dict[str, str]] = []

    def add_metric(category: str, metric: str, value: float, series: str = "", feature: str = "", lag: str = "", n: int | str = "") -> None:
        metric_rows.append({
            "category": category,
            "metric": metric,
            "series": series,
            "feature": feature,
            "lag": lag,
            "value": safe_metric_value(value),
            "n": str(n),
        })

    add_metric("summary", "rows_loaded", float(original_rows), target, n=original_rows)
    add_metric("summary", "rows_used", float(len(observations)), target, n=len(observations))
    add_metric("summary", "durbin_watson", dw_target, target, n=len(observations))
    add_metric("summary", "durbin_watson", dw_residual, "residuo_linear", n=len(observations))
    add_metric("summary", "effective_sample_size", ess_target, target, n=len(observations))
    add_metric("summary", "effective_sample_size", ess_residual, "residuo_linear", n=len(observations))
    add_metric("summary", "effective_sample_size_ratio", ess_ratio_target, target, n=len(observations))
    add_metric("summary", "effective_sample_size_ratio", ess_ratio_residual, "residuo_linear", n=len(observations))
    add_metric("summary", "max_abs_acf_lag_1_10", max_abs_acf10_target, target, n=len(observations))
    add_metric("summary", "max_abs_acf_lag_1_10", max_abs_acf10_residual, "residuo_linear", n=len(observations))

    for series_name, rows in ((target, acf_target), ("residuo_linear", acf_residual)):
        for lag, value, pairs in rows:
            add_metric("temporal", "acf", value, series_name, lag=str(lag), n=pairs)
    for series_name, rows in ((target, mi_target), ("residuo_linear", mi_residual)):
        for lag, value, pairs in rows:
            add_metric("temporal", "normalized_mutual_information", value, series_name, lag=str(lag), n=pairs)

    pressure_rows: list[dict[str, str]] = []
    feature_rows: list[dict[str, str]] = []
    target_values = [finite_float(row.get("target_value")) for row in observations]
    for feature in DEPENDENCE_PRESSURE_FEATURES:
        values = [finite_float(row.get(feature)) for row in observations]
        spearman = spearman_corr(values, target_values)
        nmi = normalized_mutual_information(values, target_values)
        row_s = {
            "category": "pressure",
            "metric": "spearman",
            "series": target,
            "feature": feature,
            "lag": "",
            "value": safe_metric_value(spearman),
            "n": str(len(observations)),
        }
        row_mi = {
            "category": "pressure",
            "metric": "normalized_mutual_information",
            "series": target,
            "feature": feature,
            "lag": "",
            "value": safe_metric_value(nmi),
            "n": str(len(observations)),
        }
        pressure_rows.extend([row_s, row_mi])
        metric_rows.extend([row_s, row_mi])

    for feature in DEPENDENCE_HEATMAP_FEATURES:
        values = [finite_float(row.get(feature)) for row in observations]
        spearman = spearman_corr(values, target_values)
        row = {
            "category": "feature_target",
            "metric": "spearman",
            "series": target,
            "feature": feature,
            "lag": "",
            "value": safe_metric_value(spearman),
            "n": str(len(observations)),
        }
        feature_rows.append(row)
        metric_rows.append(row)

    write_dependency_metrics(metrics_path, metric_rows)
    acf_path = plot_dependency_acf(output_dir, target, acf_target, acf_residual)
    lag_mi_path = plot_dependency_lag_mi(output_dir, target, mi_target, mi_residual)
    pressure_path = plot_dependency_pressure(output_dir, target, pressure_rows)
    rolling_path = plot_dependency_rolling(output_dir, target, observations)
    heatmap_path = plot_dependency_feature_heatmap(output_dir, target, feature_rows)

    return {
        "dependency_metrics_csv": str(metrics_path),
        "dependency_acf_plot": str(acf_path),
        "dependency_lag_mi_plot": str(lag_mi_path),
        "dependency_pressure_plot": str(pressure_path),
        "dependency_rolling_plot": str(rolling_path),
        "dependency_feature_heatmap": str(heatmap_path),
        "dependency_dw_target": safe_metric_value(dw_target),
        "dependency_dw_residual": safe_metric_value(dw_residual),
        "dependency_ess_ratio_target": safe_metric_value(ess_ratio_target),
        "dependency_ess_ratio_residual": safe_metric_value(ess_ratio_residual),
        "dependency_max_abs_acf10_target": safe_metric_value(max_abs_acf10_target),
        "dependency_max_abs_acf10_residual": safe_metric_value(max_abs_acf10_residual),
    }


def dependency_summary_counts(metrics_csv: str) -> tuple[str, str]:
    rows_loaded = ""
    rows_used = ""
    try:
        with Path(metrics_csv).open("r", encoding="utf-8", newline="") as file:
            for row in csv.DictReader(file):
                if row.get("category") != "summary":
                    continue
                if row.get("metric") == "rows_loaded":
                    rows_loaded = row.get("value", "")
                elif row.get("metric") == "rows_used":
                    rows_used = row.get("value", "")
    except FileNotFoundError:
        pass
    return rows_loaded, rows_used


def dependency_only_result(paths: list[Path], target: str, output_dir: Path, max_rows: int) -> dict[str, str]:
    if not paths:
        raise SystemExit("Nenhum CSV de resultados encontrado.")
    output_dir.mkdir(parents=True, exist_ok=True)
    dependency_paths = save_dependency_analysis(paths, target, output_dir, max_rows)
    rows_loaded, rows_used = dependency_summary_counts(dependency_paths["dependency_metrics_csv"])
    return {
        "source_files": str(len(paths)),
        "rows_loaded": rows_loaded,
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
        **dependency_paths,
    }


def train_and_plot(
    paths: list[Path], target: str, output_dir: Path, max_rows: int, test_fraction: float,
    seed: int, knn_k: int, knn_train_limit: int, cv_folds: int = 5,
    optimize_hyperparams: bool = False, optuna_trials: int = 20,
    enable_tabpfn: bool = False, tabpfn_train_limit: int = 1000, tabpfn_device: str = "auto",
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir = output_dir / "trained_models"
    models_dir.mkdir(parents=True, exist_ok=True)

    if not paths:
        raise SystemExit("Nenhum CSV de resultados encontrado.")
    x, y, feature_names = load_matrix_compare(paths, target)
    if len(y) == 0:
        raise SystemExit("Nenhuma linha valida apos filtrar CSVs.")

    set_global_seed(seed)
    original_rows = len(y)
    x, y = deterministic_sample(x, y, max_rows, seed)
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_fraction, seed)
    train_x_std, test_x_std, mean, scale = standardize(train_x, test_x)

    resolved_tabpfn_device = resolve_tabpfn_device(tabpfn_device) if enable_tabpfn and TABPFN_AVAILABLE else tabpfn_device
    model_signature: dict[str, object] = {
        "target": target,
        "seed": seed,
        "max_rows": max_rows,
        "test_fraction": test_fraction,
        "cv_folds": cv_folds,
        "optimize_hyperparams": optimize_hyperparams,
        "optuna_trials": optuna_trials,
        "knn_k": knn_k,
        "knn_train_limit": knn_train_limit,
        "enable_tabpfn": enable_tabpfn,
        "tabpfn_train_limit": tabpfn_train_limit,
        "tabpfn_device": tabpfn_device,
        "resolved_tabpfn_device": resolved_tabpfn_device,
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
        if optimize_hyperparams:
            tree_params = optimize_tree_params(train_x, train_y, cv_folds, seed, optuna_trials)
            model = DecisionTreeRegressor(**tree_params, random_state=seed)
        else:
            model = DecisionTreeRegressor(max_depth=10, min_samples_leaf=100, random_state=seed)
        model.fit(train_x, train_y)
        return model, model.predict(test_x)

    tree_model, tree_pred = fit_or_resume("decision_tree", train_decision_tree, test_x)
    add_result("Decision Tree", tree_pred)

    def train_random_forest():
        if optimize_hyperparams:
            forest_params = optimize_forest_params(train_x, train_y, cv_folds, seed, optuna_trials)
            model = RandomForestRegressor(**forest_params, random_state=seed, n_jobs=-1)
        else:
            model = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_leaf=100, random_state=seed, n_jobs=-1)
        model.fit(train_x, train_y)
        return model, model.predict(test_x)

    forest_model, forest_pred = fit_or_resume("random_forest", train_random_forest, test_x)
    add_result("Random Forest", forest_pred)

    def train_gradient_boosting():
        if optimize_hyperparams:
            boosting_params = optimize_boosting_params(train_x, train_y, cv_folds, seed, optuna_trials)
            model = GradientBoostingRegressor(**boosting_params, random_state=seed)
        else:
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
            optimize_knn_params(train_x_std, train_y, cv_folds, seed, optuna_trials) if optimize_hyperparams else None,
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
                    params=optimize_lightgbm_params(train_x, train_y, cv_folds, seed, optuna_trials) if optimize_hyperparams else None,
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
                    params=optimize_xgboost_params(train_x, train_y, cv_folds, seed, optuna_trials) if optimize_hyperparams else None,
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
                    params=optimize_catboost_params(train_x, train_y, cv_folds, seed, optuna_trials) if optimize_hyperparams else None,
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
        q_params = optimize_lightgbm_params(train_x, train_y, cv_folds, seed, optuna_trials, objective="quantile") if optimize_hyperparams else None
        for q in [0.90, 0.95, 0.99]:
            try:
                model_name = f"lightgbm_quantile_p{int(q * 100)}"
                q_model, q_pred = fit_or_resume(
                    model_name,
                    lambda q=q: train_lightgbm_quantile(train_x, train_y, test_x, q, seed, params=q_params, feature_names=feature_names),
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
        q_params = optimize_xgboost_params(train_x, train_y, cv_folds, seed, optuna_trials, objective="reg:quantileerror") if optimize_hyperparams else None
        for q in [0.90, 0.95, 0.99]:
            try:
                model_name = f"xgboost_quantile_p{int(q * 100)}"
                q_model, q_pred = fit_or_resume(
                    model_name,
                    lambda q=q: train_xgboost_quantile(train_x, train_y, test_x, q, seed, params=q_params, feature_names=feature_names),
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

    if enable_tabpfn and TABPFN_AVAILABLE:
        try:
            pfn_params = optimize_tabpfn_params(seed, optuna_trials) if optimize_hyperparams else None
            pfn_model, pfn_pred = fit_or_resume(
                "tabpfn",
                lambda: train_tabpfn(
                    train_x,
                    train_y,
                    test_x,
                    seed,
                    params=pfn_params,
                    train_limit=tabpfn_train_limit,
                    requested_device=tabpfn_device,
                ),
                test_x,
            )
            add_result("TabPFN", pfn_pred)
        except Exception as e:
            print(f"Warning: TabPFN training failed: {e}")
    elif enable_tabpfn:
        print("Warning: TabPFN not available")

    saved_model_paths = sorted(set(saved_model_paths))

    metrics_path = save_metrics_csv(output_dir, results)
    plot_paths = [plot_metric(output_dir, results, metric) for metric in ("MAE", "RMSE", "R2")]
    dependency_paths = save_dependency_analysis(paths, target, output_dir, max_rows)
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
        **dependency_paths,
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
            result = dependency_only_result(paths, job["target"], Path(job["output_dir"]), args.max_rows)
        else:
            result = train_and_plot(
                paths, job["target"], Path(job["output_dir"]), args.max_rows, args.test_fraction,
                args.seed, args.knn_k, args.knn_train_limit, args.cv_folds, args.optimize_hyperparams, args.optuna_trials,
                args.enable_tabpfn, args.tabpfn_train_limit, args.tabpfn_device,
            )
        row = {"label": job["label"], "target": job["target"], **result}
        rows.append(row)
        if args.dependency_only:
            print(f"{job['label']} {job['target']}: files={result['source_files']} rows={result['rows_loaded']} used={result['rows_used']} dependency_metrics={result['dependency_metrics_csv']}")
        else:
            print(f"{job['label']} {job['target']}: files={result['source_files']} rows={result['rows_loaded']} used={result['rows_used']} best={result['best_model']} rmse={float(result['best_rmse']):.3f} r2={float(result['best_r2']):.3f} models_dir={result['models_dir']}")

    summary_path = args.analysis_dir / "training_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "label", "target", "source_files", "rows_loaded", "rows_used", "train_rows", "test_rows",
        "best_model", "best_mae", "best_rmse", "best_r2", "metrics_csv", "mae_plot", "rmse_plot",
        "r2_plot", "models_dir", "dependency_metrics_csv", "dependency_acf_plot", "dependency_lag_mi_plot",
        "dependency_pressure_plot", "dependency_rolling_plot", "dependency_feature_heatmap",
        "dependency_dw_target", "dependency_dw_residual", "dependency_ess_ratio_target",
        "dependency_ess_ratio_residual", "dependency_max_abs_acf10_target", "dependency_max_abs_acf10_residual",
    ]
    with summary_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"training_summary: {summary_path}")
    return 0


def mode_compare(args: argparse.Namespace) -> int:
    if args.jobs_file is not None:
        return run_compare_jobs(args, args.jobs_file)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = result_paths(args.results_dir, args.first_sweep, args.include_regex)
    if args.dependency_only:
        result = dependency_only_result(paths, args.target, args.output_dir, args.max_rows)
    else:
        result = train_and_plot(
            paths, args.target, args.output_dir, args.max_rows, args.test_fraction,
            args.seed, args.knn_k, args.knn_train_limit, args.cv_folds, args.optimize_hyperparams, args.optuna_trials,
            args.enable_tabpfn, args.tabpfn_train_limit, args.tabpfn_device,
        )
    print(f"target: {args.target}")
    print(f"cv_folds: {args.cv_folds}")
    print(f"optimize_hyperparams: {args.optimize_hyperparams}")
    print(f"dependency_only: {args.dependency_only}")
    for key in (
        "source_files", "rows_loaded", "rows_used", "train_rows", "test_rows", "metrics_csv",
        "mae_plot", "rmse_plot", "r2_plot", "models_dir", "dependency_metrics_csv",
        "dependency_acf_plot", "dependency_lag_mi_plot", "dependency_pressure_plot",
        "dependency_rolling_plot", "dependency_feature_heatmap", "dependency_dw_target",
        "dependency_dw_residual", "dependency_ess_ratio_target", "dependency_ess_ratio_residual",
        "dependency_max_abs_acf10_target", "dependency_max_abs_acf10_residual", "best_model",
        "best_rmse", "best_r2",
    ):
        print(f"{key}: {result[key]}")
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

#!/usr/bin/env python3
"""Unified regression analysis for CUDA timing CSVs using scikit-learn.

Supports two modes:
1. compare: Compare multiple regression models
2. baseline: Train linear/ridge/quadratic regression models with cross-validation
"""

from __future__ import annotations

import argparse
import csv
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


def train_lightgbm(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, seed: int = 42, objective: str = "regression", params: dict | None = None) -> tuple:
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
    model.fit(train_x, train_y)
    return model, model.predict(test_x)


def train_xgboost(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, seed: int = 42, objective: str = "reg:squarederror", params: dict | None = None) -> tuple:
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
    model.fit(train_x, train_y)
    return model, model.predict(test_x)


def train_catboost(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, seed: int = 42, params: dict | None = None) -> tuple:
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
    model.fit(train_x, train_y)
    return model, model.predict(test_x)


def _make_tabpfn(seed: int, n_ensemble: int = 10):
    signature = inspect.signature(TabPFNRegressor)
    kwargs = {}
    if "device" in signature.parameters:
        kwargs["device"] = "cpu"
    if "n_ensemble" in signature.parameters:
        kwargs["n_ensemble"] = n_ensemble
    elif "N_ensemble_configurations" in signature.parameters:
        kwargs["N_ensemble_configurations"] = n_ensemble
    if "seed" in signature.parameters:
        kwargs["seed"] = seed
    elif "random_state" in signature.parameters:
        kwargs["random_state"] = seed
    return TabPFNRegressor(**kwargs)


def train_tabpfn(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, seed: int = 42, params: dict | None = None) -> tuple:
    """Train TabPFN model."""
    if not TABPFN_AVAILABLE:
        return None, None
    try:
        n_ensemble = int((params or {}).get("n_ensemble", 10))
        model = _make_tabpfn(seed, n_ensemble=n_ensemble)
        model.fit(train_x, train_y)
        return model, model.predict(test_x)
    except Exception as e:
        print(f"Warning: TabPFN training failed: {e}")
        return None, None


def train_lightgbm_quantile(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, quantile: float, seed: int = 42, params: dict | None = None) -> tuple:
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
    model.fit(train_x, train_y)
    return model, model.predict(test_x)


def train_xgboost_quantile(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, quantile: float, seed: int = 42, params: dict | None = None) -> tuple:
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
    model.fit(train_x, train_y)
    return model, model.predict(test_x)


def save_metrics_csv(output_dir: Path, rows: list[dict[str, float | str]]) -> Path:
    path = output_dir / "regression_metrics.csv"
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["model", "MAE", "RMSE", "R2"])
        writer.writeheader()
        writer.writerows(rows)
    return path


def save_model(model, model_dir: Path, model_name: str) -> Path:
    """Save trained model to disk using pickle or keras."""
    model_path = model_dir / f"{model_name}.pkl"
    try:
        with model_path.open("wb") as f:
            pickle.dump(model, f)
    except Exception as e:
        print(f"Warning: Could not save {model_name}: {e}")
    return model_path


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


def train_and_plot(
    paths: list[Path], target: str, output_dir: Path, max_rows: int, test_fraction: float,
    seed: int, knn_k: int, knn_train_limit: int, cv_folds: int = 5,
    optimize_hyperparams: bool = False, optuna_trials: int = 20,
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

    results: list[dict[str, float | str]] = []
    trained_models: dict[str, object] = {}

    lr_model = fit_linear(train_x_std, train_y)
    trained_models["linear_regression"] = lr_model
    results.append({"model": "Linear Regression", **metrics(test_y, predict_linear(lr_model, test_x_std))})

    ridge_model = fit_ridge(train_x_std, train_y, alpha=1.0)
    trained_models["ridge_regression"] = ridge_model
    results.append({"model": "Ridge Regression", **metrics(test_y, predict_linear(ridge_model, test_x_std))})

    train_quad = quadratic_features(train_x_std)
    test_quad = quadratic_features(test_x_std)
    poly_model = fit_ridge(train_quad, train_y, alpha=1.0)
    trained_models["polynomial_ridge"] = poly_model
    results.append({"model": "Polynomial Ridge", **metrics(test_y, predict_linear(poly_model, test_quad))})

    if optimize_hyperparams:
        tree_params = optimize_tree_params(train_x, train_y, cv_folds, seed, optuna_trials)
        tree_model = DecisionTreeRegressor(**tree_params, random_state=seed)
    else:
        tree_model = DecisionTreeRegressor(max_depth=10, min_samples_leaf=100, random_state=seed)
    tree_model.fit(train_x, train_y)
    trained_models["decision_tree"] = tree_model
    results.append({"model": "Decision Tree", **metrics(test_y, tree_model.predict(test_x))})

    if optimize_hyperparams:
        forest_params = optimize_forest_params(train_x, train_y, cv_folds, seed, optuna_trials)
        forest_model = RandomForestRegressor(**forest_params, random_state=seed, n_jobs=-1)
    else:
        forest_model = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_leaf=100, random_state=seed, n_jobs=-1)
    forest_model.fit(train_x, train_y)
    trained_models["random_forest"] = forest_model
    results.append({"model": "Random Forest", **metrics(test_y, forest_model.predict(test_x))})

    if optimize_hyperparams:
        boosting_params = optimize_boosting_params(train_x, train_y, cv_folds, seed, optuna_trials)
        boosting_model = GradientBoostingRegressor(**boosting_params, random_state=seed)
    else:
        boosting_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=seed)
    boosting_model.fit(train_x, train_y)
    trained_models["gradient_boosting"] = boosting_model
    results.append({"model": "Gradient Boosting", **metrics(test_y, boosting_model.predict(test_x))})

    knn_params = optimize_knn_params(train_x_std, train_y, cv_folds, seed, optuna_trials) if optimize_hyperparams else None
    knn_model, knn_pred = train_knn(train_x_std, train_y, test_x_std, knn_k, knn_train_limit, knn_params)
    trained_models["knn_regression"] = knn_model
    results.append({"model": "kNN Regression", **metrics(test_y, knn_pred)})

    if LIGHTGBM_AVAILABLE:
        try:
            lgb_params = optimize_lightgbm_params(train_x, train_y, cv_folds, seed, optuna_trials) if optimize_hyperparams else None
            lgb_model, lgb_pred = train_lightgbm(train_x, train_y, test_x, seed, params=lgb_params)
            if lgb_model is not None and lgb_pred is not None:
                trained_models["lightgbm"] = lgb_model
                results.append({"model": "LightGBM", **metrics(test_y, lgb_pred)})
        except Exception as e:
            print(f"Warning: LightGBM training failed: {e}")
    else:
        print("Warning: LightGBM not available")

    if XGBOOST_AVAILABLE:
        try:
            xgb_params = optimize_xgboost_params(train_x, train_y, cv_folds, seed, optuna_trials) if optimize_hyperparams else None
            xgb_model, xgb_pred = train_xgboost(train_x, train_y, test_x, seed, params=xgb_params)
            if xgb_model is not None and xgb_pred is not None:
                trained_models["xgboost"] = xgb_model
                results.append({"model": "XGBoost", **metrics(test_y, xgb_pred)})
        except Exception as e:
            print(f"Warning: XGBoost training failed: {e}")
    else:
        print("Warning: XGBoost not available")

    if CATBOOST_AVAILABLE:
        try:
            cb_params = optimize_catboost_params(train_x, train_y, cv_folds, seed, optuna_trials) if optimize_hyperparams else None
            cb_model, cb_pred = train_catboost(train_x, train_y, test_x, seed, params=cb_params)
            if cb_model is not None and cb_pred is not None:
                trained_models["catboost"] = cb_model
                results.append({"model": "CatBoost", **metrics(test_y, cb_pred)})
        except Exception as e:
            print(f"Warning: CatBoost training failed: {e}")
    else:
        print("Warning: CatBoost not available")

    if LIGHTGBM_AVAILABLE:
        quantile_preds = []
        q_params = optimize_lightgbm_params(train_x, train_y, cv_folds, seed, optuna_trials, objective="quantile") if optimize_hyperparams else None
        for q in [0.90, 0.95, 0.99]:
            try:
                q_model, q_pred = train_lightgbm_quantile(train_x, train_y, test_x, q, seed, params=q_params)
                if q_model is not None and q_pred is not None:
                    trained_models[f"lightgbm_quantile_p{int(q * 100)}"] = q_model
                    quantile_preds.append(q_pred)
            except Exception as e:
                print(f"Warning: LightGBM Quantile {q} training failed: {e}")
        if quantile_preds:
            lgb_q_pred = np.mean(quantile_preds, axis=0)
            results.append({"model": "LightGBM Quantile (p90/p95/p99)", **metrics(test_y, lgb_q_pred)})

    if XGBOOST_AVAILABLE:
        quantile_preds = []
        q_params = optimize_xgboost_params(train_x, train_y, cv_folds, seed, optuna_trials, objective="reg:quantileerror") if optimize_hyperparams else None
        for q in [0.90, 0.95, 0.99]:
            try:
                q_model, q_pred = train_xgboost_quantile(train_x, train_y, test_x, q, seed, params=q_params)
                if q_model is not None and q_pred is not None:
                    trained_models[f"xgboost_quantile_p{int(q * 100)}"] = q_model
                    quantile_preds.append(q_pred)
            except Exception as e:
                print(f"Warning: XGBoost Quantile {q} training failed: {e}")
        if quantile_preds:
            xgb_q_pred = np.mean(quantile_preds, axis=0)
            results.append({"model": "XGBoost Quantile (p90/p95/p99)", **metrics(test_y, xgb_q_pred)})

    if TABPFN_AVAILABLE:
        try:
            pfn_params = optimize_tabpfn_params(seed, optuna_trials) if optimize_hyperparams else None
            pfn_model, pfn_pred = train_tabpfn(train_x, train_y, test_x, seed, params=pfn_params)
            if pfn_model is not None and pfn_pred is not None:
                trained_models["tabpfn"] = pfn_model
                results.append({"model": "TabPFN", **metrics(test_y, pfn_pred)})
        except Exception as e:
            print(f"Warning: TabPFN training failed: {e}")
    else:
        print("Warning: TabPFN not available")

    # Save everything only after training/evaluation finishes.
    save_preprocessing(models_dir, feature_names, mean, scale)
    saved_model_paths = [save_model(model, models_dir, model_name) for model_name, model in sorted(trained_models.items())]

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
        result = train_and_plot(
            paths, job["target"], Path(job["output_dir"]), args.max_rows, args.test_fraction,
            args.seed, args.knn_k, args.knn_train_limit, args.cv_folds, args.optimize_hyperparams, args.optuna_trials,
        )
        row = {"label": job["label"], "target": job["target"], **result}
        rows.append(row)
        print(f"{job['label']} {job['target']}: files={result['source_files']} rows={result['rows_loaded']} used={result['rows_used']} best={result['best_model']} rmse={float(result['best_rmse']):.3f} r2={float(result['best_r2']):.3f} models_dir={result['models_dir']}")

    summary_path = args.analysis_dir / "training_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["label", "target", "source_files", "rows_loaded", "rows_used", "train_rows", "test_rows", "best_model", "best_mae", "best_rmse", "best_r2", "metrics_csv", "mae_plot", "rmse_plot", "r2_plot", "models_dir"]
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
    result = train_and_plot(
        paths, args.target, args.output_dir, args.max_rows, args.test_fraction,
        args.seed, args.knn_k, args.knn_train_limit, args.cv_folds, args.optimize_hyperparams, args.optuna_trials,
    )
    print(f"target: {args.target}")
    print(f"cv_folds: {args.cv_folds}")
    print(f"optimize_hyperparams: {args.optimize_hyperparams}")
    for key in ("source_files", "rows_loaded", "rows_used", "train_rows", "test_rows", "metrics_csv", "mae_plot", "rmse_plot", "r2_plot", "models_dir", "best_model", "best_rmse", "best_r2"):
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

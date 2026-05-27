#!/usr/bin/env python3
"""Compare regression baselines on CUDA timing CSVs without sklearn.

The tree/forest/boosting implementations are intentionally compact baselines.
They are useful for a first comparison against linear regression, not a
replacement for production-grade implementations from scikit-learn.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


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


@dataclass
class TreeNode:
    value: float
    feature: int = -1
    threshold: float = 0.0
    left: "TreeNode | None" = None
    right: "TreeNode | None" = None


class SimpleDecisionTreeRegressor:
    def __init__(
        self,
        max_depth: int = 8,
        min_samples_leaf: int = 80,
        max_features: int | None = None,
        n_thresholds: int = 16,
        rng: np.random.Generator | None = None,
    ):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.n_thresholds = n_thresholds
        self.rng = rng if rng is not None else np.random.default_rng(42)
        self.root: TreeNode | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "SimpleDecisionTreeRegressor":
        self.root = self._build(x, y, 0)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.root is None:
            raise RuntimeError("Tree is not fitted.")
        return np.asarray([self._predict_one(row, self.root) for row in x], dtype=float)

    def _predict_one(self, row: np.ndarray, node: TreeNode) -> float:
        while node.feature >= 0 and node.left is not None and node.right is not None:
            node = node.left if row[node.feature] <= node.threshold else node.right
        return node.value

    def _build(self, x: np.ndarray, y: np.ndarray, depth: int) -> TreeNode:
        value = float(np.mean(y))
        if depth >= self.max_depth or len(y) < 2 * self.min_samples_leaf:
            return TreeNode(value=value)

        split = self._best_split(x, y)
        if split is None:
            return TreeNode(value=value)

        feature, threshold, mask = split
        return TreeNode(
            value=value,
            feature=feature,
            threshold=threshold,
            left=self._build(x[mask], y[mask], depth + 1),
            right=self._build(x[~mask], y[~mask], depth + 1),
        )

    def _best_split(self, x: np.ndarray, y: np.ndarray):
        n_features = x.shape[1]
        if self.max_features is None or self.max_features >= n_features:
            feature_indices = np.arange(n_features)
        else:
            feature_indices = self.rng.choice(n_features, self.max_features, replace=False)

        best_score = math.inf
        best = None
        total_count = len(y)

        for feature in feature_indices:
            column = x[:, feature]
            if np.all(column == column[0]):
                continue

            thresholds = np.unique(np.quantile(column, np.linspace(0.05, 0.95, self.n_thresholds)))
            for threshold in thresholds:
                mask = column <= threshold
                left_count = int(np.sum(mask))
                right_count = total_count - left_count
                if left_count < self.min_samples_leaf or right_count < self.min_samples_leaf:
                    continue

                left_y = y[mask]
                right_y = y[~mask]
                score = float(np.var(left_y) * left_count + np.var(right_y) * right_count)
                if score < best_score:
                    best_score = score
                    best = (int(feature), float(threshold), mask)

        return best


class SimpleRandomForestRegressor:
    def __init__(self, n_estimators=12, max_depth=9, min_samples_leaf=80, sample_fraction=0.65, seed=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.sample_fraction = sample_fraction
        self.seed = seed
        self.trees: list[SimpleDecisionTreeRegressor] = []

    def fit(self, x: np.ndarray, y: np.ndarray) -> "SimpleRandomForestRegressor":
        rng = np.random.default_rng(self.seed)
        n = len(y)
        sample_size = max(2 * self.min_samples_leaf, int(n * self.sample_fraction))
        max_features = max(1, int(math.sqrt(x.shape[1])))
        self.trees = []
        for _ in range(self.n_estimators):
            idx = rng.integers(0, n, sample_size)
            tree = SimpleDecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                max_features=max_features,
                rng=rng,
            )
            tree.fit(x[idx], y[idx])
            self.trees.append(tree)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        predictions = np.column_stack([tree.predict(x) for tree in self.trees])
        return predictions.mean(axis=1)


class SimpleGradientBoostingRegressor:
    def __init__(self, n_estimators=24, learning_rate=0.08, max_depth=3, min_samples_leaf=120, seed=42):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.seed = seed
        self.base_value = 0.0
        self.trees: list[SimpleDecisionTreeRegressor] = []

    def fit(self, x: np.ndarray, y: np.ndarray) -> "SimpleGradientBoostingRegressor":
        rng = np.random.default_rng(self.seed)
        self.base_value = float(np.mean(y))
        prediction = np.full_like(y, self.base_value, dtype=float)
        self.trees = []
        for _ in range(self.n_estimators):
            residual = y - prediction
            tree = SimpleDecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                max_features=None,
                n_thresholds=12,
                rng=rng,
            )
            tree.fit(x, residual)
            update = tree.predict(x)
            prediction += self.learning_rate * update
            self.trees.append(tree)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        prediction = np.full(x.shape[0], self.base_value, dtype=float)
        for tree in self.trees:
            prediction += self.learning_rate * tree.predict(x)
        return prediction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, nargs="+", default=[DEFAULT_RESULTS_DIR])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--analysis-dir", type=Path, default=None)
    parser.add_argument("--jobs-file", type=Path, default=None)
    parser.add_argument("--target", choices=TARGETS, default="response_time_us")
    parser.add_argument("--first-sweep", action="store_true", help="Use first occurrence of each sweep config/rank.")
    parser.add_argument(
        "--include-regex",
        default="",
        help="Only use result CSV paths matching this regular expression.",
    )
    parser.add_argument("--max-rows", type=int, default=120000, help="Deterministic sample size for model comparison.")
    parser.add_argument("--test-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--knn-k", type=int, default=15)
    parser.add_argument("--knn-train-limit", type=int, default=12000)
    return parser.parse_args()


def result_paths(results_dirs: Path | list[Path], first_sweep: bool, include_regex: str = "") -> list[Path]:
    if isinstance(results_dirs, Path):
        results_dirs = [results_dirs]
    paths = sorted(
        path
        for results_dir in results_dirs
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


def to_float(row: dict[str, str], name: str, default: float = math.nan) -> float:
    value = row.get(name, "")
    if value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def row_features(row: dict[str, str]) -> dict[str, float]:
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
    target_gpu_demand_percent = (
        float(target_match.group(1).replace("p", ".")) if target_match is not None else 0.0
    )

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
        "workers_x_requested_busy_wait_us": to_float(
            row,
            "workers_x_requested_busy_wait_us",
            requested_us * effective_workers,
        ),
        "workers_x_blocks_per_sm": to_float(row, "workers_x_blocks_per_sm", blocks_per_sm * effective_workers),
        "workers_x_total_warps": to_float(row, "workers_x_total_warps", total_warps * effective_workers),
        "requested_busy_wait_us_per_arrival_ms": to_float(
            row,
            "requested_busy_wait_us_per_arrival_ms",
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


def load_matrix(paths: list[Path], target: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
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
                values = row_features(row)
                y = values[target] if target in values else to_float(row, target)
                x = [values[name] for name in features]
                if math.isfinite(y) and all(math.isfinite(value) for value in x):
                    x_rows.append(x)
                    y_values.append(y)
    return np.asarray(x_rows, dtype=float), np.asarray(y_values, dtype=float), features


def deterministic_sample(x: np.ndarray, y: np.ndarray, max_rows: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if max_rows <= 0 or len(y) <= max_rows:
        return x, y
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(y), max_rows, replace=False)
    return x[idx], y[idx]


def train_test_split(x: np.ndarray, y: np.ndarray, test_fraction: float, seed: int):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    rng.shuffle(idx)
    test_size = max(1, int(round(len(y) * test_fraction)))
    test_idx = idx[:test_size]
    train_idx = idx[test_size:]
    return x[train_idx], x[test_idx], y[train_idx], y[test_idx]


def standardize(train_x: np.ndarray, test_x: np.ndarray):
    mean = train_x.mean(axis=0)
    scale = train_x.std(axis=0)
    scale[scale == 0.0] = 1.0
    return (train_x - mean) / scale, (test_x - mean) / scale


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    residual = y_true - y_pred
    mae = float(np.mean(np.abs(residual)))
    rmse = math.sqrt(float(np.mean(residual * residual)))
    denom = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - float(np.sum(residual * residual)) / denom if denom > 0 else math.nan
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


def fit_linear(train_x: np.ndarray, train_y: np.ndarray):
    design = np.column_stack([np.ones(train_x.shape[0]), train_x])
    coef, *_ = np.linalg.lstsq(design, train_y, rcond=None)
    return coef


def fit_ridge(train_x: np.ndarray, train_y: np.ndarray, alpha: float = 1.0):
    design = np.column_stack([np.ones(train_x.shape[0]), train_x])
    penalty = np.eye(design.shape[1]) * alpha
    penalty[0, 0] = 0.0
    return np.linalg.solve(design.T @ design + penalty, design.T @ train_y)


def predict_linear(x: np.ndarray, coef: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(x.shape[0]), x]) @ coef


def quadratic_features(x: np.ndarray) -> np.ndarray:
    return np.column_stack([x, x * x])


def predict_knn(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, k: int, train_limit: int) -> np.ndarray:
    if len(train_y) > train_limit:
        train_x = train_x[:train_limit]
        train_y = train_y[:train_limit]
    k = max(1, min(k, len(train_y)))
    out = []
    for row in test_x:
        distances = np.sum((train_x - row) ** 2, axis=1)
        nearest = np.argpartition(distances, k - 1)[:k]
        out.append(float(np.mean(train_y[nearest])))
    return np.asarray(out, dtype=float)


def save_metrics_csv(output_dir: Path, rows: list[dict[str, float | str]]) -> Path:
    path = output_dir / "regression_metrics.csv"
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["model", "MAE", "RMSE", "R2"])
        writer.writeheader()
        writer.writerows(rows)
    return path


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
    paths: list[Path],
    target: str,
    output_dir: Path,
    max_rows: int,
    test_fraction: float,
    seed: int,
    knn_k: int,
    knn_train_limit: int,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if not paths:
        raise SystemExit("Nenhum CSV de resultados encontrado.")

    x, y, _ = load_matrix(paths, target)
    if len(y) == 0:
        raise SystemExit("Nenhuma linha valida apos filtrar CSVs.")

    original_rows = len(y)
    x, y = deterministic_sample(x, y, max_rows, seed)
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_fraction, seed)
    train_x_std, test_x_std = standardize(train_x, test_x)

    results: list[dict[str, float | str]] = []

    coef = fit_linear(train_x_std, train_y)
    results.append({"model": "Linear Regression", **metrics(test_y, predict_linear(test_x_std, coef))})

    coef = fit_ridge(train_x_std, train_y, alpha=1.0)
    results.append({"model": "Ridge Regression", **metrics(test_y, predict_linear(test_x_std, coef))})

    train_quad = quadratic_features(train_x_std)
    test_quad = quadratic_features(test_x_std)
    coef = fit_ridge(train_quad, train_y, alpha=1.0)
    results.append({"model": "Polynomial Ridge", **metrics(test_y, predict_linear(test_quad, coef))})

    tree = SimpleDecisionTreeRegressor(max_depth=10, min_samples_leaf=100, rng=np.random.default_rng(seed))
    tree.fit(train_x, train_y)
    results.append({"model": "Decision Tree", **metrics(test_y, tree.predict(test_x))})

    forest = SimpleRandomForestRegressor(n_estimators=12, max_depth=10, min_samples_leaf=100, seed=seed)
    forest.fit(train_x, train_y)
    results.append({"model": "Random Forest", **metrics(test_y, forest.predict(test_x))})

    boosting = SimpleGradientBoostingRegressor(n_estimators=24, learning_rate=0.08, max_depth=3, min_samples_leaf=120, seed=seed)
    boosting.fit(train_x, train_y)
    results.append({"model": "Gradient Boosting", **metrics(test_y, boosting.predict(test_x))})

    knn_pred = predict_knn(train_x_std, train_y, test_x_std, knn_k, knn_train_limit)
    results.append({"model": "kNN Regression", **metrics(test_y, knn_pred)})

    metrics_path = save_metrics_csv(output_dir, results)
    plot_paths = [plot_metric(output_dir, results, metric) for metric in ("MAE", "RMSE", "R2")]
    best = min(results, key=lambda row: float(row["RMSE"]))

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
    }


def load_jobs(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def run_jobs(args: argparse.Namespace, jobs_file: Path) -> int:
    if args.analysis_dir is None:
        raise SystemExit("--analysis-dir e obrigatorio quando --jobs-file e usado.")
    jobs = load_jobs(jobs_file)
    rows: list[dict[str, str]] = []
    for job in jobs:
        paths = result_paths(args.results_dir, args.first_sweep, job["include_regex"])
        result = train_and_plot(
            paths,
            job["target"],
            Path(job["output_dir"]),
            args.max_rows,
            args.test_fraction,
            args.seed,
            args.knn_k,
            args.knn_train_limit,
        )
        row = {"label": job["label"], "target": job["target"], **result}
        rows.append(row)
        print(
            f"{job['label']} {job['target']}: files={result['source_files']} "
            f"rows={result['rows_loaded']} used={result['rows_used']} "
            f"best={result['best_model']} rmse={float(result['best_rmse']):.3f} "
            f"r2={float(result['best_r2']):.3f}"
        )

    summary_path = args.analysis_dir / "training_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "label",
        "target",
        "source_files",
        "rows_loaded",
        "rows_used",
        "train_rows",
        "test_rows",
        "best_model",
        "best_mae",
        "best_rmse",
        "best_r2",
        "metrics_csv",
        "mae_plot",
        "rmse_plot",
        "r2_plot",
    ]
    with summary_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"training_summary: {summary_path}")
    return 0


def main() -> int:
    args = parse_args()
    if args.jobs_file is not None:
        return run_jobs(args, args.jobs_file)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = result_paths(args.results_dir, args.first_sweep, args.include_regex)
    result = train_and_plot(
        paths,
        args.target,
        args.output_dir,
        args.max_rows,
        args.test_fraction,
        args.seed,
        args.knn_k,
        args.knn_train_limit,
    )

    print(f"target: {args.target}")
    for key in (
        "source_files",
        "rows_loaded",
        "rows_used",
        "train_rows",
        "test_rows",
        "metrics_csv",
        "mae_plot",
        "rmse_plot",
        "r2_plot",
        "best_model",
        "best_rmse",
        "best_r2",
    ):
        print(f"{key}: {result[key]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

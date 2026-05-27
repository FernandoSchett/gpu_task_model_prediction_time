#!/usr/bin/env python3
"""Train regression models and generate plots from a sweep analysis manifest."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

import compare_regressors as cr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--max-rows", type=int, default=120000)
    parser.add_argument("--test-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--knn-k", type=int, default=15)
    parser.add_argument("--knn-train-limit", type=int, default=12000)
    return parser.parse_args()


def load_jobs(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def train_job(args: argparse.Namespace, job: dict[str, str]) -> dict[str, str]:
    output_dir = Path(job["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = cr.result_paths(args.results_dir, first_sweep=True, include_regex=job["include_regex"])
    if not paths:
        raise SystemExit(f"Nenhum CSV encontrado para job {job}")

    x, y, _ = cr.load_matrix(paths, job["target"])
    if len(y) == 0:
        raise SystemExit(f"Nenhuma linha valida para job {job}")

    original_rows = len(y)
    x, y = cr.deterministic_sample(x, y, args.max_rows, args.seed)
    train_x, test_x, train_y, test_y = cr.train_test_split(x, y, args.test_fraction, args.seed)
    train_x_std, test_x_std = cr.standardize(train_x, test_x)

    results: list[dict[str, float | str]] = []

    coef = cr.fit_linear(train_x_std, train_y)
    results.append({"model": "Linear Regression", **cr.metrics(test_y, cr.predict_linear(test_x_std, coef))})

    coef = cr.fit_ridge(train_x_std, train_y, alpha=1.0)
    results.append({"model": "Ridge Regression", **cr.metrics(test_y, cr.predict_linear(test_x_std, coef))})

    train_quad = cr.quadratic_features(train_x_std)
    test_quad = cr.quadratic_features(test_x_std)
    coef = cr.fit_ridge(train_quad, train_y, alpha=1.0)
    results.append({"model": "Polynomial Ridge", **cr.metrics(test_y, cr.predict_linear(test_quad, coef))})

    tree = cr.SimpleDecisionTreeRegressor(max_depth=10, min_samples_leaf=100, rng=np.random.default_rng(args.seed))
    tree.fit(train_x, train_y)
    results.append({"model": "Decision Tree", **cr.metrics(test_y, tree.predict(test_x))})

    forest = cr.SimpleRandomForestRegressor(n_estimators=12, max_depth=10, min_samples_leaf=100, seed=args.seed)
    forest.fit(train_x, train_y)
    results.append({"model": "Random Forest", **cr.metrics(test_y, forest.predict(test_x))})

    boosting = cr.SimpleGradientBoostingRegressor(
        n_estimators=24,
        learning_rate=0.08,
        max_depth=3,
        min_samples_leaf=120,
        seed=args.seed,
    )
    boosting.fit(train_x, train_y)
    results.append({"model": "Gradient Boosting", **cr.metrics(test_y, boosting.predict(test_x))})

    knn_pred = cr.predict_knn(train_x_std, train_y, test_x_std, args.knn_k, args.knn_train_limit)
    results.append({"model": "kNN Regression", **cr.metrics(test_y, knn_pred)})

    metrics_path = cr.save_metrics_csv(output_dir, results)
    plot_paths = [cr.plot_metric(output_dir, results, metric) for metric in ("MAE", "RMSE", "R2")]

    best = min(results, key=lambda row: float(row["RMSE"]))
    print(
        f"{job['label']} {job['target']}: files={len(paths)} rows={original_rows} "
        f"used={len(y)} best={best['model']} rmse={float(best['RMSE']):.3f} r2={float(best['R2']):.3f}"
    )
    return {
        "label": job["label"],
        "target": job["target"],
        "source_files": str(len(paths)),
        "rows_loaded": str(original_rows),
        "rows_used": str(len(y)),
        "best_model": str(best["model"]),
        "best_mae": f"{float(best['MAE']):.6f}",
        "best_rmse": f"{float(best['RMSE']):.6f}",
        "best_r2": f"{float(best['R2']):.6f}",
        "metrics_csv": str(metrics_path),
        "mae_plot": str(plot_paths[0]),
        "rmse_plot": str(plot_paths[1]),
        "r2_plot": str(plot_paths[2]),
    }


def main() -> int:
    args = parse_args()
    jobs_path = args.analysis_dir / "analysis_jobs.csv"
    if not jobs_path.exists():
        raise SystemExit(f"Manifesto nao encontrado: {jobs_path}")

    rows = [train_job(args, job) for job in load_jobs(jobs_path)]
    summary_path = args.analysis_dir / "training_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as file:
        fieldnames = [
            "label",
            "target",
            "source_files",
            "rows_loaded",
            "rows_used",
            "best_model",
            "best_mae",
            "best_rmse",
            "best_r2",
            "metrics_csv",
            "mae_plot",
            "rmse_plot",
            "r2_plot",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"training_summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

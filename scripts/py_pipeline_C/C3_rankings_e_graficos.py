#!/usr/bin/env python3
"""Create rankings and plots for Pipeline C CNN2D results."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-root", type=Path, required=True)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def collect_rows(analysis_root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for summary in sorted(analysis_root.glob("*_sweep_moderado_sem_estimativas_agrupado/pipeline_c_cnn2d_training_summary.csv")):
        condition = summary.parent.name.replace("_sweep_moderado_sem_estimativas_agrupado", "")
        for row in read_rows(summary):
            row = dict(row)
            row["condition"] = condition
            rows.append(row)
    return rows


def best_per_cut(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault((row["condition"], row["label"], row["target"]), []).append(row)
    best_rows = []
    for (_condition, _label, _target), group in grouped.items():
        best = sorted(group, key=lambda row: (-float(row["R2"]), float(row["RMSE"])))[0]
        best_rows.append(best)
    best_rows.sort(key=lambda row: (-float(row["R2"]), float(row["RMSE"])))
    return best_rows


def architecture_summary(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["architecture"], []).append(row)
    out = []
    for arch, group in grouped.items():
        mean_mae = sum(float(row["MAE"]) for row in group) / len(group)
        mean_rmse = sum(float(row["RMSE"]) for row in group) / len(group)
        mean_r2 = sum(float(row["R2"]) for row in group) / len(group)
        out.append({
            "architecture": arch,
            "runs": str(len(group)),
            "mean_MAE": f"{mean_mae:.6f}",
            "mean_RMSE": f"{mean_rmse:.6f}",
            "mean_R2": f"{mean_r2:.6f}",
        })
    out.sort(key=lambda row: (-float(row["mean_R2"]), float(row["mean_RMSE"])))
    return out


def plot_top(path: Path, rows: list[dict[str, str]], metric: str) -> None:
    top = rows[:20]
    if not top:
        return
    labels = [f"{row['condition']}:{row['label']}:{row['target']}\n{row['architecture']}" for row in top]
    values = [float(row[metric]) for row in top]
    fig, ax = plt.subplots(figsize=(11, max(4, len(top) * 0.35)))
    order = range(len(top))
    ax.barh(list(order), values, color="#4c78a8")
    ax.set_yticks(list(order))
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel(metric)
    ax.set_title(f"Pipeline C CNN2D top por {metric}")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_architecture_summary(path: Path, rows: list[dict[str, str]]) -> None:
    top = rows[:20]
    if not top:
        return
    labels = [row["architecture"] for row in top]
    values = [float(row["mean_R2"]) for row in top]
    fig, ax = plt.subplots(figsize=(10, max(4, len(top) * 0.32)))
    ax.barh(range(len(top)), values, color="#59a14f")
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("R2 medio")
    ax.set_title("Pipeline C CNN2D arquiteturas - R2 medio")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    output_dir = args.analysis_root / "2d_models"
    rows = collect_rows(args.analysis_root)
    if not rows:
        raise SystemExit(f"Nenhum resultado da Pipeline C encontrado em {args.analysis_root}")

    fieldnames = [
        "condition", "label", "target", "architecture", "MAE", "RMSE", "R2",
        "train_samples", "test_samples", "workers", "window_size", "features",
        "model_path", "plots_dir", "cached",
    ]
    all_path = output_dir / "cnn2d_all_architecture_metrics.csv"
    write_csv(all_path, rows, fieldnames)

    best_rows = best_per_cut(rows)
    best_path = output_dir / "best_cnn2d_rankings.csv"
    write_csv(best_path, best_rows, fieldnames)

    arch_rows = architecture_summary(rows)
    arch_path = output_dir / "cnn2d_architecture_rankings.csv"
    write_csv(arch_path, arch_rows, ["architecture", "runs", "mean_MAE", "mean_RMSE", "mean_R2"])

    plot_top(output_dir / "best_cnn2d_top_r2.png", best_rows, "R2")
    plot_top(output_dir / "best_cnn2d_top_rmse.png", sorted(best_rows, key=lambda row: float(row["RMSE"])), "RMSE")
    plot_architecture_summary(output_dir / "cnn2d_architecture_top_mean_r2.png", arch_rows)

    print(f"pipeline_c_all_metrics: {all_path}")
    print(f"pipeline_c_best_rankings: {best_path}")
    print(f"pipeline_c_architecture_rankings: {arch_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

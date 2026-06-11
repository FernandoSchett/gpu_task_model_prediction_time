#!/usr/bin/env python3
"""Plot rankings of the best ML model found in each analysis subfolder."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_ANALYSIS_ROOT = REPO_ROOT / "resultados" / "analises_regressao"
TARGETS = ("response_time_us", "queueing_delay_us", "slowdown")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-root", type=Path, default=DEFAULT_ANALYSIS_ROOT)
    parser.add_argument("--top-n", type=int, default=20)
    return parser.parse_args()


def condition_from_dir(path: Path) -> str:
    name = path.name
    if name.startswith("sem_telemetria"):
        return "sem_telemetria"
    if name.startswith("com_telemetria"):
        return "com_telemetria"
    return name


def better_row(candidate: dict[str, str], current: dict[str, str] | None) -> bool:
    if current is None:
        return True
    candidate_r2 = float(candidate.get("best_r2", "nan"))
    current_r2 = float(current.get("best_r2", "nan"))
    if candidate_r2 != current_r2:
        return candidate_r2 > current_r2
    return float(candidate.get("best_rmse", "inf")) < float(current.get("best_rmse", "inf"))


def load_classical_rows(analysis_root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for summary_path in sorted(analysis_root.glob("*_sweep_moderado_sem_estimativas_agrupado/training_summary.csv")):
        condition = condition_from_dir(summary_path.parent)
        with summary_path.open("r", encoding="utf-8", newline="") as file:
            for row in csv.DictReader(file):
                row = dict(row)
                row["condition"] = condition
                row["model_family"] = "classical"
                rows.append(row)
    return rows


def load_sequential_rows(analysis_root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    pattern = "*_sweep_moderado_sem_estimativas_agrupado/*/*/sequential_models/sequential_metrics.csv"
    for metrics_path in sorted(analysis_root.glob(pattern)):
        target_dir = metrics_path.parent.parent
        label_dir = target_dir.parent
        condition_dir = label_dir.parent
        with metrics_path.open("r", encoding="utf-8", newline="") as file:
            metrics_rows = list(csv.DictReader(file))
        if not metrics_rows:
            continue
        best = max(metrics_rows, key=lambda row: (float(row["R2"]), -float(row["RMSE"])))
        train_sequences = best.get("train_sequences", "")
        test_sequences = best.get("test_sequences", "")
        rows.append({
            "condition": condition_from_dir(condition_dir),
            "label": label_dir.name,
            "target": target_dir.name,
            "best_model": best["model"],
            "best_mae": best["MAE"],
            "best_rmse": best["RMSE"],
            "best_r2": best["R2"],
            "rows_used": str(int(train_sequences or 0) + int(test_sequences or 0)),
            "metrics_csv": str(metrics_path),
            "model_family": "sequential",
        })
    return rows


def load_rows(analysis_root: Path) -> list[dict[str, str]]:
    candidates = load_classical_rows(analysis_root) + load_sequential_rows(analysis_root)
    if not candidates:
        raise SystemExit(f"Nenhuma metrica de ML encontrada em {analysis_root}")

    best_by_slice: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in candidates:
        key = (row["condition"], row["label"], row["target"])
        if better_row(row, best_by_slice.get(key)):
            best_by_slice[key] = row
    return list(best_by_slice.values())


def write_rankings_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fields = [
        "condition",
        "label",
        "target",
        "model_family",
        "best_model",
        "best_mae",
        "best_rmse",
        "best_r2",
        "rows_used",
        "metrics_csv",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in sorted(rows, key=lambda item: (item["target"], -float(item["best_r2"]), float(item["best_rmse"]))):
            writer.writerow({field: row.get(field, "") for field in fields})


def short_label(row: dict[str, str]) -> str:
    condition = "sem" if row["condition"] == "sem_telemetria" else "com"
    return f"{condition}:{row['label']}"


def plot_target_rankings(output_dir: Path, rows: list[dict[str, str]], top_n: int) -> None:
    for target in TARGETS:
        target_rows = [row for row in rows if row["target"] == target]
        target_rows.sort(key=lambda row: (float(row["best_r2"]), -float(row["best_rmse"])), reverse=True)
        target_rows = target_rows[:top_n]
        labels = [short_label(row) for row in target_rows]
        values = [float(row["best_r2"]) for row in target_rows]
        colors = ["#4c78a8" if row["condition"] == "sem_telemetria" else "#f58518" for row in target_rows]

        fig, ax = plt.subplots(figsize=(13, max(6, len(target_rows) * 0.36)))
        y = np.arange(len(target_rows))
        ax.barh(y, values, color=colors)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("R2 do melhor modelo")
        ax.set_title(f"Top {len(target_rows)} recortes onde os modelos predizem melhor - {target}")
        ax.axvline(0.0, color="black", linewidth=0.8)
        for index, row in enumerate(target_rows):
            value = float(row["best_r2"])
            model_suffix = "" if row.get("model_family") == "classical" else " (seq)"
            text = f" {value:.3f} | {row['best_model']}{model_suffix} | RMSE {float(row['best_rmse']):.2f}"
            ax.text(
                value,
                index,
                text,
                va="center",
                fontsize=8,
            )
        fig.tight_layout()
        fig.savefig(output_dir / f"best_model_top_{target}.png", dpi=160)
        plt.close(fig)


def plot_condition_overview(output_dir: Path, rows: list[dict[str, str]]) -> None:
    grouped: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        grouped.setdefault((row["condition"], row["target"]), []).append(float(row["best_r2"]))

    labels = []
    values = []
    colors = []
    for target in TARGETS:
        for condition in ("sem_telemetria", "com_telemetria"):
            labels.append(f"{condition}\n{target}")
            values.append(float(np.mean(grouped.get((condition, target), [np.nan]))))
            colors.append("#4c78a8" if condition == "sem_telemetria" else "#f58518")

    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(labels))
    ax.bar(x, values, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("R2 medio do melhor modelo por recorte")
    ax.set_title("Comparacao geral de previsibilidade por condicao")
    ax.axhline(0.0, color="black", linewidth=0.8)
    for index, value in enumerate(values):
        ax.text(index, value, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "best_model_condition_overview.png", dpi=160)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    rows = load_rows(args.analysis_root)
    output_dir = args.analysis_root
    write_rankings_csv(output_dir / "best_model_rankings.csv", rows)
    plot_target_rankings(output_dir, rows, args.top_n)
    plot_condition_overview(output_dir, rows)
    print(f"rankings_csv: {output_dir / 'best_model_rankings.csv'}")
    print(f"plots_dir: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

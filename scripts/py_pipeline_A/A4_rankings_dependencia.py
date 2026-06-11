#!/usr/bin/env python3
"""Plot rankings of analysis cases with strongest temporal dependence."""

from __future__ import annotations

import argparse
import csv
import math
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
    if path.name.startswith("sem_telemetria"):
        return "sem_telemetria"
    if path.name.startswith("com_telemetria"):
        return "com_telemetria"
    return path.name


def finite_float(value: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def load_rows(analysis_root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for summary_path in sorted(analysis_root.glob("*_sweep_moderado_sem_estimativas_agrupado/dependency_summary.csv")):
        condition = condition_from_dir(summary_path.parent)
        with summary_path.open("r", encoding="utf-8", newline="") as file:
            for row in csv.DictReader(file):
                row = dict(row)
                row["condition"] = condition
                rows.append(row)
    if rows:
        return rows

    for metrics_path in sorted(analysis_root.glob("*_sweep_moderado_sem_estimativas_agrupado/*/*/dependency_metrics.csv")):
        condition = condition_from_dir(metrics_path.parents[2])
        label = metrics_path.parents[1].name
        target = metrics_path.parent.name
        row = {
            "condition": condition,
            "label": label,
            "target": target,
            "rows_used": "",
            "dependency_metrics_csv": str(metrics_path),
            "dependency_lag1_autocorrelation": "",
            "dependency_abs_lag1_autocorrelation": "",
            "dependency_durbin_watson": "",
            "dependency_effective_sample_ratio_lag1": "",
        }
        with metrics_path.open("r", encoding="utf-8", newline="") as file:
            for metric_row in csv.DictReader(file):
                if metric_row.get("category") != "temporal":
                    continue
                metric = metric_row.get("metric", "")
                value = metric_row.get("value", "")
                if metric == "lag1_autocorrelation":
                    row["dependency_lag1_autocorrelation"] = value
                    row["rows_used"] = metric_row.get("n", "")
                elif metric == "abs_lag1_autocorrelation":
                    row["dependency_abs_lag1_autocorrelation"] = value
                elif metric == "durbin_watson":
                    row["dependency_durbin_watson"] = value
                elif metric == "effective_sample_size_ratio_lag1":
                    row["dependency_effective_sample_ratio_lag1"] = value
        rows.append(row)
    if not rows:
        raise SystemExit(
            "Nenhum dependency_summary.csv ou dependency_metrics.csv encontrado. Rode primeiro com DEPENDENCY_ONLY=true."
        )
    return rows


def write_rankings_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fields = [
        "condition",
        "label",
        "target",
        "dependency_lag1_autocorrelation",
        "dependency_abs_lag1_autocorrelation",
        "dependency_durbin_watson",
        "dependency_effective_sample_ratio_lag1",
        "rows_used",
        "dependency_metrics_csv",
    ]
    ranked = sorted(
        rows,
        key=lambda row: finite_float(row.get("dependency_abs_lag1_autocorrelation", "")),
        reverse=True,
    )
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in ranked:
            writer.writerow({field: row.get(field, "") for field in fields})


def short_label(row: dict[str, str]) -> str:
    condition = "sem" if row["condition"] == "sem_telemetria" else "com"
    return f"{condition}:{row['label']}"


def plot_rankings(output_dir: Path, rows: list[dict[str, str]], top_n: int) -> None:
    for target in TARGETS:
        target_rows = [
            row for row in rows
            if row.get("target") == target and math.isfinite(finite_float(row.get("dependency_abs_lag1_autocorrelation", "")))
        ]
        target_rows.sort(key=lambda row: finite_float(row["dependency_abs_lag1_autocorrelation"]), reverse=True)
        target_rows = target_rows[:top_n]
        if not target_rows:
            continue
        labels = [short_label(row) for row in target_rows]
        values = [finite_float(row["dependency_abs_lag1_autocorrelation"]) for row in target_rows]
        colors = ["#4c78a8" if row["condition"] == "sem_telemetria" else "#f58518" for row in target_rows]

        fig, ax = plt.subplots(figsize=(13, max(6, len(target_rows) * 0.36)))
        y = np.arange(len(target_rows))
        ax.barh(y, values, color=colors)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlim(0.0, 1.0)
        ax.set_xlabel("|autocorrelacao lag-1| do alvo")
        ax.set_title(f"Top {len(target_rows)} recortes com maior dependencia temporal - {target}")
        for index, row in enumerate(target_rows):
            acf = finite_float(row["dependency_lag1_autocorrelation"])
            dw = finite_float(row.get("dependency_durbin_watson", ""))
            ess = finite_float(row.get("dependency_effective_sample_ratio_lag1", ""))
            ax.text(
                values[index],
                index,
                f" acf {acf:.3f} | DW {dw:.2f} | ESS {ess:.2f}",
                va="center",
                fontsize=8,
            )
        fig.tight_layout()
        fig.savefig(output_dir / f"dependency_top_{target}.png", dpi=160)
        plt.close(fig)


def plot_overview(output_dir: Path, rows: list[dict[str, str]]) -> None:
    grouped: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        value = finite_float(row.get("dependency_abs_lag1_autocorrelation", ""))
        if math.isfinite(value):
            grouped.setdefault((row["condition"], row["target"]), []).append(value)
    labels = []
    values = []
    colors = []
    for target in TARGETS:
        for condition in ("sem_telemetria", "com_telemetria"):
            group_values = grouped.get((condition, target), [])
            if not group_values:
                continue
            labels.append(f"{condition}\n{target}")
            values.append(float(np.mean(group_values)))
            colors.append("#4c78a8" if condition == "sem_telemetria" else "#f58518")
    if not values:
        return
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(labels))
    ax.bar(x, values, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("|autocorrelacao lag-1| media")
    ax.set_title("Dependencia temporal media por condicao")
    for index, value in enumerate(values):
        ax.text(index, value, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "dependency_condition_overview.png", dpi=160)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    rows = load_rows(args.analysis_root)
    write_rankings_csv(args.analysis_root / "dependency_rankings.csv", rows)
    plot_rankings(args.analysis_root, rows, args.top_n)
    plot_overview(args.analysis_root, rows)
    print(f"dependency_rankings_csv: {args.analysis_root / 'dependency_rankings.csv'}")
    print(f"plots_dir: {args.analysis_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

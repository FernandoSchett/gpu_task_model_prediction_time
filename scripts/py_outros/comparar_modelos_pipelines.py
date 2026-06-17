#!/usr/bin/env python3
"""Compare best models from Pipeline A and Pipeline C."""

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


def normalize_row(row: dict[str, str], condition: str, model_family: str, model_key: str = "best_model") -> dict[str, str]:
    out = dict(row)
    out["condition"] = condition
    out["model_family"] = model_family
    if model_key != "best_model":
        out["best_model"] = row.get(model_key, "")
    return out


def load_classical_rows(analysis_root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for summary_path in sorted((analysis_root / "pipeline_A").glob("*/training_summary.csv")):
        condition = condition_from_dir(summary_path.parent)
        with summary_path.open("r", encoding="utf-8", newline="") as file:
            for row in csv.DictReader(file):
                row = normalize_row(row, condition, "classical")
                metrics_path = (
                    analysis_root / "pipeline_A" / condition / row["label"] / row["target"]
                    / "nao_sequenciais" / "regression_metrics.csv"
                )
                if metrics_path.exists():
                    row["metrics_csv"] = str(metrics_path)
                rows.append(row)
    return rows


def load_sequential_rows(analysis_root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    pattern = "*/*/*/sequenciais/sequential_metrics.csv"
    for metrics_path in sorted((analysis_root / "pipeline_A").glob(pattern)):
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


def load_cnn2d_rows(analysis_root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    pattern = "*/*/*/2d_models/cnn2d_architecture_metrics.csv"
    for metrics_path in sorted((analysis_root / "pipeline_C").glob(pattern)):
        target_dir = metrics_path.parent.parent
        label_dir = target_dir.parent
        condition_dir = label_dir.parent
        with metrics_path.open("r", encoding="utf-8", newline="") as file:
            metrics_rows = list(csv.DictReader(file))
        if not metrics_rows:
            continue
        best = max(metrics_rows, key=lambda row: (float(row["R2"]), -float(row["RMSE"])))
        rows.append({
            "condition": condition_from_dir(condition_dir),
            "label": label_dir.name,
            "target": target_dir.name,
            "best_model": best["architecture"],
            "best_mae": best["MAE"],
            "best_rmse": best["RMSE"],
            "best_r2": best["R2"],
            "rows_used": str(int(best.get("train_samples", 0) or 0) + int(best.get("test_samples", 0) or 0)),
            "metrics_csv": str(metrics_path),
            "model_family": "cnn2d",
        })
    return rows


def best_by_slice(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    best: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in rows:
        key = (row["condition"], row["label"], row["target"])
        if better_row(row, best.get(key)):
            best[key] = row
    return list(best.values())


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
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in sorted(rows, key=lambda item: (item["target"], -float(item["best_r2"]), float(item["best_rmse"]))):
            writer.writerow({field: row.get(field, "") for field in fields})


def short_label(row: dict[str, str]) -> str:
    condition = "sem" if row["condition"] == "sem_telemetria" else "com"
    family = row.get("model_family", "")
    suffix = "" if family == "classical" else f" ({family})"
    return f"{condition}:{row['label']}{suffix}"


def family_title(model_family: str | None) -> str:
    titles = {
        "classical": "modelos nao sequenciais",
        "sequential": "modelos sequenciais",
        "cnn2d": "modelos CNN 2D",
    }
    return titles.get(model_family or "", "todos os modelos")


def plot_target_rankings(
    output_dir: Path,
    rows: list[dict[str, str]],
    top_n: int,
    model_family: str | None = None,
    file_prefix: str = "best_model_top_",
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
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
        ax.set_title(f"Top {len(target_rows)} recortes onde {family_title(model_family)} predizem melhor - {target}")
        ax.axvline(0.0, color="black", linewidth=0.8)
        for index, row in enumerate(target_rows):
            value = float(row["best_r2"])
            text = f" {value:.3f} | {row['best_model']} | RMSE {float(row['best_rmse']):.2f}"
            ax.text(value, index, text, va="center", fontsize=8)
        fig.tight_layout()
        fig.savefig(output_dir / f"{file_prefix}{target}.png", dpi=160)
        plt.close(fig)


def plot_condition_overview(
    output_dir: Path,
    rows: list[dict[str, str]],
    model_family: str | None = None,
    file_name: str = "best_model_condition_overview.png",
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    grouped: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        grouped.setdefault((row["condition"], row["target"]), []).append(float(row["best_r2"]))

    labels = []
    values = []
    colors = []
    for target in TARGETS:
        for condition in ("sem_telemetria", "com_telemetria"):
            group = grouped.get((condition, target), [])
            if not group:
                continue
            labels.append(f"{condition}\n{target}")
            values.append(float(np.mean(group)))
            colors.append("#4c78a8" if condition == "sem_telemetria" else "#f58518")
    if not values:
        return

    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(labels))
    ax.bar(x, values, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("R2 medio do melhor modelo por recorte")
    ax.set_title(f"Comparacao geral de previsibilidade por condicao - {family_title(model_family)}")
    ax.axhline(0.0, color="black", linewidth=0.8)
    for index, value in enumerate(values):
        ax.text(index, value, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / file_name, dpi=160)
    plt.close(fig)


def write_outputs(
    output_dir: Path,
    rows: list[dict[str, str]],
    top_n: int,
    model_family: str | None = None,
    csv_name: str = "best_model_rankings.csv",
    plot_prefix: str = "best_model_top_",
    overview_name: str = "best_model_condition_overview.png",
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_rankings_csv(output_dir / csv_name, rows)
    plot_target_rankings(output_dir, rows, top_n, model_family, plot_prefix)
    plot_condition_overview(output_dir, rows, model_family, overview_name)


def main() -> int:
    args = parse_args()
    classical_rows = load_classical_rows(args.analysis_root)
    sequential_rows = load_sequential_rows(args.analysis_root)
    cnn2d_rows = load_cnn2d_rows(args.analysis_root)
    pipeline_a_rows = best_by_slice(classical_rows + sequential_rows)
    all_rows = best_by_slice(classical_rows + sequential_rows + cnn2d_rows)

    if not all_rows:
        raise SystemExit(f"Nenhuma metrica de modelos encontrada em {args.analysis_root}")

    pipeline_a_rankings = args.analysis_root / "pipeline_A" / "rankings"
    pipeline_c_rankings = args.analysis_root / "pipeline_C" / "rankings"
    global_rankings = args.analysis_root / "comparacoes_pipelines"
    write_outputs(
        pipeline_a_rankings,
        classical_rows,
        args.top_n,
        "classical",
        "melhores_modelos_nao_sequenciais.csv",
        "top_nao_sequenciais_",
        "overview_nao_sequenciais.png",
    )
    write_outputs(
        pipeline_a_rankings,
        sequential_rows,
        args.top_n,
        "sequential",
        "melhores_modelos_sequenciais.csv",
        "top_sequenciais_",
        "overview_sequenciais.png",
    )
    write_outputs(
        pipeline_a_rankings,
        pipeline_a_rows,
        args.top_n,
        None,
        "melhores_modelos_pipeline_A.csv",
        "top_pipeline_A_",
        "overview_pipeline_A.png",
    )
    write_outputs(
        pipeline_c_rankings,
        cnn2d_rows,
        args.top_n,
        "cnn2d",
        "melhores_modelos_2d.csv",
        "top_cnn2d_",
        "overview_cnn2d.png",
    )
    write_outputs(global_rankings, all_rows, args.top_n, None)
    print(f"rankings_csv: {global_rankings / 'best_model_rankings.csv'}")
    print(f"pipeline_a_rankings_dir: {pipeline_a_rankings}")
    print(f"cnn2d_rankings_csv: {pipeline_c_rankings / 'melhores_modelos_2d.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

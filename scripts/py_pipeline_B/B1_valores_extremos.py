#!/usr/bin/env python3
"""Extreme-value pipeline with declustering plus GEV/GPD diagnostics."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REGRESSOR_SCRIPT = SCRIPT_DIR.parent / "py_pipeline_A" / "A2_regressores_classicos.py"


def load_regressor_module():
    spec = importlib.util.spec_from_file_location("regressor_analysis", REGRESSOR_SCRIPT)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Nao foi possivel carregar {REGRESSOR_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


REG = load_regressor_module()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, nargs="+", required=True)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--jobs-file", type=Path, required=True)
    parser.add_argument("--first-sweep", action="store_true")
    parser.add_argument("--block-size", type=int, default=int(os.getenv("EVT_BLOCK_SIZE", "1024")))
    parser.add_argument("--threshold-quantile", type=float, default=float(os.getenv("EVT_THRESHOLD_QUANTILE", "0.95")))
    parser.add_argument("--decluster-run-length", type=int, default=int(os.getenv("EVT_DECLUSTER_RUN_LENGTH", "50")))
    parser.add_argument("--return-quantiles", nargs="+", type=float, default=[0.95, 0.99, 0.999])
    parser.add_argument("--no-cache", action="store_true", help="Recompute even when EVT outputs already exist.")
    parser.add_argument("--only-model", choices=("gev", "gpd", "gumbel"), default=os.getenv("EVT_MODEL_ONLY", ""))
    parser.add_argument("--force-model", action="store_true", default=os.getenv("EVT_FORCE_MODEL", "").lower() in {"1", "true", "yes", "on", "sim"})
    return parser.parse_args()


def import_scipy_stats():
    try:
        from scipy import stats
    except ImportError as exc:
        raise SystemExit(
            "SciPy nao esta instalado. Instale com `pip install scipy` "
            "para rodar a pipeline B de GEV/GPD."
        ) from exc
    return stats


def load_jobs(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def load_target_series(paths: list[Path], target: str) -> np.ndarray:
    values: list[float] = []
    for path in paths:
        with path.open("r", encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            if reader.fieldnames is None or "response_time_us" not in reader.fieldnames:
                continue
            for row in reader:
                if REG.to_float(row, "cuda_error_code", 0.0) != 0.0:
                    continue
                features = REG.row_features_compare(row)
                value = features[target] if target in features else REG.to_float(row, target)
                if math.isfinite(value):
                    values.append(value)
    return np.asarray(values, dtype=float)


def block_maxima(values: np.ndarray, block_size: int) -> np.ndarray:
    if block_size < 2:
        raise SystemExit("--block-size precisa ser >= 2.")
    n_blocks = len(values) // block_size
    if n_blocks <= 1:
        return np.empty((0,), dtype=float)
    trimmed = values[: n_blocks * block_size]
    return np.max(trimmed.reshape(n_blocks, block_size), axis=1)


def decluster_exceedances(values: np.ndarray, threshold: float, run_length: int) -> np.ndarray:
    if run_length < 1:
        raise SystemExit("--decluster-run-length precisa ser >= 1.")
    exceedance_indices = np.flatnonzero(values > threshold)
    if len(exceedance_indices) == 0:
        return np.empty((0,), dtype=float)
    cluster_peaks: list[float] = []
    current_peak = float(values[exceedance_indices[0]])
    previous_index = int(exceedance_indices[0])
    for index in exceedance_indices[1:]:
        index = int(index)
        value = float(values[index])
        if index - previous_index <= run_length:
            current_peak = max(current_peak, value)
        else:
            cluster_peaks.append(current_peak)
            current_peak = value
        previous_index = index
    cluster_peaks.append(current_peak)
    return np.asarray(cluster_peaks, dtype=float)


def finite_or_empty(value: float) -> str:
    return f"{value:.12g}" if math.isfinite(value) else ""


def plot_hist_fit(output_dir: Path, name: str, data: np.ndarray, distribution, params: tuple[float, ...], title: str) -> Path:
    path = output_dir / f"{name}_hist_fit.png"
    fig, ax = plt.subplots(figsize=(7, 4))
    counts, bins, _patches = ax.hist(data, bins=40, density=True, color="#4c78a8", alpha=0.75)
    xs = np.linspace(float(np.min(data)), float(np.max(data)), 300)
    ax.plot(xs, distribution.pdf(xs, *params), color="#f58518", linewidth=1.8, label="ajuste")
    ax.set_title(title)
    ax.set_xlabel("Valor")
    ax.set_ylabel("Densidade")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_qq(output_dir: Path, name: str, data: np.ndarray, distribution, params: tuple[float, ...], title: str) -> Path:
    path = output_dir / f"{name}_qq.png"
    sorted_data = np.sort(data)
    probabilities = (np.arange(1, len(sorted_data) + 1) - 0.5) / len(sorted_data)
    theoretical = distribution.ppf(probabilities, *params)
    min_value = float(min(np.min(sorted_data), np.min(theoretical)))
    max_value = float(max(np.max(sorted_data), np.max(theoretical)))
    fig, ax = plt.subplots(figsize=(5.8, 5.8))
    ax.scatter(theoretical, sorted_data, s=14, alpha=0.65, color="#4c78a8")
    ax.plot([min_value, max_value], [min_value, max_value], color="black", linewidth=1.0)
    ax.set_title(title)
    ax.set_xlabel("Quantis teoricos")
    ax.set_ylabel("Quantis observados")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_pp(output_dir: Path, name: str, data: np.ndarray, distribution, params: tuple[float, ...], title: str) -> Path:
    path = output_dir / f"{name}_pp.png"
    sorted_data = np.sort(data)
    empirical = (np.arange(1, len(sorted_data) + 1) - 0.5) / len(sorted_data)
    fitted = distribution.cdf(sorted_data, *params)
    fig, ax = plt.subplots(figsize=(5.8, 5.8))
    ax.scatter(fitted, empirical, s=14, alpha=0.65, color="#2f7f7f")
    ax.plot([0.0, 1.0], [0.0, 1.0], color="black", linewidth=1.0)
    ax.set_title(title)
    ax.set_xlabel("Probabilidade ajustada")
    ax.set_ylabel("Probabilidade empirica")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_return_levels(output_dir: Path, name: str, quantile_rows: list[dict[str, str]], title: str) -> Path:
    path = output_dir / f"{name}_return_levels.png"
    probs = [float(row["quantile"]) for row in quantile_rows]
    values = [float(row["predicted_value"]) for row in quantile_rows]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(probs, values, marker="o", color="#4c78a8")
    ax.set_title(title)
    ax.set_xlabel("Quantil")
    ax.set_ylabel("Valor estimado")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def expected_metadata(args: argparse.Namespace, job: dict[str, str], paths: list[Path]) -> dict[str, object]:
    return {
        "label": job["label"],
        "target": job["target"],
        "source_files": len(paths),
        "source_paths": [str(path) for path in paths],
        "block_size": args.block_size,
        "threshold_quantile": args.threshold_quantile,
        "decluster_run_length": args.decluster_run_length,
        "return_quantiles": args.return_quantiles,
    }


def metadata_matches(existing: dict[str, object], expected: dict[str, object]) -> bool:
    return all(existing.get(key) == value for key, value in expected.items())


def existing_evt_result(
    job: dict[str, str],
    paths: list[Path],
    output_dir: Path,
    expected: dict[str, object],
) -> dict[str, str] | None:
    metadata_path = output_dir / "evt_metadata.json"
    metrics_path = output_dir / "evt_fit_metrics.csv"
    quantiles_path = output_dir / "evt_quantile_estimates.csv"
    required_plots = [
        output_dir / name
        for name in (
            "gev_hist_fit.png", "gev_qq.png", "gev_pp.png",
            "gumbel_hist_fit.png", "gumbel_qq.png", "gumbel_pp.png",
            "gpd_hist_fit.png", "gpd_qq.png", "gpd_pp.png",
            "evt_return_levels.png",
        )
    ]
    if not metadata_path.exists() or not metrics_path.exists() or not quantiles_path.exists():
        return None
    if not all(path.exists() for path in required_plots):
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        with metrics_path.open("r", encoding="utf-8", newline="") as file:
            rows = list(csv.DictReader(file))
    except (OSError, csv.Error, json.JSONDecodeError):
        return None
    if not metadata_matches(metadata, expected):
        return None
    values = {
        "gev_ks_pvalue": "",
        "gumbel_ks_pvalue": "",
        "gpd_ks_pvalue": "",
    }
    for row in rows:
        if row.get("method") == "block_maxima_gev":
            values["gev_ks_pvalue"] = row.get("ks_pvalue", "")
        elif row.get("method") == "block_maxima_gumbel":
            values["gumbel_ks_pvalue"] = row.get("ks_pvalue", "")
        elif row.get("method") == "declustered_pot_gpd":
            values["gpd_ks_pvalue"] = row.get("ks_pvalue", "")
    return {
        "label": job["label"],
        "target": job["target"],
        "source_files": str(len(paths)),
        "rows_loaded": str(metadata.get("rows_loaded", "")),
        "block_maxima": str(metadata.get("block_maxima", "")),
        "cluster_excesses": str(metadata.get("cluster_excesses", "")),
        "threshold": finite_or_empty(float(metadata.get("threshold", math.nan))),
        "gev_ks_pvalue": values["gev_ks_pvalue"],
        "gumbel_ks_pvalue": values["gumbel_ks_pvalue"],
        "gpd_ks_pvalue": values["gpd_ks_pvalue"],
        "metrics_csv": str(metrics_path),
        "quantiles_csv": str(quantiles_path),
        "output_dir": str(output_dir),
        "cached": "true",
    }


def fit_evt_for_job(args: argparse.Namespace, job: dict[str, str], stats) -> dict[str, str]:
    paths = REG.result_paths(args.results_dir, args.first_sweep, job["include_regex"])
    output_dir = Path(job["output_dir"]) / "extreme_values"
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_signature = expected_metadata(args, job, paths)

    if not args.no_cache and not args.only_model and not args.force_model:
        cached = existing_evt_result(job, paths, output_dir, metadata_signature)
        if cached is not None:
            print(f"{job['label']} {job['target']}: cached extreme_values={output_dir}")
            return cached

    values = load_target_series(paths, job["target"])
    maxima = block_maxima(values, args.block_size)
    if len(maxima) < 20:
        raise SystemExit(f"Poucos maximos de bloco para {job['label']} {job['target']}.")

    threshold = float(np.quantile(values, args.threshold_quantile))
    cluster_peaks = decluster_exceedances(values, threshold, args.decluster_run_length)
    excesses = cluster_peaks - threshold
    excesses = excesses[excesses > 0]
    if len(excesses) < 20:
        raise SystemExit(f"Poucos clusters/excessos para {job['label']} {job['target']}.")

    gev_params = stats.genextreme.fit(maxima)
    gumbel_params = stats.gumbel_r.fit(maxima)
    gpd_params = stats.genpareto.fit(excesses, floc=0.0)
    gev_ks = stats.kstest(maxima, "genextreme", args=gev_params)
    gumbel_ks = stats.kstest(maxima, "gumbel_r", args=gumbel_params)
    gpd_ks = stats.kstest(excesses, "genpareto", args=gpd_params)

    run_gev = args.only_model in ("", "gev")
    run_gumbel = args.only_model in ("", "gumbel")
    run_gpd = args.only_model in ("", "gpd")
    quantile_rows: list[dict[str, str]] = []
    for quantile in args.return_quantiles:
        gev_value = float(stats.genextreme.ppf(quantile, *gev_params)) if run_gev else math.nan
        gumbel_value = float(stats.gumbel_r.ppf(quantile, *gumbel_params)) if run_gumbel else math.nan
        if quantile <= args.threshold_quantile:
            gpd_value = threshold
            conditional_quantile = 0.0
        else:
            conditional_quantile = (quantile - args.threshold_quantile) / (1.0 - args.threshold_quantile)
            conditional_quantile = min(max(conditional_quantile, 0.0), 1.0 - 1e-12)
            gpd_value = threshold + float(stats.genpareto.ppf(conditional_quantile, *gpd_params))
        if run_gev:
            quantile_rows.append({
                "method": "block_maxima_gev",
                "quantile": f"{quantile:.6f}",
                "predicted_value": finite_or_empty(gev_value),
                "threshold": "",
                "conditional_excess_quantile": "",
                "block_size": str(args.block_size),
                "decluster_run_length": "",
            })
        if run_gumbel:
            quantile_rows.append({
                "method": "block_maxima_gumbel",
                "quantile": f"{quantile:.6f}",
                "predicted_value": finite_or_empty(gumbel_value),
                "threshold": "",
                "conditional_excess_quantile": "",
                "block_size": str(args.block_size),
                "decluster_run_length": "",
            })
        if run_gpd:
            quantile_rows.append({
                "method": "declustered_pot_gpd",
                "quantile": f"{quantile:.6f}",
                "predicted_value": finite_or_empty(gpd_value),
                "threshold": finite_or_empty(threshold),
                "conditional_excess_quantile": finite_or_empty(conditional_quantile),
                "block_size": "",
                "decluster_run_length": str(args.decluster_run_length),
            })

    write_csv(
        output_dir / "evt_quantile_estimates.csv",
        quantile_rows,
        [
            "method", "quantile", "predicted_value", "threshold",
            "conditional_excess_quantile", "block_size", "decluster_run_length",
        ],
    )

    metrics_rows = []
    if run_gev:
        metrics_rows.append({
            "method": "block_maxima_gev",
            "samples": str(len(maxima)),
            "shape": finite_or_empty(float(gev_params[0])),
            "loc": finite_or_empty(float(gev_params[1])),
            "scale": finite_or_empty(float(gev_params[2])),
            "threshold": "",
            "ks_statistic": finite_or_empty(float(gev_ks.statistic)),
            "ks_pvalue": finite_or_empty(float(gev_ks.pvalue)),
        })
    if run_gumbel:
        metrics_rows.append({
            "method": "block_maxima_gumbel",
            "samples": str(len(maxima)),
            "shape": "0",
            "loc": finite_or_empty(float(gumbel_params[0])),
            "scale": finite_or_empty(float(gumbel_params[1])),
            "threshold": "",
            "ks_statistic": finite_or_empty(float(gumbel_ks.statistic)),
            "ks_pvalue": finite_or_empty(float(gumbel_ks.pvalue)),
        })
    if run_gpd:
        metrics_rows.append({
            "method": "declustered_pot_gpd",
            "samples": str(len(excesses)),
            "shape": finite_or_empty(float(gpd_params[0])),
            "loc": finite_or_empty(float(gpd_params[1])),
            "scale": finite_or_empty(float(gpd_params[2])),
            "threshold": finite_or_empty(threshold),
            "ks_statistic": finite_or_empty(float(gpd_ks.statistic)),
            "ks_pvalue": finite_or_empty(float(gpd_ks.pvalue)),
        })
    write_csv(
        output_dir / "evt_fit_metrics.csv",
        metrics_rows,
        ["method", "samples", "shape", "loc", "scale", "threshold", "ks_statistic", "ks_pvalue"],
    )

    if run_gev:
        plot_hist_fit(output_dir, "gev", maxima, stats.genextreme, gev_params, "GEV sobre maximos de bloco")
        plot_qq(output_dir, "gev", maxima, stats.genextreme, gev_params, "QQ plot - GEV")
        plot_pp(output_dir, "gev", maxima, stats.genextreme, gev_params, "PP plot - GEV")
    if run_gumbel:
        plot_hist_fit(output_dir, "gumbel", maxima, stats.gumbel_r, gumbel_params, "Gumbel sobre maximos de bloco")
        plot_qq(output_dir, "gumbel", maxima, stats.gumbel_r, gumbel_params, "QQ plot - Gumbel")
        plot_pp(output_dir, "gumbel", maxima, stats.gumbel_r, gumbel_params, "PP plot - Gumbel")
    if run_gpd:
        plot_hist_fit(output_dir, "gpd", excesses, stats.genpareto, gpd_params, "GPD sobre excessos declusterizados")
        plot_qq(output_dir, "gpd", excesses, stats.genpareto, gpd_params, "QQ plot - GPD")
        plot_pp(output_dir, "gpd", excesses, stats.genpareto, gpd_params, "PP plot - GPD")
    plot_return_levels(output_dir, "evt", quantile_rows, "Estimativas de pior caso")

    metadata = {
        **metadata_signature,
        "rows_loaded": int(len(values)),
        "threshold": threshold,
        "block_maxima": int(len(maxima)),
        "cluster_excesses": int(len(excesses)),
    }
    (output_dir / "evt_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    return {
        "label": job["label"],
        "target": job["target"],
        "source_files": str(len(paths)),
        "rows_loaded": str(len(values)),
        "block_maxima": str(len(maxima)),
        "cluster_excesses": str(len(excesses)),
        "threshold": finite_or_empty(threshold),
        "gev_ks_pvalue": finite_or_empty(float(gev_ks.pvalue)),
        "gumbel_ks_pvalue": finite_or_empty(float(gumbel_ks.pvalue)),
        "gpd_ks_pvalue": finite_or_empty(float(gpd_ks.pvalue)),
        "metrics_csv": str(output_dir / "evt_fit_metrics.csv"),
        "quantiles_csv": str(output_dir / "evt_quantile_estimates.csv"),
        "output_dir": str(output_dir),
        "cached": "false",
    }


def main() -> int:
    args = parse_args()
    stats = import_scipy_stats()
    rows = [fit_evt_for_job(args, job, stats) for job in load_jobs(args.jobs_file)]
    summary_path = args.analysis_dir / "extreme_value_summary.csv"
    write_csv(
        summary_path,
        rows,
        [
            "label", "target", "source_files", "rows_loaded", "block_maxima",
            "cluster_excesses", "threshold", "gev_ks_pvalue", "gumbel_ks_pvalue", "gpd_ks_pvalue",
            "metrics_csv", "quantiles_csv", "output_dir", "cached",
        ],
    )
    print(f"extreme_value_summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

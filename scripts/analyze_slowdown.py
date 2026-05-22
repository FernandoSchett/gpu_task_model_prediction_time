#!/usr/bin/env python3
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
RESULTS_DIR = REPO_ROOT / "resultados"
REQUIRED_COLUMNS = {
    "experiment_name",
    "cuda_error_code",
    "response_time_us",
    "requested_busy_wait_us",
}


def percentile(series: pd.Series, q: float) -> float:
    return float(series.quantile(q))


def load_result_csvs() -> pd.DataFrame:
    frames = []

    if not RESULTS_DIR.exists():
        print("Pasta resultados/ nao encontrada.")
        return pd.DataFrame()

    for csv_path in sorted(RESULTS_DIR.glob("*.csv")):
        try:
            df = pd.read_csv(csv_path)
        except Exception as exc:
            print(f"Ignorando {csv_path}: erro ao ler CSV: {exc}")
            continue

        missing = REQUIRED_COLUMNS.difference(df.columns)
        if missing:
            continue

        df["source_file"] = csv_path.name
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["cuda_error_code"] = pd.to_numeric(df["cuda_error_code"], errors="coerce")
    df["response_time_us"] = pd.to_numeric(df["response_time_us"], errors="coerce")
    df["requested_busy_wait_us"] = pd.to_numeric(df["requested_busy_wait_us"], errors="coerce")

    df = df[df["cuda_error_code"] == 0]
    df = df.dropna(subset=["response_time_us", "requested_busy_wait_us"])
    df = df[df["requested_busy_wait_us"] > 0]

    df["slowdown"] = df["response_time_us"] / df["requested_busy_wait_us"]
    df["queueing_delay_us"] = df["response_time_us"] - df["requested_busy_wait_us"]
    return df


def summary_row(df: pd.DataFrame) -> dict:
    return {
        "total_kernels": int(len(df)),
        "mean_slowdown": df["slowdown"].mean(),
        "median_slowdown": df["slowdown"].median(),
        "p95_slowdown": percentile(df["slowdown"], 0.95),
        "p99_slowdown": percentile(df["slowdown"], 0.99),
        "max_slowdown": df["slowdown"].max(),
        "mean_response_time_us": df["response_time_us"].mean(),
        "p95_response_time_us": percentile(df["response_time_us"], 0.95),
        "p99_response_time_us": percentile(df["response_time_us"], 0.99),
        "mean_queueing_delay_us": df["queueing_delay_us"].mean(),
        "p95_queueing_delay_us": percentile(df["queueing_delay_us"], 0.95),
        "p99_queueing_delay_us": percentile(df["queueing_delay_us"], 0.99),
    }


def print_summary(title: str, df: pd.DataFrame) -> None:
    print(f"\n=== {title} ===")
    if df.empty:
        print("Sem dados validos.")
        return

    summary = summary_row(df)
    for key, value in summary.items():
        if key == "total_kernels":
            print(f"{key}: {value}")
        else:
            print(f"{key}: {value:.6f}")


def print_grouped_summary(df: pd.DataFrame) -> None:
    print("\n=== Resumo por experiment_name ===")
    if df.empty:
        print("Sem dados validos.")
        return

    grouped = []
    for experiment_name, group in df.groupby("experiment_name", sort=True):
        row = {"experiment_name": experiment_name}
        row.update(summary_row(group))
        grouped.append(row)

    summary_df = pd.DataFrame(grouped)
    with pd.option_context(
        "display.max_rows",
        None,
        "display.max_columns",
        None,
        "display.width",
        240,
    ):
        print(summary_df.to_string(index=False, float_format=lambda value: f"{value:.6f}"))


def compare_baseline_stress(df: pd.DataFrame) -> None:
    experiment_names = set(df["experiment_name"].astype(str).unique())
    if "baseline" not in experiment_names or "stress" not in experiment_names:
        return

    baseline = df[df["experiment_name"] == "baseline"]
    stress = df[df["experiment_name"] == "stress"]
    if baseline.empty or stress.empty:
        return

    baseline_p95 = percentile(baseline["slowdown"], 0.95)
    stress_p95 = percentile(stress["slowdown"], 0.95)
    ratio = stress_p95 / baseline_p95 if baseline_p95 != 0 else float("inf")

    print("\n=== Comparacao baseline vs stress ===")
    print(f"baseline_p95_slowdown: {baseline_p95:.6f}")
    print(f"stress_p95_slowdown: {stress_p95:.6f}")
    print(f"stress_p95 / baseline_p95: {ratio:.6f}")

    if ratio < 1.5:
        print("Pouco sinal de slowdown. A carga pode estar leve demais.")
    elif ratio < 5:
        print("Ha sinal de slowdown moderado.")
    else:
        print("Ha sinal forte de slowdown/concorrencia.")


def main() -> int:
    df = load_result_csvs()
    if df.empty:
        print("Nenhum CSV de resultados valido encontrado em resultados/.")
        return 1

    df = prepare_data(df)
    if df.empty:
        print("Nenhuma linha valida apos filtrar cuda_error_code == 0.")
        return 1

    print_summary("Resumo geral", df)
    print_grouped_summary(df)
    compare_baseline_stress(df)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

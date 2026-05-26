#!/usr/bin/env python3
import csv
from collections import defaultdict
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
RESULTS_DIR = REPO_ROOT / "resultados"
REQUIRED_COLUMNS = {
    "experiment_name",
    "cuda_error_code",
    "response_time_us",
    "requested_busy_wait_us",
}


def to_float(row, key):
    try:
        return float(row.get(key, ""))
    except ValueError:
        return None


def percentile(values, q):
    if not values:
        return 0.0

    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def mean(values):
    return sum(values) / len(values) if values else 0.0


def median(values):
    return percentile(values, 0.5)


def load_result_rows():
    rows = []

    if not RESULTS_DIR.exists():
        print("Pasta resultados/ nao encontrada.")
        return rows

    for csv_path in sorted(RESULTS_DIR.glob("*.csv")):
        try:
            with csv_path.open("r", encoding="utf-8", newline="") as file:
                reader = csv.DictReader(file)
                if reader.fieldnames is None:
                    continue
                if REQUIRED_COLUMNS.difference(reader.fieldnames):
                    continue
                for row in reader:
                    row["source_file"] = csv_path.name
                    rows.append(row)
        except Exception as exc:
            print(f"Ignorando {csv_path}: erro ao ler CSV: {exc}")

    return rows


def prepare_data(rows):
    prepared = []
    for row in rows:
        cuda_error_code = to_float(row, "cuda_error_code")
        response_time_us = to_float(row, "response_time_us")
        requested_busy_wait_us = to_float(row, "requested_busy_wait_us")
        if cuda_error_code != 0 or response_time_us is None or requested_busy_wait_us is None:
            continue
        if requested_busy_wait_us <= 0:
            continue

        row = dict(row)
        row["response_time_us_value"] = response_time_us
        row["requested_busy_wait_us_value"] = requested_busy_wait_us
        row["slowdown_value"] = response_time_us / requested_busy_wait_us
        row["queueing_delay_us_value"] = response_time_us - requested_busy_wait_us
        prepared.append(row)

    return prepared


def summary_row(rows):
    slowdowns = [row["slowdown_value"] for row in rows]
    response_times = [row["response_time_us_value"] for row in rows]
    queueing_delays = [row["queueing_delay_us_value"] for row in rows]
    return {
        "total_kernels": len(rows),
        "mean_slowdown": mean(slowdowns),
        "median_slowdown": median(slowdowns),
        "p95_slowdown": percentile(slowdowns, 0.95),
        "p99_slowdown": percentile(slowdowns, 0.99),
        "max_slowdown": max(slowdowns) if slowdowns else 0.0,
        "mean_response_time_us": mean(response_times),
        "p95_response_time_us": percentile(response_times, 0.95),
        "p99_response_time_us": percentile(response_times, 0.99),
        "mean_queueing_delay_us": mean(queueing_delays),
        "p95_queueing_delay_us": percentile(queueing_delays, 0.95),
        "p99_queueing_delay_us": percentile(queueing_delays, 0.99),
    }


def print_summary(title, rows):
    print(f"\n=== {title} ===")
    if not rows:
        print("Sem dados validos.")
        return

    summary = summary_row(rows)
    for key, value in summary.items():
        if key == "total_kernels":
            print(f"{key}: {value}")
        else:
            print(f"{key}: {value:.6f}")


def print_grouped_summary(rows):
    print("\n=== Resumo por experiment_name ===")
    if not rows:
        print("Sem dados validos.")
        return

    grouped = defaultdict(list)
    for row in rows:
        grouped[row["experiment_name"]].append(row)

    columns = [
        "experiment_name",
        "total_kernels",
        "mean_slowdown",
        "median_slowdown",
        "p95_slowdown",
        "p99_slowdown",
        "max_slowdown",
        "mean_response_time_us",
        "p95_response_time_us",
        "p99_response_time_us",
        "mean_queueing_delay_us",
        "p95_queueing_delay_us",
        "p99_queueing_delay_us",
    ]
    print(" ".join(f"{column:>24}" for column in columns))
    for experiment_name in sorted(grouped):
        row = {"experiment_name": experiment_name}
        row.update(summary_row(grouped[experiment_name]))
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, int):
                values.append(f"{value:>24}")
            elif isinstance(value, float):
                values.append(f"{value:>24.6f}")
            else:
                values.append(f"{value:>24}")
        print(" ".join(values))


def compare_baseline_stress(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["experiment_name"]].append(row)

    baseline = grouped.get("baseline", [])
    stress = grouped.get("stress", [])
    if not baseline or not stress:
        return

    baseline_p95 = percentile([row["slowdown_value"] for row in baseline], 0.95)
    stress_p95 = percentile([row["slowdown_value"] for row in stress], 0.95)
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


def main():
    rows = load_result_rows()
    if not rows:
        print("Nenhum CSV de resultados valido encontrado em resultados/.")
        return 1

    rows = prepare_data(rows)
    if not rows:
        print("Nenhuma linha valida apos filtrar cuda_error_code == 0.")
        return 1

    print_summary("Resumo geral", rows)
    print_grouped_summary(rows)
    compare_baseline_stress(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

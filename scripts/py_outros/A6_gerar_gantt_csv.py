#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_RESULTS_DIR = REPO_ROOT / "resultados"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a Gantt-friendly CSV from CUDA benchmark result CSVs."
    )
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_RESULTS_DIR / "gantt_intervals.csv")
    return parser.parse_args()


def to_int(row, key, default=0):
    try:
        value = row.get(key, "")
        return int(value) if value != "" else default
    except ValueError:
        return default


def to_float(row, key, default=0.0):
    try:
        value = row.get(key, "")
        return float(value) if value != "" else default
    except ValueError:
        return default


def result_csv_paths(results_dir):
    return sorted(path for path in results_dir.rglob("*.csv") if path.name.startswith("resultados_experimentos_"))


def add_interval(intervals, row, interval_type, start_ns, end_ns, origin_ns, source_file):
    if start_ns <= 0 or end_ns <= 0 or end_ns < start_ns:
        return

    start_us = (start_ns - origin_ns) / 1000.0
    end_us = (end_ns - origin_ns) / 1000.0
    intervals.append(
        {
            "interval_type": interval_type,
            "experiment_name": row.get("experiment_name", ""),
            "global_kernel_id": row.get("global_kernel_id", ""),
            "mpi_rank": row.get("mpi_rank", ""),
            "host_thread_id": row.get("host_thread_id", ""),
            "logical_stream_id": row.get("logical_stream_id", row.get("host_thread_id", "")),
            "lane": f"rank{row.get('mpi_rank', '')}.thread{row.get('host_thread_id', '')}",
            "kernel_type": row.get("kernel_type", ""),
            "blocks_x": row.get("blocks_x", ""),
            "threads_per_block": row.get("threads_per_block", ""),
            "requested_busy_wait_us": row.get("requested_busy_wait_us", ""),
            "start_time_ns": start_ns,
            "end_time_ns": end_ns,
            "start_time_us": f"{start_us:.3f}",
            "end_time_us": f"{end_us:.3f}",
            "duration_us": f"{end_us - start_us:.3f}",
            "response_time_us": row.get("response_time_us", ""),
            "queueing_delay_us": row.get("queueing_delay_us", ""),
            "slowdown": row.get("slowdown", ""),
            "cuda_error_code": row.get("cuda_error_code", ""),
            "source_file": source_file,
        }
    )


def main():
    args = parse_args()
    rows = []
    for csv_path in result_csv_paths(args.results_dir):
        with csv_path.open("r", encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            if reader.fieldnames is None or "submit_time_ns" not in reader.fieldnames:
                continue
            for row in reader:
                row["source_file"] = csv_path.name
                rows.append(row)

    if not rows:
        print(f"Nenhum CSV de resultado encontrado em {args.results_dir}.")
        return 1

    origin_ns = min(to_int(row, "measurement_start_time_ns", to_int(row, "submit_time_ns")) for row in rows)
    intervals = []
    for row in rows:
        submit_ns = to_int(row, "submit_time_ns")
        completion_ns = to_int(row, "completion_time_ns")
        source_file = row["source_file"]

        add_interval(intervals, row, "host_response", submit_ns, completion_ns, origin_ns, source_file)

    intervals.sort(key=lambda item: (float(item["start_time_us"]), item["lane"], item["interval_type"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "interval_type",
        "experiment_name",
        "global_kernel_id",
        "mpi_rank",
        "host_thread_id",
        "logical_stream_id",
        "lane",
        "kernel_type",
        "blocks_x",
        "threads_per_block",
        "requested_busy_wait_us",
        "start_time_ns",
        "end_time_ns",
        "start_time_us",
        "end_time_us",
        "duration_us",
        "response_time_us",
        "queueing_delay_us",
        "slowdown",
        "cuda_error_code",
        "source_file",
    ]
    with args.output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(intervals)

    print(f"Escrevi {len(intervals)} intervalos em {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

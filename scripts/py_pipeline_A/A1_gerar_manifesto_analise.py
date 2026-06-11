#!/usr/bin/env python3
"""Prepare regression analysis folders and manifests for one or more sweep directories."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


DEFAULT_TARGETS = ("response_time_us", "queueing_delay_us", "slowdown")
DEFAULT_GPU_TARGETS = ("10", "50", "100", "120")
DEFAULT_KERNEL_TYPES = ("busy_wait", "compute", "memory", "mixed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, nargs="+", required=True)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--targets", nargs="+", default=list(DEFAULT_TARGETS))
    parser.add_argument("--gpu-targets", nargs="+", default=list(DEFAULT_GPU_TARGETS))
    parser.add_argument("--kernel-types", nargs="+", default=list(DEFAULT_KERNEL_TYPES))
    return parser.parse_args()


def count_csv_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as file:
        return max(0, sum(1 for _ in file) - 1)


def target_from_name(name: str) -> str:
    match = re.search(r"_gputarget([0-9]+(?:p[0-9]+)?)_", name)
    return match.group(1).replace("p", ".") if match else ""


def kernel_type_from_name(name: str) -> str:
    match = re.search(r"_kt([^_]+(?:_[^_]+)*)_bx", name)
    return match.group(1) if match else ""


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    result_files = sorted(
        path
        for results_dir in args.results_dir
        for path in results_dir.rglob("resultados_experimentos_*gputarget*.csv")
    )
    if not result_files:
        raise SystemExit(f"Nenhum CSV gputarget encontrado em {', '.join(str(path) for path in args.results_dir)}")

    args.analysis_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[tuple[str, str], dict[str, int]] = {}
    total_rows = 0
    for path in result_files:
        gpu_target = target_from_name(path.name) or "unknown"
        kernel_type = kernel_type_from_name(path.name) or "unknown"
        rows = count_csv_rows(path)
        total_rows += rows
        for key in ((gpu_target, "all"), ("all", kernel_type), (gpu_target, kernel_type)):
            summary.setdefault(key, {"files": 0, "rows": 0})
            summary[key]["files"] += 1
            summary[key]["rows"] += rows

    summary_rows = [
        {
            "label": "geral",
            "gpu_target": "all",
            "kernel_type": "all",
            "files": str(len(result_files)),
            "rows": str(total_rows),
        }
    ]

    for (gpu_target, kernel_type), values in sorted(summary.items()):
        if gpu_target == "all" and kernel_type == "all":
            continue
        if gpu_target == "all":
            label = f"kernel_{kernel_type}"
        elif kernel_type == "all":
            label = f"perfil_gpu_{gpu_target}"
        else:
            label = f"perfil_gpu_{gpu_target}_kernel_{kernel_type}"
        summary_rows.append(
            {
                "label": label,
                "gpu_target": gpu_target,
                "kernel_type": kernel_type,
                "files": str(values["files"]),
                "rows": str(values["rows"]),
            }
        )
    write_csv(
        args.analysis_dir / "dataset_summary.csv",
        summary_rows,
        ["label", "gpu_target", "kernel_type", "files", "rows"],
    )

    jobs: list[dict[str, str]] = []
    labels = [("geral", "gputarget")]
    labels.extend(
        (f"perfil_gpu_{gpu_target}", rf"gputarget{gpu_target}(_|[^0-9])")
        for gpu_target in args.gpu_targets
    )
    labels.extend(
        (f"kernel_{kernel_type}", rf"_kt{kernel_type}_")
        for kernel_type in args.kernel_types
    )
    labels.extend(
        (
            f"perfil_gpu_{gpu_target}_kernel_{kernel_type}",
            rf"gputarget{gpu_target}(_|[^0-9]).*_kt{kernel_type}_",
        )
        for gpu_target in args.gpu_targets
        for kernel_type in args.kernel_types
    )
    for label, include_regex in labels:
        for target in args.targets:
            output_dir = args.analysis_dir / label / target
            output_dir.mkdir(parents=True, exist_ok=True)
            jobs.append(
                {
                    "label": label,
                    "target": target,
                    "include_regex": include_regex,
                    "output_dir": str(output_dir),
                }
            )

    write_csv(args.analysis_dir / "analysis_jobs.csv", jobs, ["label", "target", "include_regex", "output_dir"])
    print(f"results_dir: {', '.join(str(path) for path in args.results_dir)}")
    print(f"analysis_dir: {args.analysis_dir}")
    print(f"source_files: {len(result_files)}")
    print(f"source_rows: {total_rows}")
    print(f"jobs: {len(jobs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

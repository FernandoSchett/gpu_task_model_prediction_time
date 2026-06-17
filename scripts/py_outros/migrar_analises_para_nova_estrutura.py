#!/usr/bin/env python3
"""Migrate old analysis folders to the pipeline-oriented result structure."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


OLD_CONDITIONS = {
    "sem_telemetria_sweep_moderado_sem_estimativas_agrupado": "sem_telemetria",
    "com_telemetria_sweep_moderado_sem_estimativas_agrupado": "com_telemetria",
}
TARGETS = {"response_time_us", "queueing_delay_us", "slowdown"}
DEPENDENCY_FILES = {
    "dependency_metrics.csv",
    "dependency_feature_pearson.png",
    "dependency_acf.png",
}
PIPELINE_A_ROOT_FILES = {
    "analysis_jobs.csv",
    "dataset_summary.csv",
    "training_summary.csv",
    "sequential_summary.csv",
}
PIPELINE_C_ROOT_FILES = {
    "pipeline_c_preprocess_summary.csv",
    "pipeline_c_cnn2d_training_summary.csv",
}
PIPELINE_B_ROOT_FILES = {
    "extreme_value_summary.csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-root", type=Path, default=Path("resultados/analises_regressao"))
    return parser.parse_args()


def move_path(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        if src.is_dir() and dst.is_dir():
            moved = False
            for child in sorted(src.iterdir()):
                moved = move_path(child, dst / child.name) or moved
            return moved
        print(f"skip exists: {dst}")
        return False
    shutil.move(str(src), str(dst))
    print(f"moved: {src} -> {dst}")
    return True


def move_root_rankings(root: Path) -> None:
    mappings = [
        (
            root / "melhores_modelos_nao_sequenciais" / "best_model_rankings.csv",
            root / "pipeline_A" / "rankings" / "melhores_modelos_nao_sequenciais.csv",
        ),
        (
            root / "melhores_modelos_sequenciais" / "best_model_rankings.csv",
            root / "pipeline_A" / "rankings" / "melhores_modelos_sequenciais.csv",
        ),
        (
            root / "pipeline_a_model_rankings" / "best_model_rankings.csv",
            root / "pipeline_A" / "rankings" / "melhores_modelos_pipeline_A.csv",
        ),
        (
            root / "2d_models" / "best_model_rankings.csv",
            root / "pipeline_C" / "rankings" / "melhores_modelos_2d.csv",
        ),
    ]
    for src, dst in mappings:
        move_path(src, dst)

    old_rank_dirs = [
        (root / "melhores_modelos_nao_sequenciais", root / "pipeline_A" / "rankings"),
        (root / "melhores_modelos_sequenciais", root / "pipeline_A" / "rankings"),
        (root / "pipeline_a_model_rankings", root / "pipeline_A" / "rankings"),
        (root / "2d_models", root / "pipeline_C" / "rankings"),
    ]
    for src_dir, dst_dir in old_rank_dirs:
        if not src_dir.exists():
            continue
        for child in sorted(src_dir.iterdir()):
            move_path(child, dst_dir / child.name)

    for child in sorted(root.iterdir()):
        if child.name == "best_model_rankings.csv" or child.name.startswith("best_model_"):
            move_path(child, root / "comparacoes_pipelines" / child.name)
        elif child.name == "dependency_rankings.csv" or child.name.startswith("dependency_"):
            move_path(child, root / "analise_dependencia" / child.name)


def migrate_target_dir(old_target_dir: Path, root: Path, condition: str, label: str, target: str) -> None:
    pipeline_a_target = root / "pipeline_A" / condition / label / target
    pipeline_b_target = root / "pipeline_B" / condition / label / target
    pipeline_c_target = root / "pipeline_C" / condition / label / target
    dependency_target = root / "analise_dependencia" / condition / label / target

    move_path(old_target_dir / "sequential_models", pipeline_a_target / "sequenciais")
    move_path(old_target_dir / "sequenciais", pipeline_a_target / "sequenciais")
    move_path(old_target_dir / "extreme_value", pipeline_b_target / "extreme_values")
    move_path(old_target_dir / "extreme_values", pipeline_b_target / "extreme_values")
    move_path(old_target_dir / "2d_models", pipeline_c_target / "2d_models")

    for name in DEPENDENCY_FILES:
        move_path(old_target_dir / name, dependency_target / name)

    classical_dir = pipeline_a_target / "nao_sequenciais"
    for child in sorted(old_target_dir.iterdir()):
        if child.name in DEPENDENCY_FILES:
            continue
        move_path(child, classical_dir / child.name)


def migrate_condition(root: Path, old_name: str, condition: str) -> None:
    old_dir = root / old_name
    if not old_dir.exists():
        return
    pipeline_a_condition = root / "pipeline_A" / condition
    pipeline_b_condition = root / "pipeline_B" / condition
    pipeline_c_condition = root / "pipeline_C" / condition
    dependency_condition = root / "analise_dependencia" / condition

    for name in PIPELINE_A_ROOT_FILES:
        move_path(old_dir / name, pipeline_a_condition / name)
    for name in PIPELINE_B_ROOT_FILES:
        move_path(old_dir / name, pipeline_b_condition / name)
    for name in PIPELINE_C_ROOT_FILES:
        move_path(old_dir / name, pipeline_c_condition / name)
    move_path(old_dir / "dependency_summary.csv", dependency_condition / "dependency_summary.csv")

    for label_dir in sorted(old_dir.iterdir()):
        if not label_dir.is_dir():
            continue
        for target_dir in sorted(label_dir.iterdir()):
            if target_dir.is_dir() and target_dir.name in TARGETS:
                migrate_target_dir(target_dir, root, condition, label_dir.name, target_dir.name)


def main() -> int:
    args = parse_args()
    root = args.analysis_root
    root.mkdir(parents=True, exist_ok=True)
    for old_name, condition in OLD_CONDITIONS.items():
        migrate_condition(root, old_name, condition)
    move_root_rankings(root)
    print(f"migration_done: {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

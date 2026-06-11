#!/usr/bin/env python3
"""Train sequence models (LSTM, GRU and Temporal CNN) for CUDA timing targets."""

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
REPO_ROOT = SCRIPT_DIR.parent.parent
REGRESSOR_SCRIPT = SCRIPT_DIR / "A2_regressores_classicos.py"
DEFAULT_ANALYSIS_ROOT = REPO_ROOT / "resultados" / "analises_sequenciais"


def load_regressor_module():
    spec = importlib.util.spec_from_file_location("regressor_analysis", REGRESSOR_SCRIPT)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Nao foi possivel carregar {REGRESSOR_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


REG = load_regressor_module()
FEATURES = list(REG.BASE_FEATURES) + [f"kernel_type_{name}" for name in REG.KERNEL_TYPES]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, nargs="+", required=True)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--jobs-file", type=Path, required=True)
    parser.add_argument("--first-sweep", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-fraction", type=float, default=0.25)
    parser.add_argument("--sequence-length", type=int, default=int(os.getenv("SEQUENCE_LENGTH", "16")))
    parser.add_argument("--sequence-stride", type=int, default=int(os.getenv("SEQUENCE_STRIDE", "4")))
    parser.add_argument("--max-sequences", type=int, default=int(os.getenv("SEQUENCE_MAX_SEQUENCES", "200000")))
    parser.add_argument("--epochs", type=int, default=int(os.getenv("SEQUENCE_EPOCHS", "5")))
    parser.add_argument("--batch-size", type=int, default=int(os.getenv("SEQUENCE_BATCH_SIZE", "256")))
    parser.add_argument("--no-cache", action="store_true", help="Recompute even when sequential outputs already exist.")
    return parser.parse_args()


def import_tensorflow():
    try:
        import tensorflow as tf
        from tensorflow import keras
    except ImportError as exc:
        raise SystemExit(
            "TensorFlow nao esta instalado. Instale com `pip install tensorflow` "
            "ou adicione tensorflow ao ambiente antes de rodar a pipeline A."
        ) from exc
    return tf, keras


def load_jobs(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def load_ordered_matrix(paths: list[Path], target: str) -> tuple[np.ndarray, np.ndarray]:
    x_rows: list[list[float]] = []
    y_values: list[float] = []
    for path in paths:
        with path.open("r", encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            if reader.fieldnames is None or "response_time_us" not in reader.fieldnames:
                continue
            for row in reader:
                if REG.to_float(row, "cuda_error_code", 0.0) != 0.0:
                    continue
                values = REG.row_features_compare(row)
                y = values[target] if target in values else REG.to_float(row, target)
                x = [values[name] for name in FEATURES]
                if math.isfinite(y) and all(math.isfinite(value) for value in x):
                    x_rows.append(x)
                    y_values.append(y)
    return np.asarray(x_rows, dtype=float), np.asarray(y_values, dtype=float)


def make_sequences(
    x: np.ndarray,
    y: np.ndarray,
    sequence_length: int,
    stride: int,
    max_sequences: int,
) -> tuple[np.ndarray, np.ndarray]:
    if sequence_length < 2:
        raise SystemExit("--sequence-length precisa ser >= 2.")
    if stride < 1:
        raise SystemExit("--sequence-stride precisa ser >= 1.")
    count = max(0, (len(y) - sequence_length) // stride + 1)
    if count <= 0:
        return np.empty((0, sequence_length, x.shape[1]), dtype=float), np.empty((0,), dtype=float)
    indices = np.arange(0, count * stride, stride, dtype=int)
    if max_sequences > 0 and len(indices) > max_sequences:
        indices = np.linspace(0, indices[-1], num=max_sequences, dtype=int)
    seq_x = np.empty((len(indices), sequence_length, x.shape[1]), dtype=np.float32)
    seq_y = np.empty((len(indices),), dtype=np.float32)
    for out_index, start in enumerate(indices):
        end = start + sequence_length
        seq_x[out_index] = x[start:end]
        seq_y[out_index] = y[end - 1]
    return seq_x, seq_y


def chronological_split(
    x: np.ndarray,
    y: np.ndarray,
    test_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not 0.0 < test_fraction < 1.0:
        raise SystemExit("--test-fraction precisa estar entre 0 e 1.")
    test_size = max(1, int(round(len(y) * test_fraction)))
    split = len(y) - test_size
    if split <= 0:
        raise SystemExit("Poucas sequencias para criar treino/teste.")
    return x[:split], x[split:], y[:split], y[split:]


def standardize_sequence(
    train_x: np.ndarray,
    test_x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = train_x.reshape(-1, train_x.shape[-1]).mean(axis=0)
    scale = train_x.reshape(-1, train_x.shape[-1]).std(axis=0)
    scale[scale == 0.0] = 1.0
    return (train_x - mean) / scale, (test_x - mean) / scale, mean, scale


def build_models(keras, sequence_length: int, n_features: int) -> dict[str, object]:
    def compile_model(model):
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="mse", metrics=["mae"])
        return model

    return {
        "lstm": compile_model(keras.Sequential([
            keras.layers.Input(shape=(sequence_length, n_features)),
            keras.layers.LSTM(64),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dense(1),
        ])),
        "gru": compile_model(keras.Sequential([
            keras.layers.Input(shape=(sequence_length, n_features)),
            keras.layers.GRU(64),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dense(1),
        ])),
        "temporal_cnn": compile_model(keras.Sequential([
            keras.layers.Input(shape=(sequence_length, n_features)),
            keras.layers.Conv1D(64, kernel_size=3, padding="causal", activation="relu"),
            keras.layers.Conv1D(64, kernel_size=3, padding="causal", activation="relu"),
            keras.layers.GlobalAveragePooling1D(),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dense(1),
        ])),
    }


def plot_loss(output_dir: Path, model_name: str, history) -> Path:
    path = output_dir / f"{model_name}_training_loss.png"
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(history.history.get("loss", []), label="treino")
    if "val_loss" in history.history:
        ax.plot(history.history["val_loss"], label="validacao")
    ax.set_title(f"Loss - {model_name}")
    ax.set_xlabel("Epoca")
    ax.set_ylabel("MSE")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def write_metrics(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model", "MAE", "RMSE", "R2", "train_sequences", "test_sequences",
        "sequence_length", "sequence_stride", "diagnostics_dir",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def expected_metadata(args: argparse.Namespace, job: dict[str, str], paths: list[Path]) -> dict[str, object]:
    return {
        "target": job["target"],
        "label": job["label"],
        "source_files": len(paths),
        "source_paths": [str(path) for path in paths],
        "sequence_length": args.sequence_length,
        "sequence_stride": args.sequence_stride,
        "max_sequences": args.max_sequences,
        "test_fraction": args.test_fraction,
        "features": FEATURES,
    }


def metadata_matches(existing: dict[str, object], expected: dict[str, object]) -> bool:
    return all(existing.get(key) == value for key, value in expected.items())


def existing_sequential_result(
    job: dict[str, str],
    paths: list[Path],
    seq_output_dir: Path,
    expected: dict[str, object],
) -> dict[str, str] | None:
    metrics_path = seq_output_dir / "sequential_metrics.csv"
    metadata_path = seq_output_dir / "sequence_metadata.json"
    required_models = [seq_output_dir / f"{name}.keras" for name in ("lstm", "gru", "temporal_cnn")]
    required_plots = [
        seq_output_dir / "model_diagnostics" / f"{name}_{suffix}.png"
        for name in ("lstm", "gru", "temporal_cnn")
        for suffix in ("error_distribution", "predicted_vs_actual", "error_qq_normal")
    ]
    if not metrics_path.exists() or not metadata_path.exists():
        return None
    if not all(path.exists() for path in required_models + required_plots):
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        with metrics_path.open("r", encoding="utf-8", newline="") as file:
            rows = list(csv.DictReader(file))
    except (OSError, csv.Error, json.JSONDecodeError):
        return None
    if not metadata_matches(metadata, expected) or not rows:
        return None
    best = min(rows, key=lambda row: float(row["RMSE"]))
    return {
        "label": job["label"],
        "target": job["target"],
        "source_files": str(len(paths)),
        "rows_loaded": str(metadata.get("rows_loaded", "")),
        "train_sequences": str(metadata.get("train_sequences", "")),
        "test_sequences": str(metadata.get("test_sequences", "")),
        "best_model": best["model"],
        "best_rmse": best["RMSE"],
        "best_r2": best["R2"],
        "metrics_csv": str(metrics_path),
        "output_dir": str(seq_output_dir),
        "cached": "true",
    }


def run_job(args: argparse.Namespace, job: dict[str, str], keras) -> dict[str, str]:
    paths = REG.result_paths(args.results_dir, args.first_sweep, job["include_regex"])
    output_dir = Path(job["output_dir"])
    seq_output_dir = output_dir / "sequential_models"
    diagnostics_dir = seq_output_dir / "model_diagnostics"
    seq_output_dir.mkdir(parents=True, exist_ok=True)
    metadata_signature = expected_metadata(args, job, paths)

    if not args.no_cache:
        cached = existing_sequential_result(job, paths, seq_output_dir, metadata_signature)
        if cached is not None:
            print(f"{job['label']} {job['target']}: cached sequential_models={seq_output_dir}")
            return cached

    x, y = load_ordered_matrix(paths, job["target"])
    seq_x, seq_y = make_sequences(x, y, args.sequence_length, args.sequence_stride, args.max_sequences)
    if len(seq_y) == 0:
        raise SystemExit(f"Nenhuma sequencia valida para {job['label']} {job['target']}.")
    train_x, test_x, train_y, test_y = chronological_split(seq_x, seq_y, args.test_fraction)
    train_x, test_x, mean, scale = standardize_sequence(train_x, test_x)

    metadata = {
        **metadata_signature,
        "rows_loaded": int(len(y)),
        "train_sequences": int(len(train_y)),
        "test_sequences": int(len(test_y)),
        "standardize": {"mean": mean.tolist(), "scale": scale.tolist()},
    }
    (seq_output_dir / "sequence_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    rows: list[dict[str, str]] = []
    residual_metrics_path = diagnostics_dir / "residual_distribution_metrics.csv"
    if residual_metrics_path.exists():
        residual_metrics_path.unlink()
    for model_name, model in build_models(keras, args.sequence_length, train_x.shape[-1]).items():
        history = model.fit(
            train_x,
            train_y,
            validation_split=0.1,
            epochs=args.epochs,
            batch_size=args.batch_size,
            verbose=0,
            shuffle=False,
        )
        prediction = model.predict(test_x, batch_size=args.batch_size, verbose=0).reshape(-1)
        metric_row = REG.metrics(test_y, prediction)
        REG.plot_prediction_diagnostics(seq_output_dir, model_name, test_y, prediction)
        plot_loss(seq_output_dir, model_name, history)
        model.save(seq_output_dir / f"{model_name}.keras")
        rows.append({
            "model": model_name,
            "MAE": f"{metric_row['MAE']:.6f}",
            "RMSE": f"{metric_row['RMSE']:.6f}",
            "R2": f"{metric_row['R2']:.6f}",
            "train_sequences": str(len(train_y)),
            "test_sequences": str(len(test_y)),
            "sequence_length": str(args.sequence_length),
            "sequence_stride": str(args.sequence_stride),
            "diagnostics_dir": str(diagnostics_dir),
        })

    metrics_path = seq_output_dir / "sequential_metrics.csv"
    write_metrics(metrics_path, rows)
    best = min(rows, key=lambda row: float(row["RMSE"]))
    return {
        "label": job["label"],
        "target": job["target"],
        "source_files": str(len(paths)),
        "rows_loaded": str(len(y)),
        "train_sequences": rows[0]["train_sequences"],
        "test_sequences": rows[0]["test_sequences"],
        "best_model": best["model"],
        "best_rmse": best["RMSE"],
        "best_r2": best["R2"],
        "metrics_csv": str(metrics_path),
        "output_dir": str(seq_output_dir),
        "cached": "false",
    }


def main() -> int:
    args = parse_args()
    tf, keras = import_tensorflow()
    tf.keras.utils.set_random_seed(args.seed)
    rows = [run_job(args, job, keras) for job in load_jobs(args.jobs_file)]
    summary_path = args.analysis_dir / "sequential_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as file:
        fieldnames = [
            "label", "target", "source_files", "rows_loaded", "train_sequences",
            "test_sequences", "best_model", "best_rmse", "best_r2", "metrics_csv", "output_dir", "cached",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"sequential_summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
